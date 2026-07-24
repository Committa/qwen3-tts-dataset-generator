"""Step 2: batch audio generation with Qwen3-TTS, checkpoint/resume and OOM handling."""

from __future__ import annotations

import logging
import time
from typing import Any

import soundfile as sf
from tqdm import tqdm

from . import common

logger = logging.getLogger(__name__)


def _build_dtype(dtype_str: str):
    import torch

    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_str not in mapping:
        raise ValueError(f"Invalid dtype: {dtype_str}. Use bfloat16 or float16.")
    return mapping[dtype_str]


def load_tts_model(cfg: common.Config):
    """Load the Qwen3-TTS model.

    Verifies CUDA, loads the model selected by ``cfg.model_hub_id`` with the
    configured dtype/device/attention backend, seeds torch for reproducibility,
    and asserts the loaded model type matches the config. On OOM it logs the
    actionable hint and exits with code 2.

    Args:
        cfg: Pipeline configuration.

    Returns:
        The loaded ``Qwen3TTSModel`` instance.

    Raises:
        SystemExit(2): On GPU out-of-memory.
    """
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise RuntimeError("qwen-tts is not installed. Run `poetry install`.") from e

    common.check_cuda_or_die(logger)
    model_id = cfg.model_hub_id
    logger.info(
        "Loading TTS model %s (dtype=%s, device_map=%s)",
        model_id,
        cfg.dtype,
        cfg.device_map,
    )
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=cfg.device_map,
            dtype=_build_dtype(cfg.dtype),
            attn_implementation=cfg.attn_implementation,
        )
    except MemoryError as e:
        common.exit_on_oom(e, logger)
    except Exception as e:
        if common.is_oom_error(e):
            common.exit_on_oom(e, logger)
        logger.error("Error loading TTS model: %s", e)
        raise
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    actual_type = getattr(model.model, "tts_model_type", None)
    if actual_type != cfg.model_type:
        raise RuntimeError(
            f"Model type mismatch: config model_type='{cfg.model_type}' "
            f"but loaded model tts_model_type='{actual_type}'. "
            f"Check model_type/model_size in config.yaml."
        )
    if cfg.model_type == "custom_voice":
        logger.info("Available speakers: %s", ", ".join(model.get_supported_speakers()))
    else:
        mode = "x-vector-only" if cfg.x_vector_only_mode else "ICL"
        logger.info(
            "Voice clone mode: mode=%s speaker=%s (reference resolved per-voice)",
            mode,
            cfg.speaker,
        )
    return model


def _rebuild_prompt_item(d: dict) -> "object":
    """Rebuild a VoiceClonePromptItem from a serialized dict.

    Args:
        d: Dict with ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text.

    Returns:
        A VoiceClonePromptItem instance.
    """
    import torch
    from qwen_tts import VoiceClonePromptItem

    ref_code = d.get("ref_code", None)
    if ref_code is not None and not torch.is_tensor(ref_code):
        ref_code = torch.tensor(ref_code)
    ref_spk = d.get("ref_spk_embedding", None)
    if ref_spk is None:
        raise ValueError("Missing ref_spk_embedding in cached voice prompt.")
    if not torch.is_tensor(ref_spk):
        ref_spk = torch.tensor(ref_spk)
    return VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=ref_spk,
        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
        icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
        ref_text=d.get("ref_text", None),
    )


def get_voice_clone_prompt(model, cfg: common.Config, use_cache: bool = True) -> list:
    """Build or load a cached voice-clone prompt for the configured custom voice.

    In ICL mode (x_vector_only_mode=False) the reference transcript ref.txt is
    required; in x-vector-only mode it is ignored. The resulting
    VoiceClonePromptItem is cached to cfg.paths.prompt_cache as
    <speaker>_<model_size>.pt, keyed by a fingerprint of the
    reference audio, transcript, mode and model, so that
    re-runs skip the prompt extraction.

    Args:
        model: Loaded Qwen3TTSModel (must be of base type).
        cfg: Pipeline configuration.
        use_cache: If True, read/write the on-disk prompt cache. If False the
            prompt is rebuilt from the reference audio without touching the cache.

    Returns:
        A single-element list containing one VoiceClonePromptItem. The library
        broadcasts a single prompt over the whole text batch.
    """
    from dataclasses import asdict

    import torch

    ref_wav, ref_text_path = common.resolve_voice_paths(cfg)
    xvec = cfg.x_vector_only_mode
    ref_text: str | None = None
    if not xvec:
        if not ref_text_path.exists():
            raise FileNotFoundError(
                f"Reference transcript not found: {ref_text_path}. "
                f"ICL mode requires inputs/voices/{cfg.speaker}.txt. "
                f"Set x_vector_only_mode: true to skip the transcript "
                f"(lower quality)."
            )
        ref_text = ref_text_path.read_text(encoding="utf-8").strip()

    cache_dir = cfg.paths.prompt_cache
    cache_path = cache_dir / f"{cfg.speaker}_{cfg.model_size}.pt"
    fp = common.voice_fingerprint(cfg)
    if use_cache and cache_path.exists():
        try:
            payload = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            if payload.get("fingerprint") == fp:
                item = _rebuild_prompt_item(payload["items"][0])
                logger.info("Loaded cached voice prompt from %s", cache_path)
                return [item]
            logger.info("Voice prompt cache stale (fingerprint mismatch), rebuilding.")
        except Exception as e:
            logger.warning("Failed to read voice prompt cache (%s), rebuilding.", e)

    logger.info(
        "Building voice-clone prompt from %s (mode=%s)",
        ref_wav,
        "x-vector-only" if xvec else "ICL",
    )
    items = model.create_voice_clone_prompt(
        ref_audio=str(ref_wav),
        ref_text=ref_text,
        x_vector_only_mode=xvec,
    )
    if use_cache:
        payload = {"fingerprint": fp, "items": [asdict(it) for it in items]}
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(cache_path))
        logger.info("Cached voice prompt -> %s", cache_path)
    return [items[0]]


def _sampling_kwargs(cfg: common.Config) -> dict:
    """Build the sampling-generation kwargs forwarded to the Qwen3-TTS model.

    Centralises the sampling parameters so both generate_custom_voice and
    generate_voice_clone call sites stay in sync. Lower temperature / top_p /
    top_k reduce variance across clips (see config.yaml comments).

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict of generation kwargs (max_new_tokens, do_sample, temperature,
        top_k, top_p, repetition_penalty) to unpack into the model call.
    """
    return dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
    )


def _generate_batch(
    model,
    texts: list[str],
    cfg: common.Config,
    voice_clone_prompt: list | None = None,
    speaker_override: str | None = None,
) -> list[tuple[Any, int]]:
    """Generate a single batch of audio clips (one model call).

    In custom_voice mode uses generate_custom_voice with the configured speaker
    (overridable via speaker_override); in base mode uses generate_voice_clone
    with a precomputed voice prompt. Expects exactly len(texts) outputs back.
    The sampling parameters (do_sample, temperature, top_k, top_p,
    repetition_penalty) from cfg are forwarded to the model to control variance
    across clips.

    Args:
        model: Loaded Qwen3TTSModel.
        texts: List of sentences to synthesize (one model call, batched).
        cfg: Pipeline configuration.
        voice_clone_prompt: Single-element list of VoiceClonePromptItem (base mode).
        speaker_override: Speaker name (custom_voice) overriding cfg.speaker.
            Used by the speaker test to sweep multiple preset speakers without
            mutating cfg.

    Returns:
        List of (wav, sample_rate) tuples, one per input text.
    """
    n = len(texts)
    sampling = _sampling_kwargs(cfg)
    if cfg.model_type == "base":
        out, sr = model.generate_voice_clone(
            text=texts,
            language=[cfg.language] * n,
            voice_clone_prompt=voice_clone_prompt,
            **sampling,
        )
    else:
        speaker = speaker_override if speaker_override is not None else cfg.speaker
        out, sr = model.generate_custom_voice(
            text=texts,
            language=[cfg.language] * n,
            speaker=[speaker] * n,
            instruct=[cfg.instruct] * n if cfg.instruct else None,
            **sampling,
        )
    return [(w, sr) for w in out]


def generate_phrases(
    model,
    cfg: common.Config,
    texts: list[str],
    *,
    speaker_override: str | None = None,
    voice_clone_prompt: list | None = None,
    on_skip=None,
) -> list[tuple[Any, int] | None]:
    """Generate audio for a list of phrases, batched by cfg.batch_size.

    Shared helper used by the speaker/voice test utility (test_speaker) and
    available for any batched inference use case. Wraps the lower-level
    _generate_batch, chunking the input texts into batches of cfg.batch_size and
    producing one entry per input sentence, aligned in order.

    Error handling follows the same policy used by run_generate:
      - GPU out-of-memory (MemoryError or OOM-shaped exception) is fatal and
        exits the process with code 2 via common.exit_on_oom.
      - Generic errors on a batch yield None for each clip in the batch plus a
        warning log, then the loop continues with the next batch.
      - Empty / zero-length clips inside a successful batch yield None plus a
        warning.

    Args:
        model: Loaded Qwen3TTSModel.
        cfg: Pipeline configuration. Uses batch_size, language, model_type,
            max_new_tokens and (for custom_voice) instruct.
        texts: Phrases to synthesize.
        speaker_override: Optional speaker name overriding cfg.speaker
            (custom_voice only). Used by the speaker test sweep.
        voice_clone_prompt: Single-element VoiceClonePromptItem list (base mode).
        on_skip: Optional callback(i_phrase, reason: str) invoked for each
            skipped clip (empty output, or every clip in a failed batch).
            Useful for the test to count failed clips.

    Returns:
        List of (wav, sample_rate) tuples or None per input sentence, aligned
        with texts by position.
    """
    results: list[tuple[Any, int] | None] = [None] * len(texts)
    batch = max(1, cfg.batch_size)

    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        try:
            wavs = _generate_batch(
                model,
                chunk,
                cfg,
                voice_clone_prompt=voice_clone_prompt,
                speaker_override=speaker_override,
            )
        except SystemExit:
            raise
        except Exception as e:
            if common.is_oom_error(e):
                common.exit_on_oom(e, logger)
            logger.warning("Batch failed (start=%d, n=%d): %s", start, len(chunk), e)
            for i in range(len(chunk)):
                if on_skip is not None:
                    on_skip(start + i, f"batch error: {e}")
            continue

        for i, (wav, sr) in enumerate(wavs):
            if wav is None or len(wav) == 0:
                logger.warning("Empty clip at offset %d, skipped.", start + i)
                if on_skip is not None:
                    on_skip(start + i, "empty")
                continue
            results[start + i] = (wav, sr)

    return results


def run_generate(cfg: common.Config, only_rejected: bool = False) -> dict[str, Any]:
    """Generate audio for all sentences, resuming from checkpoint.

    Args:
        cfg: Pipeline configuration.
        only_rejected: If True, regenerate only previously rejected clips.

    Returns:
        Dict with generated/skipped/already_done/total counts and elapsed time.
    """
    common.setup_logging(cfg.paths.log_file)
    common.ensure_dirs(cfg.paths.raw_wav, cfg.paths.log_file.parent)

    # --- Load corpus ---
    if not cfg.paths.input_sentences.exists():
        raise FileNotFoundError(f"Corpus not found: {cfg.paths.input_sentences}")
    sentences = common.load_sentences(cfg)
    total = len(sentences)
    logger.info("Corpus: %d sentences", total)

    # --- Determine which indices still need generation ---
    done = common.read_checkpoint(cfg.paths.checkpoint)
    if only_rejected:
        rejected_idx = common.read_rejected_indices(cfg)
        if not rejected_idx:
            logger.info("No rejected clips to regenerate.")
            return {
                "generated": 0,
                "skipped": 0,
                "already_done": len(done),
                "total": total,
                "time_seconds": 0.0,
            }
        done -= rejected_idx
        common.write_checkpoint(cfg.paths.checkpoint, done)
        # Record the regenerated indices in `workspace/.regenerated.json`.
        # The next `pronunciation` run will:
        #   1. pick up exactly these indices (intersected with accepted_wav/),
        #      skipping resumability (the new audio has no PER score yet);
        #   2. delete the file once consumed (one-shot, not cumulative).
        # This is robust to manual-accept / --only-rejected operations between
        # the regen and the pronunciation run, because the manifest is
        # read once and then deleted, independent of pronunciation's `done`
        # checkpoint (which other operations can modify).
        regen_path = cfg.paths.regenerated
        existing: set[int] = set()
        if regen_path.exists():
            try:
                existing = set(common.read_checkpoint(regen_path))
            except Exception:
                existing = set()
        existing.update(rejected_idx)
        common.write_checkpoint(regen_path, existing)
        logger.info(
            "Recorded %d regenerated indices in %s (total in file: %d)",
            len(rejected_idx),
            regen_path,
            len(existing),
        )
        rejected_dir = cfg.paths.rejected
        for idx in rejected_idx:
            # Wav in raw_wav/ is the old source we generated from — drop it
            # so the new generation produces a fresh file (not overwriting).
            old_raw_wav = cfg.paths.raw_wav / f"{idx:06d}.wav"
            if old_raw_wav.exists():
                old_raw_wav.unlink()
            # Wav in rejected/ is the rejected version we're about to
            # regenerate — drop it too. The sidecar JSON stays so
            # pronunciation --only-rejected still sees the queue.
            old_rejected_wav = rejected_dir / f"{idx:06d}.wav"
            if old_rejected_wav.exists():
                old_rejected_wav.unlink()
        pending_idx = sorted(rejected_idx)
        logger.info("Only-rejected mode: %d clips to regenerate", len(pending_idx))
    else:
        pending_idx = [i for i in range(total) if i not in done]
        logger.info("Already processed: %d | pending: %d", len(done), len(pending_idx))

    if not pending_idx:
        logger.info("Nothing to do: all sentences have already been generated.")
        return {
            "generated": 0,
            "skipped": 0,
            "already_done": len(done),
            "total": total,
            "time_seconds": 0.0,
        }

    # --- Load model and (for base mode) the voice-clone prompt ---
    model = load_tts_model(cfg)
    voice_clone_prompt: list | None = None
    if cfg.model_type == "base":
        voice_clone_prompt = get_voice_clone_prompt(model, cfg)

    # --- Batch generation loop ---
    generated = 0
    skipped = 0
    start_time = time.time()
    batch = cfg.batch_size
    progress = tqdm(
        pending_idx,
        desc="generate",
        unit="sent",
        dynamic_ncols=True,
        total=total,
        initial=len(done),
    )

    batch_num = 0
    try:
        for start in range(0, len(pending_idx), batch):
            batch_idxs = pending_idx[start : start + batch]
            batch_texts = [sentences[i] for i in batch_idxs]
            try:
                wavs = _generate_batch(
                    model, batch_texts, cfg, voice_clone_prompt=voice_clone_prompt
                )
            except SystemExit:
                raise
            except Exception as e:
                if common.is_oom_error(e):
                    common.write_checkpoint(cfg.paths.checkpoint, done)
                    common.exit_on_oom(e, logger)
                logger.warning("Batch failed (idx %s): %s", batch_idxs, e)
                skipped += len(batch_idxs)
                progress.update(len(batch_idxs))
                continue

            for idx, (wav, sr) in zip(batch_idxs, wavs):
                if wav is None or len(wav) == 0:
                    logger.warning("Empty clip at idx %d, skipped.", idx)
                    skipped += 1
                    continue
                out_path = cfg.paths.raw_wav / f"{idx:06d}.wav"
                try:
                    sf.write(str(out_path), wav, sr)
                except Exception as e:
                    logger.warning("Save failed for idx %d: %s", idx, e)
                    skipped += 1
                    continue
                generated += 1
                done.add(idx)
            progress.update(len(batch_idxs))
            common.write_checkpoint(cfg.paths.checkpoint, done)
            batch_num += 1
            if batch_num % cfg.mem_cleanup_every_n_batches == 0:
                common.cleanup_gpu(log=logger)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving checkpoint and exiting...")
        common.write_checkpoint(cfg.paths.checkpoint, done)
        progress.close()
        raise SystemExit(1)

    progress.close()
    elapsed = time.time() - start_time
    logger.info(
        "Generation completed in %.1fs | new=%d skipped=%d", elapsed, generated, skipped
    )
    return {
        "generated": generated,
        "skipped": skipped,
        "already_done": len(done) - generated,
        "total": total,
        "time_seconds": elapsed,
    }
