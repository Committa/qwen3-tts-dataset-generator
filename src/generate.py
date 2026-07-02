"""Step 2: batch audio generation with Qwen3-TTS, checkpoint/resume and OOM handling."""

from __future__ import annotations

import logging
import time
from typing import Any

import soundfile as sf
from tqdm import tqdm

from . import common

_logger: logging.Logger | None = None


def _build_dtype(dtype_str: str):
    import torch

    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_str not in mapping:
        raise ValueError(f"dtype non valido: {dtype_str}. Usa bfloat16 o float16.")
    return mapping[dtype_str]


def load_tts_model(cfg: common.Config):
    """Load the Qwen3-TTS model. Raises SystemExit with clear message on OOM."""
    global _logger
    assert _logger is not None
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise RuntimeError("qwen-tts is not installed. Run `poetry install`.") from e

    common.check_cuda_or_die(_logger)
    model_id = cfg.model_hub_id
    _logger.info(
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
        _logger.error(common.OOM_HINT)
        raise SystemExit(2) from e
    except Exception as e:
        if common.is_oom_error(e):
            _logger.error(common.OOM_HINT)
            raise SystemExit(2) from e
        _logger.error("Error loading TTS model: %s", e)
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
        _logger.info(
            "Available speakers: %s", ", ".join(model.get_supported_speakers())
        )
    else:
        ref_wav, _ = common.resolve_voice_paths(cfg)
        mode = "x-vector-only" if cfg.voice.x_vector_only_mode else "ICL"
        _logger.info(
            "Voice clone: voice='%s' mode=%s ref=%s", cfg.voice.name, mode, ref_wav
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


def get_voice_clone_prompt(
    model, cfg: common.Config, logger: logging.Logger, use_cache: bool = True
) -> list:
    """Build or load a cached voice-clone prompt for the configured custom voice.

    In ICL mode (x_vector_only_mode=False) the reference transcript ref.txt is
    required; in x-vector-only mode it is ignored. The resulting
    VoiceClonePromptItem is cached to cfg.voice.prompt_cache keyed by a
    fingerprint of the reference audio, transcript, mode and model, so that
    re-runs skip the prompt extraction.

    Args:
        model: Loaded Qwen3TTSModel (must be of base type).
        cfg: Pipeline configuration.
        logger: Logger instance for diagnostic messages.
        use_cache: If True, read/write the on-disk prompt cache. If False the
            prompt is rebuilt from the reference audio without touching the cache.

    Returns:
        A single-element list containing one VoiceClonePromptItem. The library
        broadcasts a single prompt over the whole text batch.
    """
    from dataclasses import asdict

    import torch

    ref_wav, ref_text_path = common.resolve_voice_paths(cfg)
    xvec = cfg.voice.x_vector_only_mode
    ref_text: str | None = None
    if not xvec:
        if not ref_text_path.exists():
            raise FileNotFoundError(
                f"Reference transcript not found: {ref_text_path}. "
                f"ICL mode requires inputs/voices/{cfg.voice.name}.txt. "
                f"Set voice.x_vector_only_mode: true to skip the transcript "
                f"(lower quality)."
            )
        ref_text = ref_text_path.read_text(encoding="utf-8").strip()

    cache_dir = cfg.voice.prompt_cache
    cache_path = cache_dir / f"{cfg.voice.name}.pt"
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


def _generate_batch(
    model,
    texts: list[str],
    cfg: common.Config,
    voice_clone_prompt: list | None = None,
) -> list[tuple[Any, int]]:
    """Generate a batch of audio clips.

    In custom_voice mode uses generate_custom_voice with the configured speaker;
    in base mode uses generate_voice_clone with a precomputed voice prompt.

    Args:
        model: Loaded Qwen3TTSModel.
        texts: List of sentences to synthesize.
        cfg: Pipeline configuration.
        voice_clone_prompt: Single-element list of VoiceClonePromptItem (base mode).

    Returns:
        List of (wav, sample_rate) tuples.
    """
    global _logger
    assert _logger is not None
    n = len(texts)
    wavs: list[Any] = []
    try:
        if cfg.model_type == "base":
            out, sr = model.generate_voice_clone(
                text=texts,
                language=[cfg.language] * n,
                voice_clone_prompt=voice_clone_prompt,
                max_new_tokens=cfg.max_new_tokens,
            )
        else:
            out, sr = model.generate_custom_voice(
                text=texts,
                language=[cfg.language] * n,
                speaker=[cfg.speaker] * n,
                instruct=[cfg.instruct] * n if cfg.instruct else None,
                max_new_tokens=cfg.max_new_tokens,
            )
        for w in out:
            wavs.append((w, sr))
        return wavs
    except MemoryError:
        raise
    except Exception as e:
        if common.is_oom_error(e):
            raise
        raise


def run_generate(cfg: common.Config, only_rejected: bool = False) -> dict[str, Any]:
    """Generate audio for all sentences, resuming from checkpoint.

    Args:
        cfg: Pipeline configuration.
        only_rejected: If True, regenerate only previously rejected clips.
    """
    global _logger
    _logger = common.setup_logging(cfg.paths.log_file)
    common.ensure_dirs(cfg.paths.raw_wav, cfg.paths.log_file.parent)

    sentences_path = cfg.paths.input_sentences
    if not sentences_path.exists():
        raise FileNotFoundError(f"Corpus not found: {sentences_path}")
    sentences = [
        ln.strip() for ln in sentences_path.read_text(encoding="utf-8").splitlines()
    ]
    sentences = [s for s in sentences if s and not s.startswith("#")]
    total = len(sentences)
    _logger.info("Corpus: %d sentences", total)

    done = common.read_checkpoint(cfg.paths.checkpoint)

    if only_rejected:
        rejected_idx = common.read_rejected_indices(cfg)
        if not rejected_idx:
            _logger.info("No rejected clips to regenerate.")
            return {
                "generated": 0,
                "skipped": 0,
                "already_done": len(done),
                "total": total,
                "time_seconds": 0.0,
            }
        done -= rejected_idx
        common.write_checkpoint(cfg.paths.checkpoint, done)
        for idx in rejected_idx:
            old_wav = cfg.paths.raw_wav / f"{idx:06d}.wav"
            if old_wav.exists():
                old_wav.unlink()
        pending_idx = sorted(rejected_idx)
        _logger.info("Only-rejected mode: %d clips to regenerate", len(pending_idx))
    else:
        pending_idx = [i for i in range(total) if i not in done]
        _logger.info("Already processed: %d | pending: %d", len(done), len(pending_idx))

    if not pending_idx:
        _logger.info("Nothing to do: all sentences have already been generated.")
        return {
            "generated": 0,
            "skipped": 0,
            "already_done": len(done),
            "total": total,
            "time_seconds": 0.0,
        }

    model = load_tts_model(cfg)

    voice_clone_prompt: list | None = None
    if cfg.model_type == "base":
        voice_clone_prompt = get_voice_clone_prompt(model, cfg, _logger)

    generated = 0
    skipped = 0
    start_time = time.time()
    batch = cfg.batch_size
    progress = tqdm(pending_idx, desc="generate", unit="sent")

    for start in range(0, len(pending_idx), batch):
        batch_idxs = pending_idx[start : start + batch]
        batch_texts = [sentences[i] for i in batch_idxs]
        try:
            wavs = _generate_batch(
                model, batch_texts, cfg, voice_clone_prompt=voice_clone_prompt
            )
        except (MemoryError, SystemExit):
            raise
        except Exception as e:
            if common.is_oom_error(e):
                _logger.error(common.OOM_HINT)
                common.write_checkpoint(cfg.paths.checkpoint, done)
                raise SystemExit(2) from e
            _logger.warning("Batch failed (idx %s): %s", batch_idxs, e)
            skipped += len(batch_idxs)
            progress.update(len(batch_idxs))
            continue

        for idx, (wav, sr) in zip(batch_idxs, wavs):
            if wav is None or len(wav) == 0:
                _logger.warning("Empty clip at idx %d, skipped.", idx)
                skipped += 1
                continue
            out_path = cfg.paths.raw_wav / f"{idx:06d}.wav"
            try:
                sf.write(str(out_path), wav, sr)
            except Exception as e:
                _logger.warning("Save failed for idx %d: %s", idx, e)
                skipped += 1
                continue
            generated += 1
            done.add(idx)
        progress.update(len(batch_idxs))
        common.write_checkpoint(cfg.paths.checkpoint, done)

    progress.close()
    elapsed = time.time() - start_time
    _logger.info(
        "Generation completed in %.1fs | new=%d skipped=%d", elapsed, generated, skipped
    )
    return {
        "generated": generated,
        "skipped": skipped,
        "already_done": len(done) - generated,
        "total": total,
        "time_seconds": elapsed,
    }
