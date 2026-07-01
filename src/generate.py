"""Step 2: batch audio generation with Qwen3-TTS, checkpoint/resume and OOM handling."""
from __future__ import annotations

import logging
import time
from pathlib import Path
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
        raise RuntimeError(
            "qwen-tts is not installed. Run `poetry install`."
        ) from e

    common.check_cuda_or_die(_logger)
    model_id = cfg.model_hub_id
    _logger.info("Loading TTS model %s (dtype=%s, device_map=%s)", model_id, cfg.dtype, cfg.device_map)
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
    _logger.info("Available speakers: %s", ", ".join(model.get_supported_speakers()))
    return model


def _generate_batch(
    model,
    texts: list[str],
    cfg: common.Config,
) -> list[tuple[Any, int]]:
    global _logger
    assert _logger is not None
    n = len(texts)
    wavs: list[Any] = []
    try:
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
    except MemoryError as e:
        raise
    except Exception as e:
        if common.is_oom_error(e):
            raise
        raise


def run_generate(cfg: common.Config) -> dict[str, Any]:
    """Generate audio for all sentences, resuming from checkpoint."""
    global _logger
    _logger = common.setup_logging(cfg.paths.log_file)
    common.ensure_dirs(cfg.paths.raw_wav, cfg.paths.log_file.parent)

    sentences_path = cfg.paths.input_sentences
    if not sentences_path.exists():
        raise FileNotFoundError(f"Corpus not found: {sentences_path}")
    sentences = [ln.strip() for ln in sentences_path.read_text(encoding="utf-8").splitlines()]
    sentences = [s for s in sentences if s and not s.startswith("#")]
    total = len(sentences)
    _logger.info("Corpus: %d sentences", total)

    done = common.read_checkpoint(cfg.paths.checkpoint)
    pending_idx = [i for i in range(total) if i not in done]
    _logger.info("Already processed: %d | pending: %d", len(done), len(pending_idx))
    if not pending_idx:
        _logger.info("Nothing to do: all sentences have already been generated.")
        return {"generated": 0, "skipped": 0, "already_done": len(done), "total": total, "time_seconds": 0.0}

    model = load_tts_model(cfg)

    generated = 0
    skipped = 0
    start_time = time.time()
    batch = cfg.batch_size
    progress = tqdm(pending_idx, desc="generate", unit="sent")

    for start in range(0, len(pending_idx), batch):
        batch_idxs = pending_idx[start : start + batch]
        batch_texts = [sentences[i] for i in batch_idxs]
        try:
            wavs = _generate_batch(model, batch_texts, cfg)
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
    _logger.info("Generation completed in %.1fs | new=%d skipped=%d", elapsed, generated, skipped)
    return {
        "generated": generated,
        "skipped": skipped,
        "already_done": len(done) - generated,
        "total": total,
        "time_seconds": elapsed,
    }