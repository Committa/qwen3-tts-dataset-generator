"""Step 3: quality validation with faster-whisper ASR + WER (jiwer)."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import common

try:
    from text_to_num import alpha2digit as _alpha2digit
except ImportError:
    _alpha2digit = None

_logger: logging.Logger | None = None


def _normalize_text(text: str, lang_code: str | None = None) -> str:
    import unicodedata

    if _alpha2digit is not None and lang_code is not None:
        try:
            text = _alpha2digit(text, lang_code, threshold=2)
        except Exception:
            if _logger is not None:
                _logger.warning(
                    "alpha2digit failed for lang '%s', falling back", lang_code
                )

    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    text = text.lower().strip()
    text = re.sub(r"\b(\d+)[.,:;](00)\b", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wer(reference: str, hypothesis: str, lang_code: str | None = None) -> float:
    from jiwer import wer as _jiwer_wer

    ref = _normalize_text(reference, lang_code)
    hyp = _normalize_text(hypothesis, lang_code)
    if not ref:
        return 1.0 if hyp else 0.0
    return float(_jiwer_wer(ref, hyp))


def _load_sentences(cfg: common.Config) -> list[str]:
    lines = cfg.paths.input_sentences.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _load_asr(cfg: common.Config):
    assert _logger is not None
    from faster_whisper import WhisperModel

    model_name = cfg.asr_model
    device = cfg.asr_device
    compute_type = cfg.asr_compute_type
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                _logger.warning("CUDA not available for ASR, falling back to CPU.")
                device, compute_type = "cpu", "int8"
        except ImportError:
            device, compute_type = "cpu", "int8"
    _logger.info(
        "Loading ASR faster-whisper '%s' (device=%s, ct=%s)",
        model_name,
        device,
        compute_type,
    )
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _transcribe(asr_model, wav_path: Path, lang_code: str) -> str:
    segments, _info = asr_model.transcribe(
        str(wav_path),
        language=lang_code,
        beam_size=5,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments)


def run_validate(cfg: common.Config) -> dict[str, Any]:
    """Run ASR validation on all generated clips.

    Transcribes each raw wav with faster-whisper (Italian model), computes WER
    against the expected text, and moves clips to accepted_wav/ or rejected/
    based on the configured wer_threshold.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict with accepted/rejected counts, mean WER, and rejected records.
    """
    global _logger
    _logger = common.setup_logging(cfg.paths.log_file)
    common.ensure_dirs(cfg.paths.accepted_wav, cfg.paths.rejected)

    sentences = _load_sentences(cfg)
    lang_code = common.language_code(cfg.language)
    raw_dir = cfg.paths.raw_wav
    files = sorted(raw_dir.glob("*.wav"))
    if not files:
        _logger.warning("No wav files in %s. Run generate step first.", raw_dir)
        return {"accepted": 0, "rejected": 0, "mean_wer": 0.0}

    asr_model = _load_asr(cfg)

    accepted = 0
    rejected = 0
    wer_values: list[float] = []
    rejected_records: list[dict[str, Any]] = []

    progress = tqdm(files, desc="validate", unit="wav")
    for wav_path in progress:
        idx = int(wav_path.stem)
        if idx >= len(sentences):
            _logger.warning("Index %d out of range for corpus. Skipping.", idx)
            continue
        expected = sentences[idx]
        try:
            transcription = _transcribe(asr_model, wav_path, lang_code)
        except Exception as e:
            _logger.warning("ASR failed for %s: %s", wav_path.name, e)
            _move_to_rejected(wav_path, cfg, idx, expected, reason=f"asr_error: {e}")
            rejected_records.append(
                {"index": idx, "file": wav_path.name, "reason": f"asr_error: {e}"}
            )
            rejected += 1
            continue

        wer = _wer(expected, transcription, lang_code)
        wer_values.append(wer)
        if wer <= cfg.wer_threshold:
            dest = cfg.paths.accepted_wav / wav_path.name
            shutil.copy2(str(wav_path), str(dest))
            accepted += 1
            _logger.info("ACCEPTED idx=%d WER=%.3f -> %s", idx, wer, dest.name)
        else:
            reason = f"wer={wer:.3f} > {cfg.wer_threshold:.3f}"
            _move_to_rejected(wav_path, cfg, idx, expected, transcription, reason)
            rejected_records.append(
                {
                    "index": idx,
                    "file": wav_path.name,
                    "wer": wer,
                    "transcription": transcription,
                    "reason": reason,
                }
            )
            _logger.info("REJECTED idx=%d WER=%.3f", idx, wer)

    progress.close()

    mean_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    _logger.info(
        "Validation: accepted=%d rejected=%d mean WER=%.3f",
        accepted,
        rejected,
        mean_wer,
    )

    if rejected_records:
        (cfg.paths.rejected / "rejected.log").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected_records),
            encoding="utf-8",
        )

    return {
        "accepted": accepted,
        "rejected": rejected,
        "mean_wer": mean_wer,
        "rejected_records": rejected_records,
    }


def _move_to_rejected(
    wav_path: Path,
    cfg: common.Config,
    idx: int,
    expected: str,
    transcription: str = "",
    reason: str = "",
) -> None:
    dest = cfg.paths.rejected / wav_path.name
    shutil.copy2(str(wav_path), str(dest))
    meta = {
        "index": idx,
        "file": dest.name,
        "expected": expected,
        "transcription": transcription,
        "reason": reason,
    }
    (cfg.paths.rejected / f"{wav_path.stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
