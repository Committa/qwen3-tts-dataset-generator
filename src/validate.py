"""Step 3: quality validation with faster-whisper ASR + WER (jiwer)."""

from __future__ import annotations

import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import common

try:
    from text_to_num import alpha2digit as _alpha2digit
except ImportError:
    _alpha2digit = None

logger = logging.getLogger(__name__)


def _normalize_text(text: str, lang_code: str | None = None) -> str:
    """Normalize text for WER comparison: convert number words, strip accents and punctuation."""
    import unicodedata

    if _alpha2digit is not None and lang_code is not None:
        try:
            text = _alpha2digit(text, lang_code, threshold=2)
        except Exception:
            logger.warning("alpha2digit failed for lang '%s', falling back", lang_code)

    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    text = text.lower().strip()
    text = re.sub(r"\b(\d+)[.,:;](00)\b", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wer(reference: str, hypothesis: str, lang_code: str | None = None) -> float:
    """Compute the Word Error Rate between reference and hypothesis strings."""
    from jiwer import wer as _jiwer_wer

    ref = _normalize_text(reference, lang_code)
    hyp = _normalize_text(hypothesis, lang_code)
    if not ref:
        return 1.0 if hyp else 0.0
    return float(_jiwer_wer(ref, hyp))


def _load_asr(cfg: common.Config):
    """Load the faster-whisper ASR model, falling back to CPU if CUDA is unavailable.

    ``num_workers`` is forwarded to faster-whisper (maps to CTranslate2
    ``inter_threads``) so that concurrent ``transcribe()`` calls from a thread
    pool run with true parallelism, as documented by the library. Memory usage
    grows with the number of workers.
    """
    from faster_whisper import WhisperModel

    model_name = cfg.asr_model
    device = cfg.asr_device
    compute_type = cfg.asr_compute_type
    num_workers = max(1, cfg.asr_workers)
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA not available for ASR, falling back to CPU.")
                device, compute_type = "cpu", "int8"
        except ImportError:
            device, compute_type = "cpu", "int8"
    logger.info(
        "Loading ASR faster-whisper '%s' (device=%s, ct=%s, workers=%d)",
        model_name,
        device,
        compute_type,
        num_workers,
    )
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers,
    )


def _transcribe(asr_model, wav_path: Path, lang_code: str) -> str:
    """Transcribe a single wav file and return the joined segment text."""
    segments, _info = asr_model.transcribe(
        str(wav_path),
        language=lang_code,
        beam_size=5,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments)


def _validate_one(
    wav_path: Path,
    idx: int,
    expected: str,
    asr_model,
    lang_code: str,
    cfg: common.Config,
) -> dict[str, Any]:
    """Transcribe one clip, compute WER, and decide accept/reject.

    faster-whisper supports concurrent ``transcribe()`` calls on a shared model
    instance (via ``num_workers``), so no external lock is needed: this function
    is safe to submit to a thread pool.

    Args:
        wav_path: Path to the raw wav file.
        idx: Sentence index derived from the filename stem.
        expected: Reference text from the corpus.
        asr_model: Loaded faster-whisper model.
        lang_code: ISO language code for transcription/normalization.
        cfg: Pipeline configuration.

    Returns:
        Result dict with keys: accept, idx, file, wav_path, expected,
        wer, transcription, reason.
    """
    base: dict[str, Any] = {
        "idx": idx,
        "file": wav_path.name,
        "wav_path": wav_path,
        "expected": expected,
    }
    try:
        transcription = _transcribe(asr_model, wav_path, lang_code)
    except Exception as e:
        logger.warning("ASR failed for %s: %s", wav_path.name, e)
        return {
            **base,
            "accept": False,
            "wer": None,
            "transcription": "",
            "reason": f"asr_error: {e}",
        }
    wer = _wer(expected, transcription, lang_code)
    if wer <= cfg.wer_threshold:
        return {
            **base,
            "accept": True,
            "wer": wer,
            "transcription": transcription,
            "reason": "",
        }
    reason = f"wer={wer:.3f} > {cfg.wer_threshold:.3f}"
    return {
        **base,
        "accept": False,
        "wer": wer,
        "transcription": transcription,
        "reason": reason,
    }


def _handle_result(
    res: dict[str, Any],
    cfg: common.Config,
    wer_values: list[float],
    rejected_records: list[dict[str, Any]],
) -> str:
    """Apply the accept/reject file moves and update the tally lists.

    Args:
        res: Result dict from ``_validate_one``.
        cfg: Pipeline configuration.
        wer_values: Accumulator for WER values (appended when a WER was computed).
        rejected_records: Accumulator for rejected-clip metadata records.

    Returns:
        ``"accepted"`` or ``"rejected"``.
    """
    wav_path = res["wav_path"]
    idx = res["idx"]
    if res["wer"] is not None:
        wer_values.append(res["wer"])
    if res["accept"]:
        dest = cfg.paths.accepted_wav / wav_path.name
        shutil.copy2(str(wav_path), str(dest))
        logger.info("ACCEPTED idx=%d WER=%.3f -> %s", idx, res["wer"], dest.name)
        return "accepted"
    _move_to_rejected(
        wav_path, cfg, idx, res["expected"], res["transcription"], res["reason"]
    )
    rec: dict[str, Any] = {"index": idx, "file": wav_path.name, "reason": res["reason"]}
    if res["wer"] is not None:
        rec["wer"] = res["wer"]
        rec["transcription"] = res["transcription"]
        logger.info("REJECTED idx=%d WER=%.3f", idx, res["wer"])
    rejected_records.append(rec)
    return "rejected"


def _move_to_rejected(
    wav_path: Path,
    cfg: common.Config,
    idx: int,
    expected: str,
    transcription: str = "",
    reason: str = "",
) -> None:
    """Copy a rejected clip and its metadata into workspace/rejected/."""
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


def run_validate(cfg: common.Config) -> dict[str, Any]:
    """Run ASR validation on all generated clips.

    Transcribes each raw wav with faster-whisper, computes WER against the
    expected text, and moves clips to accepted_wav/ or rejected/ based on the
    configured wer_threshold. When ``cfg.asr_workers > 1`` the transcription
    runs in a thread pool with true parallelism: faster-whisper (CTranslate2)
    supports concurrent ``transcribe()`` calls via its ``num_workers`` setting,
    batching requests on the GPU or using multiple CPU cores. Memory usage grows
    with the number of workers.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict with accepted/rejected counts, mean WER, and rejected records.
    """
    common.setup_logging(cfg.paths.log_file)
    common.ensure_dirs(cfg.paths.accepted_wav, cfg.paths.rejected)

    # --- Load corpus and ASR model ---
    sentences = common.load_sentences(cfg)
    lang_code = common.language_code(cfg.language)
    raw_dir = cfg.paths.raw_wav
    files = sorted(raw_dir.glob("*.wav"))
    if not files:
        logger.warning("No wav files in %s. Run generate step first.", raw_dir)
        return {"accepted": 0, "rejected": 0, "mean_wer": 0.0}

    asr_model = _load_asr(cfg)

    # --- Build the work list: (wav_path, idx, expected_text) ---
    work: list[tuple[Path, int, str]] = []
    for wav_path in files:
        idx = int(wav_path.stem)
        if idx >= len(sentences):
            logger.warning("Index %d out of range for corpus. Skipping.", idx)
            continue
        work.append((wav_path, idx, sentences[idx]))
    if not work:
        logger.warning("No in-range clips to validate.")
        return {"accepted": 0, "rejected": 0, "mean_wer": 0.0}

    # --- Validate each clip ---
    accepted = 0
    rejected = 0
    wer_values: list[float] = []
    rejected_records: list[dict[str, Any]] = []
    progress = tqdm(total=len(work), desc="validate", unit="wav")

    if cfg.asr_workers <= 1:
        # Sequential path: simple and predictable.
        for wav_path, idx, expected in work:
            res = _validate_one(wav_path, idx, expected, asr_model, lang_code, cfg)
            if _handle_result(res, cfg, wer_values, rejected_records) == "accepted":
                accepted += 1
            else:
                rejected += 1
            progress.update(1)
    else:
        # Parallel path: faster-whisper (CTranslate2) supports concurrent
        # transcribe() calls via num_workers, so the thread pool gives real
        # throughput on GPU (batched requests) and on CPU (multiple cores).
        # Results are collected as they complete (order is not index-sorted).
        with ThreadPoolExecutor(max_workers=cfg.asr_workers) as pool:
            futures = [
                pool.submit(
                    _validate_one,
                    wav_path,
                    idx,
                    expected,
                    asr_model,
                    lang_code,
                    cfg,
                )
                for wav_path, idx, expected in work
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if _handle_result(res, cfg, wer_values, rejected_records) == "accepted":
                    accepted += 1
                else:
                    rejected += 1
                progress.update(1)

    progress.close()

    mean_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    logger.info(
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
