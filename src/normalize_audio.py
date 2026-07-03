"""Step 4: audio normalization - resample, loudness normalization, silence trimming."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from tqdm import tqdm

from . import common

logger = logging.getLogger(__name__)


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return data, sr_out
    try:
        import librosa

        data = librosa.resample(
            data.astype(np.float32), orig_sr=sr_in, target_sr=sr_out
        )
    except Exception as e:
        raise RuntimeError(f"resample {sr_in}->{sr_out} failed: {e}") from e
    return data, sr_out


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        return data.mean(axis=1)
    return data


def _trim_silence(data: np.ndarray, sr: int, top_db: float) -> np.ndarray:
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(data.astype(np.float32), top_db=top_db)
        return trimmed
    except Exception:
        return data


def _loudness_normalize(data: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data.astype(np.float32))
        if np.isneginf(loudness) or np.isnan(loudness) or loudness == -np.inf:
            return data
        return pyln.normalize.loudness(data.astype(np.float32), loudness, target_lufs)
    except Exception as e:
        logger.warning("loudness normalization failed (%s), skipping.", e)
        return data


def _process_file(src: Path, cfg: common.Config) -> tuple[bool, str]:
    try:
        data, sr = sf.read(str(src), dtype="float32")
    except Exception as e:
        return False, f"read_error: {e}"
    if data is None or len(data) == 0:
        return False, "empty_audio"

    data = _to_mono(data)
    data, sr = _resample(data, sr, cfg.target_sample_rate)
    data = _trim_silence(data, sr, cfg.trim_silence_db)
    data = _loudness_normalize(data, sr, cfg.target_lufs)
    data = np.clip(data, -1.0, 1.0)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        data = data / peak * 0.99

    sf.write(str(src), data, sr, subtype="PCM_16")
    return True, "ok"


def run_normalize(cfg: common.Config) -> dict[str, Any]:
    """Normalize all accepted audio clips in-place.

    Operations:
        - Convert to mono
        - Resample to target sample rate
        - Trim leading/trailing silence
        - Loudness normalize to target LUFS
        - Peak normalize to 0.99
        - Save as 16-bit PCM WAV

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict with counts of normalized and failed files.
    """
    common.setup_logging(cfg.paths.log_file)

    accept_dir = cfg.paths.accepted_wav
    files = sorted(accept_dir.glob("*.wav"))
    if not files:
        logger.warning("No accepted wav in %s. Run validate step first.", accept_dir)
        return {"normalized": 0, "failed": 0}

    # --- Normalize each accepted clip in-place ---
    ok = 0
    failed = 0
    progress = tqdm(files, desc="normalize", unit="wav")
    for wav_path in progress:
        success, msg = _process_file(wav_path, cfg)
        if success:
            ok += 1
        else:
            failed += 1
            logger.warning("Normalization failed %s: %s", wav_path.name, msg)
    progress.close()
    logger.info(
        "Normalization: ok=%d failed=%d (target=%dHz, %.1f LUFS)",
        ok,
        failed,
        cfg.target_sample_rate,
        cfg.target_lufs,
    )
    return {"normalized": ok, "failed": failed}
