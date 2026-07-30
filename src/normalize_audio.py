"""Step 4: audio normalization - resample, loudness normalization, silence trimming.

Reads clips from ``accepted_wav/`` (left untouched) and writes the
normalized copies into ``normalized_wav/``. This keeps the original
post-validate audio intact so normalize can be re-run (e.g. after tuning
``target_lufs``) without re-running validate, and so the publish step
can copy from ``normalized_wav/`` without emptying the workspace.
"""

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


def _trim_silence(
    data: np.ndarray,
    sr: int,
    top_db: float,
    tail_margin_ms: float = 120.0,
    tail_pad_ms: float = 80.0,
) -> np.ndarray:
    try:
        import librosa

        y = data.astype(np.float32)
        _trimmed, interval = librosa.effects.trim(y, top_db=top_db)
        start = int(interval[0])
        end = int(interval[1])
        margin = int(sr * tail_margin_ms / 1000.0)
        if margin > 0:
            end = min(len(y), end + margin)
        out = y[start:end]
        pad = int(sr * tail_pad_ms / 1000.0)
        if pad > 0:
            out = np.concatenate((out, np.zeros(pad, dtype=out.dtype)))
        return out
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


def _process_file(src: Path, dest: Path, cfg: common.Config) -> tuple[bool, str]:
    """Read ``src``, normalize the audio, and write the result to ``dest``.

    The source clip in ``accepted_wav/`` is never modified: the normalized
    copy is written to ``dest`` under ``normalized_wav/`` so the original
    can be reused for re-runs of normalize (e.g. after changing
    ``target_lufs``) without re-running validate.

    Args:
        src: Path to the input wav (in ``accepted_wav/``), read-only.
        dest: Path to the output wav (in ``normalized_wav/``), overwritten
            if it already exists.
        cfg: Pipeline configuration.

    Returns:
        A ``(success, reason)`` tuple; ``reason`` is ``"ok"`` on success or
        a short diagnostic string (``"read_error: ..."``, ``"empty_audio"``).
    """
    try:
        data, sr = sf.read(str(src), dtype="float32")
    except Exception as e:
        return False, f"read_error: {e}"
    if data is None or len(data) == 0:
        return False, "empty_audio"

    data = _to_mono(data)
    data, sr = _resample(data, sr, cfg.target_sample_rate)
    data = _trim_silence(
        data, sr, cfg.trim_silence_db, cfg.tail_margin_ms, cfg.tail_pad_ms
    )
    data = _loudness_normalize(data, sr, cfg.target_lufs)
    data = np.clip(data, -1.0, 1.0)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        data = data / peak * 0.99

    sf.write(str(dest), data, sr, subtype="PCM_16")
    return True, "ok"


def run_normalize(cfg: common.Config, force: bool = False) -> dict[str, Any]:
    """Normalize all accepted audio clips into ``normalized_wav/``.

    Operations (applied to a copy, the original in ``accepted_wav/`` is
    left untouched):
        - Convert to mono
        - Resample to target sample rate
        - Trim leading/trailing silence (preserving ``tail_margin_ms`` of
          tail, then appending ``tail_pad_ms`` of silence for clean decay
          boundaries)
        - Loudness normalize to target LUFS
        - Peak normalize to 0.99
        - Save as 16-bit PCM WAV

    Resumability: a clip whose destination file already exists in
    ``normalized_wav/`` is skipped. To force a full re-normalization
    pass ``force=True`` (or delete ``workspace/normalized_wav/`` by
    hand before re-running).

    Args:
        cfg: Pipeline configuration.
        force: If True, delete and recreate ``normalized_wav/`` before
            starting, so every clip is re-normalized from scratch.

    Returns:
        Dict with counts of normalized, skipped (already present) and
        failed files.
    """
    common.setup_logging(cfg.paths.log_file)

    accept_dir = cfg.paths.accepted_wav
    norm_dir = cfg.paths.normalized_wav

    if force:
        import shutil

        if norm_dir.exists():
            shutil.rmtree(str(norm_dir))
            logger.info("--force: wiped %s for a full re-normalization.", norm_dir)

    common.ensure_dirs(norm_dir)

    files = sorted(accept_dir.glob("*.wav"))
    if not files:
        logger.warning("No accepted wav in %s. Run validate step first.", accept_dir)
        return {"normalized": 0, "skipped": 0, "failed": 0}

    # --- Normalize each accepted clip into normalized_wav/ ---
    ok = 0
    skipped = 0
    failed = 0
    progress = tqdm(files, desc="normalize", unit="wav", dynamic_ncols=True)
    try:
        for wav_path in progress:
            dest = norm_dir / wav_path.name
            if dest.exists():
                # Resumability via filesystem state: a clip already present
                # in normalized_wav/ is considered done. Delete the dir to
                # force a full re-normalize.
                skipped += 1
                continue
            success, msg = _process_file(wav_path, dest, cfg)
            if success:
                ok += 1
            else:
                failed += 1
                logger.warning("Normalization failed %s: %s", wav_path.name, msg)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        progress.close()
        raise SystemExit(1)
    progress.close()
    logger.info(
        "Normalization: ok=%d skipped=%d failed=%d (target=%dHz, %.1f LUFS)",
        ok,
        skipped,
        failed,
        cfg.target_sample_rate,
        cfg.target_lufs,
    )
    return {"normalized": ok, "skipped": skipped, "failed": failed}
