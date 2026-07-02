"""Step 5: build LJSpeech manifest (filename|text) with deterministic train/val split."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

from . import common

_logger: logging.Logger | None = None


def _load_sentences(cfg: common.Config) -> list[str]:
    lines = cfg.paths.input_sentences.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _index_from_name(name: str) -> int:
    return int(Path(name).stem)


def _ljspeech_row(wav_path: Path, text: str) -> str:
    return f"{wav_path.resolve().as_posix()}|{text}"


def run_build_manifest(cfg: common.Config) -> dict[str, Any]:
    """Build LJSpeech-format manifest files (filename|text) with train/val split.

    The split is deterministic based on cfg.seed. Accepted wav files are
    matched to their expected text by numeric index from the filename stem.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict with train/val row counts and manifest file paths.
    """
    global _logger
    _logger = common.setup_logging(cfg.paths.log_file)
    cfg.paths.manifest_train.parent.mkdir(parents=True, exist_ok=True)

    sentences = _load_sentences(cfg)
    accept_dir = cfg.paths.accepted_wav
    files = sorted(accept_dir.glob("*.wav"), key=lambda p: _index_from_name(p.name))
    if not files:
        _logger.warning(
            "No accepted wav in %s. Run validate+normalize first.", accept_dir
        )
        return {"train": 0, "val": 0, "total": 0}

    rng = random.Random(cfg.seed)
    indices = list(range(len(files)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(files) * cfg.val_ratio))) if len(files) > 1 else 0
    val_set = set(indices[:n_val])

    train_rows: list[str] = []
    val_rows: list[str] = []
    for i, wav_path in enumerate(files):
        idx = _index_from_name(wav_path.name)
        if idx < len(sentences):
            text = sentences[idx]
        else:
            _logger.warning("Index %d out of range, using filename as text.", idx)
            text = wav_path.stem
        row = _ljspeech_row(wav_path, text)
        (val_rows if i in val_set else train_rows).append(row)

    _write_csv(cfg.paths.manifest_train, train_rows)
    _write_csv(cfg.paths.manifest_val, val_rows)
    _logger.info(
        "Manifests written -> train=%d val=%d (ratio=%.2f)",
        len(train_rows),
        len(val_rows),
        cfg.val_ratio,
    )
    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "total": len(files),
        "manifest_train": str(cfg.paths.manifest_train),
        "manifest_val": str(cfg.paths.manifest_val),
    }


def _write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            writer.writerow(r.split("|", 1))
