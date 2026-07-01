"""Shared utilities: config loading, paths, logging, VRAM/OOM handling."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

MODEL_HUB_IDS = {
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}

OOM_HINT = (
    "OutOfMemoryError: not enough VRAM for the selected model.\n"
    "  -> Set model_size: \"0.6b\" in config.yaml\n"
    "  -> OR set device_map: \"auto\" for partial CPU offload\n"
    "  -> OR reduce batch_size in config.yaml"
)

CUDA_HINT = (
    "CUDA is not available on this machine.\n"
    "  -> Check with: poetry run python -c \"import torch; print(torch.cuda.is_available())\"\n"
    "  -> On Docker make sure to use: docker run --gpus all ...\n"
    "  -> Without a GPU the pipeline will be extremely slow or will not work."
)


@dataclass
class Paths:
    """Container for all file/directory paths used by the pipeline.

    All paths are resolved relative to the project root unless absolute.
    """

    input_sentences: Path
    test_sentences: Path
    raw_wav: Path
    accepted_wav: Path
    rejected: Path
    manifest_train: Path
    manifest_val: Path
    report: Path
    checkpoint: Path
    log_file: Path


@dataclass
class Config:
    """All configuration parameters parsed from config.yaml.

    Attributes are set from the YAML file by load_config(), falling back
    to the default values defined here.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    model_size: str = "0.6b"
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    attn_implementation: str = "sdpa"
    speaker: str = "Vivian"
    language: str = "Italian"
    instruct: str = ""
    batch_size: int = 4
    seed: int = 42
    max_new_tokens: int = 2048
    asr_model: str = "medium"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    wer_threshold: float = 0.15
    target_sample_rate: int = 22050
    target_lufs: float = -23.0
    trim_silence_db: float = 40.0
    val_ratio: float = 0.1
    clean_on_full_run: bool = True
    test_phrases: list[str] = field(default_factory=list)
    paths: Paths = field(
        default_factory=lambda: Paths(
            input_sentences=Path("."),
            test_sentences=Path("."),
            raw_wav=Path("."),
            accepted_wav=Path("."),
            rejected=Path("."),
            manifest_train=Path("."),
            manifest_val=Path("."),
            report=Path("."),
            checkpoint=Path("."),
            log_file=Path("."),
        )
    )

    @property
    def model_hub_id(self) -> str:
        """Return the HuggingFace model hub ID for the selected model_size."""
        return MODEL_HUB_IDS[self.model_size.lower()]


def _resolve_path(p: str | os.PathLike[str], is_input: bool = False) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    if is_input:
        return PROJECT_ROOT / "inputs" / path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and parse config.yaml into a Config dataclass.

    Args:
        config_path: Path to the YAML config file. If None, uses config.yaml
            in the project root.

    Returns:
        A Config instance populated with values from the file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(raw=raw)
    cfg.model_size = raw.get("model_size", cfg.model_size)
    cfg.dtype = raw.get("dtype", cfg.dtype)
    cfg.device_map = raw.get("device_map", cfg.device_map)
    cfg.attn_implementation = raw.get("attn_implementation", cfg.attn_implementation)
    cfg.speaker = raw.get("speaker", cfg.speaker)
    cfg.language = raw.get("language", cfg.language)
    cfg.instruct = raw.get("instruct", cfg.instruct)
    cfg.batch_size = int(raw.get("batch_size", cfg.batch_size))
    cfg.seed = int(raw.get("seed", cfg.seed))
    cfg.max_new_tokens = int(raw.get("max_new_tokens", cfg.max_new_tokens))
    cfg.asr_model = raw.get("asr_model", cfg.asr_model)
    cfg.asr_device = raw.get("asr_device", cfg.asr_device)
    cfg.asr_compute_type = raw.get("asr_compute_type", cfg.asr_compute_type)
    cfg.wer_threshold = float(raw.get("wer_threshold", cfg.wer_threshold))
    cfg.target_sample_rate = int(raw.get("target_sample_rate", cfg.target_sample_rate))
    cfg.target_lufs = float(raw.get("target_lufs", cfg.target_lufs))
    cfg.trim_silence_db = float(raw.get("trim_silence_db", cfg.trim_silence_db))
    cfg.val_ratio = float(raw.get("val_ratio", cfg.val_ratio))
    cfg.clean_on_full_run = bool(raw.get("clean_on_full_run", cfg.clean_on_full_run))

    p = raw.get("paths", {})
    cfg.paths = Paths(
        input_sentences=_resolve_path(p.get("input_sentences", "italian_sentences.txt"), is_input=True),
        test_sentences=_resolve_path(p.get("test_sentences", "test_sentences.txt"), is_input=True),
        raw_wav=_resolve_path(p.get("raw_wav", "workspace/raw_wav")),
        accepted_wav=_resolve_path(p.get("accepted_wav", "workspace/accepted_wav")),
        rejected=_resolve_path(p.get("rejected", "workspace/rejected")),
        manifest_train=_resolve_path(p.get("manifest_train", "workspace/.manifest_train.csv")),
        manifest_val=_resolve_path(p.get("manifest_val", "workspace/.manifest_val.csv")),
        report=_resolve_path(p.get("report", "workspace/.report.json")),
        checkpoint=_resolve_path(p.get("checkpoint", "workspace/.generate_checkpoint.json")),
        log_file=_resolve_path(p.get("log_file", "logs/pipeline.log")),
    )
    return cfg


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure dual-output logging (file + stdout).

    Args:
        log_file: Path to the log file.
        level: Logging level (default: INFO).

    Returns:
        The configured logger instance.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qwen3_tts_dataset")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they do not exist (mkdir -p)."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def read_checkpoint(path: Path) -> set[int]:
    """Read the set of already-processed sentence indices from a checkpoint file.

    Returns an empty set if the file does not exist or is corrupted.
    """
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("done", []))
    except (json.JSONDecodeError, OSError):
        return set()


def write_checkpoint(path: Path, done: set[int]) -> None:
    """Write the set of processed sentence indices to a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"done": sorted(done)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_oom_error(exc: BaseException) -> bool:
    """Check whether an exception is a CUDA out-of-memory error.

    Handles MemoryError, torch.cuda.OutOfMemoryError, and generic messages
    containing "out of memory" or "cuda memory".
    """
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda" in msg and "memory" in msg
        or isinstance(exc, MemoryError)
    )


def check_cuda_or_die(logger: logging.Logger) -> None:
    """Verify CUDA is available. Exits with code 1 if not.

    Args:
        logger: Logger instance for diagnostic messages.

    Raises:
        SystemExit(1): If CUDA is not available.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch is not installed.") from e
    if not torch.cuda.is_available():
        logger.error(CUDA_HINT)
        raise SystemExit(1)
    logger.info("CUDA available: %s (%d GPU)", torch.cuda.get_device_name(0), torch.cuda.device_count())


def clean_working_dirs(cfg: Config) -> None:
    """Remove all volatile workspace content from a previous run.

    Deletes raw_wav, accepted_wav, rejected, checkpoint, and temporary
    manifest/report files. Preserves the workspace/ directory structure.
    """
    import shutil

    dirs = [cfg.paths.raw_wav, cfg.paths.accepted_wav, cfg.paths.rejected]
    for d in dirs:
        if d.exists():
            shutil.rmtree(str(d))
            d.mkdir(parents=True, exist_ok=True)

    for f in [cfg.paths.manifest_train, cfg.paths.manifest_val, cfg.paths.report, cfg.paths.checkpoint]:
        if f.exists():
            f.unlink()


def next_gen_number() -> int:
    """Return the next generation number for archiving.

    Scans output/gen{NNN}/ directories and returns max+1 (or 1 if none exist).
    """
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_num = 0
    for p in output_dir.iterdir():
        m = re.fullmatch(r"gen(\d+)", p.name)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1


def archive_generation(cfg: Config, gen_number: int) -> None:
    """Archive the current dataset as output/gen{NNN}/.

    Moves accepted wavs into gen{NNN}/wavs/, rewrites manifest paths
    to be relative to the gen directory, and copies report.

    Args:
        cfg: Pipeline configuration.
        gen_number: Archive number (e.g. 3 for gen003).
    """
    import csv
    import shutil

    gen_dir = PROJECT_ROOT / "output" / f"gen{gen_number:03d}"
    wavs_dir = gen_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Move wavs
    wavs_moved = 0
    for src in sorted(cfg.paths.accepted_wav.glob("*.wav")):
        shutil.move(str(src), str(wavs_dir / src.name))
        wavs_moved += 1

    def _rewrite_manifest(src_path: Path, dest_path: Path) -> None:
        """Rewrite manifest replacing absolute paths with 'wavs/<filename>'."""
        if not src_path.exists():
            return
        with src_path.open("r", encoding="utf-8") as fin, \
             dest_path.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            for line in fin:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    fname = Path(parts[0]).name
                    writer.writerow([f"wavs/{fname}", parts[1]])

    _rewrite_manifest(cfg.paths.manifest_train, gen_dir / "metadata_train.csv")
    _rewrite_manifest(cfg.paths.manifest_val, gen_dir / "metadata_val.csv")

    # Copy live report
    live_report = cfg.paths.report
    if live_report.exists():
        import shutil as shutil2
        shutil2.copy2(str(live_report), str(gen_dir / "report.json"))

    logger = logging.getLogger("qwen3_tts_dataset")
    logger.info("Archived generation %03d: %d wavs -> %s", gen_number, wavs_moved, gen_dir)