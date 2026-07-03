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

MODEL_HUB_IDS: dict[str, dict[str, str]] = {
    "custom_voice": {
        "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    },
    "base": {
        "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    },
}

VALID_MODEL_TYPES = sorted(MODEL_HUB_IDS.keys())

LANGUAGE_CODE_MAP: dict[str, str] = {
    "italian": "it",
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "portuguese": "pt",
    "german": "de",
    "dutch": "nl",
}


def language_code(language_name: str) -> str:
    code = LANGUAGE_CODE_MAP.get(language_name.lower())
    if code is not None:
        return code
    return language_name.lower()[:2]


OOM_HINT = (
    "OutOfMemoryError: not enough VRAM for the selected model.\n"
    '  -> Set model_size: "0.6b" in config.yaml\n'
    '  -> OR set device_map: "auto" for partial CPU offload\n'
    "  -> OR reduce batch_size in config.yaml"
)

CUDA_HINT = (
    "CUDA is not available on this machine.\n"
    '  -> Check with: poetry run python -c "import torch; print(torch.cuda.is_available())"\n'
    "  -> On Docker make sure to use: docker run --gpus all ...\n"
    "  -> Without a GPU the pipeline will be extremely slow or will not work."
)


def _resolve_path(p: str | os.PathLike[str], is_input: bool = False) -> Path:
    """Resolve a path relative to the project root (or inputs/ if is_input).

    Absolute paths are returned unchanged. Relative paths are joined to
    PROJECT_ROOT; with is_input=True they are joined to PROJECT_ROOT/inputs/.

    Args:
        p: Path string or PathLike.
        is_input: If True, resolve relative to inputs/ rather than root.

    Returns:
        An absolute Path.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    if is_input:
        return PROJECT_ROOT / "inputs" / path
    return PROJECT_ROOT / path


@dataclass
class Paths:
    """Container for all file/directory paths used by the pipeline.

    All paths are resolved relative to the project root unless absolute.
    Only `input_sentences` and `test_sentences` are configurable from
    config.yaml; every other path is a fixed default defined here and not
    exposed to users, since the workspace layout is an internal contract of
    the pipeline (the user-facing output lives under output/gen{NNN}/).
    """

    input_sentences: Path = field(
        default_factory=lambda: _resolve_path("sentences.txt", is_input=True)
    )
    test_sentences: Path = field(
        default_factory=lambda: _resolve_path("test_sentences.txt", is_input=True)
    )
    raw_wav: Path = field(default_factory=lambda: _resolve_path("workspace/raw_wav"))
    accepted_wav: Path = field(
        default_factory=lambda: _resolve_path("workspace/accepted_wav")
    )
    rejected: Path = field(default_factory=lambda: _resolve_path("workspace/rejected"))
    manifest_train: Path = field(
        default_factory=lambda: _resolve_path("workspace/.manifest_train.csv")
    )
    manifest_val: Path = field(
        default_factory=lambda: _resolve_path("workspace/.manifest_val.csv")
    )
    report: Path = field(
        default_factory=lambda: _resolve_path("workspace/.report.json")
    )
    checkpoint: Path = field(
        default_factory=lambda: _resolve_path("workspace/.generate_checkpoint.json")
    )
    log_file: Path = field(default_factory=lambda: _resolve_path("logs/pipeline.log"))
    prompt_cache: Path = field(
        default_factory=lambda: _resolve_path("workspace/.voice_cache")
    )


@dataclass
class Config:
    """All configuration parameters parsed from config.yaml.

    The `speaker` field is the voice identity for both modes: a built-in
    preset speaker name when model_type="custom_voice", or a custom voice
    name under inputs/voices/<speaker>.wav (+ optional <speaker>.txt) when
    model_type="base".

    Attributes are set from the YAML file by load_config(), falling back
    to the default values defined here.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    model_size: str = "0.6b"
    model_type: str = "custom_voice"
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    attn_implementation: str = "sdpa"
    speaker: str = "Vivian"
    language: str = "Auto"
    instruct: str = ""
    x_vector_only_mode: bool = False
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
    paths: Paths = field(default_factory=Paths)

    @property
    def model_hub_id(self) -> str:
        """Return the HuggingFace model hub ID for the selected model_type and model_size."""
        try:
            return MODEL_HUB_IDS[self.model_type.lower()][self.model_size.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Invalid model_type/model_size: '{self.model_type}'/'{self.model_size}'. "
                f"Valid model_type: {VALID_MODEL_TYPES}; valid model_size: 0.6b, 1.7b."
            ) from exc


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
    cfg.model_type = raw.get("model_type", cfg.model_type)
    if cfg.model_type.lower() not in MODEL_HUB_IDS:
        raise ValueError(
            f"Invalid model_type: '{cfg.model_type}'. Use one of: {VALID_MODEL_TYPES}."
        )
    cfg.dtype = raw.get("dtype", cfg.dtype)
    cfg.device_map = raw.get("device_map", cfg.device_map)
    cfg.attn_implementation = raw.get("attn_implementation", cfg.attn_implementation)
    cfg.speaker = raw.get("speaker", cfg.speaker)
    cfg.language = raw.get("language", cfg.language)
    cfg.instruct = raw.get("instruct", cfg.instruct)
    cfg.x_vector_only_mode = bool(raw.get("x_vector_only_mode", cfg.x_vector_only_mode))
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

    cfg.paths = Paths(
        input_sentences=_resolve_path(
            raw.get("input_sentences", "sentences.txt"), is_input=True
        ),
        test_sentences=_resolve_path(
            raw.get("test_sentences", "test_sentences.txt"), is_input=True
        ),
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
        or "cuda" in msg
        and "memory" in msg
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
    logger.info(
        "CUDA available: %s (%d GPU)",
        torch.cuda.get_device_name(0),
        torch.cuda.device_count(),
    )


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

    for f in [
        cfg.paths.manifest_train,
        cfg.paths.manifest_val,
        cfg.paths.report,
        cfg.paths.checkpoint,
    ]:
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
        with src_path.open("r", encoding="utf-8") as fin, dest_path.open(
            "w", encoding="utf-8", newline=""
        ) as fout:
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
    logger.info(
        "Archived generation %03d: %d wavs -> %s", gen_number, wavs_moved, gen_dir
    )


def read_rejected_indices(cfg: Config) -> set[int]:
    """Read the set of rejected sentence indices from workspace/rejected/*.json.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Set of rejected sentence indices.
    """
    rejected_dir = cfg.paths.rejected
    if not rejected_dir.exists():
        return set()
    indices: set[int] = set()
    for p in rejected_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            idx = data.get("index")
            if idx is not None:
                indices.add(int(idx))
        except (json.JSONDecodeError, OSError):
            continue
    return indices


def accept_clips(cfg: Config, indices: list[int]) -> dict[str, int]:
    """Manually accept rejected clips by copying them to accepted_wav/.

    Args:
        cfg: Pipeline configuration.
        indices: List of sentence indices to accept.

    Returns:
        Dict with accepted and not_found counts.
    """
    import shutil

    logger = logging.getLogger("qwen3_tts_dataset")
    accepted = 0
    not_found = 0
    for idx in indices:
        src = cfg.paths.raw_wav / f"{idx:06d}.wav"
        if not src.exists():
            logger.warning("Clip %s not found in raw_wav, skipping.", src.name)
            not_found += 1
            continue
        dest = cfg.paths.accepted_wav / src.name
        cfg.paths.accepted_wav.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
        for p in (
            cfg.paths.rejected / f"{src.stem}.json",
            cfg.paths.rejected / src.name,
        ):
            if p.exists():
                p.unlink()
        logger.info("Manually accepted idx=%d -> %s", idx, dest.name)
        accepted += 1
    logger.info("Manual accept: %d accepted, %d not found", accepted, not_found)
    return {"accepted": accepted, "not_found": not_found}


def resolve_voice_paths(cfg: Config) -> tuple[Path, Path]:
    """Resolve the reference audio and transcript paths for the configured speaker.

    The voice files live directly under inputs/voices/ as <speaker>.wav (always)
    and <speaker>.txt (required only for ICL mode, i.e. when x_vector_only_mode is False).

    Args:
        cfg: Pipeline configuration.

    Returns:
        A (ref_wav, ref_text) tuple of absolute paths.

    Raises:
        ValueError: If speaker is empty.
        FileNotFoundError: If <speaker>.wav does not exist.
    """
    if not cfg.speaker:
        raise ValueError(
            "speaker is required when model_type='base'. "
            "Set speaker in config.yaml to a voice name under inputs/voices/."
        )
    voices_dir = PROJECT_ROOT / "inputs" / "voices"
    ref_wav = voices_dir / f"{cfg.speaker}.wav"
    ref_text = voices_dir / f"{cfg.speaker}.txt"
    if not ref_wav.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {ref_wav}. "
            f"Place a wav file at inputs/voices/{cfg.speaker}.wav."
        )
    return ref_wav, ref_text


def list_available_voices(cfg: Config) -> list[str]:
    """List available custom voice names under inputs/voices/.

    A voice is any .wav file directly in inputs/voices/. The filename stem
    (without .wav) is the voice name.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Sorted list of voice names. Empty list if inputs/voices/ does not exist.
    """
    voices_dir = PROJECT_ROOT / "inputs" / "voices"
    if not voices_dir.exists():
        return []
    voices: list[str] = []
    for p in sorted(voices_dir.iterdir()):
        if p.suffix.lower() == ".wav":
            voices.append(p.stem)
    return voices


def voice_fingerprint(cfg: Config) -> str:
    """Compute a stable fingerprint of the current voice-clone configuration.

    Used to invalidate the cached VoiceClonePromptItem when the reference audio,
    transcript, cloning mode, model type, or model size change.

    Args:
        cfg: Pipeline configuration.

    Returns:
        A SHA-256 hex digest string.
    """
    import hashlib

    ref_wav, ref_text = resolve_voice_paths(cfg)
    h = hashlib.sha256()
    h.update(cfg.model_type.encode("utf-8"))
    h.update(cfg.model_size.encode("utf-8"))
    h.update(str(cfg.x_vector_only_mode).encode("utf-8"))
    h.update(ref_wav.resolve().as_posix().encode("utf-8"))
    st = ref_wav.stat()
    h.update(str(st.st_mtime).encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    if ref_text.exists():
        h.update(ref_text.read_bytes())
    return h.hexdigest()
