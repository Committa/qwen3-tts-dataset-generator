"""Shared utilities: config loading, paths, logging, VRAM/OOM handling."""

from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tqdm as _tqdm
import yaml

logger = logging.getLogger(__name__)

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

VALID_ASR_LANG_CODES: frozenset[str] = frozenset(
    {
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "zh",
        "yue",
    }
)

LANGUAGE_CODE_MAP: dict[str, str] = {
    "italian": "it",
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "portuguese": "pt",
    "german": "de",
    "dutch": "nl",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
}


def language_code(language_name: str) -> str:
    """Convert a language name or ISO code to a faster-whisper-compatible ISO 639-1 code.

    Only explicit languages are accepted — automatic detection ("Auto") is
    refused because it would disable WER text normalization (alpha2digit),
    producing silently inaccurate results.

    Args:
        language_name: Language name (e.g. ``"Italian"``) or ISO 639-1
            2-letter code (e.g. ``"it"``).

    Returns:
        A valid faster-whisper ISO 639-1 2-letter code.

    Raises:
        ValueError: If the value is ``"Auto"`` (or any auto-detect variant)
            or is not a recognised language name or ISO code.
    """
    AUTO_VALUES = {"auto", "autodetect", "auto-detect", "auto detect"}
    lang = language_name.strip().lower()
    if lang in AUTO_VALUES:
        raise ValueError(
            f"language '{language_name}' is not supported by this pipeline.\n"
            "  Auto-detection ('Auto') disables WER text normalisation "
            "(alpha2digit) and is not accepted.\n"
            '  Set an explicit language in config.yaml, e.g. language: "Italian".'
        )

    code = LANGUAGE_CODE_MAP.get(lang)
    if code is not None:
        return code

    if len(lang) == 2 and lang in VALID_ASR_LANG_CODES:
        return lang

    supported_names = sorted(LANGUAGE_CODE_MAP.keys())
    raise ValueError(
        f"language '{language_name}' is not recognised.\n"
        "  Use one of the supported language names or an ISO 639-1 2-letter code.\n"
        f"  Supported names: {', '.join(supported_names)}.\n"
        '  Example: language: "Italian" or language: "it".'
    )


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
    validate_checkpoint: Path = field(
        default_factory=lambda: _resolve_path("workspace/.validate_checkpoint.json")
    )
    review_checkpoint: Path = field(
        default_factory=lambda: _resolve_path("workspace/.review_checkpoint.json")
    )
    pronunciation_checkpoint: Path = field(
        default_factory=lambda: _resolve_path(
            "workspace/.pronunciation_checkpoint.json"
        )
    )
    regenerated: Path = field(
        default_factory=lambda: _resolve_path("workspace/.regenerated.json")
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

    model_size: str = "1.7b"
    model_type: str = "custom_voice"
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    attn_implementation: str = "sdpa"
    speaker: str = "Vivian"
    language: str = "Italian"
    instruct: str = ""
    x_vector_only_mode: bool = False
    batch_size: int = 4
    seed: int = 42
    max_new_tokens: int = 2048
    # Sampling parameters forwarded to the Qwen3-TTS model. `temperature` is
    # the main lever for cross-clip consistency (lower => less variance); the
    # model is tuned at 0.9 and going well below ~0.6 can trigger EOS-collapse
    # (premature end-of-speech => truncated clips), recovered via validate ->
    # `--only-rejected`. `top_p` is kept at 1.0: at low temperature, top_p < 1.0
    # cuts the continuation tail and increases truncation with no consistency
    # gain. do_sample=False is greedy (deterministic, but may loop on codec
    # TTS models); prefer low temperature with do_sample=True.
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    asr_model: str = "medium"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_workers: int = 1
    asr_beam_size: int = 5
    wer_threshold: float = 0.15
    # Pronunciation verification (phoneme-level). Catches clips that pass WER
    # but whose actual pronunciation is wrong: the audio is recognised to
    # espeak phonemes by a wav2vec2 CTC model and compared (PER) against the
    # espeak-ng text->phoneme rendering of the reference sentence. See
    # src/pronunciation.py. `phoneme_check` gates the step in the full
    # pipeline run; an explicit `--step pronunciation` always runs it.
    phoneme_check: bool = False
    phoneme_model: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    phoneme_device: str = "cuda"
    phoneme_batch_size: int = 8
    # Cadence (in batches) at which the wav2vec2 recognize loop frees the
    # PyTorch CUDA allocator pool. Unlike TTS generation (which holds a single
    # huge model for the whole run), the wav2vec2 forward is short and runs many
    # times over hundreds of clips; the caching allocator never returns memory
    # to the driver without empty_cache(), so reserved VRAM grows monotonically
    # until OOM. 10 batches (~80 clips at batch_size=8) is a good balance between
    # the overhead of empty_cache() (~100-500 ms) and keeping the pool bounded.
    phoneme_cleanup_every_n_batches: int = 10
    phoneme_threshold: float = 0.30
    phoneme_word_report: bool = True
    phoneme_word_top_n: int = 20
    # Minimum number of occurrences for a word to appear in the "worst words"
    # log and pronunciation report. Words that appear only once in the corpus
    # usually have a high PER by chance (a single bad TTS render) and are not
    # useful for deciding which words to remove or reword.
    phoneme_report_min_occurrences: int = 3
    target_sample_rate: int = 22050
    target_lufs: float = -23.0
    trim_silence_db: float = 60.0
    tail_margin_ms: float = 120.0
    tail_pad_ms: float = 80.0
    val_ratio: float = 0.1
    mem_cleanup_every_n_batches: int = 100
    clean_on_full_run: bool = True
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
    cfg = Config()
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
    language_code(
        cfg.language
    )  # validate early — raises ValueError on "Auto" / gibberish
    cfg.instruct = raw.get("instruct", cfg.instruct)
    cfg.x_vector_only_mode = bool(raw.get("x_vector_only_mode", cfg.x_vector_only_mode))
    cfg.batch_size = int(raw.get("batch_size", cfg.batch_size))
    cfg.seed = int(raw.get("seed", cfg.seed))
    cfg.max_new_tokens = int(raw.get("max_new_tokens", cfg.max_new_tokens))
    cfg.do_sample = bool(raw.get("do_sample", cfg.do_sample))
    cfg.temperature = float(raw.get("temperature", cfg.temperature))
    cfg.top_k = int(raw.get("top_k", cfg.top_k))
    cfg.top_p = float(raw.get("top_p", cfg.top_p))
    cfg.repetition_penalty = float(
        raw.get("repetition_penalty", cfg.repetition_penalty)
    )
    cfg.asr_model = raw.get("asr_model", cfg.asr_model)
    cfg.asr_device = raw.get("asr_device", cfg.asr_device)
    cfg.asr_compute_type = raw.get("asr_compute_type", cfg.asr_compute_type)
    cfg.asr_workers = int(raw.get("asr_workers", cfg.asr_workers))
    cfg.asr_beam_size = int(raw.get("asr_beam_size", cfg.asr_beam_size))
    cfg.wer_threshold = float(raw.get("wer_threshold", cfg.wer_threshold))
    cfg.phoneme_check = bool(raw.get("phoneme_check", cfg.phoneme_check))
    cfg.phoneme_model = raw.get("phoneme_model", cfg.phoneme_model)
    cfg.phoneme_device = raw.get("phoneme_device", cfg.phoneme_device)
    cfg.phoneme_batch_size = int(raw.get("phoneme_batch_size", cfg.phoneme_batch_size))
    cfg.phoneme_cleanup_every_n_batches = int(
        raw.get("phoneme_cleanup_every_n_batches", cfg.phoneme_cleanup_every_n_batches)
    )
    cfg.phoneme_threshold = float(raw.get("phoneme_threshold", cfg.phoneme_threshold))
    cfg.phoneme_word_report = bool(
        raw.get("phoneme_word_report", cfg.phoneme_word_report)
    )
    cfg.phoneme_word_top_n = int(raw.get("phoneme_word_top_n", cfg.phoneme_word_top_n))
    cfg.phoneme_report_min_occurrences = int(
        raw.get("phoneme_report_min_occurrences", cfg.phoneme_report_min_occurrences)
    )
    cfg.target_sample_rate = int(raw.get("target_sample_rate", cfg.target_sample_rate))
    cfg.target_lufs = float(raw.get("target_lufs", cfg.target_lufs))
    cfg.trim_silence_db = float(raw.get("trim_silence_db", cfg.trim_silence_db))
    cfg.tail_margin_ms = float(raw.get("tail_margin_ms", cfg.tail_margin_ms))
    cfg.tail_pad_ms = float(raw.get("tail_pad_ms", cfg.tail_pad_ms))
    cfg.val_ratio = float(raw.get("val_ratio", cfg.val_ratio))
    cfg.mem_cleanup_every_n_batches = int(
        raw.get("mem_cleanup_every_n_batches", cfg.mem_cleanup_every_n_batches)
    )
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


class _TqdmStreamHandler(logging.StreamHandler):
    """StreamHandler that routes messages through ``tqdm.write()`` so they
    never corrupt an active tqdm progress bar.

    ``tqdm.write()`` prints the message, then redraws any active progress bar
    on the next line. When no bar is active it falls back to a normal write,
    so this handler is safe to use throughout the pipeline regardless of step.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _tqdm.tqdm.write(msg, file=self.stream)
        except Exception:
            self.handleError(record)


class _SuppressPadTokenWarning(logging.Filter):
    """Drop the per-batch ``Setting pad_token_id to eos_token_id`` warning.

    ``transformers.generation.utils`` emits this ``WARNING`` on every
    ``generate()`` call when the model's ``generation_config`` has no
    ``pad_token_id`` (qwen3-tts leaves it unset and defaults to the eos
    token). It is constant noise repeated once per batch and garbles the
    tqdm progress bar on stderr. A level-based fix cannot drop a WARNING
    without also hiding legitimate warnings, so a message filter is the
    surgical fix. Real warnings/errors are unaffected.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Setting `pad_token_id` to `eos_token_id`" not in record.getMessage()


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure dual-output logging (file + stdout) for the whole package.

    Configures the shared parent logger ``"src"`` once. Every module obtains its
    own child logger via ``logging.getLogger(__name__)`` (e.g. ``src.generate``)
    and inherits these handlers through propagation, so no module needs to keep a
    mutable global logger or pass one around. The parent does not propagate to
    the root logger, avoiding duplicate output.

    Args:
        log_file: Path to the log file.
        level: Logging level (default: INFO).

    Returns:
        The configured parent logger instance (``"src"``).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    parent = logging.getLogger("src")
    parent.setLevel(level)
    parent.handlers.clear()
    parent.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = _TqdmStreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    parent.addHandler(fh)
    parent.addHandler(sh)
    for noisy in ("transformers", "qwen_tts"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("transformers.generation.utils").addFilter(
        _SuppressPadTokenWarning()
    )
    return parent


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
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or ("cuda" in msg and "memory" in msg)
        or isinstance(exc, MemoryError)
    )


def exit_on_oom(exc: BaseException, log: logging.Logger) -> None:
    """Log the OOM hint and raise SystemExit(2).

    Centralizes the repeated OOM handling so every call site stays consistent:
    prints the actionable hint and exits with code 2 (OOM), never code 1.

    Args:
        exc: The exception that triggered the OOM detection.
        log: Logger instance for the diagnostic message.

    Raises:
        SystemExit(2): Always.
    """
    log.error(OOM_HINT)
    raise SystemExit(2) from exc


def cleanup_gpu(log: logging.Logger | None = None) -> None:
    """Run garbage collection and flush the PyTorch CUDA allocator pool.

    This resets GPU memory fragmentation that builds up across hundreds
    of batches with variable-length outputs. Called periodically from
    the generation loop; the overhead is negligible (~100-500 ms).
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if log is not None:
            reserved = torch.cuda.memory_reserved() / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            log.debug(
                "GPU cache flushed: allocated=%.2f GiB, reserved=%.2f GiB",
                allocated,
                reserved,
            )


def load_sentences(cfg: Config) -> list[str]:
    """Load the input corpus, skipping blank and comment lines.

    Reads the file at ``cfg.paths.input_sentences`` and returns every non-empty
    line that does not start with ``#``. This is the single source of truth for
    corpus loading shared by the generate, validate, manifest and report steps.

    Args:
        cfg: Pipeline configuration.

    Returns:
        List of sentences in file order.
    """
    lines = cfg.paths.input_sentences.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


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
        cfg.paths.validate_checkpoint,
        cfg.paths.review_checkpoint,
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
        shutil.copy2(str(live_report), str(gen_dir / "report.json"))

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
