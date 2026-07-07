"""Generate test phrases with all available speakers to pick the best one.

Also verifies that the selected model (1.7b/0.6b) runs on the GPU before the
full batch. Test phrases are read from inputs/test_sentences.txt. Generation
is batched by cfg.batch_size (override with --batch-size).

Usage:
    poetry run test-gen-dataset
    poetry run test-gen-dataset --model-size 0.6b
    poetry run test-gen-dataset --model-type base
    poetry run test-gen-dataset --batch-size 8
    poetry run test-gen-dataset --instruct "Speak in a calm, neutral, declarative tone."
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import soundfile as sf

from . import common
from . import generate as gen_mod

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "config_path", default=None, help="Path to config.yaml.")
@click.option(
    "--model-size",
    "model_size",
    default=None,
    type=click.Choice(["1.7b", "0.6b"], case_sensitive=False),
    help="Override model_size from config for this test.",
)
@click.option(
    "--out-dir", default="output/test_speaker", help="Output directory for test wavs."
)
@click.option(
    "--speaker",
    "speaker_filter",
    default=None,
    help="Test only this speaker (custom_voice) or voice name (base). "
    "Case-insensitive. Without this flag, all available speakers/voices are tested.",
)
@click.option(
    "--instruct",
    "instruct",
    default=None,
    help="Override instruct text from config.yaml (custom_voice only). "
    "Pass an empty string to disable instruct for this test.",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=None,
    type=click.IntRange(min=1),
    help="Override batch_size from config.yaml for this test "
    "(higher = faster but more VRAM). Default: use config.yaml batch_size.",
)
@click.option(
    "--model-type",
    "model_type",
    default=None,
    type=click.Choice(["custom_voice", "base"], case_sensitive=False),
    help="Override model_type from config.yaml for this test. "
    "custom_voice: sweep preset speakers. "
    "base: sweep custom voices under inputs/voices/.",
)
def main(
    config_path: str | None,
    model_size: str | None,
    out_dir: str,
    speaker_filter: str | None,
    instruct: str | None,
    batch_size: int | None,
    model_type: str | None,
) -> None:
    """Run speaker/voice test: generate sample audio for all available voices.

    In custom_voice mode, tests every built-in speaker with the phrases from
    inputs/test_sentences.txt. In base mode, tests every custom voice found
    under inputs/voices/. Use --speaker to test a single one. Generation is
    batched (see batch_size in config.yaml or --batch-size).
    """
    cfg = common.load_config(config_path)
    if model_size:
        cfg.model_size = model_size.lower()
    if model_type:
        cfg.model_type = model_type.lower()
    if instruct is not None:
        cfg.instruct = instruct
    if batch_size is not None:
        cfg.batch_size = batch_size
    common.setup_logging(cfg.paths.log_file)
    logger.info(
        "Test batch size: %d (from %s)",
        cfg.batch_size,
        "--batch-size" if batch_size is not None else "config.yaml",
    )

    # --- Resolve output directory ---
    out_dir_path = (
        (common.PROJECT_ROOT / out_dir)
        if not Path(out_dir).is_absolute()
        else Path(out_dir)
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # --- Load model (reuses the shared loader: CUDA check, seed, type assert) ---
    model = gen_mod.load_tts_model(cfg)

    # --- Load test phrases ---
    phrases_path = cfg.paths.test_sentences
    phrases = [
        ln.strip()
        for ln in phrases_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not phrases:
        logger.warning("No test phrases found in %s", phrases_path)
        return

    # --- Run the appropriate test sweep ---
    if cfg.model_type == "custom_voice":
        _test_custom_voices(model, cfg, phrases, out_dir_path, speaker_filter)
    else:
        _test_base_voices(model, cfg, phrases, out_dir_path, speaker_filter)
    logger.info("Test completed. Files in: %s", out_dir_path)


def _persist(results, phrases: list[str], out_dir: Path, prefix: str) -> int:
    """Write successful clips to out_dir as {prefix}_{i:02d}.wav plus a matching
    {prefix}_{i:02d}.txt with the exact transcript.

    The .txt travels with the .wav so a chosen clip can be copied straight into
    inputs/voices/ as a voice-cloning reference: the transcript is the ICL
    reference text and must match the audio exactly, so it is written from the
    same source phrase that was synthesized (not transcribed post-hoc).

    Args:
        results: List of (wav, sr) tuples or None, one per phrase.
        phrases: The phrases synthesized, aligned with results by position.
        out_dir: Output directory (already exists).
        prefix: Speaker or voice name used for the filename stem.

    Returns:
        Number of clips actually written to disk.
    """
    written = 0
    for i, res in enumerate(results):
        if res is None:
            continue
        wav, sr = res
        fname = out_dir / f"{prefix}_{i:02d}.wav"
        tname = out_dir / f"{prefix}_{i:02d}.txt"
        try:
            sf.write(str(fname), wav, sr)
            tname.write_text(phrases[i].strip(), encoding="utf-8")
            logger.info("OK %s [%d] -> %s", prefix, i, fname.name)
            written += 1
        except Exception as e:
            logger.warning("Save failed %s [%d]: %s", prefix, i, e)
    return written


def _test_custom_voices(
    model,
    cfg: common.Config,
    phrases: list[str],
    out_dir: Path,
    speaker_filter: str | None,
) -> None:
    """Generate test phrases for every built-in speaker (custom_voice mode).

    Uses generate_phrases (batched by cfg.batch_size) to synthesize all phrases
    for one speaker in a single sweep, then persists the results.

    Args:
        model: Loaded Qwen3TTSModel.
        cfg: Pipeline configuration.
        phrases: Test phrases to synthesize.
        out_dir: Output directory for test wavs.
        speaker_filter: Optional speaker name to test (case-insensitive).
    """
    speakers = model.get_supported_speakers()
    if speaker_filter:
        speakers = [s for s in speakers if s.lower() == speaker_filter.lower()]
        if not speakers:
            logger.warning(
                "Speaker '%s' not found. Available: %s",
                speaker_filter,
                model.get_supported_speakers(),
            )
            return
    logger.info("Testing speakers: %s", speakers)
    logger.info("Instruct: %s", cfg.instruct if cfg.instruct else "(disabled)")
    for speaker in speakers:
        results = gen_mod.generate_phrases(
            model,
            cfg,
            phrases,
            speaker_override=speaker,
            on_skip=lambda i, r: logger.warning("Skip %s [%d] %s", speaker, i, r),
        )
        _persist(results, phrases, out_dir, prefix=speaker)


def _test_base_voices(
    model,
    cfg: common.Config,
    phrases: list[str],
    out_dir: Path,
    speaker_filter: str | None,
) -> None:
    """Generate test phrases for every custom voice under inputs/voices/ (base mode).

    Temporarily sets cfg.speaker to each voice name (needed by
    get_voice_clone_prompt) and restores it afterwards. Uses generate_phrases
    (batched by cfg.batch_size) to synthesize all phrases for one voice in a
    single sweep, then persists the results.

    Args:
        model: Loaded Qwen3TTSModel (base type).
        cfg: Pipeline configuration.
        phrases: Test phrases to synthesize.
        out_dir: Output directory for test wavs.
        speaker_filter: Optional voice name to test (case-insensitive).
    """
    voices = common.list_available_voices(cfg)
    if speaker_filter:
        voices = [v for v in voices if v.lower() == speaker_filter.lower()]
        if not voices:
            logger.warning(
                "Voice '%s' not found. Available: %s",
                speaker_filter,
                common.list_available_voices(cfg),
            )
            return
    if not voices:
        logger.warning(
            "No custom voices found under inputs/voices/. "
            "Create inputs/voices/<name>.wav first."
        )
        return
    if cfg.instruct:
        logger.warning("instruct is set but ignored in base mode (custom_voice only).")
    logger.info("Testing voices: %s", voices)

    original_speaker = cfg.speaker
    try:
        for voice_name in voices:
            cfg.speaker = voice_name
            try:
                prompt_items = gen_mod.get_voice_clone_prompt(model, cfg)
            except Exception as e:
                logger.warning(
                    "Failed to build prompt for voice '%s': %s", voice_name, e
                )
                continue
            results = gen_mod.generate_phrases(
                model,
                cfg,
                phrases,
                voice_clone_prompt=prompt_items,
                on_skip=lambda i, r: logger.warning(
                    "Skip %s [%d] %s", voice_name, i, r
                ),
            )
            _persist(results, phrases, out_dir, prefix=voice_name)
    finally:
        cfg.speaker = original_speaker


if __name__ == "__main__":
    main()
