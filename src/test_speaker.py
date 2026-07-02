"""Generate test phrases with all available Italian speakers to pick the best one.
Also verifies that the selected model (1.7b/0.6b) runs on the GPU before the full batch.
Test phrases are read from inputs/test_sentences.txt.

Usage:
    poetry run test-gen-dataset
    poetry run test-gen-dataset --model-size 0.6b
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import soundfile as sf
import torch

from . import common


def _load_tts(model_id: str, cfg: common.Config, logger: logging.Logger):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    logger.info(
        "Loading %s (dtype=%s, device_map=%s, attn=%s)",
        model_id,
        cfg.dtype,
        cfg.device_map,
        cfg.attn_implementation,
    )
    try:
        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=cfg.device_map,
            dtype=dtype_map.get(cfg.dtype, torch.bfloat16),
            attn_implementation=cfg.attn_implementation,
        )
    except MemoryError as e:
        logger.error(common.OOM_HINT)
        raise SystemExit(2) from e
    except Exception as e:
        if common.is_oom_error(e):
            logger.error(common.OOM_HINT)
            raise SystemExit(2) from e
        raise


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
def main(
    config_path: str | None,
    model_size: str | None,
    out_dir: str,
    speaker_filter: str | None,
) -> None:
    """Run speaker/voice test: generate sample audio for all available voices.

    In custom_voice mode, tests every built-in speaker with the phrases from
    inputs/test_sentences.txt. In base mode, tests every custom voice found
    under inputs/voices/. Use --speaker to test a single one.

    Args:
        config_path: Path to config.yaml (optional).
        model_size: Override model size (1.7b or 0.6b).
        out_dir: Output directory for generated test wavs.
        speaker_filter: Optional speaker/voice name to test (case-insensitive).
    """
    cfg = common.load_config(config_path)
    if model_size:
        cfg.model_size = model_size.lower()
    logger = common.setup_logging(cfg.paths.log_file)
    common.check_cuda_or_die(logger)
    out_dir_path = (
        (common.PROJECT_ROOT / out_dir)
        if not Path(out_dir).is_absolute()
        else Path(out_dir)
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    model = _load_tts(cfg.model_hub_id, cfg, logger)
    actual_type = getattr(model.model, "tts_model_type", None)
    if actual_type != cfg.model_type:
        raise RuntimeError(
            f"Model type mismatch: config model_type='{cfg.model_type}' "
            f"but loaded model tts_model_type='{actual_type}'. "
            f"Check model_type/model_size in config.yaml."
        )

    # Load test phrases from inputs/test_sentences.txt
    phrases_path = cfg.paths.test_sentences
    phrases = [
        ln.strip()
        for ln in phrases_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not phrases:
        logger.warning("No test phrases found in %s", phrases_path)
        return

    if cfg.model_type == "custom_voice":
        _test_custom_voices(model, cfg, logger, phrases, out_dir_path, speaker_filter)
    else:
        _test_base_voices(model, cfg, logger, phrases, out_dir_path, speaker_filter)
    logger.info("Test completed. Files in: %s", out_dir_path)


def _test_custom_voices(
    model,
    cfg: common.Config,
    logger: logging.Logger,
    phrases: list[str],
    out_dir: Path,
    speaker_filter: str | None,
) -> None:
    """Generate test phrases for every built-in speaker (custom_voice mode).

    Args:
        model: Loaded Qwen3TTSModel.
        cfg: Pipeline configuration.
        logger: Logger instance.
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
    for speaker in speakers:
        for i, sentence in enumerate(phrases):
            try:
                wavs, sr = model.generate_custom_voice(
                    text=sentence,
                    language=cfg.language,
                    speaker=speaker,
                    max_new_tokens=cfg.max_new_tokens,
                )
                fname = out_dir / f"{speaker}_{i:02d}.wav"
                sf.write(str(fname), wavs[0], sr)
                logger.info("OK %s [%d] -> %s", speaker, i, fname.name)
            except MemoryError as e:
                logger.error(common.OOM_HINT)
                raise SystemExit(2) from e
            except Exception as e:
                if common.is_oom_error(e):
                    logger.error(common.OOM_HINT)
                    raise SystemExit(2) from e
                logger.warning("Failed %s [%d]: %s", speaker, i, e)
                continue


def _test_base_voices(
    model,
    cfg: common.Config,
    logger: logging.Logger,
    phrases: list[str],
    out_dir: Path,
    speaker_filter: str | None,
) -> None:
    """Generate test phrases for every custom voice under inputs/voices/ (base mode).

    Args:
        model: Loaded Qwen3TTSModel (base type).
        cfg: Pipeline configuration.
        logger: Logger instance.
        phrases: Test phrases to synthesize.
        out_dir: Output directory for test wavs.
        speaker_filter: Optional voice name to test (case-insensitive).
    """
    from . import generate as gen_mod

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
            "Create inputs/voices/<name>/ref.wav first."
        )
        return
    logger.info("Testing voices: %s", voices)
    for voice_name in voices:
        cfg.voice.name = voice_name
        try:
            prompt_items = gen_mod.get_voice_clone_prompt(
                model, cfg, logger, use_cache=False
            )
        except Exception as e:
            logger.warning("Failed to build prompt for voice '%s': %s", voice_name, e)
            continue
        for i, sentence in enumerate(phrases):
            try:
                wavs, sr = model.generate_voice_clone(
                    text=sentence,
                    language=cfg.language,
                    voice_clone_prompt=prompt_items,
                    max_new_tokens=cfg.max_new_tokens,
                )
                fname = out_dir / f"{voice_name}_{i:02d}.wav"
                sf.write(str(fname), wavs[0], sr)
                logger.info("OK %s [%d] -> %s", voice_name, i, fname.name)
            except MemoryError as e:
                logger.error(common.OOM_HINT)
                raise SystemExit(2) from e
            except Exception as e:
                if common.is_oom_error(e):
                    logger.error(common.OOM_HINT)
                    raise SystemExit(2) from e
                logger.warning("Failed %s [%d]: %s", voice_name, i, e)
                continue


if __name__ == "__main__":
    main()
