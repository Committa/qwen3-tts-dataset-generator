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
    logger.info("Loading %s (dtype=%s, device_map=%s, attn=%s)",
                model_id, cfg.dtype, cfg.device_map, cfg.attn_implementation)
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
@click.option("--model-size", "model_size", default=None, type=click.Choice(["1.7b", "0.6b"], case_sensitive=False),
              help="Override model_size from config for this test.")
@click.option("--out-dir", default="output/test_speaker", help="Output directory for test wavs.")
def main(config_path: str | None, model_size: str | None, out_dir: str) -> None:
    """Run speaker test: generate sample audio for all speakers.

    Tests every available speaker with the phrases from inputs/test_sentences.txt.
    Useful for choosing the best speaker and verifying the model runs on the GPU
    before launching the full pipeline.

    Args:
        config_path: Path to config.yaml (optional).
        model_size: Override model size (1.7b or 0.6b).
        out_dir: Output directory for generated test wavs.
    """
    cfg = common.load_config(config_path)
    if model_size:
        cfg.model_size = model_size.lower()
    logger = common.setup_logging(cfg.paths.log_file)
    common.check_cuda_or_die(logger)
    out_dir_path = (common.PROJECT_ROOT / out_dir) if not Path(out_dir).is_absolute() else Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    model = _load_tts(cfg.model_hub_id, cfg, logger)
    speakers = model.get_supported_speakers()
    logger.info("Tested speakers: %s", speakers)

    # Load test phrases from inputs/test_sentences.txt
    phrases_path = cfg.paths.test_sentences
    phrases = [ln.strip() for ln in phrases_path.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.startswith("#")]
    if not phrases:
        logger.warning("No test phrases found in %s", phrases_path)
        return

    for speaker in speakers:
        for i, sentence in enumerate(phrases):
            try:
                wavs, sr = model.generate_custom_voice(
                    text=sentence,
                    language=cfg.language,
                    speaker=speaker,
                    max_new_tokens=cfg.max_new_tokens,
                )
                fname = out_dir_path / f"{speaker}_{i:02d}.wav"
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
    logger.info("Test completed. Files in: %s", out_dir_path)


if __name__ == "__main__":
    main()