"""CLI pipeline orchestrator."""

from __future__ import annotations

import logging

import click

from . import build_manifest as man_mod
from . import common
from . import generate as gen_mod
from . import normalize_audio as norm_mod
from . import report as rep_mod
from . import validate as val_mod

logger = logging.getLogger(__name__)

STEPS = ["generate", "validate", "normalize", "publish", "all"]
STEPS_ORDER = ["generate", "validate", "normalize", "publish"]


@click.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config.yaml (default: config.yaml in project root).",
)
@click.option(
    "--step",
    type=click.Choice(STEPS, case_sensitive=False),
    default="all",
    help="Run a single pipeline step: generate, validate, normalize, publish "
    "(manifest+report+archive). Default: all.",
)
@click.option(
    "--from",
    "from_step",
    type=click.Choice(STEPS_ORDER, case_sensitive=False),
    default=None,
    help="Run all steps from this point onward (no auto-clean). "
    "E.g. --from validate runs validate+normalize+publish.",
)
@click.option(
    "--no-clean",
    is_flag=True,
    default=False,
    help="Skip auto-clean of workspace on full pipeline run.",
)
@click.option(
    "--only-rejected",
    is_flag=True,
    default=False,
    help="With --step generate: regenerate only previously rejected clips.",
)
@click.option(
    "--accept",
    "accept_indices",
    type=str,
    default=None,
    help="Manually accept rejected clips by index (comma-separated, e.g. '7,13').",
)
def main(
    config_path: str | None,
    step: str,
    from_step: str | None,
    no_clean: bool,
    only_rejected: bool,
    accept_indices: str | None,
) -> None:
    """Orchestrate the TTS dataset pipeline.

    Steps: generate -> validate -> normalize -> publish

    - generate:  create audio from text corpus via Qwen3-TTS
    - validate:  check clips with ASR (faster-whisper) + WER, accept/reject
    - normalize: resample, loudness, trim silence, 16-bit PCM (in-place on accepted)
    - publish:   build LJSpeech manifest + report + archive to output/gen{NNN}/

    Default (no flags): full run with auto-clean + archive.
    Use --from to resume from a specific step without auto-cleaning.
    """
    step = step.lower()

    # --- Manual accept (standalone, no steps) ---
    if accept_indices is not None:
        cfg = common.load_config(config_path)
        common.ensure_dirs(
            cfg.paths.raw_wav,
            cfg.paths.accepted_wav,
            cfg.paths.rejected,
            cfg.paths.log_file.parent,
        )
        common.setup_logging(cfg.paths.log_file)
        indices = [int(i.strip()) for i in accept_indices.split(",") if i.strip()]
        common.accept_clips(cfg, indices)
        return

    # --- Determine steps to run ---
    if from_step is not None:
        start_idx = STEPS_ORDER.index(from_step.lower())
        steps_to_run = STEPS_ORDER[start_idx:]
        do_clean = False
    elif step == "all":
        steps_to_run = list(STEPS_ORDER)
        do_clean = True
    else:
        steps_to_run = [step]
        do_clean = False

    # --- Validate flags ---
    if only_rejected and "generate" not in steps_to_run:
        raise click.UsageError(
            "--only-rejected can only be used with --step generate "
            "or --from generate."
        )

    # --- Load config and set up ---
    cfg = common.load_config(config_path)
    common.ensure_dirs(
        cfg.paths.raw_wav,
        cfg.paths.accepted_wav,
        cfg.paths.rejected,
        cfg.paths.manifest_train.parent,
        cfg.paths.report.parent,
        cfg.paths.log_file.parent,
    )
    common.setup_logging(cfg.paths.log_file)

    # Auto-clean workspace on full run
    if do_clean and cfg.clean_on_full_run and not no_clean:
        logger.info("Auto-clean: clearing workspace for a fresh run.")
        common.clean_working_dirs(cfg)

    logger.info(
        "Starting pipeline steps=%s model_size=%s",
        steps_to_run,
        cfg.model_size,
    )

    # --- Run pipeline steps ---
    gen_stats: dict | None = None
    val_stats: dict | None = None
    norm_stats: dict | None = None
    man_stats: dict | None = None

    try:
        for s in steps_to_run:
            if s == "generate":
                gen_stats = gen_mod.run_generate(cfg, only_rejected=only_rejected)
            elif s == "validate":
                val_stats = val_mod.run_validate(cfg)
            elif s == "normalize":
                norm_stats = norm_mod.run_normalize(cfg)
            elif s == "publish":
                # publish = manifest + report + archive
                man_stats = man_mod.run_build_manifest(cfg)
                rep_mod.run_report(cfg, gen_stats, val_stats, norm_stats, man_stats)
                gen_number = common.next_gen_number()
                common.archive_generation(cfg, gen_number)

    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()
