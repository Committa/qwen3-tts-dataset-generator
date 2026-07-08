"""CLI pipeline orchestrator."""

from __future__ import annotations

import logging
import sys

import click

from . import build_manifest as man_mod
from . import common
from . import generate as gen_mod
from . import normalize_audio as norm_mod
from . import pronunciation as pron_mod
from . import report as rep_mod
from . import validate as val_mod

logger = logging.getLogger(__name__)

STEPS = ["generate", "validate", "pronunciation", "normalize", "publish", "all"]
STEPS_ORDER = ["generate", "validate", "pronunciation", "normalize", "publish"]


def _maybe_clean_workspace(cfg: common.Config, do_clean: bool, no_clean: bool) -> None:
    """Run the auto-clean logic for a full pipeline run, with resume detection.

    When a full run would normally auto-clean the workspace, check for an
    incomplete generation checkpoint. If one exists (some sentences done and
    some still pending), prompt the user to choose between resuming the
    previous run (skip clean) or starting a fresh clean run. In
    non-interactive contexts (no TTY on stdin) the default is to resume, so
    progress is never lost silently. When no checkpoint exists or the
    previous generation is complete, the workspace is cleaned as usual.

    Args:
        cfg: Pipeline configuration.
        do_clean: True when the selected step combination requests a clean
            (i.e. a full run without ``--step``/``--from``).
        no_clean: True when the user passed ``--no-clean``; clean is skipped
            entirely regardless of resume detection.
    """
    if not do_clean or not cfg.clean_on_full_run or no_clean:
        return

    done = common.read_checkpoint(cfg.paths.checkpoint)
    try:
        total = len(common.load_sentences(cfg))
    except FileNotFoundError:
        total = 0

    if not (0 < len(done) < total):
        logger.info("Auto-clean: clearing workspace for a fresh run.")
        common.clean_working_dirs(cfg)
        return

    pending = total - len(done)
    interactive = bool(sys.stdin) and sys.stdin.isatty()
    if interactive:
        choice = click.prompt(
            f"An incomplete generation was found "
            f"({len(done)}/{total} done, {pending} pending).\n"
            "  [r] resume previous generation (skip clean)\n"
            "  [f] start a fresh clean run\n"
            "Choose",
            type=click.Choice(["r", "f"], case_sensitive=False),
            default="r",
            show_default=False,
        )
        resume = choice == "r"
    else:
        logger.info(
            "Non-interactive run with an incomplete checkpoint "
            "(%d/%d done, %d pending): resuming to preserve progress "
            "(delete the checkpoint file or use --no-clean to change).",
            len(done),
            total,
            pending,
        )
        resume = True

    if resume:
        logger.info(
            "Resuming previous generation: %d/%d done, %d pending.",
            len(done),
            total,
            pending,
        )
    else:
        logger.info("Starting a fresh run: clearing workspace.")
        common.clean_working_dirs(cfg)


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
    help="Run a single pipeline step: generate, validate, pronunciation, "
    "normalize, publish (manifest+report+archive). Default: all.",
)
@click.option(
    "--from",
    "from_step",
    type=click.Choice(STEPS_ORDER, case_sensitive=False),
    default=None,
    help="Run all steps from this point onward (no auto-clean). "
    "E.g. --from validate runs validate+pronunciation+normalize+publish.",
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
    "--calibrate",
    is_flag=True,
    default=False,
    help="With --step pronunciation: measure the PER distribution without "
    "rejecting anything, to help tune phoneme_threshold.",
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
    calibrate: bool,
    accept_indices: str | None,
) -> None:
    """Orchestrate the TTS dataset pipeline.

    Steps: generate -> validate -> pronunciation -> normalize -> publish

    - generate:      create audio from text corpus via Qwen3-TTS
    - validate:      check clips with ASR (faster-whisper) + WER, accept/reject
    - pronunciation: phoneme-level check (wav2vec2 CTC + espeak-ng PER) on the
                     WER survivors; rejects bad-pronunciation clips. Gated by
                     the `phoneme_check` config flag in a full run; an explicit
                     `--step pronunciation` always runs it (use `--calibrate`
                     to measure the PER distribution without rejecting).
    - normalize:     resample, loudness, trim silence, 16-bit PCM (in-place)
    - publish:       build LJSpeech manifest + report + archive to output/gen{NNN}/

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
    if calibrate and step != "pronunciation":
        raise click.UsageError(
            "--calibrate can only be used with --step pronunciation."
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

    # Gate the pronunciation step by `phoneme_check` unless the user asked
    # for it explicitly (e.g. calibrating with phoneme_check disabled).
    if (
        "pronunciation" in steps_to_run
        and not cfg.phoneme_check
        and step != "pronunciation"
    ):
        steps_to_run = [s for s in steps_to_run if s != "pronunciation"]

    # Auto-clean workspace on full run (with resume detection)
    _maybe_clean_workspace(cfg, do_clean, no_clean)

    logger.info(
        "Starting pipeline steps=%s model_size=%s",
        steps_to_run,
        cfg.model_size,
    )

    # --- Run pipeline steps ---
    gen_stats: dict | None = None
    val_stats: dict | None = None
    pron_stats: dict | None = None
    norm_stats: dict | None = None
    man_stats: dict | None = None

    try:
        for s in steps_to_run:
            if s == "generate":
                gen_stats = gen_mod.run_generate(cfg, only_rejected=only_rejected)
            elif s == "validate":
                val_stats = val_mod.run_validate(cfg)
            elif s == "pronunciation":
                pron_stats = pron_mod.run_pronunciation(cfg, calibrate=calibrate)
            elif s == "normalize":
                norm_stats = norm_mod.run_normalize(cfg)
            elif s == "publish":
                # publish = manifest + report + archive
                man_stats = man_mod.run_build_manifest(cfg)
                rep_mod.run_report(
                    cfg, gen_stats, val_stats, pron_stats, norm_stats, man_stats
                )
                gen_number = common.next_gen_number()
                common.archive_generation(cfg, gen_number)

    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()
