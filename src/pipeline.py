"""CLI pipeline orchestrator."""
from __future__ import annotations

import click

from . import common
from . import generate as gen_mod
from . import validate as val_mod
from . import normalize_audio as norm_mod
from . import build_manifest as man_mod
from . import report as rep_mod

STEPS = ["generate", "validate", "normalize", "manifest", "report", "all"]


@click.command()
@click.option("--config", "config_path", default=None, help="Path to config.yaml (default: config.yaml in project root).")
@click.option("--step", type=click.Choice(STEPS, case_sensitive=False), default="all", help="Run a single pipeline step.")
@click.option("--no-clean", is_flag=True, default=False, help="Skip auto-clean of workspace on full pipeline run.")
def main(config_path: str | None, step: str, no_clean: bool) -> None:
    """Orchestrate the TTS dataset pipeline.

    Loads config, sets up logging, and runs the requested pipeline step(s).
    Steps are: generate, validate, normalize, manifest, report, or all.

    Args:
        config_path: Path to config.yaml (optional, uses default if None).
        step: Pipeline step or "all" to run the complete pipeline.
        no_clean: If True, skip auto-clean of workspace on full run.

    Raises:
        SystemExit: If a step raises SystemExit (e.g. OOM error).
        click.ClickException: On other pipeline failures.
    """
    step = step.lower()
    cfg = common.load_config(config_path)
    common.ensure_dirs(
        cfg.paths.raw_wav,
        cfg.paths.accepted_wav,
        cfg.paths.rejected,
        cfg.paths.manifest_train.parent,
        cfg.paths.report.parent,
        cfg.paths.log_file.parent,
    )
    logger = common.setup_logging(cfg.paths.log_file)

    # Auto-clean workspace on full run
    if step == "all" and cfg.clean_on_full_run and not no_clean:
        logger.info("Auto-clean: clearing workspace for a fresh run.")
        common.clean_working_dirs(cfg)

    logger.info("Starting pipeline step='%s' model_size=%s", step, cfg.model_size)

    gen_stats: dict | None = None
    val_stats: dict | None = None
    norm_stats: dict | None = None
    man_stats: dict | None = None

    try:
        if step in ("generate", "all"):
            gen_stats = gen_mod.run_generate(cfg)
        if step in ("validate", "all"):
            val_stats = val_mod.run_validate(cfg)
        if step in ("normalize", "all"):
            norm_stats = norm_mod.run_normalize(cfg)
        if step in ("manifest", "all"):
            man_stats = man_mod.run_build_manifest(cfg)
        if step in ("report", "all"):
            rep_mod.run_report(cfg, gen_stats, val_stats, norm_stats, man_stats)

        # Archive to output/gen{NNN}/ after successful full run
        if step == "all":
            gen_number = common.next_gen_number()
            common.archive_generation(cfg, gen_number)

    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()