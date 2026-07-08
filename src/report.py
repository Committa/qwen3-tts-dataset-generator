"""Step 6: final report with statistics and total audio duration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import common

logger = logging.getLogger(__name__)


def _audio_duration_seconds(wav_path: Path) -> float:
    """Return the duration of a wav file in seconds, or 0.0 on error."""
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def run_report(
    cfg: common.Config,
    generate_stats: dict | None = None,
    validate_stats: dict | None = None,
    pronunciation_stats: dict | None = None,
    normalize_stats: dict | None = None,
    manifest_stats: dict | None = None,
) -> dict[str, Any]:
    """Generate the final pipeline report with all statistics.

    Computes totals from disk state (accepted/rejected wavs, durations) and
    merges optional stats dicts passed from earlier pipeline steps. Writes the
    report to JSON and prints a human-readable summary.

    Args:
        cfg: Pipeline configuration.
        generate_stats: Output from run_generate, if available.
        validate_stats: Output from run_validate, if available.
        pronunciation_stats: Output from run_pronunciation, if available.
        normalize_stats: Output from run_normalize, if available.
        manifest_stats: Output from run_build_manifest, if available.

    Returns:
        The full report dict.
    """
    common.setup_logging(cfg.paths.log_file)

    # --- Gather disk state ---
    sentences = common.load_sentences(cfg)
    accepted = sorted(
        cfg.paths.accepted_wav.glob("*.wav"), key=lambda p: int(Path(p.name).stem)
    )
    rejected_files = list(cfg.paths.rejected.glob("*.wav"))

    total_duration = 0.0
    for w in accepted:
        total_duration += _audio_duration_seconds(w)

    # --- Assemble report ---
    mean_wer = (validate_stats or {}).get("mean_wer", 0.0)
    skipped_gen = (generate_stats or {}).get("skipped", 0)
    pron = pronunciation_stats or {}

    report = {
        "totals": {
            "input_sentences": len(sentences),
            "accepted": len(accepted),
            "rejected": len(rejected_files),
            "skipped_during_generation": skipped_gen,
        },
        "quality": {
            "mean_wer": round(mean_wer, 4),
            "wer_threshold": cfg.wer_threshold,
            "mean_per": round(pron.get("mean_per", 0.0), 4),
            "phoneme_threshold": cfg.phoneme_threshold,
        },
        "audio": {
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_hhmmss": _seconds_to_hhmmss(total_duration),
            "sample_rate_hz": cfg.target_sample_rate,
            "target_lufs": cfg.target_lufs,
        },
        "manifest": manifest_stats or {},
        "generation_time_seconds": round(
            (generate_stats or {}).get("time_seconds", 0.0), 2
        ),
        "pronunciation": {
            "enabled": cfg.phoneme_check,
            "model": cfg.phoneme_model,
            "checked": pron.get("checked", 0),
            "phoneme_rejected": pron.get("phoneme_rejected", 0),
            "mean_per": round(pron.get("mean_per", 0.0), 4),
            "threshold": cfg.phoneme_threshold,
            "calibration": pron.get("calibration"),
        },
        "model": {
            "model_size": cfg.model_size,
            "model_type": cfg.model_type,
            "speaker": cfg.speaker,
            "x_vector_only_mode": (
                cfg.x_vector_only_mode if cfg.model_type == "base" else None
            ),
            "language": cfg.language,
            "instruct": cfg.instruct,
            "dtype": cfg.dtype,
            "sampling": {
                "do_sample": cfg.do_sample,
                "temperature": cfg.temperature,
                "top_k": cfg.top_k,
                "top_p": cfg.top_p,
                "repetition_penalty": cfg.repetition_penalty,
                "max_new_tokens": cfg.max_new_tokens,
            },
        },
    }

    # --- Write report to disk and print summary ---
    cfg.paths.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Report written to %s", cfg.paths.report)
    _print_report(report)
    return report


def _seconds_to_hhmmss(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.sss."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _print_report(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the final report to stdout."""
    print("\n" + "=" * 60)
    print("FINAL REPORT - Synthetic TTS dataset")
    print("=" * 60)
    t = report["totals"]
    print(f"  Input sentences       : {t['input_sentences']}")
    print(f"  Accepted clips        : {t['accepted']}")
    print(f"  Rejected clips        : {t['rejected']}")
    print(f"  Skipped (generation)  : {t['skipped_during_generation']}")
    print("-" * 60)
    q = report["quality"]
    print(
        f"  Mean WER              : {q['mean_wer']:.4f}  (threshold {q['wer_threshold']})"
    )
    print(
        f"  Mean PER              : {q['mean_per']:.4f}  (threshold {q['phoneme_threshold']})"
    )
    a = report["audio"]
    print(
        f"  Total audio duration  : {a['total_duration_hhmmss']} ({a['total_duration_seconds']} s)"
    )
    print(
        f"  Sample rate           : {a['sample_rate_hz']} Hz | LUFS target {a['target_lufs']}"
    )
    m = report.get("manifest", {})
    if m:
        print("-" * 60)
        print(f"  Train manifest        : {m.get('train', 0)} rows")
        print(f"  Val manifest          : {m.get('val', 0)} rows")
        print(f"  Train file            : {m.get('manifest_train', '')}")
        print(f"  Val file              : {m.get('manifest_val', '')}")
    print("=" * 60 + "\n")
