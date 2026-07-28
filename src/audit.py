"""Post-publish interactive audit of accepted clips.

A single-command tool that helps the user spot clips that passed the
threshold-based ``validate``/``pronunciation`` filters but still sound bad
(mispronunciation, flat intonation, breathy artefacts like sighs, clicks,
short EOS-collapse truncations). The pronunciation step already logs
per-clip metrics in ``workspace/.audit_per.csv`` (``per`` and stress-aware
``per_pitch``); this tool adds cheap audio-signal scan features on top
(``normalized_wav/`` is the source), joins them into a composite suspect
score, and walks the user through the top-N interactively.

The ``r`` key mirrors the on-disk contract used by ``review-rejected`` and
the validate/pronunciation reject path: a sidecar JSON is written under
``workspace/rejected/<idx>.json`` (with ``reason="manual_audit"`` and
``audited=true``) and the wavs are removed from both ``normalized_wav/``
and ``accepted_wav/``. The next run of ``poetry run gen-dataset --step
generate --only-rejected`` then regenerates those clips with a fresh RNG
draw, followed by ``poetry run gen-dataset --from validate`` to re-validate
and re-publish as ``output/gen{NNN+1}/``.

Workflow:

    poetry run gen-dataset --from validate    # writes .audit_per.csv
    poetry run audit                          # this tool
    # ... mark K clips bad with 'r' ...
    poetry run gen-dataset --step generate --only-rejected
    poetry run gen-dataset --from validate   # re-publish as gen002

Reuses the cross-platform UI helpers (``_Player``, ``_getch``, ``_Colors``)
from :mod:`src.review_rejected` so the look-and-feel matches the existing
``review-rejected`` triage tool.

Usage:
    poetry run audit
    poetry run audit --top 200
    poetry run audit --rank pitch
    poetry run audit --restart
    poetry run audit --dry-run
"""

from __future__ import annotations

import csv
import enum
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import soundfile as sf

from . import common
from .review_rejected import (
    _Colors,
    _getch,
    _maybe_clear,
    _Player,
    _show_feedback,
    _show_minimal_prompt,
    _show_prompt,
    _wants_color,
)

logger = logging.getLogger(__name__)


# Composite score weights. per_pitch dominates because it is the only
# signal the threshold-based PER step cannot see (it strips stress on
# purpose); lf_burst catches breathy artefacts (sighs); crest catches
# clicks/glitches; per catches borderline word-level mispronunciations.
_WEIGHTS: dict[str, float] = {
    "per_pitch": 0.40,
    "per": 0.20,
    "lf_burst": 0.25,
    "crest": 0.15,
}

# Audio scan parameters.
# lf_burst measures broadband low-frequency (80-300 Hz) energy ratio: sighs
# and breathy artefacts concentrate energy in this band. We approximate the
# band via a crude FFT magnitude ratio (no scipy dependency required); the
# number is comparable across clips because normalized_wav has uniform
# loudness and sample rate.
_LF_BAND_HZ = (80.0, 300.0)
_LF_BURST_FFT_SIZE = 2048


@dataclass
class _AuditRow:
    """One candidate row in the audit queue.

    Attributes:
        idx: Sentence index (wav filename stem).
        wav_path: Absolute path of the clip in ``normalized_wav/``.
        expected: Reference sentence from the corpus.
        per: Standard PER (stress stripped). 0.0 if missing from CSV.
        per_pitch: Stress-aware PER. 0.0 if missing from CSV.
        crest: Peak / RMS of the normalized clip (high = clicks/glitches).
        lf_burst: Low-frequency energy ratio (high = sighs/breathy).
        clf_ratio: Fraction of samples with ``|x| >= 0.99`` (clipping).
        dur: Clip duration in seconds.
        suspect_score: Composite rank score in ``[0, 1]`` (1 = worst).
    """

    idx: int
    wav_path: Path
    expected: str
    per: float
    per_pitch: float
    crest: float
    lf_burst: float
    clf_ratio: float
    dur: float
    suspect_score: float


@dataclass
class _AudioFeatures:
    """Cheap per-clip audio-scan features (no model inference)."""

    dur: float
    crest: float
    lf_burst: float
    clf_ratio: float


def _audio_features(wav_path: Path) -> _AudioFeatures | None:
    """Compute cheap audio-signal features for one normalized clip.

    Reads the file in float32 (no full-resample: normalized clips are
    already mono at ``target_sample_rate``, so we use the file's native
    sample rate as-is). Returns ``None`` on read failure so the caller can
    skip the clip without crashing the whole audit run.

    Args:
        wav_path: Absolute path of the clip in ``normalized_wav/``.

    Returns:
        An :class:`_AudioFeatures` instance, or ``None`` on read/decode
        error.
    """
    try:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as exc:
        logger.warning("Audio scan failed for %s: %s", wav_path.name, exc)
        return None
    if data is None or data.size == 0:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)

    dur = float(len(data)) / float(sr)
    abs_data = np.abs(data)
    peak = float(abs_data.max()) if abs_data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    crest = peak / rms if rms > 0 else 0.0

    # Clipping fraction: samples pegged near the 0.99 peak-normalize ceiling.
    clf_ratio = float(np.mean(abs_data >= 0.99)) if abs_data.size else 0.0

    # lf_burst: ratio of low-band energy to total energy via short-time FFT.
    # Crude but stable across clips because normalize enforces uniform
    # loudness and sample rate.
    lf_burst = _lf_burst_ratio(data, sr)

    return _AudioFeatures(dur=dur, crest=crest, lf_burst=lf_burst, clf_ratio=clf_ratio)


def _lf_burst_ratio(data: np.ndarray, sr: float) -> float:
    """Return the fraction of total spectral energy in the 80-300 Hz band.

    Uses a single non-overlapped FFT over the whole clip (clips are short,
    ~3-15 s). For longer clips this is a coarse approximation but the goal
    is *ranking*, not absolute calibration, so the trade-off (one FFT, no
    scipy) is acceptable. The ratio is in ``[0, 1]``; sigh/breathy
    artefacts concentrate energy here and stand out from clean speech.

    Args:
        data: Mono float32 audio.
        sr: Sample rate.

    Returns:
        A float in ``[0, 1]`` (``0.0`` if the spectrum is empty).
    """
    if data.size < 2:
        return 0.0
    n_fft = min(_LF_BURST_FFT_SIZE, data.size)
    # Trim or pad to n_fft (we trim: pad would inject energy at DC).
    segment = data[:n_fft].astype(np.float32)
    # Apply a Hann window to reduce edge leakage.
    window = np.hanning(n_fft).astype(np.float32)
    windowed = segment * window
    spectrum = np.abs(np.fft.rfft(windowed))
    if spectrum.size == 0:
        return 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    total_energy = float(np.sum(spectrum))
    if total_energy <= 0.0:
        return 0.0
    lo, hi = _LF_BAND_HZ
    band_energy = float(np.sum(spectrum[(freqs >= lo) & (freqs <= hi)]))
    return band_energy / total_energy


def _read_audit_per_csv(path: Path) -> dict[int, dict[str, Any]]:
    """Load ``workspace/.audit_per.csv`` into a ``{idx: row}`` dict.

    Tolerant of missing, empty, or partially-corrupted files: a warning is
    logged and an empty dict is returned so the caller can decide whether
    to abort or proceed with audio-only ranking.

    Args:
        path: Path to ``cfg.paths.audit_per_csv``.

    Returns:
        A ``{idx: {"per": float, "per_pitch": float, "ref_phonemes": str,
        "hyp_phonemes": str}}`` dict. Empty on missing/corrupted file.
    """
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin, delimiter="|")
            for row in reader:
                raw_idx = row.get("idx", "")
                if not raw_idx:
                    continue
                try:
                    idx = int(raw_idx)
                except ValueError:
                    continue
                try:
                    per = float(row.get("per") or 0.0)
                    per_pitch = float(row.get("per_pitch") or 0.0)
                except ValueError:
                    per, per_pitch = 0.0, 0.0
                out[idx] = {
                    "per": per,
                    "per_pitch": per_pitch,
                    "ref_phonemes": row.get("ref_phonemes", ""),
                    "hyp_phonemes": row.get("hyp_phonemes", ""),
                }
    except (OSError, csv.Error) as exc:
        logger.warning("Cannot read %s (%s); proceeding audio-only.", path, exc)
        return {}
    return out


def _check_csv_completeness(
    per_rows: dict[int, dict[str, Any]],
    normalized_indices: set[int],
    csv_path: Path,
) -> None:
    """Abort with a clear error if any normalized clip is missing from the CSV.

    Args:
        per_rows: Rows loaded from the PER CSV.
        normalized_indices: Indices present in ``normalized_wav/``.
        csv_path: Path of the CSV (for the error message).

    Raises:
        click.ClickException: if any idx is missing.
    """
    missing = sorted(normalized_indices - set(per_rows.keys()))
    if missing:
        sample = ", ".join(str(i) for i in missing[: min(10, len(missing))])
        more = "" if len(missing) <= 10 else f" (...{len(missing)} total)"
        raise click.ClickException(
            f"PER audit CSV ({csv_path}) is missing {len(missing)} clip(s) "
            f"that exist in normalized_wav/ (e.g. {sample}{more}). "
            "Re-run `poetry run gen-dataset --step pronunciation` to refresh "
            "the CSV, then retry `poetry run audit`."
        )


def _check_csv_staleness(
    per_rows: dict[int, dict[str, Any]],
    normalized_indices: set[int],
    csv_path: Path,
) -> None:
    """Warn (non-blocking) if the PER CSV is older than ``normalized_wav/``.

    A stale CSV may still be usable: the user can proceed audio-only (the
    composite ranking will still use ``per``/``per_pitch`` from the old
    rows); the warning simply surfaces the discrepancy so the user can
    decide whether to re-run pronunciation first.

    Uses a single stat of the ``normalized_wav/`` directory as canary: the
    directory mtime changes on add/remove/rename inside it on most
    filesystems, so a directory newer than the CSV almost always means a
    freshly normalized clip whose pronunciation row is stale.
    """
    if not per_rows:
        return
    try:
        csv_mtime = csv_path.stat().st_mtime
    except OSError:
        return
    nw_dir = csv_path.parent / "normalized_wav"
    try:
        if not nw_dir.exists():
            return
        nw_mtime = nw_dir.stat().st_mtime
    except OSError:
        return
    if nw_mtime > csv_mtime:
        logger.warning(
            "PER audit CSV (%s) may be stale: its mtime %.0f is older than "
            "normalized_wav/ dir mtime %.0f. Re-run "
            "`poetry run gen-dataset --step pronunciation` to refresh the "
            "CSV, or proceed (audio-only signals will still rank on the most "
            "recent wavs).",
            csv_path,
            csv_mtime,
            nw_mtime,
        )


def _percentile_rank(values: dict[int, float]) -> dict[int, float]:
    """Return ``{idx: percentile_rank_in_[0,1]}`` for a sparse idx→value map.

    Higher value → higher rank (i.e. higher = more suspect). Used to
    normalize heterogeneous signals onto the ``[0, 1]`` domain before the
    weighted sum that produces ``suspect_score``.

    Args:
        values: ``{idx: signal_value}`` (may be empty).

    Returns:
        ``{idx: rank_in_[0,1]}``. Missing idx (sparse input handled in
        :func:`_build_queue`).
    """
    if not values:
        return {}
    sorted_vals = sorted(values.values())
    n = len(sorted_vals)
    ranks: dict[int, float] = {}
    for idx, v in values.items():
        # Average rank within ties (Python's bisect would over-credit ties;
        # the simpler linear-interp rank is enough for ranking purposes).
        position = sorted_vals.index(v)  # leftmost match
        # Use position / (n-1) so max value gets rank 1.0 (when n > 1).
        ranks[idx] = position / (n - 1) if n > 1 else 0.0
    return ranks


def _composite_score(
    per: float,
    per_pitch: float,
    crest: float,
    lf_burst: float,
    rank_per: float | None,
    rank_per_pitch: float | None,
    rank_crest: float | None,
    rank_lf_burst: float | None,
) -> float:
    """Combine per-signal percentile ranks into a single ``[0, 1]`` score.

    Missing ranks (CSV incomplete — although we abort before getting here)
    fall back to the raw value itself, normalised to ``[0, 1]`` by clipping:
    such a clip is down-ranked, never up-ranked, so it does not pollute the
    top of the queue.

    Args:
        per, per_pitch, crest, lf_burst: Raw per-clip signal values.
        rank_per, rank_per_pitch, rank_crest, rank_lf_burst: Per-clip
            percentile ranks in ``[0, 1]`` (``None`` if no per-CSV was
            loaded for this signal; currently only for the audio-only
            fallback, which we do not reach because the CSV-completeness
            check is hard-fail).

    Returns:
        A composite ``suspect_score`` in ``[0, 1]``.
    """
    rp = rank_per if rank_per is not None else max(0.0, min(1.0, per))
    rpp = (
        rank_per_pitch if rank_per_pitch is not None else max(0.0, min(1.0, per_pitch))
    )
    rc = rank_crest if rank_crest is not None else 0.0
    rl = rank_lf_burst if rank_lf_burst is not None else 0.0
    return (
        _WEIGHTS["per_pitch"] * rpp
        + _WEIGHTS["per"] * rp
        + _WEIGHTS["lf_burst"] * rl
        + _WEIGHTS["crest"] * rc
    )


def _build_queue(
    cfg: common.Config,
    per_rows: dict[int, dict[str, Any]],
    normalized_indices: set[int],
    rank_by: str,
) -> list[_AuditRow]:
    """Build the sorted audit queue from per-clip audio + PER signals.

    Args:
        cfg: Pipeline configuration (uses ``cfg.paths.normalized_wav``,
            ``common.load_sentences(cfg)``).
        per_rows: Per-clip PER rows from the audit CSV (already
            completeness-checked by the caller).
        normalized_indices: ``{idx}`` set derived from ``normalized_wav/``.
        rank_by: One of ``composite``, ``per``, ``pitch``, ``audio``
            (chooses the sort key).

    Returns:
        A list of :class:`_AuditRow` sorted by the requested ranking
        (worst first). May be empty.
    """
    sentences = common.load_sentences(cfg)
    norm_dir = cfg.paths.normalized_wav

    # Pass 1: scan audio features + pull PER values.
    clips: list[_AuditRow] = []
    per_vals: dict[int, float] = {}
    per_pitch_vals: dict[int, float] = {}
    crest_vals: dict[int, float] = {}
    lf_vals: dict[int, float] = {}
    for idx in normalized_indices:
        wav_path = norm_dir / f"{idx:06d}.wav"
        features = _audio_features(wav_path)
        if features is None:
            continue
        per_row = per_rows.get(idx, {})
        per = float(per_row.get("per", 0.0))
        per_pitch = float(per_row.get("per_pitch", 0.0))
        expected = sentences[idx] if idx < len(sentences) else ""
        row = _AuditRow(
            idx=idx,
            wav_path=wav_path,
            expected=expected,
            per=per,
            per_pitch=per_pitch,
            crest=features.crest,
            lf_burst=features.lf_burst,
            clf_ratio=features.clf_ratio,
            dur=features.dur,
            suspect_score=0.0,
        )
        clips.append(row)
        per_vals[idx] = per
        per_pitch_vals[idx] = per_pitch
        crest_vals[idx] = features.crest
        lf_vals[idx] = features.lf_burst

    # Pass 2: percentile ranks + composite score.
    rank_per = _percentile_rank(per_vals)
    rank_per_pitch = _percentile_rank(per_pitch_vals)
    rank_crest = _percentile_rank(crest_vals)
    rank_lf = _percentile_rank(lf_vals)
    for row in clips:
        row.suspect_score = _composite_score(
            row.per,
            row.per_pitch,
            row.crest,
            row.lf_burst,
            rank_per.get(row.idx),
            rank_per_pitch.get(row.idx),
            rank_crest.get(row.idx),
            rank_lf.get(row.idx),
        )

    # Sort by selected key (worst first). For per/per_pitch, "worst" = higher
    # raw value (we use the raw value rather than its rank so ties resolve by
    # idx); for composite we use suspect_score; for audio we use the simple
    # mean of the two audio ranks.
    if rank_by == "composite":
        clips.sort(key=lambda r: (-r.suspect_score, r.idx))
    elif rank_by == "per":
        clips.sort(key=lambda r: (-r.per, r.idx))
    elif rank_by == "pitch":
        clips.sort(key=lambda r: (-r.per_pitch, r.idx))
    elif rank_by == "audio":
        clips.sort(
            key=lambda r: (
                -(rank_crest.get(r.idx, 0.0) * 0.4 + rank_lf.get(r.idx, 0.0) * 0.6),
                r.idx,
            )
        )
    else:
        # Unreached: Click choices constrain this, but keep a safe default.
        clips.sort(key=lambda r: (-r.suspect_score, r.idx))
    return clips


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Read the audit review checkpoint (decisions only).

    Args:
        path: ``cfg.paths.audit_checkpoint``.

    Returns:
        ``{"<idx>": {"action": "keep"|"bad", "ts": "<iso>"}}``. Empty dict
        on missing/corrupted file.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupted audit checkpoint (%s); ignoring it.", exc)
        return {}
    decisions = data.get("decisions", {})
    if not isinstance(decisions, dict):
        return {}
    return decisions


def _save_checkpoint(path: Path, decisions: dict[str, dict]) -> None:
    """Persist the audit checkpoint atomically (temp-file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"decisions": decisions}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _mark_bad(
    cfg: common.Config,
    row: _AuditRow,
    dry_run: bool,
    decisions: dict[str, dict],
) -> None:
    """Apply the ``r`` (bad) decision: write sidecar + remove wavs.

    Atomic from the user's perspective: the sidecar is written before the
    wavs are removed so a crash between the two steps leaves a
    ``rejected/<idx>.json`` whose corresponding wav is still on disk
    (the next ``generate --only-rejected`` will simply discard the stale
    wav and regenerate). The checkpoint is updated last.

    The sidecar format matches what :func:`common.read_rejected_indices`
    expects (an integer ``index`` field) plus the optional ``audited``
    marker so downstream consumers can filter manual-audit rejects if
    they ever need to.

    Args:
        cfg: Pipeline configuration.
        row: The audit row the user marked bad.
        dry_run: Skip filesystem mutation; only persist the decision.
        decisions: Live decisions dict (mutated in place).
    """
    if dry_run:
        logger.info("[dry-run] would mark bad idx=%d", row.idx)
    else:
        sidecar_path = cfg.paths.rejected / f"{row.idx:06d}.json"
        cfg.paths.rejected.mkdir(parents=True, exist_ok=True)
        meta = {
            "index": row.idx,
            "file": f"{row.idx:06d}.wav",
            "expected": row.expected,
            "reason": "manual_audit",
            "audited": True,
            "audit_signals": {
                "per": round(row.per, 4),
                "per_pitch": round(row.per_pitch, 4),
                "crest": round(row.crest, 4),
                "lf_burst": round(row.lf_burst, 4),
                "clf_ratio": round(row.clf_ratio, 4),
                "dur": round(row.dur, 3),
                "suspect_score": round(row.suspect_score, 4),
            },
        }
        sidecar_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Remove the wavs so `generate --only-rejected` regenerates them
        # (and so `normalize` re-creates the normalized copy from the new
        # accepted_wav). The audio is no longer "good": drop it everywhere
        # the pipeline could pick it up.
        for p in (row.wav_path, cfg.paths.accepted_wav / row.wav_path.name):
            if p.exists():
                p.unlink()
        logger.info("MARK-BAD idx=%d -> %s", row.idx, sidecar_path.name)

    decisions[str(row.idx)] = {"action": "bad", "ts": _now_iso()}
    if not dry_run:
        _save_checkpoint(cfg.paths.audit_checkpoint, decisions)


def _mark_keep(
    cfg: common.Config, row: _AuditRow, dry_run: bool, decisions: dict[str, dict]
) -> None:
    """Apply the ``a`` (keep) decision: persist the decision (no FS change)."""
    if dry_run:
        logger.info("[dry-run] would mark keep idx=%d", row.idx)
    decisions[str(row.idx)] = {"action": "keep", "ts": _now_iso()}
    if not dry_run:
        _save_checkpoint(cfg.paths.audit_checkpoint, decisions)


def _now_iso() -> str:
    """Return an ISO-8601 timestamp of the current UTC time."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _display_clip(row: _AuditRow, position: int, total: int) -> None:
    """Print the per-clip header + signal summary above the prompt."""
    delta_pitch = row.per_pitch - row.per
    delta_str = f"{delta_pitch:+.3f}"
    print(
        f"[{position:>{len(str(total))}}/{total}]  "
        f"idx={row.idx:06d}  suspect={row.suspect_score:.3f}  "
        f"per={row.per:.3f}  per_pitch={row.per_pitch:.3f} "
        f"({_Colors.dim}delta {delta_str}{_Colors.reset})  "
        f"crest={row.crest:.2f}  lf_burst={row.lf_burst:.3f}  "
        f"clip={row.clf_ratio:.3f}  dur={row.dur:.2f}s"
    )
    print(f"  {_Colors.bold}expected{_Colors.reset} : {row.expected}")


class _Outcome(enum.Enum):
    """Per-keypress outcome of the audit review loop."""

    ADVANCE = "advance"
    REWIND = "rewind"
    REPLAY = "replay"
    UNKNOWN = "unknown"
    QUIT = "quit"


def _handle_key(
    cfg: common.Config,
    row: _AuditRow,
    decisions: dict[str, dict],
    dry_run: bool,
    player: _Player,
    cursor: int,
) -> _Outcome:
    """Read one keypress and apply its side effects; return the loop outcome.

    Args:
        cfg: Pipeline configuration.
        row: The clip currently on display.
        decisions: Live decisions dict (mutated in place).
        dry_run: Skip filesystem mutations.
        player: Audio player (for ``p`` replay; stopped on ``b``/``q``).
        cursor: Current 0-based position in the queue (only used to detect
            ``b`` at the first clip).

    Returns:
        The :class:`_Outcome` the caller should dispatch on.
    """
    key = _getch().lower()
    if key == "a":
        _mark_keep(cfg, row, dry_run, decisions)
        _show_feedback("keep", _Colors.green)
        return _Outcome.ADVANCE
    if key == "r":
        _mark_bad(cfg, row, dry_run, decisions)
        _show_feedback("bad", _Colors.red)
        return _Outcome.ADVANCE
    if key == "p":
        player.play(row.wav_path)
        _show_minimal_prompt()
        return _Outcome.REPLAY
    if key == "b":
        if cursor == 0:
            _show_feedback(
                "already at the first clip; nothing to go back to",
                _Colors.dim,
            )
            _show_minimal_prompt()
            return _Outcome.REPLAY
        player.stop()
        _show_feedback("rewound one clip", _Colors.dim)
        return _Outcome.REWIND
    if key == "q":
        player.stop()
        _show_feedback("quit", _Colors.dim)
        return _Outcome.QUIT
    if not key:
        return _Outcome.REPLAY
    _show_feedback(f"unknown key {key!r}", _Colors.dim)
    _show_prompt()
    return _Outcome.UNKNOWN


_HELP_LINE = "[a] keep  [r] bad  [p] play  [b] back  [q] quit"


def _print_summary(decisions: dict[str, dict], queue: list[_AuditRow]) -> None:
    """Print the final tally of keep/bad decisions and a follow-up hint."""
    decided = sum(1 for r in queue if str(r.idx) in decisions)
    bad = sum(1 for r in queue if decisions.get(str(r.idx), {}).get("action") == "bad")
    keep = decided - bad
    print(
        f"\n{_Colors.bold}Audit summary{_Colors.reset}: "
        f"reviewed={decided}/{len(queue)}  keep={keep}  bad={bad}  "
        f"queue={len(queue)}"
    )
    if bad:
        bad_indices = sorted(
            int(i) for i, d in decisions.items() if d.get("action") == "bad"
        )
        sample = ", ".join(str(i) for i in bad_indices[: min(20, len(bad_indices))])
        more = "" if len(bad_indices) <= 20 else f" (...{len(bad_indices)} total)"
        print(f"  marked bad (idx): {sample}{more}")
        print(
            f"\nNext: regenerate the {bad} bad clip(s) with a fresh RNG draw, then\n"
            f"      re-validate and re-publish:\n"
            f"  poetry run gen-dataset --step generate --only-rejected\n"
            f"  poetry run gen-dataset --from validate"
        )
    else:
        print("\nNo clips marked bad. Nothing to regenerate.")


# --------------------------------------------------------------------------- #
# Click CLI                                                                    #
# --------------------------------------------------------------------------- #


@click.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config.yaml (default: config.yaml in project root).",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=500,
    show_default=True,
    help="Review only the top-N suspects by the selected ranking.",
)
@click.option(
    "--rank",
    "rank_by",
    type=click.Choice(["composite", "per", "pitch", "audio"], case_sensitive=False),
    default="composite",
    show_default=True,
    help="Rank the queue by composite score (default), raw PER, raw PER-pitch, "
    "or audio-only (crest + lf_burst).",
)
@click.option(
    "--restart",
    is_flag=True,
    default=False,
    help="Ignore the audit checkpoint and start from the first clip.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Walk the queue without moving files or writing the checkpoint.",
)
@click.option(
    "--clear/--no-clear",
    "clear_screen",
    default=True,
    show_default=True,
    help="Clear the terminal before each clip so the previous one does not "
    "scroll into view (default: enabled). Use --no-clear to keep the "
    "scrollback of recent decisions.",
)
def main(
    config_path: str | None,
    top_n: int,
    rank_by: str,
    restart: bool,
    dry_run: bool,
    clear_screen: bool,
) -> None:
    """Audit accepted clips that passed the threshold filters but sound bad.

    Loads the per-clip PER CSV (written by the ``pronunciation`` step),
    scans cheap audio-signal features on ``normalized_wav/`` (no model
    inference), ranks candidates by a composite suspect score, and walks
    the user through the top-N interactively. Bad clips are moved into the
    standard ``rejected/`` + sidecar contract so the existing regeneration
    pipeline picks them up unchanged.

    Pre-flight checks (hard-fail with a clear error):

    - ``normalized_wav/`` is non-empty (else: run the normalize step).
    - ``.audit_per.csv`` exists and covers every normalized clip (else:
      re-run the pronunciation step).

    Soft warnings (proceed anyway):

    - PER CSV older than the ``normalized_wav/`` directory (stale); the
      composite ranking may be inaccurate but the audio-only path still
      works.
    """
    cfg = common.load_config(config_path)
    common.setup_logging(cfg.paths.log_file)
    _Colors.enable(_wants_color())

    if not sys.stdin.isatty():
        raise click.UsageError(
            "audit needs an interactive terminal (TTY on stdin). Run it "
            "directly in a terminal, not via piping or a non-interactive shell."
        )

    # --- Pre-flight: normalized_wav/ must exist and be non-empty ---
    norm_dir = cfg.paths.normalized_wav
    if not norm_dir.exists() or not any(norm_dir.glob("*.wav")):
        raise click.ClickException(
            f"'{norm_dir}' is empty. Run the normalize step before "
            "auditing:\n  poetry run gen-dataset --step normalize"
        )
    normalized_indices = {int(p.stem) for p in norm_dir.glob("*.wav")}
    logger.info(
        "Normalized workspace: %d clips in %s", len(normalized_indices), norm_dir
    )

    # --- Pre-flight: PER CSV must exist and cover every normalized clip ---
    csv_path = cfg.paths.audit_per_csv
    if not csv_path.exists():
        raise click.ClickException(
            f"'{csv_path}' does not exist. Run the pronunciation step "
            "to log per-clip PER before auditing:\n"
            "  poetry run gen-dataset --step pronunciation"
        )
    per_rows = _read_audit_per_csv(csv_path)
    _check_csv_completeness(per_rows, normalized_indices, csv_path)
    _check_csv_staleness(per_rows, normalized_indices, csv_path)
    logger.info("PER audit CSV: %d rows loaded from %s", len(per_rows), csv_path)

    # --- Build the queue and slice to top-N ---
    queue = _build_queue(cfg, per_rows, normalized_indices, rank_by)
    if not queue:
        logger.warning("No clips to review.")
        return
    queue = queue[:top_n]
    logger.info("Audit queue: top %d by rank=%s (worst first)", len(queue), rank_by)

    # --- Resume from checkpoint ---
    decisions = {} if restart else _load_checkpoint(cfg.paths.audit_checkpoint)
    if restart:
        logger.info("--restart: audit checkpoint ignored.")
    first_pending = next(
        (i for i, r in enumerate(queue) if str(r.idx) not in decisions), None
    )
    if first_pending is None:
        logger.info(
            "All %d top-%d suspects already decided. Use --restart to redo.",
            len(queue),
            top_n,
        )
        _print_summary(decisions, queue)
        return

    # --- Interactive loop ---
    cursor = first_pending
    total = len(queue)
    # The header counter (``[N/total]``) counts decisions made so far +1,
    # NOT the position in the queue. This mirrors review-rejected's
    # ``decision_position``: on resume the queue may have shrunk (clips
    # marked 'r' have their normalized_wav/accepted_wav removed, so they
    # don't reappear in the queue when it is rebuilt), and the
    # queue-position counter would jump backwards. Counting decisions
    # keeps the header stable across sessions. The progress banner above
    # it uses the queue position (``cursor + 1``) so the two convey
    # complementary information.
    decision_position = len(decisions) + 1
    player = _Player()
    try:
        while cursor < total:
            row = queue[cursor]
            idx_key = str(row.idx)
            note = ""
            if idx_key in decisions:
                note = f"already {decisions[idx_key]['action']}"
            if clear_screen:
                _maybe_clear()
            decided = sum(1 for r in queue if str(r.idx) in decisions)
            sys.stdout.write(
                f"{_Colors.dim}"
                f"─── {cursor + 1}/{total} · {decided} decided "
                f"· {total - decided} to go ───"
                f"{_Colors.reset}\n"
            )
            sys.stdout.flush()
            _display_clip(row, decision_position, total)
            if not note:
                print(f"  {_Colors.dim}[playing...]{_Colors.reset}")
            player.play(row.wav_path)
            _show_prompt()
            while True:
                outcome = _handle_key(cfg, row, decisions, dry_run, player, cursor)
                if outcome is _Outcome.ADVANCE:
                    cursor += 1
                    decision_position += 1
                    break
                if outcome is _Outcome.REWIND:
                    cursor = max(0, cursor - 1)
                    decision_position = max(1, decision_position - 1)
                    break
                if outcome is _Outcome.QUIT:
                    _print_summary(decisions, queue)
                    return
                # REPLAY and UNKNOWN: handler already re-printed prompt.
    except KeyboardInterrupt:
        logger.info("Interrupted; saving current state.")
    finally:
        player.stop()
        _print_summary(decisions, queue)


def cli() -> None:
    """Entry-point wrapper for Click+Windows console display errors.

    Mirrors :func:`src.review_rejected.cli`: on Windows, Click's exception
    handler crashes if the console handle is invalid (OSError 6). This
    wrapper extracts the original Click error and prints it cleanly.
    """
    try:
        main()
    except SystemExit:
        raise
    except OSError as exc:
        cause = exc.__context__ if exc.__context__ is not None else exc
        if isinstance(cause, click.ClickException):
            print(f"audit: {cause.format_message()}", file=sys.stderr)
        else:
            print(
                "audit: error while displaying usage. Use --help to see "
                "valid options.",
                file=sys.stderr,
            )
        sys.exit(2)


if __name__ == "__main__":
    cli()
