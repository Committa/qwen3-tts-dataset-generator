"""Merge archived generations (output/genNNN) into a single new archive.

Combines two published-style archives (e.g. gen001 + gen002) into one
uniform archive (output/genNNN): the secondary archive's clips are
re-indexed after the primary's highest index, all metadata CSVs are
rewritten with the standard ``wavs/<first>-<last>/`` bucket layout, and
``report.json`` is merged (totals, duration, manifest counts) with a
``merge`` section documenting the sources. Transcripts must not overlap
between the archives (hard fail otherwise).

Besides the split CSVs (``metadata_train.csv``/``metadata_val.csv``), a
concatenated ``metadata.csv`` (train + val) is emitted as a convenience
for Piper-style training, which expects a single metadata file and does
its own random split via ``--validation-split``.

The secondary archive is added entirely to the train split; the primary's
train/val split is preserved unchanged. Manual artifacts (README.md,
LICENSE) are copied from the primary archive.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from copy import deepcopy
from pathlib import Path

import click

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
GEN_RE = re.compile(r"^gen(\d{3})$")
METADATA_FILES = ("metadata_train.csv", "metadata_val.csv")
ARCHIVE_METADATA = (*METADATA_FILES, "metadata.csv")


def _list_generations() -> list[Path]:
    """Return the existing output/genNNN directories sorted by number.

    Returns:
        Sorted list of archive directories.
    """
    gens = []
    for child in OUTPUT_DIR.iterdir():
        if child.is_dir() and GEN_RE.match(child.name):
            gens.append(child)
    return sorted(gens, key=lambda p: int(GEN_RE.match(p.name).group(1)))


def _read_metadata_indexed(path: Path) -> list[tuple[int, str]]:
    """Read a metadata CSV into (numeric stem, transcript) rows.

    Args:
        path: Metadata CSV (``wavs/.../NNNNNN.wav|<transcript>``).

    Returns:
        Rows as (int, str) tuples in file order.

    Raises:
        ValueError: On malformed lines or non-numeric stems.
    """
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for rec in csv.reader(fh, delimiter="|"):
            if len(rec) != 2:
                raise ValueError(f"Malformed metadata line in {path}: {rec!r}")
            stem = Path(rec[0]).stem
            try:
                idx = int(stem)
            except ValueError:
                raise ValueError(f"Non-numeric wav stem in {path}: {stem!r}")
            rows.append((idx, rec[1]))
    return rows


def _wav_files(gen_dir: Path) -> list[Path]:
    """Return the wav files of an archive sorted by numeric stem.

    Args:
        gen_dir: Archive directory.

    Returns:
        Sorted wav file paths.
    """
    wavs = list(gen_dir.rglob("*.wav"))
    wavs.sort(key=lambda p: int(p.stem))
    return wavs


def _fmt_hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    frac = int(round((seconds - total) * 1000))
    if frac == 1000:
        s += 1
        frac = 0
    return f"{h:02d}:{m:02d}:{s:02d}.{frac:03d}"


def _rel_path(idx: int, per_dir: int, max_idx: int) -> str:
    """Archive-relative wav path, subfoldered if per_dir > 0.

    Args:
        idx: Clip index.
        per_dir: Max wav files per subdirectory (0 for flat).
        max_idx: Highest index in the merged archive (caps the last bucket).

    Returns:
        Relative path like ``wavs/000000-008999/000123.wav``.
    """
    if per_dir <= 0:
        return f"wavs/{idx:06d}.wav"
    start = (idx // per_dir) * per_dir
    end = min(start + per_dir - 1, max_idx)
    return f"wavs/{start:06d}-{end:06d}/{idx:06d}.wav"


def _write_metadata(
    rows: list[tuple[int, str]], dest: Path, per_dir: int, max_idx: int
) -> None:
    """Write a metadata CSV with archive-relative wav paths.

    Args:
        rows: (index, transcript) rows.
        dest: Destination CSV path.
        per_dir: Max wav files per subdirectory (0 for flat).
        max_idx: Highest index in the merged archive.
    """
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for idx, text in rows:
            writer.writerow([_rel_path(idx, per_dir, max_idx), text])


def _merge_report(
    prim_report: dict,
    sec_report: dict,
    *,
    out_train: int,
    out_val: int,
    sources: list[str],
    primary: str,
    secondary: str,
    offset: int,
    per_dir: int,
    overlap: int,
) -> dict:
    """Merge two archive report.json files into one.

    Totals, duration and manifest counts are summed/recomputed; the
    quality/pronunciation/model sections come from the primary archive
    (both archives are generated with the same model and settings). A
    ``merge`` block records the provenance of the merged archive.

    Args:
        prim_report: Primary archive report dict.
        sec_report: Secondary archive report dict.
        out_train: Merged train clip count.
        out_val: Merged val clip count.
        sources: Archive folder names, primary first.
        primary: Primary archive folder name.
        secondary: Secondary archive folder name.
        offset: Index offset applied to the secondary clips.
        per_dir: wavs_per_dir used for the merged layout.
        overlap: Number of transcripts shared between the archives.

    Returns:
        The merged report dict.
    """
    merged = deepcopy(prim_report)
    for key in ("input_sentences", "accepted", "rejected", "skipped_during_generation"):
        merged["totals"][key] = int(prim_report.get("totals", {}).get(key, 0)) + int(
            sec_report.get("totals", {}).get(key, 0)
        )
    duration = float(
        prim_report.get("audio", {}).get("total_duration_seconds", 0.0)
    ) + float(sec_report.get("audio", {}).get("total_duration_seconds", 0.0))
    merged["audio"]["total_duration_seconds"] = round(duration, 2)
    merged["audio"]["total_duration_hhmmss"] = _fmt_hhmmss(duration)
    merged["manifest"] = {
        "train": out_train,
        "val": out_val,
        "total": out_train + out_val,
        "manifest_train": "metadata_train.csv",
        "manifest_val": "metadata_val.csv",
    }
    merged["generation_time_seconds"] = float(
        prim_report.get("generation_time_seconds", 0.0)
    ) + float(sec_report.get("generation_time_seconds", 0.0))
    merged["merge"] = {
        "sources": sources,
        "primary": primary,
        "secondary": secondary,
        "secondary_split": "train",
        "index_offset": offset,
        "wavs_per_dir": per_dir,
        "overlapping_transcripts": overlap,
    }
    return merged


def merge_archives(
    primary: Path, secondary: Path, output: Path, per_dir: int, force: bool
) -> dict:
    """Merge two archives into a new uniform archive.

    The primary archive keeps its indices and its train/val split; the
    secondary archive is re-indexed after the primary's highest index and
    added entirely to the train split. Wav files are copied (never moved),
    so both source archives remain intact.

    Args:
        primary: Archive kept as-is (indices, splits).
        secondary: Archive re-indexed after the primary and added to train.
        output: Destination archive directory.
        per_dir: Max wav files per subdirectory (0 for flat).
        force: Overwrite an existing output directory.

    Returns:
        Summary dict (clips, duration, splits, offset, overlap).

    Raises:
        click.ClickException: On user-facing errors (missing files,
            transcript overlap, existing output without --force).
        RuntimeError: On archive integrity violations.
    """
    for gen_dir, role in ((primary, "primary"), (secondary, "secondary")):
        for meta in METADATA_FILES:
            if not (gen_dir / meta).exists():
                raise click.ClickException(
                    f"{role} archive {gen_dir} is missing {meta}"
                )

    prim_train = _read_metadata_indexed(primary / "metadata_train.csv")
    prim_val = _read_metadata_indexed(primary / "metadata_val.csv")
    prim_rows = prim_train + prim_val
    sec_train = _read_metadata_indexed(secondary / "metadata_train.csv")
    sec_val = _read_metadata_indexed(secondary / "metadata_val.csv")
    sec_rows = sec_train + sec_val

    prim_wavs = _wav_files(primary)
    sec_wavs = _wav_files(secondary)

    if len(prim_rows) != len(prim_wavs):
        raise RuntimeError(
            f"primary archive {primary}: {len(prim_rows)} metadata rows but "
            f"{len(prim_wavs)} wav files"
        )
    if len(sec_rows) != len(sec_wavs):
        raise RuntimeError(
            f"secondary archive {secondary}: {len(sec_rows)} metadata rows but "
            f"{len(sec_wavs)} wav files"
        )

    prim_texts = dict(prim_rows)
    sec_texts = dict(sec_rows)
    if len(prim_texts) != len(prim_rows) or len(sec_texts) != len(sec_rows):
        raise RuntimeError("duplicate wav stems within an archive")
    if {int(p.stem) for p in prim_wavs} != set(prim_texts):
        raise RuntimeError(
            f"primary archive {primary}: wav stems do not match metadata"
        )
    if {int(p.stem) for p in sec_wavs} != set(sec_texts):
        raise RuntimeError(
            f"secondary archive {secondary}: wav stems do not match metadata"
        )

    overlap = set(prim_texts.values()) & set(sec_texts.values())
    if overlap:
        sample = ", ".join(sorted(overlap)[:10])
        raise click.ClickException(
            f"archives share {len(overlap)} transcripts (first: {sample}); "
            "merge refused to avoid duplicate audio"
        )

    offset = max(int(p.stem) for p in prim_wavs) + 1
    max_idx = offset + len(sec_wavs) - 1

    if output.exists() and not force:
        raise click.ClickException(
            f"output folder {output} already exists; use --force to overwrite"
        )
    for stale in ("wavs", *ARCHIVE_METADATA, "report.json"):
        stale_path = output / stale
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.exists():
            stale_path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in prim_wavs:
        dest = output / _rel_path(int(src.stem), per_dir, max_idx)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
        copied += 1
    for i, src in enumerate(sec_wavs):
        dest = output / _rel_path(offset + i, per_dir, max_idx)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
        copied += 1
    logger.info("Copied %d wav files to %s", copied, output / "wavs")

    train_stems = {idx for idx, _ in prim_train}
    val_stems = {idx for idx, _ in prim_val}
    if train_stems & val_stems:
        raise RuntimeError("primary archive: stem appears in both train and val")

    out_train = [
        (int(p.stem), prim_texts[int(p.stem)])
        for p in prim_wavs
        if int(p.stem) in train_stems
    ]
    out_val = [
        (int(p.stem), prim_texts[int(p.stem)])
        for p in prim_wavs
        if int(p.stem) in val_stems
    ]
    out_train += [(offset + i, sec_texts[int(p.stem)]) for i, p in enumerate(sec_wavs)]

    _write_metadata(out_train, output / "metadata_train.csv", per_dir, max_idx)
    _write_metadata(out_val, output / "metadata_val.csv", per_dir, max_idx)
    _write_metadata(out_train + out_val, output / "metadata.csv", per_dir, max_idx)

    for meta in ("README.md", "LICENSE"):
        src_meta = primary / meta
        if src_meta.exists():
            shutil.copy2(str(src_meta), str(output / meta))

    prim_report = json.loads((primary / "report.json").read_text(encoding="utf-8"))
    sec_report = json.loads((secondary / "report.json").read_text(encoding="utf-8"))
    merged_report = _merge_report(
        prim_report,
        sec_report,
        out_train=len(out_train),
        out_val=len(out_val),
        sources=[primary.name, secondary.name],
        primary=primary.name,
        secondary=secondary.name,
        offset=offset,
        per_dir=per_dir,
        overlap=len(overlap),
    )
    (output / "report.json").write_text(
        json.dumps(merged_report, indent=2), encoding="utf-8"
    )

    # --- Verify the merged archive round-trips ---
    wavs_after = len(_wav_files(output))
    if wavs_after != copied:
        raise RuntimeError(
            f"verification failed: {wavs_after} wavs found but {copied} expected"
        )
    for meta in ARCHIVE_METADATA:
        rows = _read_metadata_indexed(output / meta)
        for idx, _ in rows:
            dest = output / _rel_path(idx, per_dir, max_idx)
            if not dest.exists():
                raise RuntimeError(
                    f"verification failed: {dest} referenced by {meta} does not exist"
                )

    return {
        "primary": primary.name,
        "secondary": secondary.name,
        "output": output.name,
        "primary_clips": len(prim_wavs),
        "secondary_clips": len(sec_wavs),
        "total_clips": copied,
        "train": len(out_train),
        "val": len(out_val),
        "primary_duration_seconds": float(
            prim_report.get("audio", {}).get("total_duration_seconds", 0.0)
        ),
        "secondary_duration_seconds": float(
            sec_report.get("audio", {}).get("total_duration_seconds", 0.0)
        ),
        "duration_seconds": merged_report["audio"]["total_duration_seconds"],
        "duration_hhmmss": merged_report["audio"]["total_duration_hhmmss"],
        "offset": offset,
        "overlap": len(overlap),
    }


@click.command()
@click.option(
    "--archives",
    default=None,
    help="Comma-separated gen folders to merge (e.g. gen001,gen002). "
    "Defaults to the two most recent archives in output/.",
)
@click.option(
    "--output",
    "output_name",
    default=None,
    help="Target gen folder (e.g. gen003). Defaults to the next free number.",
)
@click.option(
    "--wavs-per-dir",
    type=int,
    default=9000,
    show_default=True,
    help="Max wav files per archive subdirectory (0 for flat).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the output gen folder if it already exists.",
)
def main(
    archives: str | None,
    output_name: str | None,
    wavs_per_dir: int,
    force: bool,
) -> None:
    """Merge two published-style archives into a single uniform archive."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if archives:
        names = [name.strip() for name in archives.split(",") if name.strip()]
    else:
        gens = _list_generations()
        if len(gens) < 2:
            raise click.ClickException(
                f"need at least 2 archives under {OUTPUT_DIR}; found {len(gens)}"
            )
        names = [gens[-2].name, gens[-1].name]

    if len(names) != 2:
        raise click.ClickException(
            "--archives must name exactly two gen folders (e.g. gen001,gen002)"
        )
    primary = OUTPUT_DIR / names[0]
    secondary = OUTPUT_DIR / names[1]
    for gen_dir in (primary, secondary):
        if not (gen_dir.is_dir() and GEN_RE.match(gen_dir.name)):
            raise click.ClickException(f"not a gen archive: {gen_dir}")

    if output_name:
        if not GEN_RE.match(output_name):
            raise click.ClickException("--output must look like genNNN (e.g. gen003)")
        output = OUTPUT_DIR / output_name
    else:
        gens = _list_generations()
        next_num = (int(GEN_RE.match(gens[-1].name).group(1)) if gens else 0) + 1
        output = OUTPUT_DIR / f"gen{next_num:03d}"

    summary = merge_archives(primary, secondary, output, wavs_per_dir, force)

    logger.info(
        "Merged %s (%d clips, %s) + %s (%d clips, %s) -> %s (%d clips, %s)",
        summary["primary"],
        summary["primary_clips"],
        _fmt_hhmmss(summary["primary_duration_seconds"]),
        summary["secondary"],
        summary["secondary_clips"],
        _fmt_hhmmss(summary["secondary_duration_seconds"]),
        summary["output"],
        summary["total_clips"],
        summary["duration_hhmmss"],
    )
    logger.info(
        "Train %d / val %d, index offset %d, overlapping transcripts %d",
        summary["train"],
        summary["val"],
        summary["offset"],
        summary["overlap"],
    )
    print(
        f"\nDataset card checklist (update {output / 'README.md'}):\n"
        f"  - pretty_name / title: switch to 28h\n"
        f"  - Clips (train / val): {summary['train']:,} / {summary['val']:,}\n"
        f"  - Total duration: ~{summary['duration_seconds'] / 3600:.1f} h "
        f"({int(summary['duration_seconds'])} s)\n"
        f"  - File structure: update the wavs/ bucket listing"
    )


if __name__ == "__main__":
    main()
