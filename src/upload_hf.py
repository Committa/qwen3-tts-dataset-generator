"""Upload an HF-ready gen folder (output/genNNN/) to the Hugging Face Hub.

Usage (from the project root):

    poetry run upload-dataset --folder output/gen003 --repo Committa/serena-synthetic-it-28h

Uses ``upload_large_folder``: resumable (state in ``<folder>/.cache/.huggingface/``),
parallel workers, and Xet storage (hf_xet is installed) so the content phase
does not consume the per-file API quota that git-LFS hits on free-tier accounts.

A live progress bar is rendered in the console (committed/hashed/pre-uploaded
counters, snapshot from the uploader internals - no extra API calls). A status
line is appended to ``logs/upload_hf.log`` every 30s, plus every milestone
message is mirrored to the log.

If the commit phase hits the 1000-requests/5-min rate limit, the script waits
6 minutes and resumes automatically. Stop with Ctrl+C (exits cleanly; rerun
the same command to resume).

The target repo must already exist on the Hub: tokens that can write to a
repo are often not allowed to create one under an organization namespace.
A pre-flight check fails fast with a link to create it manually.

After the upload a small set of representative files is verified on the hub:
the top-level artifacts plus the first clip of every wavs/ bucket and the
last clip overall, all derived from the folder's metadata CSVs (layout
agnostic, no hardcoded bucket names).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = PROJECT_ROOT / "logs" / "upload_hf.log"
RATE_LIMIT_WAIT_S = 360

STATUS = {"total": 0, "hashed": 0, "pre_uploaded": 0, "committed": 0, "waiting": False}
_status_lock = threading.Lock()
_status_instance = []
_stop = threading.Event()

logf = open(LOG, "w", encoding="utf-8")


def _emit(msg: str) -> None:
    """Print a message to the console and append it to the log file.

    ``sys.stdout`` is never replaced (click's stream factory would wrap the
    replacement and swallow its output), so the log is fed explicitly.

    Args:
        msg: Message text (no trailing newline required).
    """
    print(msg)
    try:
        logf.write(msg + "\n")
        logf.flush()
    except Exception:
        pass


def _patch_uploader() -> None:
    """Monkeypatch upload_large_folder internals: capture the upload status
    for the progress bar, and pace the commit retry loop so a rate-limit 429
    cannot spiral (each failed retry would otherwise consume more quota)."""
    import huggingface_hub._upload_large_folder as ulf

    _orig_init = ulf.LargeUploadStatus.__init__

    def _patched_init(self, items, upload_batch_size=1):
        _orig_init(self, items, upload_batch_size)
        _status_instance.append(self)

    ulf.LargeUploadStatus.__init__ = _patched_init

    _orig_report = ulf.LargeUploadStatus.current_report

    def _patched_report(self):
        try:
            with self.lock:
                total = hashed = pre = comm = 0
                for _, m in self.items:
                    if m.should_ignore:
                        continue
                    total += 1
                    if m.sha256 is not None:
                        hashed += 1
                    if m.is_uploaded:
                        pre += 1
                    if m.is_committed:
                        comm += 1
            with _status_lock:
                STATUS.update(
                    total=total, hashed=hashed, pre_uploaded=pre, committed=comm
                )
        except Exception:
            pass
        return ""

    ulf.LargeUploadStatus.current_report = _patched_report

    _orig_update = ulf.LargeUploadStatus.update_chunk

    def _paced_update(self, success: bool, nb_items: int, duration: float) -> None:
        if not success:
            with _status_lock:
                STATUS["waiting"] = True
            _emit(
                f"commit batch failed; waiting {RATE_LIMIT_WAIT_S}s "
                "for the rate-limit window to reset before retrying..."
            )
            time.sleep(RATE_LIMIT_WAIT_S)
            with _status_lock:
                STATUS["waiting"] = False
        return _orig_update(self, success, nb_items, duration)

    ulf.LargeUploadStatus.update_chunk = _paced_update


def _renderer() -> None:
    """Draw the console progress bar in place; append a status line to the
    log file (not the console) every 30s."""
    last_log_line = 0.0
    while not _stop.is_set():
        if _status_instance:
            try:
                _status_instance[0].current_report()
            except Exception:
                pass
        with _status_lock:
            s = dict(STATUS)
        total = s["total"]
        committed = s["committed"]
        if total:
            pct = committed / total
            bar = "#" * int(pct * 30) + "." * (30 - int(pct * 30))
            line = (
                f"\r\x1b[KUpload: [{bar}] {committed}/{total} committed ({pct * 100:.0f}%) "
                f"| pre-uploaded {s['pre_uploaded']}/{total} | hashed {s['hashed']}/{total}"
            )
            if s["waiting"]:
                line += " | waiting for rate limit..."
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            sys.stdout.write("\r\x1b[KUpload: preparing... (hashing files)")
            sys.stdout.flush()
        now = time.time()
        if now - last_log_line >= 30 and total:
            last_log_line = now
            try:
                logf.write(
                    f"status: committed {committed}/{total} | "
                    f"pre-uploaded {s['pre_uploaded']}/{total} | hashed {s['hashed']}/{total}\n"
                )
                logf.flush()
            except Exception:
                pass
        time.sleep(1)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _check_paths(folder: Path) -> list[str]:
    """Derive representative files to verify on the hub from the folder layout.

    Top-level artifacts plus the first clip of every wavs bucket and the last
    clip of the archive, read from the metadata CSVs (layout-agnostic).

    Args:
        folder: The uploaded gen folder.

    Returns:
        Repository-relative paths (forward slashes) to verify.
    """
    paths = [
        "README.md",
        "LICENSE",
        "report.json",
        "metadata_train.csv",
        "metadata_val.csv",
    ]
    if (
        not (folder / "metadata_train.csv").exists()
        or not (folder / "metadata_val.csv").exists()
    ):
        return paths
    rows = []
    for meta in ("metadata_train.csv", "metadata_val.csv"):
        with (folder / meta).open("r", encoding="utf-8", newline="") as fh:
            rows.extend(line.strip().split("|", 1)[0] for line in fh if line.strip())
    if not rows:
        return paths
    rows = sorted(set(rows))
    buckets = sorted({"/".join(p.split("/")[:-1]) for p in rows})
    for bucket in buckets:
        paths.append(next(p for p in rows if p.startswith(bucket + "/")))
    paths.append(rows[-1])
    return paths


@click.command()
@click.option(
    "--folder",
    required=True,
    type=click.Path(path_type=Path),
    help="Gen folder to upload (e.g. output/gen003).",
)
@click.option(
    "--repo",
    required=True,
    help="Target dataset repo id (e.g. Committa/serena-synthetic-it-28h).",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    show_default=True,
    help="Upload worker count.",
)
@click.option(
    "--no-verify",
    is_flag=True,
    help="Skip the post-upload file verification.",
)
def main(folder: Path, repo: str, num_workers: int, no_verify: bool) -> None:
    """Upload an HF-ready gen folder to the Hugging Face Hub."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    _patch_uploader()

    api = HfApi()
    if not api.repo_exists(repo, repo_type="dataset"):
        raise click.ClickException(
            f"repo {repo} does not exist and this token cannot create it; "
            "create it first at https://huggingface.co/new/dataset "
            "(owner, name, license), then re-run this command"
        )
    try:
        while True:
            _emit(f"uploading {folder} -> {repo}")
            threading.Thread(target=_renderer, daemon=True).start()
            t0 = time.time()
            api.upload_large_folder(
                folder_path=str(folder),
                repo_id=repo,
                repo_type="dataset",
                num_workers=num_workers,
                print_report=False,
                print_report_every=15,
            )
            _emit(f"UPLOAD DONE in {time.time() - t0:.0f}s")
            break
    except KeyboardInterrupt:
        _emit("interrupted by user; rerun the same command to resume")
        logf.flush()
        os._exit(0)  # force-exit: uploader worker threads may not be daemonic
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            _emit(f"rate limited (429); waiting {RATE_LIMIT_WAIT_S}s and resuming...")
            time.sleep(RATE_LIMIT_WAIT_S)
            main(folder=folder, repo=repo, num_workers=num_workers, no_verify=no_verify)
            return
        _emit(f"ERR: {e!r}")
        return
    except Exception as e:
        _emit(f"ERR: {e!r}")
        return

    _stop.set()
    if no_verify:
        _emit("verification skipped (--no-verify)")
        return
    for path in _check_paths(folder):
        ok = api.file_exists(repo, path, repo_type="dataset")
        _emit(f"  {path}: {'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
    main()

logf.flush()
logf.close()
