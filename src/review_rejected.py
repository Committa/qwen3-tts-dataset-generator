"""Interactive triage of rejected clips.

Walks every clip in ``workspace/rejected/``, plays the audio, and lets the user
decide whether to accept (move to ``accepted_wav/``) or reject (keep as-is) with
a single keypress. Designed to make it cheap to recover the false positives that
``validate`` (WER) and ``pronunciation`` (PER) drop because the ASR is not
perfect: typically borderline clips at WER ~0.22-0.25 where the audio is fine
but faster-whisper misheard a word or two.

The script is cross-platform: audio via ``sounddevice`` (PortAudio), keyboard
input via ``msvcrt`` on Windows and ``tty``/``termios`` on POSIX. State is
persisted in ``workspace/.review_checkpoint.json`` so sessions are resumable;
``a`` and ``r`` apply immediately (and cannot be undone from the keyboard — to
undo, edit the checkpoint or move the file back by hand).

Usage:
    poetry run review-rejected
    poetry run review-rejected --restart
    poetry run review-rejected --sort index
    poetry run review-rejected --dry-run
"""

from __future__ import annotations

import enum
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import click
import sounddevice as sd
import soundfile as sf

from . import common

logger = logging.getLogger(__name__)


_HELP_LINE = (
    "Keys: [a]ccept  [r]eject  [p]lay again  [b]ack  [q]uit  "
    "(a/r apply immediately; 'b' rewinds one clip but its prior decision, "
    "if any, stands)"
)


# --------------------------------------------------------------------------- #
# ANSI colors                                                                  #
# --------------------------------------------------------------------------- #


class _Colors:
    """ANSI escape codes; harmless no-ops when stdout is not a TTY."""

    reset = ""
    red = ""
    yellow = ""
    green = ""
    dim = ""
    bold = ""

    @classmethod
    def enable(cls, use_color: bool) -> None:
        """Populate the color attributes with real escape sequences when on."""
        if use_color:
            cls.reset = "\033[0m"
            cls.red = "\033[31m"
            cls.yellow = "\033[33m"
            cls.green = "\033[32m"
            cls.dim = "\033[2m"
            cls.bold = "\033[1m"


def _wants_color() -> bool:
    """Return True iff stdout looks like an interactive TTY that handles ANSI.

    Windows 10+ supports ANSI in the new console; legacy cmd.exe does not. We
    optimistically enable it; the worst case is a few stray escape sequences,
    which is preferable to silently stripping useful emphasis.
    """
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


# --------------------------------------------------------------------------- #
# Cross-platform single-key input                                             #
# --------------------------------------------------------------------------- #


def _getch() -> str:
    """Read one character from stdin without waiting for Enter.

    Windows: ``msvcrt.getch()``. POSIX: ``tty.setcbreak()`` + ``sys.stdin.read(1)``
    with the original terminal attributes restored in a ``finally`` block so
    Ctrl-C / EOF / exceptions never leave the user's terminal in raw mode.
    """
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            # Function / arrow key prefix: swallow the trailing byte and ignore.
            try:
                msvcrt.getch()
            except Exception:
                pass
            return ""
        try:
            return ch.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        # setcbreak is the right choice: Ctrl-C still raises KeyboardInterrupt
        # (we want a clean exit), but the line discipline is otherwise bypassed
        # so single keys arrive without Enter.
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# --------------------------------------------------------------------------- #
# Audio                                                                        #
# --------------------------------------------------------------------------- #


class _Player:
    """Tiny non-blocking WAV player around sounddevice.

    Tolerant of missing audio devices: ``play`` logs and returns False instead
    of raising, so the review loop keeps going on a headless box (e.g. a CI
    runner). Playback is stopped via the global ``sd.stop()``; ``sounddevice``
    does not hand out a per-clip stream handle to close.
    """

    def play(self, wav_path: Path) -> bool:
        """Start playback of the WAV file in a background stream.

        Args:
            wav_path: Absolute path to a readable WAV file.

        Returns:
            True if playback started, False if the device refused (so the
            caller can keep the loop running).
        """
        self.stop()
        try:
            data, sr = sf.read(str(wav_path), always_2d=False)
        except (OSError, RuntimeError) as e:
            logger.warning("Cannot read %s: %s", wav_path.name, e)
            return False
        try:
            sd.play(data, samplerate=sr, blocking=False)
        except Exception as e:
            logger.warning(
                "Audio playback failed for %s (%s). "
                "Continuing without sound. On Linux hosts install libportaudio2.",
                wav_path.name,
                e,
            )
            return False
        return True

    def stop(self) -> None:
        """Stop any in-flight playback; safe to call when nothing is playing."""
        try:
            sd.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Diff display                                                                 #
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    """Split a string into word + punctuation tokens for diff display.

    Keeps commas, periods, apostrophes as part of the adjacent word when
    natural (so "L'oro" is one token), but standalone punctuation is its own
    token. This gives a diff that highlights actual word-level disagreements
    instead of a forest of spurious punctuation changes.
    """
    return re.findall(r"\w+(?:['']\w+)*|[^\w\s]", text, flags=re.UNICODE)


def _format_diff(expected: str, transcribed: str) -> str:
    """Render expected vs transcribed as a colorized word-level diff.

    Equal tokens are plain. Tokens only in expected are red (ASR missed them);
    tokens only in transcribed are yellow (ASR hallucinated them).
    """
    a = _tokenize(expected)
    b = _tokenize(transcribed)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(" ".join(b[j1:j2]))
        elif tag == "delete":
            out.append(_Colors.red + " ".join(a[i1:i2]) + _Colors.reset)
        elif tag == "insert":
            out.append(_Colors.yellow + " ".join(b[j1:j2]) + _Colors.reset)
        elif tag == "replace":
            out.append(_Colors.red + " ".join(a[i1:i2]) + _Colors.reset)
            out.append(_Colors.yellow + " ".join(b[j1:j2]) + _Colors.reset)
    return " ".join(out).strip()


# --------------------------------------------------------------------------- #
# Rejected-clips model + checkpoint                                           #
# --------------------------------------------------------------------------- #


@dataclass
class RejectedClip:
    """One sidecar JSON in workspace/rejected/.

    Attributes:
        index: Sentence index from the corpus (matches the wav filename stem).
        wav_path: Absolute path to the rejected wav (read from disk; may be
            missing if the sidecar survived the wav — we tolerate that).
        sidecar_path: Absolute path to the sidecar JSON.
        expected: Reference sentence from the corpus.
        transcription: ASR hypothesis (validate-step clips only). Empty for
            pronunciation-step rejections.
        ref_phonemes: espeak reference phonemes (pronunciation clips only).
        hyp_phonemes: recognized phonemes (pronunciation clips only).
        reason: Rejection reason, e.g. ``"wer=0.222 > 0.200"``.
        metric_label: ``"wer"`` or ``"per"`` parsed from the reason field.
        metric_value: Numeric value parsed from the reason field.
        threshold: Numeric threshold parsed from the reason field.
    """

    index: int
    wav_path: Path
    sidecar_path: Path
    expected: str
    transcription: str
    ref_phonemes: str
    hyp_phonemes: str
    reason: str
    metric_label: str
    metric_value: float
    threshold: float


_REASON_RE = re.compile(r"^(?P<label>wer|per)=(?P<value>[\d.]+)\s*>\s*(?P<thr>[\d.]+)$")


def _parse_reason(reason: str) -> tuple[str, float, float]:
    """Extract (label, value, threshold) from a reason string.

    Falls back to ``("wer", 0.0, 0.0)`` for reasons we do not recognize
    (e.g. custom third-party rejects) so the UI still shows something useful.
    """
    m = _REASON_RE.match(reason.strip())
    if m:
        return m["label"], float(m["value"]), float(m["thr"])
    return "wer", 0.0, 0.0


def _load_rejected_clips(cfg: common.Config) -> list[RejectedClip]:
    """Read all sidecar JSONs from workspace/rejected/ and parse them.

    Skips files with malformed JSON or missing ``index`` fields (logs a
    warning per skip). Sidecars are read in deterministic stem order so the
    queue is stable across filesystems; the caller owns any further ordering
    (e.g. by WER or by corpus index).

    Args:
        cfg: Pipeline configuration (only ``cfg.paths.rejected`` is used).

    Returns:
        List of RejectedClip records in sidecar-stem order. May be empty.
    """
    rejected_dir = cfg.paths.rejected
    if not rejected_dir.exists():
        return []
    clips: list[RejectedClip] = []
    for sidecar in sorted(rejected_dir.glob("*.json"), key=lambda p: p.stem):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping malformed sidecar %s: %s", sidecar.name, e)
            continue
        idx = data.get("index")
        if not isinstance(idx, int):
            logger.warning("Skipping sidecar without int 'index': %s", sidecar.name)
            continue
        label, value, thr = _parse_reason(str(data.get("reason", "")))
        wav_name = data.get("file") or f"{idx:06d}.wav"
        clips.append(
            RejectedClip(
                index=idx,
                wav_path=rejected_dir / wav_name,
                sidecar_path=sidecar,
                expected=str(data.get("expected", "")),
                transcription=str(data.get("transcription", "")),
                ref_phonemes=str(data.get("ref_phonemes", "")),
                hyp_phonemes=str(data.get("hyp_phonemes", "")),
                reason=str(data.get("reason", "")),
                metric_label=label,
                metric_value=value,
                threshold=thr,
            )
        )
    return clips


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Read the review checkpoint (decisions only, no sidecar cache for v1).

    Args:
        path: Path to ``workspace/.review_checkpoint.json``.

    Returns:
        Mapping of ``str(idx) -> {"action": "accepted"|"rejected", "reason": ...}``.
        Returns an empty dict if the file does not exist or is corrupted.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Corrupted review checkpoint (%s); ignoring it.", e)
        return {}
    decisions = data.get("decisions", {})
    if not isinstance(decisions, dict):
        return {}
    return decisions


def _save_checkpoint(path: Path, decisions: dict[str, dict]) -> None:
    """Write the review checkpoint to disk atomically.

    Writes via a sibling temp file then renames, so a Ctrl-C or crash mid-write
    cannot leave a half-written JSON that would break the next resume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"decisions": decisions}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Move / restore actions                                                       #
# --------------------------------------------------------------------------- #


def _accept_clip(cfg: common.Config, clip: RejectedClip) -> None:
    """Move a clip from rejected/ to accepted_wav/ (idempotent).

    Reuses the same file-move logic as ``common.accept_clips`` so the on-disk
    layout matches what ``normalize`` and ``build_manifest`` expect, but
    operates on a single clip without re-loading config or iterating.

    For pronunciation-step rejections (sidecar carries ``ref_phonemes`` /
    ``hyp_phonemes``), the clip's index is added to the pronunciation
    ``done`` set so a subsequent ``--step pronunciation --only-rejected``
    does not re-score it. Validate-step rejects are not tracked there
    because validate already manages its own checkpoint.

    Args:
        cfg: Pipeline configuration.
        clip: The rejected clip to accept.
    """
    src = cfg.paths.raw_wav / clip.wav_path.name
    if not src.exists():
        logger.warning(
            "Cannot accept idx=%d: %s missing in raw_wav. Marking accepted anyway.",
            clip.index,
            src.name,
        )
    else:
        dest = cfg.paths.accepted_wav / src.name
        cfg.paths.accepted_wav.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
    for p in (clip.sidecar_path, clip.wav_path):
        if p.exists():
            p.unlink()
    # Sync pronunciation checkpoint: a manually-accepted pronunciation-reject
    # is "done" from pronunciation's POV — do not re-score on the next pass.
    if clip.ref_phonemes or clip.hyp_phonemes:
        done = common.read_checkpoint(cfg.paths.pronunciation_checkpoint)
        done.add(clip.index)
        common.write_checkpoint(cfg.paths.pronunciation_checkpoint, done)
        logger.info(
            "Pronunciation checkpoint updated (manual accept): idx=%d", clip.index
        )
    logger.info("ACCEPT idx=%d -> %s", clip.index, src.name)


def _restore_rejected(cfg: common.Config, clip: RejectedClip) -> None:
    """Move a previously-accepted clip back to rejected/ and re-write the sidecar.

    Used when the user goes ``back`` with ``b`` and changes their mind on a
    clip they had already accepted. The original sidecar JSON content is
    preserved so subsequent ``--only-rejected`` regeneration sees the same
    reason and transcription as before.

    Args:
        cfg: Pipeline configuration.
        clip: The clip to re-reject.
    """
    accepted_wav = cfg.paths.accepted_wav / clip.wav_path.name
    rejected_wav = cfg.paths.rejected / clip.wav_path.name
    if accepted_wav.exists():
        rejected_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(accepted_wav), str(rejected_wav))
    sidecar = {
        "index": clip.index,
        "file": clip.wav_path.name,
        "expected": clip.expected,
        "reason": clip.reason,
    }
    if clip.transcription:
        sidecar["transcription"] = clip.transcription
    if clip.ref_phonemes:
        sidecar["ref_phonemes"] = clip.ref_phonemes
        sidecar["hyp_phonemes"] = clip.hyp_phonemes
    clip.sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Undo the pronunciation-checkpoint add from the prior _accept_clip call,
    # so the clip will be re-scored on the next --only-rejected pass.
    if clip.ref_phonemes or clip.hyp_phonemes:
        done = common.read_checkpoint(cfg.paths.pronunciation_checkpoint)
        done.discard(clip.index)
        common.write_checkpoint(cfg.paths.pronunciation_checkpoint, done)
        logger.info(
            "Pronunciation checkpoint updated (re-rejected): idx=%d", clip.index
        )
    logger.info("RESTORE-REJECTED idx=%d -> %s", clip.index, rejected_wav.name)


# --------------------------------------------------------------------------- #
# Display helpers                                                              #
# --------------------------------------------------------------------------- #


def _format_header(
    clip: RejectedClip, position: int, total: int, note: str = ""
) -> str:
    """Build the one-line header shown above each clip's text."""
    base = (
        f"[{position:>{len(str(total))}}/{total}]  "
        f"idx={clip.index:06d}  "
        f"{clip.metric_label}={clip.metric_value:.3f} "
        f"(thr {clip.threshold:.3f})  "
        f"reason={clip.reason}"
    )
    if note:
        base = f"{_Colors.yellow}[{note}]{_Colors.reset}  {base}"
    return base


def _format_duration(path: Path) -> str:
    """Return a short ``"dur=4.0s"`` string or empty if the file is unreadable."""
    try:
        return f"{sf.info(str(path)).duration:.2f}s"
    except (OSError, RuntimeError):
        return ""


def _display_clip(clip: RejectedClip, position: int, total: int) -> None:
    """Print the per-clip header + texts to the terminal (no newline flush)."""
    dur = _format_duration(clip.wav_path)
    header = _format_header(clip, position, total)
    if dur:
        header = f"{header}  dur={dur}"
    print(header)
    print(f"  {_Colors.bold}expected  {_Colors.reset}: {clip.expected}")
    if clip.transcription:
        diff = _format_diff(clip.expected, clip.transcription)
        print(f"  {_Colors.bold}asr       {_Colors.reset}: {diff}")
    elif clip.ref_phonemes and clip.hyp_phonemes:
        print(
            f"  {_Colors.bold}phonemes  {_Colors.reset}: "
            f"ref={_Colors.dim}{clip.ref_phonemes}{_Colors.reset}"
        )
        print(f"  {' ' * 10}  " f"hyp={_Colors.dim}{clip.hyp_phonemes}{_Colors.reset}")


# --------------------------------------------------------------------------- #
# Prompt + feedback                                                            #
# --------------------------------------------------------------------------- #


_PROMPT_HELP = "[a]ccept  [r]eject  [p]lay  [b]ack  [q]uit"
_PROMPT_TEXT = "> "


def _show_prompt() -> None:
    """Print the two-line prompt (full command names above, ``>`` on its own
    line) and flush, so the next typed char lands on the same line as the
    ``>`` instead of after a hidden newline. Use this for the first display
    of a clip and after the user pressed an unknown key (so the legend helps
    them remember)."""
    sys.stdout.write(f"{_Colors.dim}{_PROMPT_HELP}\n{_PROMPT_TEXT}{_Colors.reset}")
    sys.stdout.flush()


def _show_minimal_prompt() -> None:
    """Print just the ``> `` marker, no legend, after transient events like a
    replay (``p``) or a no-op back (``b`` on the first clip). Avoids
    repainting the help line on every replay and producing visual spam."""
    sys.stdout.write(f"{_Colors.dim}{_PROMPT_TEXT}{_Colors.reset}")
    sys.stdout.flush()


def _show_progress(
    clips: list[RejectedClip], decisions: dict[str, dict], cursor: int
) -> None:
    """Print a one-line progress banner before each clip's display.

    Shows how many clips have been decided (this session + previous), the
    accept/reject breakdown, and how many remain. Persists across the
    clear-screen and the replay flow so the user never loses track of where
    they are in the triage — important when the queue spans multiple
    sessions and the checkpoint is the only cross-session memory.

    Args:
        clips: All rejected clips currently in the queue (the total).
        decisions: Map of already-decided ``str(idx) -> {"action": ...}``;
            mutated in place as the user decides, so its length is the
            ground truth for "decided so far".
        cursor: Position of the next clip to display in ``clips`` (0-based).
            Used only to print the current position; the counts are derived
            from ``decisions`` and ``clips`` length, not the cursor.
    """
    total = len(clips)
    decided = len(decisions)
    accepted = sum(1 for v in decisions.values() if v.get("action") == "accepted")
    rejected = decided - accepted
    left = total - decided
    position = cursor + 1
    line = (
        f"─── {position}/{total} · "
        f"{decided} decided ({accepted}a + {rejected}r) · "
        f"{left} to go ───"
    )
    sys.stdout.write(f"{_Colors.dim}{line}{_Colors.reset}\n")
    sys.stdout.flush()


def _maybe_clear() -> None:
    """Clear the screen and home the cursor.

    Emits the standard ``\\033[2J\\033[H`` sequence. No-op on terminals that
    do not handle ANSI (file redirection, CI logs, dumb consoles), so the
    output stays clean when piped to a file.
    """
    if not _Colors.reset:
        # No color was enabled (non-TTY) -> skip the clear too, so we don't
        # leak escape sequences into a redirected log.
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _show_feedback(text: str, color: str | None = None) -> None:
    """Print a short inline feedback line after the user pressed a key.

    Used to confirm decisions (``accepted``, ``rejected``, ``rewound``,
    ``quit``) without obscuring the next clip's display. The
    color argument is an ANSI class attribute (e.g. ``_Colors.green``); pass
    ``None`` for a plain message.

    Args:
        text: Message to display (no trailing newline needed).
        color: Optional ANSI color escape (``_Colors.green`` / ``.red`` / ``.dim``).
    """
    prefix = color if color else ""
    suffix = _Colors.reset if color else ""
    sys.stdout.write(f"\n  {prefix}-> {text}{suffix}\n")
    sys.stdout.flush()


def _print_summary(decisions: dict[str, dict], clips: list[RejectedClip]) -> None:
    """Print a one-shot tally of decisions made so far."""
    accepted = sum(1 for v in decisions.values() if v.get("action") == "accepted")
    rejected = sum(1 for v in decisions.values() if v.get("action") == "rejected")
    print(
        f"\n{_Colors.bold}Review summary{_Colors.reset}: "
        f"accepted={accepted}  rejected={rejected}  total_decided={len(decisions)}  "
        f"queue={len(clips)}"
    )


# --------------------------------------------------------------------------- #
# Decision application                                                         #
# --------------------------------------------------------------------------- #


def _decide_and_apply(
    cfg: common.Config,
    clip: RejectedClip,
    action: str,
    decisions: dict[str, dict],
    dry_run: bool,
) -> None:
    """Apply a single decision and update the in-memory + on-disk checkpoint.

    ``a`` and ``r`` mutate the filesystem (via ``_accept_clip`` or
    ``_restore_rejected``). In dry-run mode the mutation is skipped but the
    decision is still recorded so a subsequent run without --dry-run won't
    re-prompt.

    Args:
        cfg: Pipeline configuration.
        clip: The clip the user just decided on.
        action: One of ``"accepted"``, ``"rejected"``.
        decisions: Live decisions dict (mutated in place).
        dry_run: True to skip filesystem mutations.
    """
    idx_key = str(clip.index)
    prev = decisions.get(idx_key, {}).get("action")

    if action == "accepted":
        if dry_run:
            logger.info("[dry-run] would ACCEPT idx=%d", clip.index)
        else:
            _accept_clip(cfg, clip)
    elif action == "rejected":
        if prev == "accepted" and not dry_run:
            _restore_rejected(cfg, clip)
        elif dry_run:
            logger.info("[dry-run] would REJECT idx=%d", clip.index)
    else:
        raise ValueError(f"Unknown action: {action!r}")

    decisions[idx_key] = {"action": action, "reason": clip.reason}
    if not dry_run:
        _save_checkpoint(cfg.paths.review_checkpoint, decisions)


# --------------------------------------------------------------------------- #
# Key handler                                                                  #
# --------------------------------------------------------------------------- #


class _Outcome(enum.Enum):
    """Result of a single keypress in the review loop.

    ADVANCE: user accepted or rejected; move to the next clip.
    REWIND:  user pressed ``b``; revisit the previous clip.
    REPLAY:  user pressed ``p`` (or ``b`` at the first clip); re-prompt
        without advancing.
    UNKNOWN: user pressed an unmapped key; re-prompt after showing help.
    QUIT:    user pressed ``q``; exit the loop.
    """

    ADVANCE = "advance"
    REWIND = "rewind"
    REPLAY = "replay"
    UNKNOWN = "unknown"
    QUIT = "quit"


def _handle_review_key(
    cfg: common.Config,
    clip: RejectedClip,
    decisions: dict[str, dict],
    dry_run: bool,
    player: _Player,
    cursor: int,
) -> _Outcome:
    """Read one keypress and apply its side effects; return the loop outcome.

    Filesystem mutations (accept/restore) and feedback printing happen here;
    the caller is responsible for moving the cursor and re-prompting based
    on the returned ``_Outcome``.

    Args:
        cfg: Pipeline configuration.
        clip: The clip currently on display.
        decisions: Live decisions dict (mutated in place on ``a`` / ``r``).
        dry_run: True to skip filesystem mutations.
        player: Audio player (stopped on ``b`` / ``q``; replayed on ``p``).
        cursor: Current 0-based position in the clips queue, used only to
            detect ``b`` at the first clip (a no-op rewind).

    Returns:
        The ``_Outcome`` the caller should dispatch on.
    """
    key = _getch().lower()
    if key == "a":
        _decide_and_apply(cfg, clip, "accepted", decisions, dry_run)
        _show_feedback("accepted", _Colors.green)
        return _Outcome.ADVANCE
    if key == "r":
        _decide_and_apply(cfg, clip, "rejected", decisions, dry_run)
        _show_feedback("rejected", _Colors.red)
        return _Outcome.ADVANCE
    if key == "p":
        player.play(clip.wav_path)
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
    logger.info("%s", _HELP_LINE)
    _show_prompt()
    return _Outcome.UNKNOWN


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
    "--restart",
    is_flag=True,
    default=False,
    help="Ignore the review checkpoint and start from the first rejected clip.",
)
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(["wer", "index"], case_sensitive=False),
    default="wer",
    show_default=True,
    help='Queue order. "wer" puts the borderline (likely false-positive) clips '
    'first so the easy wins are recovered quickly. "index" walks the corpus in '
    "natural order.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Walk the queue without moving files or writing the checkpoint. "
    "Useful for a first look at the format.",
)
@click.option(
    "--clear/--no-clear",
    "clear_screen",
    default=True,
    show_default=True,
    help="Clear the terminal before each clip so the previous one does not "
    "scroll into view (default: enabled). Disable with --no-clear to keep the "
    "scrollback of recent decisions.",
)
def main(
    config_path: str | None,
    restart: bool,
    sort_by: str,
    dry_run: bool,
    clear_screen: bool,
) -> None:
    """Interactively triage every clip in ``workspace/rejected/``.

    For each clip the script shows the rejection reason, the ASR (or phoneme)
    diff against the reference sentence, and plays the audio. You then press
    one of ``a/r/p/b/q`` to act. Decisions (``a`` and ``r``) are written to
    ``workspace/.review_checkpoint.json`` and applied to the filesystem
    immediately, so quitting in the middle never loses progress.

    Two position counters live in the loop:

    - ``cursor``: 0-based index into ``clips``; the clip currently on display.
    - ``decision_position``: 1-based counter that increments on every
      ``a`` / ``r`` and decrements on every ``b``. Shown in the per-clip
      header so the user sees the decision sequence number, which is
      distinct from the queue position when the checkpoint already held
      decisions from a previous session.
    """
    cfg = common.load_config(config_path)
    common.ensure_dirs(
        cfg.paths.accepted_wav, cfg.paths.rejected, cfg.paths.log_file.parent
    )
    common.setup_logging(cfg.paths.log_file)
    _Colors.enable(_wants_color())

    if not sys.stdin.isatty():
        raise click.UsageError(
            "review-rejected needs an interactive terminal (TTY on stdin). "
            "Run it directly in a terminal, not via piping or a non-interactive shell."
        )

    clips = _load_rejected_clips(cfg)
    if not clips:
        logger.info("Nothing to review: %s is empty.", cfg.paths.rejected)
        return

    if sort_by == "wer":
        clips.sort(key=lambda c: (c.metric_value, c.index))
    else:
        clips.sort(key=lambda c: c.index)

    decisions = {} if restart else _load_checkpoint(cfg.paths.review_checkpoint)
    first_pending = next(
        (i for i, c in enumerate(clips) if str(c.index) not in decisions), None
    )
    if first_pending is None:
        logger.info(
            "All %d rejected clips already have a decision in %s. "
            "Use --restart to start over.",
            len(clips),
            cfg.paths.review_checkpoint,
        )
        _print_summary(decisions, clips)
        return

    cursor = first_pending
    total = len(clips)
    player = _Player()
    decision_position = len(decisions) + 1
    try:
        while cursor < total:
            clip = clips[cursor]
            idx_key = str(clip.index)
            note = ""
            if idx_key in decisions:
                note = f"already {decisions[idx_key]['action']}"
            if clear_screen:
                _maybe_clear()
            _show_progress(clips, decisions, cursor)
            _display_clip(clip, decision_position, total)
            if not note:
                print(f"  {_Colors.dim}[playing...]{_Colors.reset}")
            player.play(clip.wav_path)
            _show_prompt()

            while True:
                outcome = _handle_review_key(
                    cfg, clip, decisions, dry_run, player, cursor
                )
                if outcome is _Outcome.ADVANCE:
                    decision_position += 1
                    break
                if outcome is _Outcome.REWIND:
                    cursor -= 1
                    decision_position = max(1, decision_position - 1)
                    break
                if outcome is _Outcome.QUIT:
                    _print_summary(decisions, clips)
                    return
                # REPLAY and UNKNOWN: the handler already printed the
                # next prompt/feedback; stay in the inner loop.
            cursor += 1
    except KeyboardInterrupt:
        logger.info("Interrupted; saving current state.")
    finally:
        player.stop()
        _print_summary(decisions, clips)


if __name__ == "__main__":
    main()
