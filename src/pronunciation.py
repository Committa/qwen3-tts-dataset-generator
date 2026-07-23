"""Step 3b: pronunciation verification with wav2vec2 CTC + espeak-ng (PER).

Complements the WER-based ``validate`` step. faster-whisper transcribes at the
*word* level and is forgiving of pronunciation drift: a clip can match the
reference transcript (low WER) while the actual pronunciation is wrong,
producing artifacts in the downstream training set.

This step recognises the audio at the *phoneme* level with
``facebook/wav2vec2-xlsr-53-espeak-cv-ft`` (a multilingual wav2vec2 CTC model
fine-tuned to output espeak IPA phonemes) and compares it (Phoneme Error Rate,
PER) against the espeak-ng text->phoneme rendering of the reference sentence.
Both sides use the same espeak phoneme inventory, so the comparison is direct.

Runs after ``validate`` on the WER survivors in ``accepted_wav/`` and before
``normalize`` (so normalize only processes pronunciation survivors). Clips
whose PER exceeds ``cfg.phoneme_threshold`` are moved from ``accepted_wav/``
to ``rejected/`` with a ``per=... > ...`` reason, feeding back into the
``--only-rejected`` regeneration loop unchanged (``read_rejected_indices``
keys off the ``index`` field, which is preserved).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import common

logger = logging.getLogger(__name__)

ESPEAK_HINT = (
    "espeak-ng was not found by the phonemizer library.\n"
    "  -> Install espeak-ng and ensure it is on PATH.\n"
    "  -> Windows: install the MSI from "
    "https://github.com/espeak-ng/espeak-ng/releases\n"
    "     and, if needed, set PHONEMIZER_ESPEAK_LIBRARY to the espeak-ng.dll path.\n"
    "  -> Linux (Debian/Ubuntu): sudo apt-get install espeak-ng\n"
)

# wav2vec2 CTC required input sample rate (mono float).
_PHONEME_SAMPLE_RATE = 16000

# IPA diacritics stripped from both sides before PER comparison: they are not
# meaningful for a coarse pronunciation check and the wav2vec2 tokenizer may
# emit them inconsistently.
#   U+02C8 ˈ  primary stress   U+02CC ˌ  secondary stress
#   U+02D0 ː  long             U+02D1 ˑ  half-long
#   U+0329 ̩  combining syllabic mark
_PHONEME_STRIP_TABLE = str.maketrans("", "", "\u02c8\u02cc\u02d0\u02d1\u0329")

# Filename of the JSONL aggregate written under workspace/rejected/.
_REJECTED_LOG_NAME = "pronunciation.log"

# Filename of the per-word PER CSV report written under workspace/.
_WORD_REPORT_NAME = ".pronunciation_words.csv"

# Filename of the human-readable pronunciation report written under workspace/.
_PRONUNCIATION_REPORT_NAME = ".pronunciation_report.txt"

# Sentinel inserted by espeak-ng between words in the phoneme output so word
# boundaries survive tokenization. The ASCII unit separator (\x1f) is a control
# character that never appears in IPA phoneme strings.
_WORD_SENTINEL = "\x1f"


@dataclass
class _Clip:
    """A clip pending pronunciation verification (read from accepted_wav/).

    Attributes:
        wav_path: Path of the clip in accepted_wav/.
        idx: Sentence index derived from the filename stem (``int(stem)``).
        expected: Reference text from the corpus at that index.
    """

    wav_path: Path
    idx: int
    expected: str


@dataclass
class _ClipResult:
    """Outcome of verifying a single clip against its reference phonemes.

    Attributes:
        clip: The verified clip.
        per: Phoneme Error Rate between reference and hypothesis phonemes.
        ref_phonemes: Normalized reference (espeak-ng) phoneme string.
        hyp_phonemes: Normalized hypothesis (wav2vec2) phoneme string.
        accepted: True if ``per`` is within the threshold (clip kept); False
            if the clip should be rejected.
    """

    clip: _Clip
    per: float
    ref_phonemes: str
    hyp_phonemes: str
    accepted: bool


def _normalize_phonemes(s: str) -> str:
    """Strip stress/length diacritics, lowercase, and collapse whitespace.

    Normalizes both the espeak-ng reference and the wav2vec2 hypothesis to the
    same footing so PER is fair. IPA is lowercase by convention in both
    espeak-ng output and the wav2vec2 tokenizer vocab.

    Args:
        s: A space-separated phoneme string (IPA).

    Returns:
        A cleaned, space-separated phoneme string.
    """
    s = s.translate(_PHONEME_STRIP_TABLE)
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def _per(reference: str, hypothesis: str) -> float:
    """Phoneme Error Rate between two normalized phoneme strings.

    Computes the Levenshtein edit distance between the whitespace-tokenized
    phoneme sequences and divides it by the number of reference tokens. PER is
    in [0.0, +inf) and matches what ``jiwer.wer`` returns for a single sentence
    on tokenized input.

    A direct Levenshtein implementation is used (rather than ``jiwer.wer``)
    because the latter interprets a list input as a list of *sentences* and
    applies sentence-level transforms (``RemoveMultipleSpaces``, ``Strip``,
    ``ReduceToListOfListOfWords``); jiwer 4.x raises
    ``ValueError("...sentences, their lengths must match")`` whenever ref and
    hyp token lists differ in length, which is the normal case for PER
    (insertions/deletions are exactly what edit distance is supposed to score).
    Levenshtein directly mirrors that contract without the indirection.

    Args:
        reference: Normalized reference phoneme string.
        hypothesis: Normalized hypothesis phoneme string.

    Returns:
        PER in [0.0, +inf). Returns 1.0 when the reference is empty but the
        hypothesis is not, and 0.0 when both are empty.
    """
    ref_toks = reference.split()
    hyp_toks = hypothesis.split()
    if not ref_toks:
        return 1.0 if hyp_toks else 0.0
    n, m = len(ref_toks), len(hyp_toks)
    # dp[i][j] = edit distance between ref_toks[:i] and hyp_toks[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref_toks[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp_toks[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / n


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile (0..100) of a sorted list, or 0.0 if empty.

    Uses nearest-rank selection (no interpolation), which is adequate for a
    coarse calibration summary.

    Args:
        sorted_values: Values sorted ascending.
        pct: Percentile in [0, 100].

    Returns:
        The percentile value, or 0.0 for an empty list.
    """
    if not sorted_values:
        return 0.0
    k = max(
        0,
        min(
            len(sorted_values) - 1,
            int(round((pct / 100.0) * (len(sorted_values) - 1))),
        ),
    )
    return sorted_values[k]


def _maybe_set_espeak_library() -> None:
    """On Windows, auto-detect the espeak-ng shared library from common locations.

    The ``phonemizer`` library relies on ``ctypes.util.find_library`` to locate
    the espeak-ng shared library. On Windows that function does **not** search
    ``PATH`` — it only looks in standard DLL directories (``System32``, the
    Python directory, etc.). Users who installed espeak-ng in ``Program Files``
    (via the MSI installer) will have ``espeak-ng.exe`` on ``PATH`` (or not) but
    the DLL will not be found by ``ctypes.util.find_library`` either way.

    This function works around the issue by searching in:

    1. The directory of the espeak-ng executable found via ``shutil.which``
       (works when the ``eSpeak NG`` directory is on ``PATH``).
    2. Common Windows install paths:

       - ``%ProgramFiles%/eSpeak NG/``
       - ``%ProgramFiles(x86)%/eSpeak NG/``

    If a candidate DLL is found, ``EspeakWrapper.set_library()`` is called so
    phonemizer can load it directly without relying on ``ctypes.util``.

    Has no effect on Linux / macOS (``sys.platform != 'win32'``) and respects
    an explicitly set ``PHONEMIZER_ESPEAK_LIBRARY`` environment variable, which
    takes precedence.
    """
    if sys.platform != "win32":
        return

    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    # Already set programmatically or via env var — skip.
    if EspeakWrapper._ESPEAK_LIBRARY is not None:
        return
    if "PHONEMIZER_ESPEAK_LIBRARY" in os.environ:
        return

    dll_candidates: list[Path] = []

    # Step 1 — executable directory (works when on PATH).
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe:
        dll_candidates.append(Path(exe).parent)

    # Step 2 — common Windows install paths.
    for base_var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(base_var)
        if base:
            candidate = Path(base) / "eSpeak NG"
            if candidate.is_dir():
                dll_candidates.append(candidate)

    dll_names = ("espeak-ng.dll", "libespeak-ng.dll", "libespeak-ng.so.1")
    for dll_dir in dll_candidates:
        for name in dll_names:
            dll = dll_dir / name
            if dll.exists():
                resolved = str(dll.resolve())
                logger.info("Auto-detected espeak-ng DLL on Windows: %s", resolved)
                EspeakWrapper.set_library(resolved)
                return


def _make_espeak_backend(lang_code: str) -> Any:
    """Construct the espeak-ng backend used for reference phonemization.

    Centralizes backend construction so the espeak-not-found error path and the
    fixed espeak options (``with_stress=False``, ``preserve_punctuation=False``,
    ``language_switch='remove-flags'``) live in one place.

    Args:
        lang_code: ISO 639-1 language code (e.g. ``"it"``).

    Returns:
        An ``EspeakBackend`` instance.

    Raises:
        RuntimeError: If the espeak-ng binary is not found, with the actionable
            ``ESPEAK_HINT`` appended to the message.
    """
    _maybe_set_espeak_library()

    from phonemizer.backend import EspeakBackend

    try:
        return EspeakBackend(
            language=lang_code,
            with_stress=False,
            preserve_punctuation=False,
            language_switch="remove-flags",
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}\n\n{ESPEAK_HINT}") from exc


def _phonemize(clips: list[_Clip], lang_code: str) -> dict[int, list[tuple[str, str]]]:
    """Convert each clip's reference text to per-word normalized IPA phonemes.

    Uses a sentinel word separator (``_WORD_SENTINEL``) so word boundaries
    survive tokenization: the output is, per clip, a list of
    ``(word, normalized_phonemes)`` pairs in sentence order. The flat
    clip-level phoneme string consumed by :func:`_per` is derived from this
    same output via :func:`_flat_phonemes`, so espeak-ng is invoked exactly
    once per run (not once for clip-level and once for per-word).

    Args:
        clips: Clips whose ``expected`` text must be phonemized.
        lang_code: ISO 639-1 language code (e.g. ``"it"``).

    Returns:
        A ``{idx: [(word, phonemes), ...]}`` dict.
    """
    from phonemizer.separator import Separator

    backend = _make_espeak_backend(lang_code)
    sep = Separator(phone=" ", syllable="", word=_WORD_SENTINEL)
    sentences = [c.expected for c in clips]
    raw = backend.phonemize(sentences, separator=sep, strip=True)

    out: dict[int, list[tuple[str, str]]] = {}
    for clip, phonemized in zip(clips, raw):
        words = clip.expected.split()
        chunks = (phonemized or "").split(_WORD_SENTINEL)
        per_word = [
            (word, _normalize_phonemes(chunks[i]) if i < len(chunks) else "")
            for i, word in enumerate(words)
        ]
        out[clip.idx] = per_word
    return out


def _flat_phonemes(per_word: list[tuple[str, str]]) -> str:
    """Join per-word normalized phonemes into a flat space-separated string.

    This derives the clip-level phoneme string (consumed by :func:`_per`) from
    the per-word output of :func:`_phonemize`, so espeak-ng runs once and both
    the flat and per-word views come from the same call.

    Args:
        per_word: ``[(word, phonemes), ...]`` for one clip.

    Returns:
        A normalized space-separated phoneme string.
    """
    return _normalize_phonemes(" ".join(ph for _w, ph in per_word))


@dataclass
class _WordScore:
    """Per-word pronunciation score for one clip.

    Attributes:
        word: The reference word (orthographic).
        per: Per-word Phoneme Error Rate (edits assigned to the word / word
            phoneme count). 0.0 for words with no reference phonemes.
        n_tokens: Number of reference phoneme tokens in the word.
    """

    word: str
    per: float
    n_tokens: int


def _align_words(
    ref_words: list[tuple[str, str]], hyp_tokens: list[str]
) -> list[_WordScore]:
    """Align a recognized phoneme stream to per-word reference phonemes.

    Runs a Levenshtein DP between the concatenated reference phoneme tokens
    (grouped by word) and the hypothesis tokens, with a backtrace that
    attributes each edit operation to the word owning the involved reference
    token. Insertions between two words are assigned to the following word (or
    to the last word if at the end). Per-word PER is then ``edits / n_tokens``
    for words with reference phonemes, else ``0.0``.

    The result is an *approximate* per-word diagnostic: CTC decoded phonemes
    have no explicit word boundaries, so attribution at word edges is best
    effort. It is meant for ranking systematically problematic words across the
    corpus, not for exact per-word grading.

    Args:
        ref_words: ``[(word, normalized_phonemes), ...]`` from :func:`_phonemize`.
        hyp_tokens: Recognized phoneme tokens (already normalized).

    Returns:
        A list of :class:`_WordScore`, one per reference word (in order).
    """
    # Flatten reference tokens and record the owning word index for each.
    ref_flat: list[str] = []
    word_of: list[int] = []
    word_token_counts: list[int] = []
    for _word, ph in ref_words:
        toks = ph.split()
        word_token_counts.append(len(toks))
        for t in toks:
            ref_flat.append(t)
            word_of.append(len(word_token_counts) - 1)
    n_words = len(ref_words)

    n, m = len(ref_flat), len(hyp_tokens)
    # dp[i][j] = edit distance between ref_flat[:i] and hyp_tokens[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref_flat[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # match/substitution
            )

    # Backtrace, counting edits per word.
    edits_per_word = [0] * n_words
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and (
                dp[i][j]
                == dp[i - 1][j - 1] + (0 if ref_flat[i - 1] == hyp_tokens[j - 1] else 1)
            )
        ):
            # match or substitution: consumes one ref token (word = word_of[i-1])
            if ref_flat[i - 1] != hyp_tokens[j - 1]:
                edits_per_word[word_of[i - 1]] += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # deletion: ref token dropped (word = word_of[i-1])
            edits_per_word[word_of[i - 1]] += 1
            i -= 1
        else:
            # insertion: extra hyp token. Attribute to the word we're about to
            # enter (word_of[i-1] is the last consumed; the next ref token, if
            # any, belongs to word_of[i]). Assign to the following word, else
            # to the last word.
            target = word_of[i] if i < n else n_words - 1
            if n_words > 0:
                edits_per_word[target] += 1
            j -= 1

    scores: list[_WordScore] = []
    for k, (word, _ph) in enumerate(ref_words):
        nt = word_token_counts[k]
        edits = edits_per_word[k]
        per = edits / nt if nt else 0.0
        scores.append(_WordScore(word, per, nt))
    return scores


class _PhonemeRecognizer:
    """Batched audio-to-phoneme recognizer wrapping a wav2vec2 CTC model.

    The model is kept in float32: wav2vec2-large is small (~1.2 GB) and run
    alone (faster-whisper and Qwen3-TTS are freed by this point), so VRAM is
    not a concern and float32 avoids any CTC argmax numerical surprises. CUDA
    is used when available and requested, with a CPU fallback mirroring
    ``validate._load_asr``.
    """

    def __init__(self, cfg: common.Config) -> None:
        """Load the processor and model from ``cfg.phoneme_model``.

        Args:
            cfg: Pipeline configuration.
        """
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        device = cfg.phoneme_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available for phoneme model, falling back to CPU.")
            device = "cpu"
        logger.info("Loading phoneme model '%s' (device=%s)", cfg.phoneme_model, device)
        self._processor = Wav2Vec2Processor.from_pretrained(cfg.phoneme_model)
        self._model = Wav2Vec2ForCTC.from_pretrained(cfg.phoneme_model).to(device)
        self._model.eval()
        self._device = device
        self._batch_size = max(1, cfg.phoneme_batch_size)
        # Cadence at which the PyTorch CUDA allocator pool is released back to
        # the driver via common.cleanup_gpu(). The caching allocator never
        # returns memory on its own, so without this the reserved pool grows
        # monotonically across hundreds of short wav2vec2 forwards until OOM.
        # Fall back to a safe default of 10 if the config value is non-positive.
        self._cleanup_every = max(1, cfg.phoneme_cleanup_every_n_batches)
        logger.info(
            "Phoneme recognizer cleanup cadence: every %d batches",
            self._cleanup_every,
        )

    def recognize(self, clips: list[_Clip]) -> dict[int, str]:
        """Recognize each clip to normalized IPA phonemes (batched, 16 kHz mono).

        Audio is loaded mono at 16 kHz via ``librosa`` (resampling from the
        generator's native rate as needed), padded and run through the CTC model
        in one forward pass per batch. The processor's ``attention_mask`` is
        forwarded so padded timesteps are ignored; CTC blank collapsing handles
        any residual tail (standard HuggingFace batched wav2vec2 ASR recipe).

        Args:
            clips: Clips to recognize.

        Returns:
            A ``{idx: normalized_phonemes}`` dict.
        """
        import librosa

        hyp: dict[int, str] = {}
        progress = tqdm(
            total=len(clips), desc="pronunciation", unit="wav", dynamic_ncols=True
        )
        batch_num = 0
        try:
            for i in range(0, len(clips), self._batch_size):
                chunk = clips[i : i + self._batch_size]
                arrays = [
                    librosa.load(str(c.wav_path), sr=_PHONEME_SAMPLE_RATE, mono=True)[0]
                    for c in chunk
                ]
                decoded = self._decode_batch(arrays)
                for c, ph in zip(chunk, decoded):
                    hyp[c.idx] = _normalize_phonemes(ph or "")
                progress.update(len(chunk))
                batch_num += 1
                if batch_num % self._cleanup_every == 0:
                    common.cleanup_gpu(log=logger)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            progress.close()
            raise SystemExit(1)
        progress.close()
        return hyp

    def _decode_batch(self, arrays: list[Any]) -> list[str]:
        """Run a padded batch through the CTC model and decode to phoneme strings.

        Args:
            arrays: List of 16 kHz mono float32 numpy arrays (one per clip).

        Returns:
            A list of raw (un-normalized) phoneme strings, one per input array.
        """
        import torch

        inputs = self._processor(
            arrays,
            return_tensors="pt",
            sampling_rate=_PHONEME_SAMPLE_RATE,
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)
        attention_mask = inputs.attention_mask.to(self._device)
        with torch.no_grad():
            logits = self._model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return list(self._processor.batch_decode(predicted_ids))


def _build_clips(cfg: common.Config) -> list[_Clip]:
    """Build the list of clips to verify from ``accepted_wav/``.

    Globs ``*.wav`` sorted, maps each filename stem to a corpus index, and
    drops out-of-range clips with a warning. The corpus is the single source
    of truth shared with validate/manifest/report (``common.load_sentences``).

    Args:
        cfg: Pipeline configuration.

    Returns:
        A list of ``_Clip`` (possibly empty).
    """
    sentences = common.load_sentences(cfg)
    clips: list[_Clip] = []
    for wav_path in sorted(cfg.paths.accepted_wav.glob("*.wav")):
        idx = int(wav_path.stem)
        if idx >= len(sentences):
            logger.warning("Index %d out of range for corpus. Skipping.", idx)
            continue
        clips.append(_Clip(wav_path, idx, sentences[idx]))
    return clips


def _evaluate(
    clips: list[_Clip],
    ref_map: dict[int, str],
    hyp_map: dict[int, str],
    threshold: float,
) -> list[_ClipResult]:
    """Compute PER for each clip and decide accept/reject (no side effects).

    Clips with empty reference phonemes (e.g. espeak produced nothing for an
    all-punctuation sentence) are skipped with a warning and excluded from the
    results. This is a pure function: it neither moves files nor writes logs,
    so the caller can choose to act (reject) or just measure (calibrate).

    Args:
        clips: Clips to evaluate.
        ref_map: ``{idx: normalized_reference_phonemes}``.
        hyp_map: ``{idx: normalized_hypothesis_phonemes}``.
        threshold: PER cutoff; clips with ``per <= threshold`` are accepted.

    Returns:
        A list of ``_ClipResult`` (one per clip with a non-empty reference).
    """
    results: list[_ClipResult] = []
    for clip in clips:
        ref = ref_map.get(clip.idx, "")
        hyp = hyp_map.get(clip.idx, "")
        if not ref:
            logger.warning(
                "Empty reference phonemes for idx=%d ('%s'), skipping.",
                clip.idx,
                clip.expected,
            )
            continue
        per = _per(ref, hyp)
        results.append(_ClipResult(clip, per, ref, hyp, per <= threshold))
    return results


def _reject_clip(
    clip: _Clip,
    cfg: common.Config,
    reason: str,
    ref_phonemes: str,
    hyp_phonemes: str,
) -> None:
    """Move a clip from ``accepted_wav/`` to ``rejected/`` and write its sidecar.

    Unlike ``validate._move_to_rejected`` (which copies ``raw_wav`` -> rejected
    because the clip was never in ``accepted_wav``), here the clip already lives
    in ``accepted_wav/`` (put there by validate), so it is *moved* (copy +
    unlink) to avoid leaving a duplicate that ``normalize`` would process twice.

    Args:
        clip: The clip to reject (source is ``clip.wav_path``).
        cfg: Pipeline configuration.
        reason: Rejection reason string (e.g. ``"per=0.42 > 0.300"``).
        ref_phonemes: Normalized reference phoneme string.
        hyp_phonemes: Normalized hypothesis phoneme string.
    """
    dest = cfg.paths.rejected / clip.wav_path.name
    shutil.copy2(str(clip.wav_path), str(dest))
    clip.wav_path.unlink()
    meta = {
        "index": clip.idx,
        "file": dest.name,
        "expected": clip.expected,
        "ref_phonemes": ref_phonemes,
        "hyp_phonemes": hyp_phonemes,
        "reason": reason,
    }
    (cfg.paths.rejected / f"{clip.wav_path.stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _apply_rejections(
    results: list[_ClipResult], cfg: common.Config
) -> list[dict[str, Any]]:
    """Move rejected clips and write their sidecars; return JSONL records.

    Args:
        results: Evaluation results from ``_evaluate``.
        cfg: Pipeline configuration.

    Returns:
        A list of rejection records (``{index, file, reason, per,
        ref_phonemes, hyp_phonemes}``) for the ``pronunciation.log`` aggregate.
    """
    records: list[dict[str, Any]] = []
    for r in results:
        if r.accepted:
            logger.info("PRONOUNCE-OK idx=%d PER=%.3f", r.clip.idx, r.per)
            continue
        reason = f"per={r.per:.3f} > {cfg.phoneme_threshold:.3f}"
        _reject_clip(r.clip, cfg, reason, r.ref_phonemes, r.hyp_phonemes)
        records.append(
            {
                "index": r.clip.idx,
                "file": r.clip.wav_path.name,
                "reason": reason,
                "per": r.per,
                "ref_phonemes": r.ref_phonemes,
                "hyp_phonemes": r.hyp_phonemes,
            }
        )
        logger.info("PRONOUNCE-REJECT idx=%d PER=%.3f -> %s", r.clip.idx, r.per, reason)
    return records


def _calibration_summary(per_values: list[float], threshold: float) -> dict[str, Any]:
    """Build the PER distribution block (min / percentiles / max / mean).

    Args:
        per_values: PER values from ``_evaluate``.
        threshold: Current ``phoneme_threshold`` (included for reference).

    Returns:
        A dict with ``count``, ``min``, ``p25``, ``median``, ``p75``, ``p90``,
        ``max``, ``mean`` and ``threshold``.
    """
    s = sorted(per_values)
    if not s:
        return {
            k: 0.0
            for k in ("count", "min", "p25", "median", "p75", "p90", "max", "mean")
        } | {"threshold": threshold}
    return {
        "count": len(s),
        "min": s[0],
        "p25": _percentile(s, 25),
        "median": _percentile(s, 50),
        "p75": _percentile(s, 75),
        "p90": _percentile(s, 90),
        "max": s[-1],
        "mean": sum(s) / len(s),
        "threshold": threshold,
    }


def _log_calibration(cal: dict[str, Any]) -> None:
    """Log the calibration distribution and a follow-up hint."""
    logger.info(
        "Calibration: N=%d min=%.3f p25=%.3f median=%.3f p75=%.3f p90=%.3f "
        "max=%.3f mean=%.3f (current threshold=%.3f)",
        cal["count"],
        cal["min"],
        cal["p25"],
        cal["median"],
        cal["p75"],
        cal["p90"],
        cal["max"],
        cal["mean"],
        cal["threshold"],
    )
    logger.info(
        "Calibration done: no clips were rejected. Adjust phoneme_threshold "
        "in config.yaml, then run `poetry run gen-dataset --step pronunciation`."
    )


def _write_rejected_log(records: list[dict[str, Any]], cfg: common.Config) -> None:
    """Write the rejected-records JSONL aggregate to ``workspace/rejected/``.

    Args:
        records: Rejection records from ``_apply_rejections``.
        cfg: Pipeline configuration.
    """
    if not records:
        return
    (cfg.paths.rejected / _REJECTED_LOG_NAME).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


def _aggregate_word_stats(
    per_clip_word_scores: list[list[_WordScore]],
) -> list[dict[str, Any]]:
    """Aggregate per-word PER across all clips into ranked rows.

    Collects every word's PER across all clips, then returns one row per word
    sorted by mean PER descending (worst first). Words with no reference
    phonemes (empty) are excluded.

    Args:
        per_clip_word_scores: One list of :class:`_WordScore` per clip.

    Returns:
        A list of dicts with ``word``, ``occurrences``, ``mean_per``,
        ``min_per``, ``max_per``, ``median_per``.
    """
    by_word: dict[str, list[float]] = {}
    for scores in per_clip_word_scores:
        for ws in scores:
            if ws.n_tokens == 0:
                continue
            by_word.setdefault(ws.word, []).append(ws.per)

    rows: list[dict[str, Any]] = []
    for word, pers in by_word.items():
        pers_sorted = sorted(pers)
        n = len(pers_sorted)
        rows.append(
            {
                "word": word,
                "occurrences": n,
                "mean_per": round(sum(pers_sorted) / n, 4),
                "min_per": round(pers_sorted[0], 4),
                "max_per": round(pers_sorted[-1], 4),
                "median_per": round(_percentile(pers_sorted, 50), 4),
            }
        )
    rows.sort(key=lambda r: r["mean_per"], reverse=True)
    return rows


def _write_word_report(word_rows: list[dict[str, Any]], cfg: common.Config) -> None:
    """Write the per-word PER CSV report to ``workspace/.pronunciation_words.csv``.

    Args:
        word_rows: Ranked rows from :func:`_aggregate_word_stats`.
        cfg: Pipeline configuration.
    """
    import csv

    cfg.paths.report.parent.mkdir(parents=True, exist_ok=True)
    out_path = cfg.paths.report.parent / _WORD_REPORT_NAME
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ["word", "occurrences", "mean_per", "min_per", "max_per", "median_per"]
        )
        for r in word_rows:
            writer.writerow(
                [
                    r["word"],
                    r["occurrences"],
                    f"{r['mean_per']:.4f}",
                    f"{r['min_per']:.4f}",
                    f"{r['max_per']:.4f}",
                    f"{r['median_per']:.4f}",
                ]
            )
    logger.info(
        "Per-word PER report written to %s (%d words)", out_path, len(word_rows)
    )


def _log_worst_words(
    word_rows: list[dict[str, Any]],
    top_n: int,
    min_occurrences: int,
) -> None:
    """Log the top-N worst-pronounced words with at least *min_occurrences*.

    Words that appear only once (``occ=1``) often have a high PER by chance
    (a single bad TTS render) and are excluded to keep the log actionable.

    Args:
        word_rows: Ranked rows from :func:`_aggregate_word_stats`.
        top_n: Maximum number of words to log.
        min_occurrences: Minimum occurrences to qualify.
    """
    if not word_rows:
        return
    filtered = [r for r in word_rows if r["occurrences"] >= min_occurrences]
    if not filtered:
        logger.info(
            "No words with >=%d occurrences found; skipping per-word log.",
            min_occurrences,
        )
        return
    logger.info(
        "Worst-pronounced words (top %d by mean PER, occ >= %d):",
        min(top_n, len(filtered)),
        min_occurrences,
    )
    for r in filtered[:top_n]:
        logger.info(
            "  %-20s occ=%d mean=%.3f min=%.3f max=%.3f median=%.3f",
            r["word"],
            r["occurrences"],
            r["mean_per"],
            r["min_per"],
            r["max_per"],
            r["median_per"],
        )


def _write_pronunciation_report(
    cfg: common.Config,
    calibration: dict[str, Any] | None,
    word_rows: list[dict[str, Any]],
    top_n: int,
    min_occurrences: int,
) -> None:
    """Write a human-readable pronunciation report to ``workspace/``.

    The report contains the PER calibration distribution (if available), a
    summary of the per-word stats, and the top-N worst-pronounced words
    filtered by minimum occurrences (systemic issues, not one-off artefacts).

    Args:
        cfg: Pipeline configuration.
        calibration: Calibration summary dict from :func:`_calibration_summary`,
            or ``None`` (normal mode — distribution is not evaluated).
        word_rows: Ranked rows from :func:`_aggregate_word_stats`.
        top_n: Maximum number of worst-pronounced words to list.
        min_occurrences: Minimum occurrences for a word to be listed.
    """
    out_path = cfg.paths.report.parent / _PRONUNCIATION_REPORT_NAME
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("PRONUNCIATION REPORT")
    lines.append("=" * 70)

    # --- Calibration / quality section ---
    if calibration:
        lines.append("")
        lines.append("Calibration distribution (PER across all checked clips)")
        lines.append("-" * 70)
        lines.append(f"  Clips checked         : {calibration.get('count', 0)}")
        lines.append(
            f"  min / p25 / median    : {calibration.get('min', 0):.3f} / "
            f"{calibration.get('p25', 0):.3f} / {calibration.get('median', 0):.3f}"
        )
        lines.append(
            f"  p75 / p90 / max       : {calibration.get('p75', 0):.3f} / "
            f"{calibration.get('p90', 0):.3f} / {calibration.get('max', 0):.3f}"
        )
        lines.append(f"  mean                  : {calibration.get('mean', 0):.3f}")
        lines.append(f"  Current threshold     : {calibration.get('threshold', 0):.3f}")

    # --- Word summary ---
    if not word_rows:
        lines.append("")
        lines.append("No per-word data available.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Pronunciation report written to %s", out_path)
        return

    total_unique = len(word_rows)
    perfect = sum(1 for r in word_rows if r["mean_per"] == 0.0)
    systemic = [
        r
        for r in word_rows
        if r["occurrences"] >= min_occurrences and r["mean_per"] > 0.0
    ]
    one_off = [r for r in word_rows if r["occurrences"] == 1]

    lines.append("")
    lines.append("Word Summary")
    lines.append("-" * 70)
    lines.append(f"  Unique words analyzed                 : {total_unique}")
    lines.append(f"  Perfectly pronounced (mean_per=0.00)  : {perfect}")
    lines.append(f"  Evaluable words (occ >= {min_occurrences})   : {len(systemic)}")
    lines.append(f"  One-off words (occ=1, excluded)        : {len(one_off)}")

    # --- Systemic issues (filtered top-N) ---
    if systemic:
        n_show = min(top_n, len(systemic))
        lines.append("")
        lines.append(
            f"Systemic Issues — "
            f"words appearing at least {min_occurrences} times, ranked by mean PER"
        )
        lines.append("  (These are candidates for removal or rewording in the corpus.)")
        lines.append("-" * 70)
        lines.append(
            f"  {'word':<24s} {'occ':>4s} {'mean':>7s} {'min':>7s} {'max':>7s} "
            f"{'median':>7s}"
        )
        lines.append(f"  {'-' * 24} {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
        for r in systemic[:n_show]:
            lines.append(
                f"  {r['word']:<24s} "
                f"{r['occurrences']:>4d} "
                f"{r['mean_per']:>7.3f} "
                f"{r['min_per']:>7.3f} "
                f"{r['max_per']:>7.3f} "
                f"{r['median_per']:>7.3f}"
            )
    else:
        lines.append("")
        lines.append(
            f"No words with {min_occurrences}+ occurrences. "
            "Corpus may be too small for systemic diagnosis."
        )

    # --- Top one-off artefacts (curiosity, 5 worst) ---
    if one_off:
        one_off_sorted = sorted(one_off, key=lambda r: r["mean_per"], reverse=True)
        n_one = min(5, len(one_off_sorted))
        lines.append("")
        lines.append(
            "Top one-off artefacts (occ=1, worst by PER — "
            "single bad TTS renders, not word problems)"
        )
        lines.append("-" * 70)
        lines.append(f"  {'word':<24s} {'occ':>4s} {'mean':>7s}")
        lines.append(f"  {'-' * 24} {'-' * 4} {'-' * 7}")
        for r in one_off_sorted[:n_one]:
            lines.append(
                f"  {r['word']:<24s} "
                f"{r['occurrences']:>4d} "
                f"{r['mean_per']:>7.3f}"
            )

    lines.append("")
    lines.append("=" * 70)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Pronunciation report written to %s", out_path)


def run_pronunciation(
    cfg: common.Config,
    calibrate: bool = False,
    only_rejected: bool = False,
) -> dict[str, Any]:
    """Run phoneme-level pronunciation verification on accepted clips.

    For each clip in ``accepted_wav/`` (the WER survivors from ``validate``),
    the reference sentence is phonemized with espeak-ng and the audio is
    recognized to phonemes with the configured wav2vec2 CTC model; the Phoneme
    Error Rate (PER) is computed and clips whose PER exceeds
    ``cfg.phoneme_threshold`` are moved from ``accepted_wav/`` to ``rejected/``.

    In calibrate mode no clips are moved: the PER distribution is logged and
    returned so the user can pick a threshold before committing.

    The pronunciation checkpoint (``cfg.paths.pronunciation_checkpoint``)
    records indices that have been accepted (PER <= threshold) by a previous
    non-calibrate run. When ``only_rejected=True``, only clips whose index is
    NOT in the checkpoint are processed, enabling the standard
    ``generate --only-rejected`` -> ``validate --only-rejected`` ->
    ``pronunciation --only-rejected`` regeneration cycle without re-scoring
    clips that already passed. The checkpoint is updated incrementally as
    clips are accepted, mirroring the write-on-accept pattern of generate's
    checkpoint (so a crash mid-run loses at most the current batch).

    Args:
        cfg: Pipeline configuration.
        calibrate: If True, measure-only mode (no rejections); prints and
            returns the PER distribution. The checkpoint is NOT updated in
            calibrate mode.
        only_rejected: If True, skip clips already in the pronunciation
            checkpoint (process only newly-added clips in ``accepted_wav/``).

    Returns:
        A dict with ``checked``, ``phoneme_rejected``, ``mean_per``,
        ``rejected_records`` and (in calibrate mode) a ``calibration`` block
        with min/median/max/percentiles. When ``phoneme_word_report`` is
        enabled, also includes ``worst_words`` (top-N by mean PER).
    """
    common.setup_logging(cfg.paths.log_file)
    common.ensure_dirs(cfg.paths.accepted_wav, cfg.paths.rejected)
    logger.info(
        "Pronunciation %s model='%s' threshold=%.3f",
        "CALIBRATE" if calibrate else "CHECK",
        cfg.phoneme_model,
        cfg.phoneme_threshold,
    )

    clips = _build_clips(cfg)

    # --- --only-rejected filter: skip clips already accepted by a previous run ---
    accepted_checkpoint: set[int] = set()
    if only_rejected:
        accepted_checkpoint = common.read_checkpoint(cfg.paths.pronunciation_checkpoint)
        before = len(clips)
        clips = [c for c in clips if c.idx not in accepted_checkpoint]
        logger.info(
            "--only-rejected: %d already accepted (skipped), %d to process",
            before - len(clips),
            len(clips),
        )

    if not clips:
        logger.warning(
            "No clips to check in %s. Run the validate step first.",
            cfg.paths.accepted_wav,
        )
        return {
            "checked": 0,
            "phoneme_rejected": 0,
            "mean_per": 0.0,
            "rejected_records": [],
        }

    # --- Reference (espeak-ng) and hypothesis (wav2vec2) phonemes ---
    # espeak-ng runs once and returns per-word phonemes; the flat clip-level
    # string consumed by _per is derived from the same output via _flat_phonemes.
    lang_code = common.language_code(cfg.language)
    ref_words_map = _phonemize(clips, lang_code)
    ref_map = {idx: _flat_phonemes(pw) for idx, pw in ref_words_map.items()}
    hyp_map = _PhonemeRecognizer(cfg).recognize(clips)

    # --- Evaluate (pure) then act or report ---
    results = _evaluate(clips, ref_map, hyp_map, cfg.phoneme_threshold)
    per_values = [r.per for r in results]
    mean_per = sum(per_values) / len(per_values) if per_values else 0.0

    result: dict[str, Any] = {
        "checked": len(results),
        "phoneme_rejected": 0,
        "mean_per": mean_per,
        "rejected_records": [],
    }

    if calibrate:
        for r in results:
            logger.info(
                "CALIBRATE idx=%d PER=%.3f ref=[%s] hyp=[%s]",
                r.clip.idx,
                r.per,
                r.ref_phonemes,
                r.hyp_phonemes,
            )
        result["calibration"] = _calibration_summary(per_values, cfg.phoneme_threshold)
        _log_calibration(result["calibration"])
    else:
        records = _apply_rejections(results, cfg)
        _write_rejected_log(records, cfg)
        result["phoneme_rejected"] = len(records)
        result["rejected_records"] = records
        logger.info(
            "Pronunciation: checked=%d rejected=%d mean PER=%.3f",
            len(results),
            len(records),
            mean_per,
        )
        # --- Persist pronunciation checkpoint: accepted indices ---
        # Two modes:
        # - only_rejected=True: merge newly-accepted with the pre-existing
        #   checkpoint (we only re-scored previously-rejected clips, so the
        #   already-accepted ones stay accepted).
        # - only_rejected=False: replace the checkpoint with the freshly-
        #   computed accepted set (we re-scored every clip, so a threshold
        #   change can drop previously-accepted clips correctly).
        newly_accepted = {r.clip.idx for r in results if r.accepted}
        if only_rejected:
            final_accepted = accepted_checkpoint | newly_accepted
        else:
            final_accepted = newly_accepted
        common.write_checkpoint(cfg.paths.pronunciation_checkpoint, final_accepted)
        logger.info(
            "Pronunciation checkpoint updated: %d total accepted (%d newly this run)",
            len(final_accepted),
            len(newly_accepted),
        )

    # --- Per-word PER report (diagnostic, both modes) ---
    if cfg.phoneme_word_report and results:
        per_clip_word_scores: list[list[_WordScore]] = []
        for r in results:
            ref_words = ref_words_map.get(r.clip.idx, [])
            hyp_tokens = r.hyp_phonemes.split()
            per_clip_word_scores.append(_align_words(ref_words, hyp_tokens))
        word_rows = _aggregate_word_stats(per_clip_word_scores)
        _write_word_report(word_rows, cfg)
        min_occ = max(1, cfg.phoneme_report_min_occurrences)
        _log_worst_words(word_rows, cfg.phoneme_word_top_n, min_occ)
        _write_pronunciation_report(
            cfg,
            result.get("calibration"),
            word_rows,
            cfg.phoneme_word_top_n,
            min_occ,
        )
        # Store only the filtered worst-words list so pipeline report (JSON) is also clean.
        filtered_worst = [
            r for r in word_rows if r["occurrences"] >= min_occ and r["mean_per"] > 0.0
        ][: cfg.phoneme_word_top_n]
        result["worst_words"] = filtered_worst

    return result
