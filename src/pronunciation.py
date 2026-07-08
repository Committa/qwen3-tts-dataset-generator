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
import re
import shutil
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

    Reuses ``jiwer.wer`` on whitespace-tokenized phoneme sequences: WER on a
    token list is exactly the Levenshtein-based PER.

    Args:
        reference: Normalized reference phoneme string.
        hypothesis: Normalized hypothesis phoneme string.

    Returns:
        PER in [0.0, +inf). Returns 1.0 when the reference is empty but the
        hypothesis is not, and 0.0 when both are empty.
    """
    from jiwer import wer as _jiwer_wer

    ref_toks = reference.split()
    hyp_toks = hypothesis.split()
    if not ref_toks:
        return 1.0 if hyp_toks else 0.0
    return float(_jiwer_wer(ref_toks, hyp_toks))


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


def _phonemize_references(clips: list[_Clip], lang_code: str) -> dict[int, str]:
    """Convert each clip's reference text to normalized IPA phonemes via espeak-ng.

    Uses the ``phonemizer`` espeak backend with ``language_switch='remove-flags'``
    so that foreign-word flags (e.g. ``(en)``) inserted by espeak-ng are dropped,
    and ``with_stress=False`` / ``preserve_punctuation=False`` so the output is a
    clean stream of IPA phones, space-separated for direct PER tokenization.

    Args:
        clips: Clips whose ``expected`` text must be phonemized.
        lang_code: ISO 639-1 language code (e.g. ``"it"``).

    Returns:
        A ``{idx: normalized_phonemes}`` dict.

    Raises:
        RuntimeError: If the espeak-ng binary is not found, with the actionable
            ``ESPEAK_HINT`` appended to the message.
    """
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    try:
        backend = EspeakBackend(
            language=lang_code,
            with_stress=False,
            preserve_punctuation=False,
            language_switch="remove-flags",
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}\n\n{ESPEAK_HINT}") from exc

    sep = Separator(phone=" ", syllable="", word=" ")
    sentences = [c.expected for c in clips]
    raw = backend.phonemize(sentences, separator=sep, strip=True)
    return {c.idx: _normalize_phonemes(r or "") for c, r in zip(clips, raw)}


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
        if device == "cuda":
            try:
                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA not available for phoneme model, falling back to CPU."
                    )
                    device = "cpu"
            except ImportError:
                device = "cpu"
        logger.info("Loading phoneme model '%s' (device=%s)", cfg.phoneme_model, device)
        self._processor = Wav2Vec2Processor.from_pretrained(cfg.phoneme_model)
        self._model = Wav2Vec2ForCTC.from_pretrained(cfg.phoneme_model).to(device)
        self._model.eval()
        self._device = device
        self._batch_size = max(1, cfg.phoneme_batch_size)

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
        attention_mask = (
            inputs.attention_mask.to(self._device)
            if hasattr(inputs, "attention_mask")
            else None
        )
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
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "threshold": threshold,
        }
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


def run_pronunciation(cfg: common.Config, calibrate: bool = False) -> dict[str, Any]:
    """Run phoneme-level pronunciation verification on accepted clips.

    For each clip in ``accepted_wav/`` (the WER survivors from ``validate``),
    the reference sentence is phonemized with espeak-ng and the audio is
    recognized to phonemes with the configured wav2vec2 CTC model; the Phoneme
    Error Rate (PER) is computed and clips whose PER exceeds
    ``cfg.phoneme_threshold`` are moved from ``accepted_wav/`` to ``rejected/``.

    In calibrate mode no clips are moved: the PER distribution is logged and
    returned so the user can pick a threshold before committing.

    Args:
        cfg: Pipeline configuration.
        calibrate: If True, measure-only mode (no rejections); prints and
            returns the PER distribution.

    Returns:
        A dict with ``checked``, ``phoneme_rejected``, ``mean_per``,
        ``rejected_records`` and (in calibrate mode) a ``calibration`` block
        with min/median/max/percentiles.
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
    lang_code = common.language_code(cfg.language)
    ref_map = _phonemize_references(clips, lang_code)
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

    return result
