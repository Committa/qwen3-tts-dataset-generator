# Pronunciation verification (phoneme-level)

The `validate` step uses faster-whisper, a *word*-level ASR that is forgiving of
pronunciation drift: a clip can match the reference transcript (low WER) while
the actual pronunciation is wrong, producing artifacts in the downstream
training set. The `pronunciation` step catches these by checking at the
*phoneme* level:

- The audio is recognised to espeak IPA phonemes by
  [`facebook/wav2vec2-xlsr-53-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
  (a multilingual wav2vec2 CTC model fine-tuned to output espeak phoneme
  labels).
- The reference sentence is converted to phonemes with **espeak-ng** via the
  `phonemizer` library.
- The two phoneme sequences are compared with the **Phoneme Error Rate** (PER),
  computed with `jiwer` (the same library used for WER). Both sides use the
  same espeak phoneme inventory, so the comparison is direct.
- Clips whose PER exceeds `phoneme_threshold` (default `0.30` — phoneme
  recognition is noisier than word ASR) are moved from `accepted_wav/` to
  `rejected/` and feed back into the `--only-rejected` regeneration loop.

The step runs after `validate` (on the WER survivors) and before `normalize`
(so normalize only processes pronunciation survivors). It is gated by the
`phoneme_check` config flag in a full run; an explicit `--step pronunciation`
always runs it.

Reference text source: `common.load_sentences(cfg)` + `int(wav_path.stem)` —
the same single source of truth used by validate/manifest/report.

## System dependency: espeak-ng

The `phonemizer` library wraps the **espeak-ng** binary, which must be on PATH.

- **Windows**: install the MSI from the
  [espeak-ng GitHub releases](https://github.com/espeak-ng/espeak-ng/releases).
  If `phonemizer` cannot find it, set the `PHONEMIZER_ESPEAK_LIBRARY`
  environment variable to the `espeak-ng.dll` path.
- **Linux** (Debian/Ubuntu): `sudo apt-get install espeak-ng`.

If espeak-ng is missing, the step prints a hint and exits with code 1.

On the audio side, wav2vec2 requires 16 kHz mono float input; the generator's
native rate is resampled via `librosa.load(..., sr=16000, mono=True)`.
`transformers==4.57.3` (pinned by qwen-tts) is compatible with
`Wav2Vec2ForCTC`/`Wav2Vec2Processor`.

## Tuning the threshold

PER absolute values depend on the model and the phoneme normalization, so pick
the threshold empirically on a known-good set:

```bash
poetry run gen-dataset --step pronunciation --calibrate
```

This measures the PER for every clip **without rejecting anything** and prints
the distribution (min / p25 / median / p75 / p90 / max / mean). Set
`phoneme_threshold` in `config.yaml` to a value that rejects clips you
consider mispronounced while keeping the good ones, then run the step for real:

```bash
poetry run gen-dataset --step pronunciation
```

## Re-scoring already-rejected clips

`--only-rejected` (with `--step pronunciation`) re-scores pronunciation-rejected
clips still sitting in `rejected/` — e.g. after raising `phoneme_threshold`.
Unlike `generate`/`validate --only-rejected`, this flag is **step-specific**:
only sidecars carrying `ref_phonemes`/`hyp_phonemes` (pronunciation sidecars)
are eligible; validate-rejected sidecars (transcription/wer only) are skipped,
since there is no point scoring PER on a clip whose words are wrong.

- Wavs are read from `rejected/` in place (NOT from `accepted_wav/`).
- The `done` checkpoint is NOT used as a filter (you are explicitly asking to
  re-evaluate previously-rejected audio).
- **Accept** moves the wav back to `accepted_wav/` and removes the sidecar
  (clip no longer rejected).
- **Reject** just refreshes the sidecar PER in place (wav stays in `rejected/`).
- The checkpoint is updated at the end of each run (MERGE, never REPLACE —
  `done` is a monotonic set).

## Regeneration cycle

`generate --only-rejected` invalidates the pronunciation `done` checkpoint for
the regenerated indices: the new audio has not been pronunciation-checked yet,
so removing the index from `done` is exactly the right trigger — the next
standard `pronunciation` run picks up the regenerated clips via normal
resumability (`accepted_wav/ - done`). No special flag, no extra file. The
three checkpoints (`generate`, `validate`, `pronunciation`) all share the same
monotonic `done` contract: MERGE, never REPLACE. Manual accept via
`review-rejected` does NOT touch any checkpoint — it is a human judgment on
audio quality, not a PER pass.

## Per-word PER report

The clip-level PER is a single number: it tells you *that* a clip is
mispronounced, not *which* words are wrong. To help decide which corpus words
to remove or reword, the step also ranks reference words by mean PER across
all checked clips:

- espeak-ng phonemizes the reference **per word** (a sentinel separator
  preserves word boundaries in the phoneme output).
- The recognized phoneme stream is aligned to the per-word reference via a
  Levenshtein DP with backtrace, attributing each edit (substitution /
  deletion / insertion) to the word owning the involved reference token.
- Per-word PER is aggregated across all checked clips and ranked worst-first.

Output (gated by `phoneme_word_report`, default on; runs in both normal and
`--calibrate` mode — it's pure post-processing of already-decoded phonemes,
no extra inference):

- `workspace/.pronunciation_words.csv` — every word with
  `word | occurrences | mean_per | min_per | max_per | median_per` (worst first).
- `report.json` → `pronunciation.worst_words` — top `phoneme_word_top_n` words.
- A CLI/log block with the same top-N.

Use the CSV to spot words that are systematically problematic (high mean PER
with several occurrences) and either remove the offending sentences from the
corpus or reword them.

See also the [Configuration reference](Configuration-reference.md) for the
full list of `phoneme_*` parameters.