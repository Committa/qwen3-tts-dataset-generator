# Post-publish audit workflow

The pipeline's threshold-based filters (`validate` for WER,
`pronunciation` for PER) catch the obviously wrong clips, but they miss
some classes of artefacts:

- **intonation drift** — flat or wrong stress on otherwise-correct
  phonemes. The PER step is *by design* blind to this:
  `_normalize_phonemes` strips the stress/length diacritics
  (`_PHONEME_STRIP_TABLE`) before scoring because espeak-ng's stress
  output is noisy across runs.
- **breathy artefacts** (sighs, audible inhalations between words) —
  not phonemes; faster-whisper transcribes through them (WER=0),
  wav2vec2 may or may not flag them.
- **low-level clicks / glitches / clipping** — pure audio signals that
  ASR/PER are not designed to detect.
- **EOS-collapse truncations** — clips cut to a few words + silence;
  caught by WER but borderline ones can slip past the threshold.

The `audit` tool surfaces these by ranking the **survivors** of the
pipeline (the clips that *did* pass the thresholds) with three
orthogonal signals, walking the user through the top-N interactively,
and writing the bad-clip verdict back into the standard `rejected/`
sidecar contract that the existing regeneration pipeline already
understands.

## When to use it

After a publish (`output/genNNN/`) that you suspect contains artefacts
the filters let through. Typically invoked once or twice after a big
generation batch, or when spot-checking reveals recurring issues.

## Workflow

```bash
# 1. Run the regular pipeline up to publish (writes workspace/.audit_per.csv
#    as a side effect of the pronunciation step):
poetry run gen-dataset --from validate

# 2. Audit the survivors:
poetry run audit
# → walks you through the top 500 suspects, ranked by composite score,
#   playing each clip and asking [a] keep / [r] bad / [p] play /
#   [b] back / [q] quit.
# → clips marked 'r' are written to rejected/ with reason="manual_audit"
#   and removed from normalized_wav/ + accepted_wav/.

# 3. Regenerate the bad clips with a fresh RNG draw, then re-normalize and
#    re-publish (--from validate includes normalize automatically):
poetry run gen-dataset --step generate --only-rejected
poetry run gen-dataset --from validate      # → output/gen002/

# 4. Re-audit only the regenerated clips to check whether they are now OK:
poetry run audit --only-rejected
```

`--only-rejected` reads the checkpoint from your previous audit run,
filters the queue to only the clips you marked `bad`, and presents them
for re-review. The old `bad` decisions are removed from the in-memory
checkpoint so they are not skipped by the resumability logic. You press
`a` if the new clip is now acceptable, or `r` if it still needs another
regeneration attempt.

## Ranking

Each clip gets a composite `suspect_score` in `[0, 1]` (1 = worst),
made from four percentile-ranked signals:

| Signal | Weight | Catches |
|---|---|---|
| `per_pitch` | 0.40 | intonation drift (stress-aware PER kept diacritics) |
| `per` | 0.20 | borderline word-level mispronunciation |
| `lf_burst` | 0.25 | sighs, breathy artefacts (low-frequency energy ratio) |
| `crest` | 0.15 | clicks / glitches (peak-to-RMS ratio) |

Use `--rank {per|pitch|audio|composite}` to switch ranking when you
want to focus on a specific class of artefacts.

## Pre-flight checks

`audit` refuses to start without:

- `workspace/normalized_wav/` populated — the audio-scan source.
  Run `poetry run gen-dataset --step normalize` if it is empty.
- `workspace/.audit_per.csv` existing **and covering every clip** in
  `normalized_wav/`. Re-run `poetry run gen-dataset --step pronunciation`
  if missing or incomplete.

A non-blocking warning is logged if the CSV is older than the
`normalized_wav/` directory (stale). You can proceed and the audio
signals will still be computed on the most recent wavs.

## Resume

Decisions are persisted immediately to
`workspace/.audit_checkpoint.json`, so Ctrl-C or `q` never loses
progress. A second `poetry run audit` run resumes from the first
undecided clip.

Flags:

| Flag | Effect |
|---|---|
| `--top N` | Review fewer/more suspects (default 500). |
| `--rank composite|per|pitch|audio` | Choose the ranking key (default composite). |
| `--restart` | Ignore the checkpoint and start over. |
| `--only-rejected` | Re-review only clips marked `bad` in a previous audit run. Useful after regenerating and re-validating them. |
| `--dry-run` | Walk the queue without writing files or the checkpoint. |
| `--no-clear` | Keep the scrollback instead of clearing the screen per clip. |

## What `r` (bad) does on disk

When you press `r` on `<idx>`:

1. `workspace/rejected/<idx>.json` is written with
   `reason="manual_audit"`, `audited=true`, and a nested
   `audit_signals` block with the clip's PER/PER-pitch/audio features
   at the time of the decision (useful post-mortem).
2. `workspace/normalized_wav/<idx>.wav` and
   `workspace/accepted_wav/<idx>.wav` are removed, so the next
   `generate --only-rejected` regenerates them from scratch, and the
   next `normalize` re-creates the normalized copy from the freshly
   accepted audio.

The sidecar format matches what
[`common.read_rejected_indices`](https://github.com/italian-synthetic-tts/qwen3-tts-dataset-generator/blob/main/src/common.py)
expects: an integer `index` field. The existing regeneration cycle
picks it up unchanged.

## Why this instead of lowering the thresholds

Lowering `wer_threshold` / `phoneme_threshold` would flood the
rejection queue with false positives (clip borderline-but-fine,
flagged as bad) and add back the very cost the thresholds were tuned
to avoid. `audit` drops no false positives into the live pipeline: it
only marks the few hundred clips *you* listen to, after explicit
review, and leaves the rest of the 29k-clips dataset untouched.

## Files

- `workspace/.audit_per.csv` — written by the `pronunciation` step.
  Columns: `idx|per|per_pitch|ref_phonemes|hyp_phonemes`. Re-runnable
  idempotently (per-idx row refresh on every pronunciation run).
- `workspace/.audit_checkpoint.json` — `audit` decisions state.
  Delete it to start over, or use `--restart`.