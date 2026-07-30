# Retry workflow

When `validate` rejects clips, inspect and retry:

1. Check `workspace/rejected/*.json` for expected vs transcription vs WER/PER
2. Listen to the rejected wavs in `workspace/rejected/`
3. If the TTS mispronounced: regenerate + re-validate (+ re-check pronunciation)
4. If the ASR/PER was wrong (audio sounds correct): accept manually
5. Publish the final dataset

## Re-running steps after a publish

`publish` copies the normalized clips into `output/gen{NNN}/wavs/` and
leaves the workspace state (`raw_wav` + `accepted_wav` + `normalized_wav`
+ `rejected`) intact. This means any step from `validate` onward can be
re-run after a publish without losing clips:

```bash
# Re-normalize with different loudness/trim settings and re-publish
poetry run gen-dataset --step normalize          # re-writes normalized_wav/ (skipped clips are still present)
poetry run gen-dataset --step normalize --force  # wipe normalized_wav/ first, then re-normalize everything
poetry run gen-dataset --step publish            # archives a fresh gen{NNN}/

# Re-pronunciation after raising phoneme_threshold: just delete the checkpoint
# (accepted_wav/ still has all survivors, pronunciation has no FS marker for
# passing clips so the checkpoint is the only source of truth)
rm workspace/.pronunciation_checkpoint.json
poetry run gen-dataset --step pronunciation
poetry run gen-dataset --from normalize

# Re-validate from scratch: delete the validate checkpoint (raw_wav/ retains
# every generated clip as a backup because validate copies, never moves)
rm workspace/.validate_checkpoint.json
poetry run gen-dataset --from validate
```

## Regeneration options

```bash
# Option A (fast): regenerate and re-validate only the rejected clips
poetry run gen-dataset --step generate --only-rejected
poetry run gen-dataset --step validate --only-rejected
# `generate --only-rejected` automatically removes the regenerated indices
# from the pronunciation checkpoint (P2 design), so the next plain
# `pronunciation` run re-scores only those regenerated clips via resumability.
poetry run gen-dataset --step pronunciation
poetry run gen-dataset --from normalize
# (`--step pronunciation --only-rejected` is a different tool: re-score
# pronunciation-rejected clips still in rejected/, e.g. after raising
# phoneme_threshold — does NOT touch the regeneration cycle.)

# Option B (full): regenerate rejected, but re-validate everything
poetry run gen-dataset --step generate --only-rejected
poetry run gen-dataset --from validate              # validate + pronunciation + normalize + publish

# Option C: ASR/PER was wrong — accept manually
poetry run gen-dataset --accept 7,13
poetry run gen-dataset --from normalize
```

The difference between A and B: A skips re-validating already-accepted clips
(much faster on the second iteration when most clips have already been
validated). B re-runs ASR on everything, which is slower but can catch
systematic errors (e.g. if the format or clipping changed after regeneration).

For the pronunciation-side re-scoring (after raising `phoneme_threshold`),
see [Pronunciation verification](Pronunciation-verification.md#re-scoring-already-rejected-clips).

## Review rejected clips interactively

`--accept 7,13` is fine for a handful of indices, but on a batch of dozens or
hundreds of false positives the loop is faster than reading back the JSONs by
hand. `review-rejected` walks every clip in `workspace/rejected/`, plays the
audio, and lets you decide with a single keypress:

```bash
poetry run review-rejected
```

```
─── 43/198 · 42 decided (30a + 12r) · 156 to go ───
[042/198]  idx=000294  wer=0.222 (thr 0.200)  dur=4.0s
expected   : I bambini mi hanno invitata a cenare con loro.
transcribed: Ita Naka mi hanno invitata a cenare con loro.
[playing...]
[a]ccept  [r]eject  [p]lay  [b]ack  [q]uit
> a
  -> accepted
```

### Keys

| Key | Action |
|---|---|
| `a` | accept (moves the clip to `accepted_wav/`, persists decision) |
| `r` | reject (keeps the clip in `rejected/`, persists decision) |
| `p` | play again |
| `b` | back (rewind one clip — the previous decision, if any, stands) |
| `q` | quit with summary |

Decisions are applied immediately (`a` and `r` call `common.accept_clips` in
real time), so a `q` in the middle never loses progress. State is stored in
`workspace/.review_checkpoint.json`; clips you've already decided on do not
reappear in the next run.

### Flags

| Flag | Effect |
|---|---|
| `--restart` | ignore the checkpoint and start over |
| `--sort index` | walk the corpus in natural order instead of the default WER-ascending ("easy wins" first) |
| `--dry-run` | preview the queue without touching the filesystem |
| `--no-clear` | keep the scrollback instead of clearing before each clip (focus mode is the default) |

By default the terminal is cleared before each clip so the previous one does
not scroll into view. Use `--no-clear` to keep the scrollback of recent
decisions. Replaying a clip with `p` only re-prints the `> ` prompt without
the legend, to avoid visual spam on repeated replays. A one-line progress
banner (`N/M · X decided (Aa + Br) · K to go`) is shown at the top of every
clip so you can see at a glance how far the triage has progressed across
sessions.

### Audio playback dependency

On Linux hosts (not Docker) install the PortAudio runtime first:

```bash
sudo apt-get install libportaudio2
```

The Docker image already has it.