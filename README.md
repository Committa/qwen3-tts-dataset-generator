# qwen3-tts-dataset-generator

**Pipeline** that transforms a text corpus into a **synthetic TTS dataset**
validated by ASR. It generates audio with [Qwen3-TTS], filters out low-quality clips via WER,
checks pronunciation at the phoneme level (PER), normalizes
(resample, loudness, silence trim), and produces a ready-to-use LJSpeech
manifest with train/val split.

**Resumable** (JSON checkpoint), **validated** (ASR + WER + phoneme-level PER), **normalized**
(resample 22050 Hz, loudness EBU R128, silence trimming) and produces an **LJSpeech manifest**
with deterministic train/val split. Works with any language supported by Qwen3-TTS:
Italian, English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish.

> Developed and tested on **NVIDIA RTX 4070 (12 GB VRAM)**. Compatible with any NVIDIA GPU
> with CUDA 12.4+. OOM is handled with a clear message and suggestion.

---

## Features

- **Generate** audio from a text corpus via Qwen3-TTS (batch GPU inference with resumable checkpoint)
- **Validate** each clip with ASR (faster-whisper) + WER, auto-reject low-quality clips (with checkpoint/resume)
- **Verify pronunciation** at the phoneme level (wav2vec2 CTC + espeak-ng PER) to catch clips that pass WER but are mispronounced
- **Normalize** audio: convert to mono, resample to 22050 Hz, loudness normalize (-23 LUFS), trim silence, save as 16-bit PCM
- **Manifest** in LJSpeech format (filename|text) with deterministic train/val split
- **Multi-language**: works with any language supported by Qwen3-TTS (Italian, English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish)

---

## Setup

### Prerequisites

- **Python 3.11** (exactly 3.11.x — not 3.10, not 3.12)
- **Poetry** ([install guide](https://python-poetry.org/docs/#installation)) for dependency management
- **NVIDIA GPU** with CUDA 12.4+ and up-to-date drivers
- **Git** to clone the repository

### Clone and install

```bash
git clone https://github.com/Committa/qwen3-tts-dataset-generator.git
cd qwen3-tts-dataset-generator
poetry lock
poetry install
```

### Verify GPU

```bash
poetry run python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True`. If `False`, double-check your NVIDIA drivers and CUDA installation.

### Optional: FlashAttention (faster inference, lower VRAM)

```bash
pip install flash-attn --no-build-isolation
```

After installing, set `attn_implementation: "flash_attention_2"` in `config.yaml`.
Not required — the pipeline works without it (uses `sdpa` by default).

## Docker Setup

```bash
docker build -t qwen3-tts-dataset-generator .
docker run --rm --gpus all \
  -v "$PWD/inputs:/workspace/inputs:ro" \
  -v "$PWD/workspace:/workspace/workspace" \
  -v "$PWD/output:/workspace/output" \
  -v "$PWD/logs:/workspace/logs" \
  qwen3-tts-dataset-generator
```

## Preparing the input corpus

Prepare your text corpus as a plain text file, one sentence per line.
Lines starting with `#` and blank lines are ignored. The default path is
`inputs/sentences.txt` (configured via `input_sentences` in
`config.yaml`). The sample file included in the repo contains placeholder
sentences for testing. Works with any language supported by Qwen3-TTS.

## Usage

### Pipeline steps

| Step | Description |
|------|-------------|
| `generate` | Create audio from text corpus via Qwen3-TTS (batch GPU inference, resumable) |
| `validate` | Check each clip with ASR (faster-whisper) + WER, accept/reject |
| `pronunciation` | Phoneme-level check (wav2vec2 CTC + espeak-ng PER) on the WER survivors |
| `normalize` | Resample to 22050 Hz, loudness normalize (-23 LUFS), trim silence, 16-bit PCM |
| `publish` | Build LJSpeech manifest + report + archive to `output/gen{NNN}/` |

> **Language support for number normalization:** Full word-to-digit conversion
> (alpha2digit + num2words) is available for Italian, English, French, Spanish,
> Portuguese, German, and Dutch. For other languages (Chinese, Japanese, Korean,
> Russian, etc.), validation runs with basic text cleanup only — number words
> are compared verbatim without normalization.

### Commands

```bash
# Full pipeline (auto-clean + all steps + archive)
poetry run gen-dataset

# If an interrupted generation is detected you will be asked:
#   [r] resume previous generation (skip clean)
#   [f] start a fresh clean run
# Non-interactive runs (no TTY) resume by default to preserve progress.

# Full pipeline without auto-clean
poetry run gen-dataset --no-clean

# Single step
poetry run gen-dataset --step generate
poetry run gen-dataset --step validate
poetry run gen-dataset --step pronunciation
poetry run gen-dataset --step normalize
poetry run gen-dataset --step publish

# Run from a step onward (no auto-clean)
poetry run gen-dataset --from validate

# Regenerate only rejected clips
poetry run gen-dataset --step generate --only-rejected

# Pronunciation calibration: measure the PER distribution without rejecting
poetry run gen-dataset --step pronunciation --calibrate

# Manually accept rejected clips (override ASR / PER)
poetry run gen-dataset --accept 7,13

# Help
poetry run gen-dataset --help
```

### Retry workflow

When validation rejects clips, inspect and retry:

1. Check `workspace/rejected/*.json` for expected vs transcription vs WER/PER
2. Listen to the rejected wavs in `workspace/rejected/`
3. If the TTS mispronounced: regenerate + re-validate (+ re-check pronunciation)
4. If the ASR/PER was wrong (audio sounds correct): accept manually
5. Publish the final dataset

```bash
# Option A (fast): regenerate and re-validate only the rejected clips
poetry run gen-dataset --step generate --only-rejected
poetry run gen-dataset --step validate --only-rejected
poetry run gen-dataset --from pronunciation   # pronunciation + normalize + publish
# (pronunciation here is full, since `--from` does not pass --only-rejected)
# If you only want to re-score the regenerated clips (~1 minute on a 29k
# corpus instead of ~11), the right tool is the regenerated-only path:
# `generate --only-rejected` writes `workspace/.regenerated.json` with the
# indices it just regenerated; the next plain `pronunciation` consumes it
# one-shot and processes ONLY those. No flag needed.
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

The difference between A and B: A skips re-validating already-accepted clips (much faster on the second iteration when most clips have already been validated). B re-runs ASR on everything, which is slower but can catch systematic errors (e.g. if the format or clipping changed after regeneration).

### Review rejected clips interactively

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

Keys: `a` accept (moves the clip to `accepted_wav/`, persists decision), `r`
reject (keeps the clip in `rejected/`, persists decision), `p` play again,
`b` back (rewind one clip — the previous decision, if any, stands),
`q` quit with summary.

Decisions are applied immediately (`a` and `r` call `common.accept_clips` in
real time), so a `q` in the middle never loses progress. State is stored in
`workspace/.review_checkpoint.json`; clips you've already decided on do not
reappear in the next run. Use `--restart` to ignore the checkpoint and start
over, `--sort index` to walk the corpus in natural order instead of the
default WER-ascending ("easy wins" first), and `--dry-run` to preview the
queue without touching the filesystem.

By default the terminal is cleared before each clip so the previous one does
not scroll into view (focus mode). Use `--no-clear` to keep the scrollback of
recent decisions. Replaying a clip with `p` only re-prints the `> ` prompt
without the legend, to avoid visual spam on repeated replays. A one-line
progress banner (`N/M · X decided (Aa + Br) · K to go`) is shown at the top
of every clip so you can see at a glance how far the triage has progressed
across sessions.

On Linux hosts (not Docker) install the PortAudio runtime first:
`sudo apt-get install libportaudio2`. The Docker image already has it.

## Pronunciation verification (phoneme-level)

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

### System dependency: espeak-ng

The `phonemizer` library wraps the **espeak-ng** binary, which must be on PATH.

- **Windows**: install the MSI from the
  [espeak-ng GitHub releases](https://github.com/espeak-ng/espeak-ng/releases).
  If `phonemizer` cannot find it, set the `PHONEMIZER_ESPEAK_LIBRARY`
  environment variable to the `espeak-ng.dll` path.
- **Linux** (Debian/Ubuntu): `sudo apt-get install espeak-ng`.

If espeak-ng is missing, the step prints a hint and exits with code 1.

### Tuning the threshold

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

### Per-word PER report

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

## Speaker / voice test

```bash
poetry run test-gen-dataset
poetry run test-gen-dataset --model-size 0.6b
poetry run test-gen-dataset --model-type base         # override config.yaml model_type
poetry run test-gen-dataset --speaker Vivian     # custom_voice: single preset speaker
poetry run test-gen-dataset --speaker my_voice   # base: single custom voice
poetry run test-gen-dataset --batch-size 8       # override config.yaml batch_size
```

In `custom_voice` mode the sweep covers all built-in speakers; in `base` mode
it covers every custom voice found under `inputs/voices/`. `--speaker` restricts
the test to a single one. `--model-type` overrides `model_type` from
`config.yaml` so both worlds can be tested without editing the config file
(note: the two model types load different HuggingFace repos, so the model is
reloaded when switching). Generation is **batched** by `batch_size` from
`config.yaml` (override with `--batch-size`): each speaker/voice produces
`ceil(N / batch_size)` model calls instead of N, dramatically reducing the time
needed to evaluate the whole universe of voices.

Each generated clip is written next to its exact transcript
(`output/test_speaker/<speaker>_<i>.wav` + `<speaker>_<i>.txt`), so a good
candidate can be copied straight into `inputs/voices/` as a voice-cloning
reference — see [Stable voice for dataset training](#stable-voice-for-dataset-training-recommended).

## Config (`config.yaml`)

Main parameters:

| Parameter | Default | Notes |
|---|---|---|
| `model_size` | `1.7b` | `1.7b` or `0.6b` |
| `model_type` | `custom_voice` | `custom_voice` (preset speakers) or `base` (voice clone) |
| `dtype` | `bfloat16` | `bfloat16` or `float16` |
| `attn_implementation` | `sdpa` | `sdpa` (default), `flash_attention_2` (faster, needs `pip install flash-attn`), or `eager` |
| `device_map` | `cuda:0` | `"auto"` for CPU offload with 1.7B on 12 GB |
| `speaker` | `Vivian` | preset speaker name (custom_voice) or custom voice name under `inputs/voices/` (base) |
| `language` | `Italian` | Language name or ISO 639-1 2-letter code. Supported names: `italian`, `english`, `french`, `spanish`, `portuguese`, `german`, `dutch`, `russian`, `chinese`, `japanese`, `korean`. ISO codes (e.g. `it`, `en`, `zh`) are also accepted. |
| `instruct` | `""` | style instruction in natural language; works only with `model_size: "1.7b"` in `custom_voice` mode (ignored by `0.6b` and base mode) |
| `x_vector_only_mode` | `false` | base only: `false`=ICL (best quality, needs `<speaker>.txt`) \| `true`=x-vector-only |
| `input_sentences` | `sentences.txt` | corpus filename under `inputs/` |
| `test_sentences` | `test_sentences.txt` | test phrases filename under `inputs/` (used by `test-gen-dataset`) |
| `max_new_tokens` | `2048` | maximum tokens generated per clip |
| `seed` | `42` | reproducibility for train/val split and sampling |
| `batch_size` | `4` | 4–8 recommended on 12 GB. Used by both `gen-dataset` (generate step) and `test-gen-dataset` (override with `--batch-size`). |
| `asr_model` | `medium` | faster-whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`) |
| `asr_device` | `cuda` | `cuda` or `cpu` |
| `asr_compute_type` | `float16` | `float16`, `int8`, etc. — affects ASR performance |
| `asr_workers` | `1` | parallel ASR transcriptions (`1`=sequential; `>1`=thread pool, faster-whisper runs them concurrently via `num_workers`; memory grows with workers). On a single GPU the benefit is marginal — throughput is GPU-bound. |
| `asr_beam_size` | `5` | beam size for the ASR decoder (`5`=author default; `1`=greedy, ~3-5x faster on short clips with negligible WER impact). Primary lever for validate throughput on GPU. |
| `wer_threshold` | `0.20` | WER rejection threshold (clips above this are rejected) |
| `phoneme_check` | `false` | enable the `pronunciation` step in a full run (an explicit `--step pronunciation` always runs it) |
| `phoneme_model` | `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | wav2vec2 CTC model used for phoneme recognition |
| `phoneme_device` | `cuda` | `cuda` or `cpu` (falls back to CPU if CUDA unavailable) |
| `phoneme_batch_size` | `8` | wav2vec2 CTC batched inference (not thread-safe; uses batching, not workers) |
| `phoneme_cleanup_every_n_batches` | `10` | free the PyTorch CUDA allocator pool every N batches (without this, reserved VRAM grows monotonically across hundreds of short forwards until OOM) |
| `phoneme_threshold` | `0.30` | PER rejection threshold (tune with `--step pronunciation --calibrate`) |
| `phoneme_word_report` | `true` | write `workspace/.pronunciation_words.csv` ranking words by mean PER (diagnostic, both modes) |
| `phoneme_word_top_n` | `20` | number of worst-pronounced words to log and include in `report.json` |
| `phoneme_report_min_occurrences` | `3` | words must appear at least this many times in the corpus to be listed in the "worst words" report |
| `target_sample_rate` | `22050` | output sample rate in Hz |
| `target_lufs` | `-23.0` | loudness normalization target (EBU R128) |
| `trim_silence_db` | `60` | dB threshold for silence trimming (higher = less aggressive) |
| `tail_margin_ms` | `120` | ms of original signal preserved past the trim end (protects final consonants) |
| `tail_pad_ms` | `80` | ms of silence appended after trimming (clean decay boundary) |
| `val_ratio` | `0.1` | fraction of data held out for validation |
| `clean_on_full_run` | `true` | auto-clean workspace before a fresh full run; an incomplete checkpoint prompts resume vs. clean (`--no-clean` overrides) |

### Speakers (CustomVoice)

| Speaker | Native language |
|---|---|
| Vivian | Chinese |
| Serena | Chinese |
| Uncle_Fu | Chinese |
| Dylan | Chinese (dialect) |
| Eric | Chinese (dialect) |
| Ryan | English |
| Aiden | English |
| Ono_Anna | Japanese |
| Sohee | Korean |

Every speaker can speak any supported language. Use
`poetry run test-gen-dataset` to choose the best one.

### Custom voices (base / voice clone)

Set `model_type: "base"` to clone any voice from a reference audio sample
instead of using a preset speaker. The Base model
(`Qwen/Qwen3-TTS-12Hz-{0.6b,1.7b}-Base`) extracts a speaker embedding (and,
in ICL mode, reference speech codes) from the sample and reapplies it to the
whole corpus.

Each custom voice lives as a pair of files under `inputs/voices/`:

```
inputs/voices/
├── my_voice.wav   # reference audio (required)
├── my_voice.txt   # transcript (required for ICL mode, optional for x-vector-only)
├── another.wav
└── another.txt
```

Configuration:

```yaml
model_type: "base"
speaker: "my_voice"           # -> inputs/voices/my_voice.wav + my_voice.txt
x_vector_only_mode: false     # false=ICL (best quality, needs my_voice.txt) | true=x-vector-only
```

Two cloning modes are supported:

- **ICL** (`x_vector_only_mode: false`, default): uses the reference audio **and**
  its transcript. Best quality. `ref.txt` is required.
- **x-vector-only** (`x_vector_only_mode: true`): uses only the speaker embedding,
  no transcript needed. Lower quality.

The extracted `VoiceClonePromptItem` is cached per-voice per-model-size under `workspace/.voice_cache/` (e.g. `workspace/.voice_cache/my_voice_1.7b.pt`) and reused across runs; the cache is invalidated automatically when the reference audio, transcript, cloning mode, model type, or model size change. The cache directory is fixed (`workspace/.voice_cache/`) and not user-configurable.

Test every custom voice before the full run:

```bash
poetry run test-gen-dataset
poetry run test-gen-dataset --speaker my_voice   # test a single voice
```

## Stable voice for dataset training (recommended)

Generating thousands of clips with a `custom_voice` preset speaker (e.g.
`Serena`) produces audible **drift**: tone, pacing and micro-formants shift
across clips, so the dataset sounds like several people with similar voices
rather than one. This is intrinsic to `custom_voice` — each call resamples the
speaker from a learned distribution (autoregressive generation over discrete
codec tokens), so variance accumulates over thousands of draws. For
downstream training (e.g. Piper) this degrades quality and introduces
artifacts.

The fix recommended by the Qwen3-TTS team for a stable character voice over
many lines is to **clone a single fixed reference clip** instead of recalling
the preset every time:

1. Generate a clean 10-15 s reference clip once with the preset speaker.
2. Build a reusable `voice_clone_prompt` from it (ICL: reference audio +
   transcript).
3. Use that fixed prompt with the **Base** model to synthesize the whole
   corpus via `generate_voice_clone`.

The identity is then anchored to a concrete audio sample rather than a
re-sampled distribution, so every clip shares the same timbre and prosody.

### Step-by-step

The `inputs/test_sentences.txt` phrases are calibrated to produce 10-15 s
clips at Italian speech rate (~3 words/s) — the sweet spot for a
voice-cloning reference.

1. **Generate reference candidates** with the preset speaker:

   ```bash
   poetry run test-gen-dataset --model-type custom_voice --speaker Serena
   ```

   `instruct` from `config.yaml` is applied and **baked into** the resulting
   audio — the style lives in the reference clip itself, not in a per-call
   directive. The Base model ignores `instruct` (it has no such input), so
   this is your only chance to set the tone.

   Each clip is written next to its exact transcript:
   `output/test_speaker/Serena_00.wav` + `Serena_00.txt`,
   `Serena_01.wav` + `Serena_01.txt`, etc.

2. **Listen** to the candidates and pick the cleanest one (no truncation,
   natural pacing, no artifacts). 10-15 s is ideal: shorter captures too
   little of the speaker; longer bloats the ICL context with no gain.

3. **Copy** the chosen clip and its transcript into `inputs/voices/` under a
   new voice name, e.g. `serena`:

   ```bash
   cp output/test_speaker/Serena_03.wav inputs/voices/serena.wav
   cp output/test_speaker/Serena_03.txt inputs/voices/serena.txt
   ```

   The `.txt` must match the audio **exactly** (it is the ICL reference
   transcript). The file written in step 1 is already exact — do not retype
   it.

4. **Switch the config** to Base + ICL mode:

   ```yaml
   model_type: "base"
   speaker: "serena"            # inputs/voices/serena.wav + serena.txt
   x_vector_only_mode: false    # ICL (best quality, needs the .txt)
   ```

5. **Generate the dataset**:

   ```bash
   poetry run gen-dataset
   ```

   The first run extracts the `VoiceClonePromptItem` once (cached at
   `workspace/.voice_cache/serena_<model_size>.pt`, auto-invalidated by a
   fingerprint of the reference audio, transcript, cloning mode and model)
   and broadcasts it over every batch. Every clip is now conditioned on the
   same fixed reference → no drift.

### Notes

- Keep `temperature` low (e.g. `0.3`) for cross-clip consistency; see the
  sampling parameters comment in `config.yaml`. Low temperature is safe here
  because the reference, not the sampling distribution, carries the identity.
- `instruct` is ignored in Base mode — set the desired tone in step 1, it is
  permanent in the reference.
- To regenerate the reference: pick a different candidate, replace the two
  files in `inputs/voices/`, and the cache auto-invalidates. To force a clean
  extraction, delete `workspace/.voice_cache/<name>_<model_size>.pt`.
- The `test_sentences.txt` clips also serve as a quick speaker evaluation —
  just note they are optimized for reference length and neutral tone, not for
  phonetic stress-testing.

## VRAM / OOM

- **1.7B** bf16 → on RTX 4070 (12 GB) fits with `batch_size: 2` (or lower).
  If too tight, try `device_map: "auto"` (CPU offload, slower) or fall back to
  `model_size: "0.6b"`.
- On OOM/Ctrl+C: the pipeline saves both the generate and validate checkpoints
  and prints a clear suggestion.
- Full run auto-archives the result in `output/gen{NNN}/`. On a full run, if an
  incomplete generation is detected, you're prompted to resume or start fresh;
  use `--no-clean` to force a resume, or delete the checkpoints to force a clean.

## Project structure

```
.
├── pyproject.toml
├── Dockerfile
├── config.yaml
├── src/
│   ├── common.py           # shared utilities
│   ├── generate.py         # audio generation
│   ├── validate.py         # ASR validation (WER)
│   ├── pronunciation.py    # phoneme-level verification (PER)
│   ├── normalize_audio.py  # audio normalization
│   ├── build_manifest.py   # LJSpeech manifest
│   ├── report.py           # final report
│   ├── pipeline.py         # CLI orchestrator
│   └── test_speaker.py     # speaker test utility
├── inputs/                 # user-provided text corpora and voice samples
│   ├── sentences.txt
│   ├── test_sentences.txt
│   └── voices/             # custom voices for base (voice clone) mode
│       └── <name>.wav        # + <name>.txt for ICL transcript
├── workspace/              # volatile (auto-cleaned on full run)
│   ├── raw_wav/
│   ├── accepted_wav/
│   ├── rejected/
│   ├── .generate_checkpoint.json
│   └── .validate_checkpoint.json
├── output/                 # immutable dataset archives
│   ├── gen001/
│   │   ├── wavs/
│   │   ├── metadata_train.csv
│   │   ├── metadata_val.csv
│   │   └── report.json
│   └── ...
└── logs/
```

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

[Qwen3-TTS]: https://github.com/QwenLM/Qwen3-TTS