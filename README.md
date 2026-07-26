# qwen3-tts-dataset-generator

**Pipeline** that transforms a text corpus into a **synthetic TTS dataset**
validated by ASR. It generates audio with [Qwen3-TTS], filters out low-quality
clips via WER, checks pronunciation at the phoneme level (PER), normalizes
(resample, loudness, silence trim), and produces a ready-to-use LJSpeech
manifest with train/val split.

**Resumable** (JSON checkpoint), **validated** (ASR + WER + phoneme-level PER),
**normalized** (resample 22050 Hz, loudness EBU R128, silence trimming) and
produces an **LJSpeech manifest** with deterministic train/val split. Works
with any language supported by Qwen3-TTS: Italian, English, Chinese, Japanese,
Korean, German, French, Russian, Portuguese, Spanish.

> Developed and tested on **NVIDIA RTX 4070 (12 GB VRAM)**. Compatible with any
> NVIDIA GPU with CUDA 12.4+. OOM is handled with a clear message and
> suggestion.

---

## Highlights

- **Generate** audio from a text corpus via Qwen3-TTS (batch GPU inference with resumable checkpoint)
- **Validate** each clip with ASR (faster-whisper) + WER, auto-reject low-quality clips (with checkpoint/resume)
- **Verify pronunciation** at the phoneme level (wav2vec2 CTC + espeak-ng PER) to catch clips that pass WER but are mispronounced
- **Normalize** audio: convert to mono, resample to 22050 Hz, loudness normalize (-23 LUFS), trim silence, save as 16-bit PCM (writes into `normalized_wav/`, the originals in `accepted_wav/` are kept intact for re-runs)
- **Manifest** in LJSpeech format (filename|text) with deterministic train/val split
- **Multi-language**: works with any language supported by Qwen3-TTS

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

Expected output: `True`. If `False`, double-check your NVIDIA drivers and
CUDA installation.

### Optional: FlashAttention (faster inference, lower VRAM)

```bash
pip install flash-attn --no-build-isolation
```

After installing, set `attn_implementation: "flash_attention_2"` in
`config.yaml`. Not required — the pipeline works without it (uses `sdpa` by
default).

## Docker

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

Prepare your text corpus as a plain text file, one sentence per line. Lines
starting with `#` and blank lines are ignored. The default path is
`inputs/sentences.txt` (configured via `input_sentences` in `config.yaml`).
The sample file included in the repo contains placeholder sentences for
testing. Works with any language supported by Qwen3-TTS.

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

poetry run gen-dataset --no-clean              # full pipeline without auto-clean
poetry run gen-dataset --step generate         # single step
poetry run gen-dataset --from validate         # run from a step onward
poetry run gen-dataset --step generate --only-rejected  # regenerate rejected clips
poetry run gen-dataset --step pronunciation --calibrate  # measure PER distribution, no rejects
poetry run gen-dataset --accept 7,13           # manually accept rejected clips
poetry run gen-dataset --help                  # CLI help
```

When validation rejects clips, see the **[Retry workflow](docs/Retry-workflow.md)**
guide (regeneration options + interactive `review-rejected` triage).

### Pronunciation verification

The `pronunciation` step (phoneme-level PER) is gated by `phoneme_check` in a
full run; an explicit `--step pronunciation` always runs it. It requires the
**espeak-ng** binary on PATH. See
**[Pronunciation verification](docs/Pronunciation-verification.md)** for
install instructions, threshold tuning (`--calibrate`), and the per-word PER
report.

## Config essentials

Main parameters in `config.yaml` (see the full
**[Configuration reference](docs/Configuration-reference.md)** for all
parameters and defaults):

| Parameter | Default | Notes |
|---|---|---|
| `model_size` | `1.7b` | `1.7b` or `0.6b` |
| `model_type` | `custom_voice` | `custom_voice` (preset speakers) or `base` (voice clone) |
| `speaker` | `Vivian` | preset speaker name (custom_voice) or custom voice name under `inputs/voices/` (base) |
| `language` | `Italian` | Language name or ISO 639-1 2-letter code (`it`, `en`, `zh`, ...) |
| `batch_size` | `4` | 4–8 recommended on 12 GB VRAM |
| `wer_threshold` | `0.20` | WER rejection threshold (clips above this are rejected) |
| `phoneme_check` | `false` | enable the `pronunciation` step in a full run |
| `phoneme_threshold` | `0.30` | PER rejection threshold (tune with `--calibrate`) |
| `target_sample_rate` | `22050` | output sample rate in Hz |
| `seed` | `42` | reproducibility for train/val split and sampling |

## Speakers and voices

Two model types are supported:

- **`custom_voice`** — 9 preset speakers (Vivian, Serena, Ryan, Aiden,
  Ono_Anna, Sohee, ...). Every speaker can speak any supported language.
- **`base`** — voice cloning from a reference audio sample under
  `inputs/voices/`. ICL mode (best quality, needs a `.txt` transcript) or
  x-vector-only mode.

Use `poetry run test-gen-dataset` to evaluate voices before the full run.

For the preset speaker table, custom voice setup, cloning modes and caching,
see **[Speakers and custom voices](docs/Custom-voices.md)**.
For dataset-scale generation, **[Stable voice for dataset training](docs/Stable-voice.md)**
is the recommended workflow (clone a single reference clip to avoid preset
drift across thousands of clips).

## VRAM / OOM

- **1.7B** bf16 fits on RTX 4070 (12 GB) with `batch_size: 2` (or lower). If
  too tight, try `device_map: "auto"` (CPU offload, slower) or fall back to
  `model_size: "0.6b"`.
- On OOM/Ctrl+C: the pipeline saves both the generate and validate checkpoints
  and prints a clear suggestion. Exit code is **2** (not 1) on GPU OOM, **1**
  on missing CUDA.
- Full run auto-archives the result in `output/gen{NNN}/`. On a full run, if
  an incomplete generation is detected, you're prompted to resume or start
  fresh; use `--no-clean` to force a resume, or delete the checkpoints to
  force a clean. Delete `workspace/.generate_checkpoint.json` (and
  `workspace/raw_wav/`) to restart generation from scratch.

## Project structure

See the **[Project structure](docs/Configuration-reference.md#paths)** section
of the configuration reference for the full directory tree (including the
volatile `workspace/` files and the immutable `output/gen{NNN}/` archives).

## Further reading

- [Speakers and custom voices](docs/Custom-voices.md) — preset speakers, custom voice setup, cloning modes, voice cache
- [Stable voice for dataset training](docs/Stable-voice.md) — recommended workflow for large-scale datasets (clone a reference, avoid preset drift)
- [Pronunciation verification](docs/Pronunciation-verification.md) — phoneme-level PER, espeak-ng install, threshold tuning, per-word report
- [Retry workflow](docs/Retry-workflow.md) — regeneration options + interactive rejected-clip triage
- [Configuration reference](docs/Configuration-reference.md) — full parameter table, paths, sampling parameters, `test-gen-dataset`

---

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

[Qwen3-TTS]: https://github.com/QwenLM/Qwen3-TTS