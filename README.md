# qwen3-tts-dataset-generator

**Pipeline** that transforms a text corpus into a **synthetic TTS dataset**
validated by ASR. It generates audio with [Qwen3-TTS], filters out low-quality clips via WER,
normalizes them (resample, loudness, silence trim), and produces a ready-to-use LJSpeech
manifest with train/val split.

**Resumable** (JSON checkpoint), **validated** (ASR + WER), **normalized**
(resample 22050 Hz, loudness EBU R128, silence trimming) and produces an **LJSpeech manifest**
with deterministic train/val split. Works with any language supported by Qwen3-TTS:
Italian, English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish.

> Developed and tested on **NVIDIA RTX 4070 (12 GB VRAM)**. Compatible with any NVIDIA GPU
> with CUDA 12.4+. The 1.7B model ~16 GB in bf16 might not fit on 12 GB:
> use `model_size: 0.6b` or `device_map: "auto"` (partial CPU offload).
> OOM is handled with a clear message and suggestion.

---

## Features

- **Generate** audio from a text corpus via Qwen3-TTS (batch GPU inference with resumable checkpoint)
- **Validate** each clip with ASR (faster-whisper) + WER, auto-reject low-quality clips
- **Normalize** audio: convert to mono, resample to 22050 Hz, loudness normalize (-23 LUFS), trim silence, save as 16-bit PCM
- **Manifest** in LJSpeech format (filename|text) with deterministic train/val split
- **Multi-language**: works with any language supported by Qwen3-TTS (Italian, English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish)

---

## Local Setup (Poetry)

```bash
cd qwen3-tts-dataset-generator
poetry lock
poetry install
```
Verify GPU: `poetry run python -c "import torch; print(torch.cuda.is_available())"`

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
`inputs/sentences.txt` (configured via `paths.input_sentences` in
`config.yaml`). The sample file included in the repo contains placeholder
sentences for testing. Works with any language supported by Qwen3-TTS.

## Usage

### Pipeline steps

| Step | Description |
|------|-------------|
| `generate` | Create audio from text corpus via Qwen3-TTS (batch GPU inference, resumable) |
| `validate` | Check each clip with ASR (faster-whisper) + WER, accept/reject |
| `normalize` | Resample to 22050 Hz, loudness normalize (-23 LUFS), trim silence, 16-bit PCM |
| `publish` | Build LJSpeech manifest + report + archive to `output/gen{NNN}/` |

### Commands

```bash
# Full pipeline (auto-clean + all steps + archive)
poetry run gen-dataset

# Full pipeline without auto-clean
poetry run gen-dataset --no-clean

# Single step
poetry run gen-dataset --step generate
poetry run gen-dataset --step validate
poetry run gen-dataset --step normalize
poetry run gen-dataset --step publish

# Run from a step onward (no auto-clean)
poetry run gen-dataset --from validate

# Regenerate only rejected clips
poetry run gen-dataset --step generate --only-rejected

# Manually accept rejected clips (override ASR)
poetry run gen-dataset --accept 7,13

# Help
poetry run gen-dataset --help
```

### Retry workflow

When validation rejects clips, inspect and retry:

1. Check `workspace/rejected/*.json` for expected vs transcription vs WER
2. Listen to the rejected wavs in `workspace/rejected/`
3. If the TTS mispronounced: regenerate + re-validate
4. If the ASR hallucinated (audio sounds correct): accept manually
5. Publish the final dataset

```bash
# Option A: TTS was wrong — regenerate the rejected clips
poetry run gen-dataset --step generate --only-rejected
poetry run gen-dataset --step validate
poetry run gen-dataset --from normalize

# Option B: ASR was wrong — accept manually
poetry run gen-dataset --accept 7,13
poetry run gen-dataset --from normalize
```

## Speaker / voice test

```bash
poetry run test-gen-dataset
poetry run test-gen-dataset --model-size 0.6b
poetry run test-gen-dataset --speaker Vivian     # custom_voice: single preset speaker
poetry run test-gen-dataset --speaker my_voice   # base: single custom voice
```

In `custom_voice` mode the sweep covers all built-in speakers; in `base` mode
it covers every custom voice found under `inputs/voices/`. `--speaker` restricts
the test to a single one.

## Config (`config.yaml`)

Main parameters:

| Parameter | Default | Notes |
|---|---|---|
| `model_size` | `0.6b` | `1.7b` or `0.6b` |
| `model_type` | `custom_voice` | `custom_voice` (preset speakers) or `base` (voice clone) |
| `dtype` | `bfloat16` | `bfloat16` or `float16` |
| `attn_implementation` | `sdpa` | `sdpa` (default), `flash_attention_2` (faster, needs `pip install flash-attn`), or `eager` |
| `device_map` | `cuda:0` | `"auto"` for CPU offload with 1.7B on 12 GB |
| `speaker` | `Vivian` | preset speaker (custom_voice mode only) |
| `voice.name` | _(none)_ | file stem under `inputs/voices/` (base mode, e.g. `my_voice` → `my_voice.wav`) |
| `voice.x_vector_only_mode` | `false` | `false`=ICL (best quality, needs `<name>.txt`) \| `true`=x-vector-only |
| `language` | `Auto` | `Auto` for automatic detection, or a language name (`italian`, `english`, etc.) |
| `max_new_tokens` | `2048` | Maximum tokens generated per clip |
| `seed` | `42` | Reproducibility for train/val split and sampling |
| `batch_size` | `4` | 4–8 recommended on 12 GB |
| `asr_model` | `medium` | faster-whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`) |
| `wer_threshold` | `0.20` | WER rejection threshold (20%) |
| `target_sample_rate` | `22050` | Output sample rate in Hz |
| `target_lufs` | `-23.0` | Loudness normalization target (EBU R128) |
| `val_ratio` | `0.1` | Fraction of data held out for validation |

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
voice:
  name: "my_voice"            # -> inputs/voices/my_voice.wav + my_voice.txt
  x_vector_only_mode: false   # false=ICL (best quality, needs ref.txt) | true=x-vector-only
  prompt_cache: "workspace/.voice_cache"
```

Two cloning modes are supported:

- **ICL** (`x_vector_only_mode: false`, default): uses the reference audio **and**
  its transcript. Best quality. `ref.txt` is required.
- **x-vector-only** (`x_vector_only_mode: true`): uses only the speaker embedding,
  no transcript needed. Lower quality.

The extracted `VoiceClonePromptItem` is cached per-voice under `workspace/.voice_cache/` (e.g. `workspace/.voice_cache/my_voice.pt`) and reused across runs; the cache is invalidated automatically when the
reference audio, transcript, cloning mode, model type, or model size change.

Test every custom voice before the full run:

```bash
poetry run test-gen-dataset
poetry run test-gen-dataset --speaker my_voice   # test a single voice
```

## VRAM / OOM

- **1.7B** bf16 ~16 GB → on RTX 4070 (12 GB) use `device_map: "auto"`
  (slower) or `model_size: 0.6b` (recommended for 12 GB GPUs).
- On OOM: the pipeline saves the checkpoint and prints a clear suggestion.
- Full run auto-archives the result in `output/gen{NNN}/`. Use `--no-clean` to skip workspace cleanup.

## Project structure

```
.
├── pyproject.toml
├── Dockerfile
├── config.yaml
├── src/
│   ├── common.py           # shared utilities
│   ├── generate.py         # audio generation
│   ├── validate.py         # ASR validation
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
│   └── rejected/
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