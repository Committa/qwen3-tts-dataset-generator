# qwen3-tts-dataset-generator

**Production-ready** pipeline that transforms a text corpus into a **synthetic TTS dataset**
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

Before running the pipeline, prepare `inputs/italian_sentences.txt` with your own text corpus, one sentence per line. Lines starting with `#` and blank lines are ignored.

The quality of the generated dataset depends on the input text: use clean, natural sentences covering your target domain. The sample file included in the repo contains 20 placeholder Italian sentences for testing. You can use any language supported by Qwen3-TTS.

## Usage

```bash
# Full pipeline (auto-clean + archive)
poetry run gen-dataset

# Full pipeline without auto-clean
poetry run gen-dataset --no-clean

# Single step
poetry run gen-dataset --step validate

# Help
poetry run gen-dataset --help
```

## Speaker test

```bash
poetry run test-gen-dataset
poetry run test-gen-dataset --model-size 0.6b
```

## Config (`config.yaml`)

Main parameters:

| Parameter | Default | Notes |
|---|---|---|
| `model_size` | `0.6b` | `1.7b` or `0.6b` |
| `device_map` | `cuda:0` | `"auto"` for CPU offload with 1.7B on 12 GB |
| `speaker` | `Vivian` | 9 available speakers (see below) |
| `language` | `Italian` | `Auto` for automatic detection |
| `wer_threshold` | `0.20` | WER rejection threshold (20%) |
| `batch_size` | `4` | 4-8 recommended on 12 GB |
| `target_sample_rate` | `22050` | 22050 Hz standard |

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

## VRAM / OOM

- **1.7B** bf16 ~16 GB → on RTX 4070 (12 GB) use `device_map: "auto"`
  (slower) or `model_size: 0.6b` (recommended, Italian WER ~1.36%).
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
├── inputs/                 # user-provided text corpora
│   ├── italian_sentences.txt
│   └── test_sentences.txt
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