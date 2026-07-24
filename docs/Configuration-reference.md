# Configuration reference

All parameters live in `config.yaml` at the repo root. Paths in `config.yaml`
are resolved relative to `PROJECT_ROOT` (= repo root, computed as
`Path(__file__).resolve().parent.parent` in `common.py`).

## Full parameter table

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

## Paths

`Paths` dataclass holds all pipeline paths. Only `input_sentences` and
`test_sentences` are configurable in `config.yaml` (top-level keys); all
other paths are fixed defaults defined in `_RUNTIME_PATH_DEFAULTS` and
resolved relative to `PROJECT_ROOT` (= repo root).

### Project structure

```
.
├── pyproject.toml
├── Dockerfile
├── config.yaml
├── docs/                  # deeper guides (linked from README)
├── src/
│   ├── common.py           # shared utilities
│   ├── generate.py         # audio generation
│   ├── validate.py         # ASR validation (WER)
│   ├── pronunciation.py    # phoneme-level verification (PER)
│   ├── normalize_audio.py  # audio normalization
│   ├── build_manifest.py   # LJSpeech manifest
│   ├── report.py           # final report
│   ├── review_rejected.py  # review-rejected interactive triage CLI
│   ├── pipeline.py         # CLI orchestrator
│   └── test_speaker.py     # speaker test utility
├── inputs/                 # user-provided text corpora and voice samples
│   ├── sentences.txt
│   ├── test_sentences.txt
│   └── voices/             # custom voices for base (voice clone) mode
│       └── <name>.wav        # + <name>.txt for ICL transcript
├── workspace/              # volatile (auto-cleaned on full run, gitignored)
│   ├── .voice_cache/                  # per-voice per-model-size VoiceClonePromptItem cache
│   ├── raw_wav/                       # generated audio
│   ├── accepted_wav/                  # validate + pronunciation survivors
│   ├── rejected/                      # rejected clips + sidecar JSONs
│   ├── .generate_checkpoint.json      # generate step resumability (done indices)
│   ├── .validate_checkpoint.json      # validate step resumability
│   ├── .pronunciation_checkpoint.json # pronunciation step resumability
│   ├── .review_checkpoint.json        # review-rejected interactive triage state
│   ├── .manifest_train.csv            # train manifest (before publish)
│   ├── .manifest_val.csv              # validation manifest (before publish)
│   ├── .pronunciation_words.csv       # per-word PER ranking (when phoneme_word_report)
│   ├── .pronunciation_report.txt      # pronunciation step CLI/log dump
│   └── .report.json                   # final report (before publish)
├── output/                 # immutable dataset archives
│   ├── gen001/
│   │   ├── wavs/
│   │   ├── metadata_train.csv
│   │   ├── metadata_val.csv
│   │   └── report.json
│   └── ...
└── logs/
```

`workspace/` is gitignored (`workspace/*` with `!workspace/.gitkeep`) and
auto-cleaned on a fresh full run (`clean_on_full_run: true`), so you will
not see all of these files until you run the pipeline. `output/gen{NNN}/`
archives are immutable — each full run produces a new numbered directory.

## Sampling parameters

Configurable under the Generation section of `config.yaml`: `do_sample`,
`temperature`, `top_k`, `top_p`, `repetition_penalty` (plus `max_new_tokens`).
Loaded into `Config` by `common.load_config` and forwarded to the model by
`generate._sampling_kwargs` (used by both `run_generate` and `test_speaker`
via `generate_phrases`). See [Stable voice](Stable-voice.md#sampling-parameters-and-consistency)
for the consistency rationale.

## Speaker / voice test

```bash
poetry run test-gen-dataset
poetry run test-gen-dataset --model-size 0.6b
poetry run test-gen-dataset --model-type base         # override config.yaml model_type
poetry run test-gen-dataset --speaker Vivian     # custom_voice: single preset speaker
poetry run test-gen-dataset --speaker my_voice   # base: single custom voice
poetry run test-gen-dataset --batch-size 8       # override config.yaml batch_size
```

`test_speaker.py` sweeps the universe of the configured `model_type` (preset
speakers for `custom_voice`, all voices under `inputs/voices/` for `base`);
`--speaker` restricts to one. `--model-type` overrides `model_type` from
`config.yaml` so both worlds can be tested without editing the config file
(note: the two model types load different HuggingFace repos, so the model is
reloaded when switching). The two worlds cannot be tested in a single run
(different model).

Generation is **batched** by `batch_size` from `config.yaml` (override with
`--batch-size`): each speaker/voice produces `ceil(N / batch_size)` model
calls instead of N, dramatically reducing the time needed to evaluate the
whole universe of voices. Each clip is written next to its exact transcript
(`output/test_speaker/<speaker>_<i>.wav` + `<speaker>_<i>.txt`), so a good
candidate can be copied straight into `inputs/voices/` as a voice-cloning
reference — see [Stable voice](Stable-voice.md).