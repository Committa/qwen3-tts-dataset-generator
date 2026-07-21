# AGENTS.md

## Commands

```bash
poetry lock                          # generate poetry.lock (required before first install)
poetry install                       # install all deps (torch from cu124 source)
poetry run gen-dataset               # full pipeline (auto-clean + archive)
poetry run gen-dataset --no-clean    # full pipeline without auto-clean
poetry run gen-dataset --step generate                # single step generate
poetry run gen-dataset --step generate --only-rejected  # regenerate rejected clips
poetry run gen-dataset --step validate                # single step validate
poetry run gen-dataset --step pronunciation           # phoneme-level check (PER)
poetry run gen-dataset --step pronunciation --calibrate  # measure PER distribution, no rejects
poetry run gen-dataset --step normalize               # single step normalize
poetry run gen-dataset --step publish                 # manifest + report + archive
poetry run gen-dataset --from validate                # validate + pronunciation + normalize + publish
poetry run gen-dataset --accept 7,13                  # manually accept rejected clips
poetry run test-gen-dataset          # speaker test (batched by batch_size)
poetry run test-gen-dataset --model-type base  # override model_type for the test
poetry run test-gen-dataset --batch-size 8     # override batch_size for the test
poetry run review-rejected          # interactive triage of rejected clips (a/r/p/b/q)
poetry run review-rejected --restart  # ignore review checkpoint, start from the first clip
poetry run review-rejected --no-clear  # keep scrollback instead of clearing before each clip
poetry run gen-dataset --help        # CLI help
poetry run black src                 # format
poetry run isort src                 # sort imports (black profile)
poetry run ruff check src            # lint
```

## Critical dependency constraints

- `qwen-tts==0.1.1` **pins exactly** `transformers==4.57.3` and `accelerate==1.12.0`. Do not upgrade any of these independently — the qwen-tts package will break.
- `torch`/`torchaudio` must come from the `pytorch-cu124` source declared in `pyproject.toml`. Installing from default PyPI yields CPU-only wheels.
- Python: `>=3.11.9,<3.12` — do not use 3.10 or 3.12+.

## Architecture

- Entry point: `gen-dataset` (from `src/pipeline.py` via Poetry script alias `[tool.poetry.scripts]`). Do NOT run `python src/pipeline.py` directly — relative imports will fail.
- `src/common.py` defines `Config`/`Paths` dataclasses, `PROJECT_ROOT`, and all shared helpers (logging, checkpoint, OOM detection). All other modules import from here. The `speaker` field on `Config` is the voice identity for both modes (preset name for `custom_voice`, custom voice name under `inputs/voices/` for `base`); `x_vector_only_mode` is a flattened top-level field used only in `base` mode. The `Paths` dataclass holds all pipeline paths: only `input_sentences` and `test_sentences` are configurable in `config.yaml` (top-level keys); all other paths (`raw_wav`, `accepted_wav`, `rejected`, manifests, `report`, `checkpoint`, `log_file`, `prompt_cache`) are fixed defaults defined in `_RUNTIME_PATH_DEFAULTS` and resolved relative to `PROJECT_ROOT`.
- Config paths in `config.yaml` are resolved relative to `PROJECT_ROOT` (= repo root, computed as `Path(__file__).resolve().parent.parent` in `common.py`).
- Pipeline steps: `generate` → `validate` → `pronunciation` → `normalize` → `publish` (manifest + report + archive). Each can run standalone via `--step`.

## Model types & voices

- `config.model_type` selects the Qwen3-TTS model variant: `custom_voice` (preset speakers via `generate_custom_voice`) or `base` (voice cloning via `generate_voice_clone`). The `qwen-tts` library validates `model.tts_model_type` at runtime and raises `ValueError` on mismatch; `generate.load_tts_model` also asserts this up front.
- `MODEL_HUB_IDS` in `common.py` is nested `{model_type: {model_size: repo_id}}`. Base repos: `Qwen/Qwen3-TTS-12Hz-{0.6b,1.7b}-Base`.
- Custom voices (base mode) live as `<speaker>.wav` (required) and `<speaker>.txt` (required for ICL mode, optional for x-vector-only) directly under `inputs/voices/`. The `speaker` field selects which one to use. Helpers: `common.resolve_voice_paths`, `common.list_available_voices`, `common.voice_fingerprint`.
- `generate.get_voice_clone_prompt` extracts a `VoiceClonePromptItem` once (cached per-voice per-model-size under `workspace/.voice_cache/<speaker>_<model_size>.pt`, invalidated by fingerprint) and broadcasts it over every batch. The `--only-rejected` regenerate path reuses the same cache.
- `test_speaker.py` sweeps the universe of the configured `model_type` (preset speakers for `custom_voice`, all voices under `inputs/voices/` for `base`); `--speaker NAME` restricts to one. Both worlds cannot be tested in a single run (different model). Each clip is written next to its exact transcript (`<speaker>_<i>.wav` + `<speaker>_<i>.txt`) in `output/test_speaker/`, ready to be copied into `inputs/voices/` as a voice-cloning reference.
- **Drift recommendation:** for dataset-scale generation (more than ~100 clips) prefer `base` mode with a single fixed reference clip over `custom_voice` presets. `custom_voice` resamples the speaker stochastically on every call, so variance in F0/formants/pacing accumulates across thousands of clips and degrades downstream training (e.g. Piper). Recommended workflow: generate a 10-15 s reference once with the preset speaker via `test-gen-dataset --model-type custom_voice --speaker <name>`, copy the resulting `.wav`+`.txt` into `inputs/voices/<name>/`, switch to `model_type: "base"` + ICL (`x_vector_only_mode: false`), then run `gen-dataset`. `inputs/test_sentences.txt` is calibrated to produce 10-15 s clips for this purpose. See README "Stable voice for dataset training".
- Downstream steps (`validate`, `pronunciation`, `normalize`, `manifest`, `report`) are model-type-agnostic: the model always returns `(wavs, sr)`. Only `generate` and `report` (model section) branch on `model_type`.

## Pronunciation verification

- `src/pronunciation.py` is a phoneme-level check that complements the word-level WER validation. faster-whisper (grapheme ASR) is forgiving of pronunciation drift: a clip can match the reference transcript (low WER) while the actual pronunciation is wrong, producing artifacts in the downstream training set.
- The audio is recognised to espeak IPA phonemes by `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (a multilingual wav2vec2 CTC model fine-tuned on Common Voice to output espeak phoneme labels) and compared (Phoneme Error Rate, PER, via `jiwer.wer` on tokenized phoneme sequences) against the `phonemizer`+espeak-ng text→phoneme rendering of the reference sentence. Both sides use the same espeak phoneme inventory, so the comparison is direct.
- Runs after `validate` on the WER survivors in `accepted_wav/` and before `normalize` (so normalize only processes pronunciation survivors). Clips whose PER exceeds `cfg.phoneme_threshold` are **moved** from `accepted_wav/` to `rejected/` with a `per=... > ...` reason and a `pronunciation.log` JSONL aggregate. The rejected sidecar JSON carries the `index` field, so `--only-rejected` regeneration and `--accept` pick them up unchanged.
- Reference text source: `common.load_sentences(cfg)` + `int(wav_path.stem)` — the same single source of truth used by validate/manifest/report.
- Config fields (in `common.py` `Config`, gated by `phoneme_check`): `phoneme_model`, `phoneme_device`, `phoneme_batch_size` (wav2vec2 CTC is not thread-safe; batching is used instead of workers), `phoneme_threshold` (default 0.30 — phoneme recognition is noisier than word ASR), `phoneme_word_report` (default true) and `phoneme_word_top_n` (default 20) for the per-word diagnostic. In a full run the step is skipped when `phoneme_check: false`; an explicit `--step pronunciation` always runs it.
- **Per-word report** (`phoneme_word_report`): the clip-level PER is a single number; to help decide which corpus words to remove or reword, the step also ranks reference words by mean PER across all checked clips. espeak-ng phonemizes per-word (sentinel word separator preserves boundaries), the recognized phoneme stream is aligned to the per-word reference via a Levenshtein DP with backtrace (best-effort attribution of edits to words), and per-word PER is aggregated. Output: `workspace/.pronunciation_words.csv` (`word|occurrences|mean_per|min_per|max_per|median_per`, worst first), a `worst_words` block (top N) in the report JSON, and a CLI log. Runs in both normal and `--calibrate` mode (diagnostic, no extra inference — pure post-processing of already-decoded phonemes).
- `--calibrate` (only with `--step pronunciation`): measure-only mode — computes PER for every clip without rejecting anything, prints min/p25/median/p75/p90/max/mean, and returns a `calibration` block. Use it to tune `phoneme_threshold` on a known-good set before committing.
- System dependency: the `phonemizer` Python lib wraps the **espeak-ng** binary, which must be on PATH. Windows: install the MSI from the espeak-ng GitHub releases (set `PHONEMIZER_ESPEAK_LIBRARY` to `espeak-ng.dll` if needed). Linux: `sudo apt-get install espeak-ng`. On failure the step prints `ESPEAK_HINT` and exits 1.
- wav2vec2 requires 16 kHz mono float input; `librosa.load(..., sr=16000, mono=True)` resamples from the generator's native rate. `transformers==4.57.3` (pinned by qwen-tts) is compatible with `Wav2Vec2ForCTC`/`Wav2Vec2Processor` — no dependency conflict.

## Sampling parameters

- Configurable in `config.yaml` under the Generation section: `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty` (plus `max_new_tokens`). Loaded into `Config` by `common.load_config` and forwarded to the model by `generate._sampling_kwargs` (used by both `run_generate` and `test_speaker` via `generate_phrases`).
- The Qwen3-TTS checkpoint ships a `generation_config.json` with the author-recommended defaults: `do_sample=true, temperature=0.9, top_p=1.0, top_k=50, repetition_penalty=1.05, max_new_tokens=8192`. These are the values the authors used in evaluation.
- `temperature` is the main lever for cross-clip consistency. Lower => less variance in tone/pauses/intonation across clips. The model is tuned at 0.9; going well below ~0.6 takes it out of distribution and can trigger **EOS-collapse** (the model emits the end-of-speech token prematurely => clip truncated to a few words + silence).
- `top_p` must stay at `1.0` when `temperature` is low: at low temperature the distribution is already peaked, so `top_p < 1.0` cuts the tail of continuation tokens and can make EOS-collapse *more* likely (the nucleus collapses on the EOS token alone), with no consistency gain.
- Truncated clips have high WER and are rejected by the `validate` step. Recover them with `poetry run gen-dataset --step generate --only-rejected` (re-runs only the rejected subset with a fresh RNG draw, same temperature). Iterate `validate` -> `--only-rejected` until clean.
- `min_new_tokens` is NOT configurable: it is hardcoded to `2` inside `Qwen3TTSForConditionalGeneration.generate` (`modeling_qwen3_tts.py`, `talker_kwargs`), so passing it via kwargs is silently ignored.

## Resumability

- `workspace/.generate_checkpoint.json` tracks processed sentence indices. Re-running `--step generate` skips already-done clips.
- Full pipeline run (`gen-dataset` without `--no-clean` or `--step`) auto-cleans workspace/ and archives the result in `output/gen{NNN}/`. **Resume detection:** if a checkpoint with an incomplete generation is found, `pipeline._maybe_clean_workspace` prompts the user (interactive) to choose resume vs. fresh clean; in non-interactive runs it resumes by default to avoid losing progress. `--no-clean` forces resume; deleting the checkpoint forces a fresh clean.
- Delete `workspace/.generate_checkpoint.json` (and `workspace/raw_wav/`) to restart generation from scratch.

## OOM / error behavior

- On GPU OOM, pipeline saves checkpoint, prints `OOM_HINT` with fix suggestions, and exits with **code 2** (not 1).
- Missing CUDA exits with code 1. Requires NVIDIA GPU with CUDA 12.4+.

## What NOT to do

- Do not translate the Italian sentences inside `inputs/sentences.txt` — that is the TTS corpus data and must remain in Italian.
- Do not add a `[tool.ruff]` config section without checking — ruff uses defaults currently.
- No test suite exists. `src/test_speaker.py` is a utility script, not a unit test. Do not assume pytest is configured.

## Coding conventions

- All comments, docstrings, log messages, and user-facing text must be in **English**.
- Every public function, method, and class must have a **docstring** describing its purpose, parameters, and return value.

## Docker

- Dockerfile uses `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` base. Poetry is installed inside the container (pip only installs Poetry itself). Run with `--gpus all`.

## Optional: flash-attn

`flash-attn` speeds up inference and reduces VRAM usage. Not required — everything works without it (you'll see a benign warning on startup).

**Do NOT add flash-attn to pyproject.toml.** No official wheels for Windows; requires CUDA toolkit + build tools at install time and `--no-build-isolation`. Instead install manually when needed:

- **Docker / Linux:** `pip install flash-attn --no-build-isolation`
- **Windows:** not supported via pip; use WSL2 or Docker

## Git / commits

- **Commit messages**: Use Conventional Commits format without scope. Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`.
- Commit after every logical change with a description of what was done.