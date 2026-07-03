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
poetry run gen-dataset --step normalize               # single step normalize
poetry run gen-dataset --step publish                 # manifest + report + archive
poetry run gen-dataset --from validate                # validate + normalize + publish
poetry run gen-dataset --accept 7,13                  # manually accept rejected clips
poetry run test-gen-dataset          # speaker test (batched by batch_size)
poetry run test-gen-dataset --model-type base  # override model_type for the test
poetry run test-gen-dataset --batch-size 8     # override batch_size for the test
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
- Pipeline steps: `generate` → `validate` → `normalize` → `manifest` → `report`. Each can run standalone via `--step`.

## Model types & voices

- `config.model_type` selects the Qwen3-TTS model variant: `custom_voice` (preset speakers via `generate_custom_voice`) or `base` (voice cloning via `generate_voice_clone`). The `qwen-tts` library validates `model.tts_model_type` at runtime and raises `ValueError` on mismatch; `generate.load_tts_model` also asserts this up front.
- `MODEL_HUB_IDS` in `common.py` is nested `{model_type: {model_size: repo_id}}`. Base repos: `Qwen/Qwen3-TTS-12Hz-{0.6b,1.7b}-Base`.
- Custom voices (base mode) live as `<speaker>.wav` (required) and `<speaker>.txt` (required for ICL mode, optional for x-vector-only) directly under `inputs/voices/`. The `speaker` field selects which one to use. Helpers: `common.resolve_voice_paths`, `common.list_available_voices`, `common.voice_fingerprint`.
- `generate.get_voice_clone_prompt` extracts a `VoiceClonePromptItem` once (cached per-voice per-model-size under `workspace/.voice_cache/<speaker>_<model_size>.pt`, invalidated by fingerprint) and broadcasts it over every batch. The `--only-rejected` regenerate path reuses the same cache.
- `test_speaker.py` sweeps the universe of the configured `model_type` (preset speakers for `custom_voice`, all voices under `inputs/voices/` for `base`); `--speaker NAME` restricts to one. Both worlds cannot be tested in a single run (different model).
- Downstream steps (`validate`, `normalize`, `manifest`, `report`) are model-type-agnostic: the model always returns `(wavs, sr)`. Only `generate` and `report` (model section) branch on `model_type`.

## Resumability

- `workspace/.generate_checkpoint.json` tracks processed sentence indices. Re-running `--step generate` skips already-done clips.
- Full pipeline run (`gen-dataset` without `--no-clean` or `--step`) auto-cleans workspace/ and archives the result in `output/gen{NNN}/`.
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