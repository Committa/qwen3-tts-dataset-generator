# AGENTS.md

## Commands

```bash
poetry lock                          # generate poetry.lock (required before first install)
poetry install                       # install all deps (torch from cu124 source)
poetry run gen-dataset               # full pipeline (auto-clean + archive)
poetry run gen-dataset --no-clean    # full pipeline without auto-clean
poetry run gen-dataset --step generate  # single step (no clean, no archive)
poetry run test-gen-dataset          # speaker test
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
- `src/common.py` defines `Config`/`Paths` dataclasses, `PROJECT_ROOT`, and all shared helpers (logging, checkpoint, OOM detection). All other modules import from here.
- Config paths in `config.yaml` are resolved relative to `PROJECT_ROOT` (= repo root, computed as `Path(__file__).resolve().parent.parent` in `common.py`).
- Pipeline steps: `generate` → `validate` → `normalize` → `manifest` → `report`. Each can run standalone via `--step`.

## Resumability

- `workspace/.generate_checkpoint.json` tracks processed sentence indices. Re-running `--step generate` skips already-done clips.
- Full pipeline run (`gen-dataset` without `--no-clean` or `--step`) auto-cleans workspace/ and archives the result in `output/gen{NNN}/`.
- Delete `workspace/.generate_checkpoint.json` (and `workspace/raw_wav/`) to restart generation from scratch.

## OOM / error behavior

- On GPU OOM, pipeline saves checkpoint, prints `OOM_HINT` with fix suggestions, and exits with **code 2** (not 1).
- Missing CUDA exits with code 1. Requires NVIDIA GPU with CUDA 12.4+.

## What NOT to do

- Do not translate the Italian sentences inside `inputs/italian_sentences.txt` — that is the TTS corpus data and must remain in Italian.
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