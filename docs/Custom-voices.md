# Speakers and custom voices

`config.model_type` selects the Qwen3-TTS model variant:

- **`custom_voice`** — preset speakers via `generate_custom_voice`.
- **`base`** — voice cloning via `generate_voice_clone`.

The `qwen-tts` library validates `model.tts_model_type` at runtime and raises
`ValueError` on mismatch; `generate.load_tts_model` also asserts this up front.

`MODEL_HUB_IDS` in `common.py` is nested `{model_type: {model_size: repo_id}}`.
Base repos: `Qwen/Qwen3-TTS-12Hz-{0.6b,1.7b}-Base`.

Downstream steps (`validate`, `pronunciation`, `normalize`, `manifest`,
`report`) are model-type-agnostic: the model always returns `(wavs, sr)`.
Only `generate` and `report` (model section) branch on `model_type`.

For dataset-scale generation (more than ~100 clips) prefer `base` mode with a
single fixed reference clip over `custom_voice` presets — see
[Stable voice for dataset training](Stable-voice.md).

## Preset speakers (custom_voice mode)

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

## Custom voices (base / voice clone)

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

The `speaker` field on `Config` is the voice identity for both modes: preset
name for `custom_voice`, custom voice name under `inputs/voices/` for `base`;
`x_vector_only_mode` is a flattened top-level field used only in `base` mode.

### Voice prompt cache

`generate.get_voice_clone_prompt` extracts a `VoiceClonePromptItem` once
(cached per-voice per-model-size under
`workspace/.voice_cache/<speaker>_<model_size>.pt`, invalidated by a
fingerprint of the reference audio, transcript, cloning mode, model type and
model size) and broadcasts it over every batch. The `--only-rejected`
regeneration path reuses the same cache. The cache directory is fixed
(`workspace/.voice_cache/`) and not user-configurable. To force a clean
extraction, delete
`workspace/.voice_cache/<name>_<model_size>.pt`.

### Helpers

- `common.resolve_voice_paths`
- `common.list_available_voices`
- `common.voice_fingerprint`

## Testing voices

See the [Speaker / voice test](Configuration-reference.md#speaker--voice-test)
section of the configuration reference for the full `test-gen-dataset` CLI.

Test every custom voice before the full run:

```bash
poetry run test-gen-dataset
poetry run test-gen-dataset --speaker my_voice   # test a single voice
```