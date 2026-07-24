# Stable voice for dataset training (recommended)

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

## Step-by-step

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

## Notes

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

## Sampling parameters and consistency

Qwen3-TTS ships a `generation_config.json` with author-recommended defaults:
`do_sample=true, temperature=0.9, top_p=1.0, top_k=50, repetition_penalty=1.05,
max_new_tokens=8192`. These are the values the authors used in evaluation.

- `temperature` is the main lever for cross-clip consistency. Lower => less
  variance in tone/pauses/intonation across clips. The model is tuned at 0.9;
  going well below ~0.6 takes it out of distribution and can trigger
  **EOS-collapse** (the model emits the end-of-speech token prematurely =>
  clip truncated to a few words + silence).
- `top_p` must stay at `1.0` when `temperature` is low: at low temperature the
  distribution is already peaked, so `top_p < 1.0` cuts the tail of
  continuation tokens and can make EOS-collapse *more* likely (the nucleus
  collapses on the EOS token alone), with no consistency gain.
- Truncated clips have high WER and are rejected by `validate`. Recover them
  with `poetry run gen-dataset --step generate --only-rejected` (re-runs only
  the rejected subset with a fresh RNG draw, same temperature). Iterate
  `validate` → `--only-rejected` until clean.
- `min_new_tokens` is NOT configurable: hardcoded to `2` inside
  `Qwen3TTSForConditionalGeneration.generate` (`modeling_qwen3_tts.py`,
  `talker_kwargs`), so passing it via kwargs is silently ignored.

See the [Configuration reference](Configuration-reference.md) for the
sampling-related config fields.