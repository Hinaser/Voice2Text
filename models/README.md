# Models

The model weights are **not** stored in this repository — they are large
(several GB) and carry their own third-party licenses (see
[`../THIRD-PARTY-MODELS.md`](../THIRD-PARTY-MODELS.md)).

Download them into this folder with:

```powershell
# everything (CPU models + Whisper + the LLM, ~3.5 GB)
.\scripts\Fetch-Models.ps1

# or just the CPU models needed for live captions (~0.5 GB)
.\scripts\Fetch-Models.ps1 -CpuOnly
```

After fetching you should have:

```
models/
  sherpa-onnx-streaming-zipformer-en-2023-06-26/   # live streaming ASR
  sherpa-onnx-punct-ct-transformer-zh-en-...-2024-04-12/  # punctuation
  3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx       # speaker diarization
  ggml-large-v3-q5_0.bin                            # Whisper (clean transcript, GPU)
  Qwen2.5-3B-Instruct-Q4_K_M.gguf                   # local LLM (summary, GPU)
```

The app resolves this folder via `$VOICE2TEXT_MODELS`, then the repo `models/`,
then a `models/` next to the executable.
