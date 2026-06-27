# Third-party models

Voice2Text downloads and runs the following pre-trained models at runtime. They
are **not** included in this repository and are **not** covered by this project's
MIT license — each is distributed by its authors under its own terms. Review and
comply with these before redistributing any model weights or using the app
commercially.

| Model | Used for | Author / Source | License |
|---|---|---|---|
| **Whisper large-v3** (`ggml-large-v3-q5_0`) | Clean saved transcript (GPU) | OpenAI, via [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) | MIT |
| **sherpa-onnx streaming Zipformer EN** (2023-06-26) | Live streaming captions | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | Apache-2.0 |
| **sherpa-onnx punctuation CT-Transformer** (zh-en) | Punctuation/casing | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models) (FunASR) | Apache-2.0 |
| **3D-Speaker CAM++ sv en voxceleb 16k** | Speaker diarization | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) (Alibaba 3D-Speaker) | Apache-2.0 |
| **Qwen2.5-3B-Instruct** (Q4_K_M GGUF) | Meeting summary (GPU) | [Qwen](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct); GGUF quant by [bartowski](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF) | **Qwen Research License** ⚠️ |

> ⚠️ **Qwen2.5-3B is under the Qwen Research License Agreement, not Apache-2.0.**
> It restricts commercial use. If you need a permissively licensed summarizer,
> swap in an Apache-2.0 model — e.g. Qwen2.5-1.5B/7B-Instruct (Apache-2.0) — by
> changing the filename in `app/llama-sidecar` / the summary command and in
> `scripts/Fetch-Models.ps1`.

The application also depends on the **CUDA runtime** (cuDART/cuBLAS) and
**ONNX Runtime** at runtime; those are obtained from your CUDA Toolkit install
and the `sherpa-rs` crate respectively, under their own (NVIDIA / MIT) licenses.
