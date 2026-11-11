# Setup

How the development PC is set up. The code lives in the repository; tools, engines, models and results live outside it, in `LLMLAB_HOME` (default `D:\LLMLab`). Experiment files can use `${LLMLAB_HOME}` in paths.

| Path | Contents |
|---|---|
| `D:\LLMLab\tools\uv\uv.exe` | uv 0.13.7, which manages Python and packages |
| `D:\LLMLab\tools\python\` | CPython 3.13.18, installed by uv |
| `D:\LLMLab\engines\llama-b12110-cuda-13.4\` | llama.cpp v0.9.0 (build b12110) for CUDA 13.4 |
| `D:\LLMLab\models\SmolLM3-3B\` | `SmolLM3-Q4_K_M.gguf` |
| `D:\LLMLab\models\SmolLM3-3B-ladder\` | The [quantization ladder](#quantization-ladder): F16 and six quantizations made from it |
| `D:\LLMLab\cache\uv\` | uv's package cache |
| `D:\LLMLab\results\lab.sqlite` | Every recorded session |
| `D:\LLMLab\results\lab-before-schema-v3.sqlite` | A copy taken before the database was upgraded for the Claude baseline (sessions 1–4) |

## Requirements

- Windows 11 and an NVIDIA GPU, read through NVML; without a readable NVIDIA GPU, the GPU checks are skipped.
- A llama.cpp release build, with `llama-server`.
- uv. Python packages are pinned in `uv.lock`: httpx, nvidia-ml-py, psutil and PyYAML 6.0.3, plus pytest 9.2.0 for tests.
- For the Claude baseline only: Claude Code (2.1.311 on the development PC), signed in to a Claude plan. No API key is needed. The harness finds `claude` on `PATH`; set `engine.cli` in the experiment file to use another one.

```bash
D:\LLMLab\tools\uv\uv.exe sync
```

```bash
D:\LLMLab\tools\uv\uv.exe run pytest -q
```

## Downloads (SHA-256 verified)

| File | Source | SHA-256 |
|---|---|---|
| `uv-x86_64-pc-windows-msvc.zip` (0.13.7) | github.com/astral-sh/uv | `0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021` |
| `llama-b12110-bin-win-cuda-13.4-x64.zip` | github.com/ggml-org/llama.cpp | `dc910a969190b5d42e189afd73117f6bbdd8dd8ed6586342c5e0b82cab62c673` |
| `cudart-llama-bin-win-cuda-13.4-x64.zip` | github.com/ggml-org/llama.cpp | `738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668` |
| `SmolLM3-Q4_K_M.gguf` | huggingface.co/ggml-org/SmolLM3-3B-GGUF | `8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e` |
| `SmolLM3-f16.gguf` (6,158,339,424 bytes) | huggingface.co/ggml-org/SmolLM3-3B-GGUF | `8e6ad54cf7426bd8318df06597357deef1bd758e27fbed2ed87ef39ed96e6f74` |

## Quantization ladder

The quantization experiments compare one model at seven precisions. All of them are made here from the same 16-bit file with the same llama.cpp build. Files from different publishers could differ in how they were converted, which would mix that into the comparison.

```bash
cd D:\LLMLab\models\SmolLM3-3B-ladder
```

```bash
D:\LLMLab\engines\llama-b12110-cuda-13.4\llama-quantize.exe SmolLM3-f16.gguf SmolLM3-Q4_K_M-local.gguf Q4_K_M
```

Repeat with `Q2_K`, `Q3_K_M`, `Q5_K_M`, `Q6_K` and `Q8_0`, naming each file `SmolLM3-<type>.gguf`.

| File | Bytes | Share of F16 |
|---|---|---|
| `SmolLM3-Q2_K.gguf` | 1,253,301,600 | 20% |
| `SmolLM3-Q3_K_M.gguf` | 1,571,069,280 | 26% |
| `SmolLM3-Q4_K_M-local.gguf` | 1,915,305,312 | 31% |
| `SmolLM3-Q5_K_M.gguf` | 2,213,756,256 | 36% |
| `SmolLM3-Q6_K.gguf` | 2,530,860,384 | 41% |
| `SmolLM3-Q8_0.gguf` | 3,275,574,624 | 53% |
| `SmolLM3-f16.gguf` | 6,158,339,424 | 100% |

The Q4_K_M made here (SHA-256 `f09a82d599b6964b5fd4ab47c6bf70131af922c7cabcd4b0034307eeb3504c0e`) has the same size as the published one, and 0.08% of its bytes differ, spread across the weights. The experiments run both as a control: if they disagree beyond noise, the ladder isn't comparable with the published file.

## Phase 0 spike

`spike/` is the measurement check that came before the harness. It runs one model while recording NVML and nvidia-smi side by side, and passes only if they agree on utilization, VRAM and power, and if the server's speed matches what the client receives. All 8 checks passed on 2026-11-07, with 150.3 tok/s and 1.4% run-to-run variation.

```bash
D:\LLMLab\tools\uv\uv.exe run python spike/run_spike.py
```
