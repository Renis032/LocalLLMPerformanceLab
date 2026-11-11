# Experiments

Each YAML file here is one experiment: a model and its settings, a prompt, and how many runs to make. A `sweep` turns one file into several configurations. All settings are listed in [docs/reference.md](../docs/reference.md#experiment-files).

| File | Purpose |
|---|---|
| [`smollm3-baseline.yaml`](smollm3-baseline.yaml) | The reproducibility baseline. A 3B model fully on the GPU and with half its layers on the CPU, run twice to show results repeat within a few percent |
| [`smollm3-quality.yaml`](smollm3-quality.yaml) | The baseline speed runs for the local model, then answer quality on the 24 graded tasks |
| [`claude-quality.yaml`](claude-quality.yaml) | The same for Claude Haiku, Sonnet and Opus |
| [`claude-baseline.yaml`](claude-baseline.yaml) | The cloud baseline. The same prompt sent to Claude Haiku, Sonnet and Opus through Claude Code on a Claude plan, to set the local model's speed beside Claude's |
| [`gpu-offload.yaml`](gpu-offload.yaml) | Phase 5. How many layers belong on the GPU: 0 to all 37 |
| [`quantization-gpu.yaml`](quantization-gpu.yaml) | Phase 5. Speed, memory and answer quality from Q2_K to F16, all on the GPU. Needs the [quantization ladder](../docs/setup.md#quantization-ladder) |
| [`quantization-cpu.yaml`](quantization-cpu.yaml) | Phase 5. The same precisions entirely on the CPU: does quantization help both equally? |
| [`context-input-length.yaml`](context-input-length.yaml) | Phase 5. Inputs from 521 to about 31,000 tokens (`prompt_repeat`): prompt speed, generation speed and first token |
| [`context-size.yaml`](context-size.yaml) | Phase 5. What reserving a larger context costs when the input stays the same |

Run one with `uv run python -m llmlab run experiments/<file>.yaml`. Add `--dry-run` to check it without running.
