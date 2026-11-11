# Phase 0: measurement spike

**Purpose:** before building anything, check that the measurements can be trusted. The spike runs one model while recording the GPU two independent ways, through NVML (the library the harness uses) and through nvidia-smi. It passes only if they agree.

| Check | Purpose |
|---|---|
| Every run generated exactly `n_predict` tokens | Speeds are only comparable for the same work |
| The full prompt was processed every run | A cached prompt would fake prompt speed |
| Client and server counted the same tokens | Inter-token latency really is per token |
| Client and server speed within 5% | The server's timer matches what arrives over the network |
| All layers on the GPU | The configuration is what we think it is |
| NVML and nvidia-smi agree on utilization, VRAM and power | The telemetry reads the GPU correctly |

All 8 checks passed on 2026-11-07 (150.3 tok/s, 1.4% run-to-run variation). Two corrections came out of the spike, and the harness keeps both: NVML's v2 memory call, and llama-server's log level 4 (see [measurement findings](../docs/measurement-findings.md)).

```bash
uv run python spike/run_spike.py
```

Each run writes a folder under `spike/results/`, which is not committed.
