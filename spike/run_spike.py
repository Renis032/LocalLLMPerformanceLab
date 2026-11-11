"""Phase 0 spike: can we trust the numbers?

Runs one GGUF model on llama-server and measures it independently:
  1. llama-server's own timings
  2. what the client sees arriving over HTTP
  3. GPU telemetry from NVML (our sampler) and from nvidia-smi (run separately)
Then checks that they agree. Nothing else in the lab is worth building if they don't.

Usage:
  uv run spike/run_spike.py           (defaults point at D:\\LLMLab)
  uv run spike/run_spike.py --help
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import pynvml

LAB_HOME = Path(os.environ.get("LLMLAB_HOME", r"D:\LLMLab"))
SPIKE_DIR = Path(__file__).resolve().parent

SMI_QUERY = [
    "timestamp",
    "utilization.gpu",
    "memory.used",
    "power.draw.average",
    "power.draw.instant",
    "temperature.gpu",
    "clocks.gr",
]

# Pass/fail tolerances for the Phase 0 exit criteria.
TOL_RATE_PCT = 5.0  # server vs client generation speed
TOL_UTIL_POINTS = 10.0  # NVML vs nvidia-smi GPU utilization
TOL_VRAM_MIB = 64.0  # NVML vs nvidia-smi VRAM (or 2%, whichever is larger)
TOL_POWER_PCT = 10.0  # NVML vs nvidia-smi power
WARN_CV_PCT = 5.0  # run-to-run variation worth flagging

# An original, fixed passage so every run sees identical input.
_PASSAGE = (
    "Field notes on running language models on a desktop GPU. A model's weights are "
    "loaded into video memory once, and every generated token then requires reading "
    "most of those weights again. That is why generation speed on a single user's "
    "machine usually depends more on memory bandwidth than on raw compute. Prompt "
    "processing behaves differently: the whole prompt can be pushed through the network "
    "in large batches, so it keeps the arithmetic units busy and runs many times faster "
    "per token. Quantization shrinks each weight from sixteen bits to eight, six, five "
    "or four, which reduces the bytes read per token and often raises generation speed, "
    "at some cost in accuracy. The key-value cache grows with context length and "
    "competes with the weights for the same memory. When the model no longer fits, some "
    "layers stay on the CPU, and the slower side sets the pace for every token.\n\n"
)
PROMPT = _PASSAGE * 3 + "Continue these notes with a practical checklist:\n"

CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "app_clock_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clock_setting",
}


# ---------------------------------------------------------------- helpers


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--server",
        type=Path,
        default=LAB_HOME / "engines" / "llama-b12110-cuda-13.4" / "llama-server.exe",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=LAB_HOME / "models" / "SmolLM3-3B" / "SmolLM3-Q4_K_M.gguf",
    )
    ap.add_argument("--out", type=Path, help="results folder (default: spike/results/<time>)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--gpu-layers", default="all")
    ap.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interval", type=float, default=0.2, help="sampling interval, seconds")
    ap.add_argument("--baseline-seconds", type=float, default=3.0)
    ap.add_argument(
        "--busy-threshold",
        type=float,
        default=15.0,
        help="stop if mean GPU utilization before the run is above this (%%)",
    )
    ap.add_argument("--allow-busy", action="store_true", help="run even on a busy GPU")
    ap.add_argument("--skip-hash", action="store_true", help="don't hash the model file")
    return ap.parse_args()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def as_text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def pct_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100


def rnd(v, digits=1):
    if v is None:
        return None
    return int(round(v)) if digits == 0 else round(v, digits)


# ---------------------------------------------------------------- environment


def vram(handle):
    """v2 memory info: 'used' excludes driver-reserved memory, matching nvidia-smi."""
    return pynvml.nvmlDeviceGetMemoryInfo(handle, pynvml.nvmlMemory_v2)


def gpu_preflight(handle, seconds: float, interval: float) -> dict:
    utils, used = [], []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        utils.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        used.append(vram(handle).used / 2**20)
        time.sleep(interval)
    mem = vram(handle)
    return {
        "gpu_util_mean_pct": fmean(utils),
        "vram_used_mib": fmean(used),
        "vram_free_mib": mem.free / 2**20,
        "vram_total_mib": mem.total / 2**20,
    }


def env_snapshot(args, handle) -> dict:
    version = subprocess.run(
        [str(args.server), "--version"], capture_output=True, text=True, timeout=30
    )
    return {
        "captured_at": dt.datetime.now().astimezone().isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_mib": round(psutil.virtual_memory().total / 2**20),
        "gpu": as_text(pynvml.nvmlDeviceGetName(handle)),
        "gpu_driver": as_text(pynvml.nvmlSystemGetDriverVersion()),
        "cuda_driver_version": pynvml.nvmlSystemGetCudaDriverVersion(),
        "vram_total_mib": round(vram(handle).total / 2**20),
        "llama_server_version": (version.stdout + version.stderr).strip(),
        "model_path": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "model_sha256": None if args.skip_hash else sha256_file(args.model),
    }


# ---------------------------------------------------------------- processes


def start_smi(path: Path, interval: float) -> subprocess.Popen:
    cmd = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(SMI_QUERY),
        "--format=csv,noheader,nounits",
        "-lms",
        str(int(interval * 1000)),
    ]
    return subprocess.Popen(cmd, stdout=open(path, "w", encoding="utf-8"), stderr=subprocess.STDOUT)


def start_sampler(path: Path, interval: float) -> subprocess.Popen:
    cmd = [sys.executable, str(SPIKE_DIR / "sampler.py"), "--out", str(path), "--interval", str(interval)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)


def sampler_send(proc: subprocess.Popen, line: str) -> None:
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def server_command(args, port: int) -> list[str]:
    return [
        str(args.server),
        "-m", str(args.model),
        "-ngl", str(args.gpu_layers),
        "-c", str(args.ctx),
        "-np", "1",  # one slot: the whole context belongs to the benchmark request
        "-fa", args.flash_attn,
        "--fit", "off",  # never let the server change settings to fit memory
        "-lm", "none",  # no memory-mapping, so RAM figures are real
        "--no-cache-prompt",  # every run must process the full prompt
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
        "-lv", "4",  # needed for the layer-offload and buffer-size lines in the log
    ]


def parse_server_log(log: str) -> dict:
    off = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
    buffers = {}
    for kind in ("model", "KV", "compute"):
        m = re.search(rf"CUDA0 {kind} buffer size =\s*([\d.]+) MiB", log)
        buffers[kind.lower()] = float(m.group(1)) if m else None
    return {
        "offloaded": [int(off.group(1)), int(off.group(2))] if off else None,
        "gpu_buffers_mib": buffers,
        "memory_lines": [
            ln.strip() for ln in log.splitlines()
            if re.search(r"buffer size|llama_kv_cache: size", ln)
        ],
    }


def wait_healthy(base: str, proc: subprocess.Popen, log_path: Path, timeout: float = 180) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"llama-server exited with code {proc.returncode}:\n{tail}")
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError("llama-server did not become healthy in time")


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------- measuring


def stream_completion(client: httpx.Client, base: str, args) -> dict:
    body = {
        "prompt": PROMPT,
        "n_predict": args.n_predict,
        "ignore_eos": True,  # always generate exactly n_predict tokens
        "cache_prompt": False,
        "seed": args.seed,
        "temperature": 0.0,
        "stream": True,
        "return_tokens": True,  # token ids per event, so counts don't rely on text
    }
    events: list[tuple[float, float, int]] = []  # (wall, perf, tokens in event)
    final: dict = {}
    t_send_wall, t_send = time.time(), time.perf_counter()
    with client.stream("POST", f"{base}/completion", json=body, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            now_perf, now_wall = time.perf_counter(), time.time()
            msg = json.loads(payload)
            n = len(msg.get("tokens") or []) or (1 if msg.get("content") else 0)
            if n:
                events.append((now_wall, now_perf, n))
            if msg.get("stop"):
                final = msg

    if not events:
        raise RuntimeError("stream returned no tokens")
    first_wall, first_perf, first_n = events[0]
    last_wall, last_perf, _ = events[-1]
    total = sum(e[2] for e in events)
    gaps_ms = [(b[1] - a[1]) * 1000 for a, b in zip(events, events[1:])]
    timings = final.get("timings") or {}
    return {
        "t_send": t_send_wall,
        "t_first_token": first_wall,
        "t_last_token": last_wall,
        "client_ttft_ms": (first_perf - t_send) * 1000,
        "client_total_ms": (last_perf - t_send) * 1000,
        "client_tokens": total,
        "client_events": len(events),
        "client_gen_tps": (total - first_n) / (last_perf - first_perf) if last_perf > first_perf else None,
        "itl_p50_ms": percentile(gaps_ms, 50),
        "itl_p95_ms": percentile(gaps_ms, 95),
        "itl_max_ms": max(gaps_ms) if gaps_ms else None,
        "server_timings": timings,
        "stop_type": final.get("stop_type"),
    }


# ---------------------------------------------------------------- analysis


def load_samples(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_smi(path: Path) -> list[dict]:
    def num(s: str):
        s = s.strip()
        try:
            return float(s)
        except ValueError:
            return None  # "[N/A]" and similar

    rows = []
    with open(path, encoding="utf-8") as f:
        for rec in csv.reader(f):
            if len(rec) != len(SMI_QUERY):
                continue
            try:
                t = dt.datetime.strptime(rec[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            values = dict(zip(SMI_QUERY[1:], (num(x) for x in rec[1:])))
            rows.append({"t": t, **values})
    return rows


def in_windows(rows: list[dict], windows: list[tuple[float, float]]) -> list[dict]:
    return [r for r in rows if any(a <= r["t"] <= b for a, b in windows)]


def decode_clock_events(masks) -> list[str]:
    seen = 0
    for m in masks:
        if m:
            seen |= int(m)
    return [name for bit, name in CLOCK_EVENT_REASONS.items() if seen & bit]


def check(name: str, value, limit: str, passed: bool | None, note: str = "") -> dict:
    return {"check": name, "value": value, "limit": limit, "passed": passed, "note": note}


def analyze(env, preflight, server_info, runs, phases, nvml, smi, prompt_tokens, args) -> dict:
    measured = [r for r in runs if r["phase"].startswith("run")]
    gen_windows = [(r["t_first_token"], r["t_last_token"]) for r in measured]
    idle = next(p for p in phases if p["name"] == "idle")
    nvml_idle = in_windows(nvml, [(idle["t0"], idle["t1"])])
    nvml_gen = in_windows(nvml, gen_windows)
    smi_gen = in_windows(smi, gen_windows)

    # ---- speed
    per_run = []
    for r in measured:
        t = r["server_timings"]
        per_run.append({
            "run": r["phase"],
            "client_ttft_ms": rnd(r["client_ttft_ms"]),
            "server_prompt_ms": rnd(t.get("prompt_ms")),
            "prompt_tokens": t.get("prompt_n"),
            "prompt_tps": rnd(t.get("prompt_per_second")),
            "server_gen_tps": rnd(t.get("predicted_per_second"), 2),
            "client_gen_tps": rnd(r["client_gen_tps"], 2),
            "generated_tokens": t.get("predicted_n"),
            "client_tokens": r["client_tokens"],
            "client_events": r["client_events"],
            "itl_p50_ms": rnd(r["itl_p50_ms"], 2),
            "itl_p95_ms": rnd(r["itl_p95_ms"], 2),
            "itl_max_ms": rnd(r["itl_max_ms"], 2),
            "client_total_ms": rnd(r["client_total_ms"]),
        })

    gen = [p["server_gen_tps"] for p in per_run if p["server_gen_tps"]]
    cv = statistics.stdev(gen) / statistics.fmean(gen) * 100 if len(gen) > 1 else None

    def spread(key):
        vals = [p[key] for p in per_run if p[key] is not None]
        if not vals:
            return None
        return {"median": rnd(statistics.median(vals), 2), "min": min(vals), "max": max(vals)}

    # ---- hardware
    gen_seconds = sum(b - a for a, b in gen_windows)
    gen_tokens = sum(p["generated_tokens"] or 0 for p in per_run)
    power_gen = fmean(s.get("gpu_power_w") for s in nvml_gen)
    vram_idle = fmean(s.get("vram_used_mib") for s in nvml_idle)
    vram_peak = max((s["vram_used_mib"] for s in nvml if s.get("vram_used_mib")), default=None)
    hardware = {
        "gpu_util_mean_pct": rnd(fmean(s.get("gpu_util_pct") for s in nvml_gen)),
        "gpu_power_mean_w": rnd(power_gen),
        "gpu_power_peak_w": rnd(max((s["gpu_power_w"] for s in nvml_gen if s.get("gpu_power_w")), default=None)),
        "gpu_temp_max_c": max((s["gpu_temp_c"] for s in nvml if s.get("gpu_temp_c")), default=None),
        "gpu_clock_mean_mhz": rnd(fmean(s.get("gpu_clock_mhz") for s in nvml_gen), 0),
        "clock_events_during_runs": decode_clock_events(s.get("clock_event_mask") for s in nvml_gen),
        "vram_idle_mib": rnd(vram_idle, 0),
        "vram_peak_mib": rnd(vram_peak, 0),
        "vram_delta_mib": rnd(vram_peak - vram_idle, 0) if vram_peak and vram_idle else None,
        "model_file_mib": round(env["model_bytes"] / 2**20),
        "server_rss_peak_mib": rnd(max((s["proc_rss_mib"] for s in nvml if s.get("proc_rss_mib")), default=None), 0),
        "server_private_peak_mib": rnd(max((s["proc_private_mib"] for s in nvml if s.get("proc_private_mib")), default=None), 0),
        "cpu_total_mean_pct": rnd(fmean(statistics.fmean(s["cpu_pct_per_core"]) for s in nvml_gen)),
        "server_cpu_mean_pct": rnd(fmean(s.get("proc_cpu_pct") for s in nvml_gen)),
        # GPU board power only; joules per token = watts / (tokens per second).
        "gpu_energy_j_per_token": rnd(power_gen * gen_seconds / gen_tokens, 3) if power_gen and gen_tokens else None,
        "samples_nvml_in_runs": len(nvml_gen),
        "samples_smi_in_runs": len(smi_gen),
    }

    # ---- exit-criteria checks
    checks = []
    predicted = [p["generated_tokens"] for p in per_run]
    checks.append(check(
        "Every run generated exactly n_predict tokens",
        predicted, f"== {args.n_predict}", all(n == args.n_predict for n in predicted),
    ))
    prompt_ns = [p["prompt_tokens"] for p in per_run]
    no_cache = bool(prompt_ns) and None not in prompt_ns and min(prompt_ns) == max(prompt_ns) \
        and (prompt_tokens is None or min(prompt_ns) >= 0.9 * prompt_tokens)
    checks.append(check(
        "Full prompt processed every run (no cache reuse)",
        prompt_ns, f"all equal, ~{prompt_tokens} tokens", no_cache,
    ))
    checks.append(check(
        "Client counted the same tokens as the server",
        [p["client_tokens"] for p in per_run], "== server", all(p["client_tokens"] == p["generated_tokens"] for p in per_run),
    ))
    rate_diffs = [pct_diff(p["client_gen_tps"], p["server_gen_tps"]) for p in per_run]
    worst = max((abs(d) for d in rate_diffs if d is not None), default=None)
    checks.append(check(
        "Server and client generation speed agree",
        rnd(worst, 2), f"<= {TOL_RATE_PCT}% apart", None if worst is None else worst <= TOL_RATE_PCT,
    ))
    off = server_info.get("offloaded")
    checks.append(check(
        "All layers on the GPU",
        off, "N/N", None if off is None else off[0] == off[1],
    ))

    u_nvml = fmean(s.get("gpu_util_pct") for s in nvml_gen)
    u_smi = fmean(s.get("utilization.gpu") for s in smi_gen)
    d_util = None if u_nvml is None or u_smi is None else abs(u_nvml - u_smi)
    checks.append(check(
        "NVML and nvidia-smi agree: GPU utilization",
        {"nvml": rnd(u_nvml), "smi": rnd(u_smi)}, f"<= {TOL_UTIL_POINTS} points apart",
        None if d_util is None else d_util <= TOL_UTIL_POINTS,
    ))
    m_nvml = fmean(s.get("vram_used_mib") for s in nvml_gen)
    m_smi = fmean(s.get("memory.used") for s in smi_gen)
    d_mem = None if m_nvml is None or m_smi is None else abs(m_nvml - m_smi)
    mem_limit = max(TOL_VRAM_MIB, 0.02 * (m_smi or 0))
    checks.append(check(
        "NVML and nvidia-smi agree: VRAM used",
        {"nvml": rnd(m_nvml, 0), "smi": rnd(m_smi, 0)}, f"<= {mem_limit:.0f} MiB apart",
        None if d_mem is None else d_mem <= mem_limit,
    ))
    p_smi = fmean(s.get("power.draw.average") for s in smi_gen)
    d_pow = pct_diff(power_gen, p_smi)
    checks.append(check(
        "NVML and nvidia-smi agree: GPU power",
        {"nvml": rnd(power_gen), "smi_average": rnd(p_smi)}, f"<= {TOL_POWER_PCT}% apart",
        None if d_pow is None else abs(d_pow) <= TOL_POWER_PCT,
    ))

    warnings = []
    if cv is not None and cv > WARN_CV_PCT:
        warnings.append(f"Generation speed varied {cv:.1f}% between runs (more than {WARN_CV_PCT}%).")
    if preflight["gpu_util_mean_pct"] > args.busy_threshold:
        warnings.append(
            f"GPU was {preflight['gpu_util_mean_pct']:.0f}% busy before the run; results include that load."
        )
    if hardware["clock_events_during_runs"]:
        throttling = [c for c in hardware["clock_events_during_runs"] if c not in ("gpu_idle",)]
        if throttling:
            warnings.append("GPU clock limits seen during runs: " + ", ".join(throttling))

    return {
        "env": env,
        "settings": {
            "server_command": server_info["command"],
            "n_predict": args.n_predict,
            "ctx": args.ctx,
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
            "sampling_interval_s": args.interval,
            "prompt_tokens_tokenized": prompt_tokens,
        },
        "preflight": {k: rnd(v) for k, v in preflight.items()},
        "load_seconds": rnd(server_info["load_seconds"], 2),
        "gpu_buffers_mib": server_info.get("gpu_buffers_mib"),
        "memory_lines_from_server_log": server_info.get("memory_lines"),
        "phases": phases,
        "runs": per_run,
        "summary": {
            "server_gen_tps": spread("server_gen_tps"),
            "client_ttft_ms": spread("client_ttft_ms"),
            "prompt_tps": spread("prompt_tps"),
            "itl_p95_ms": spread("itl_p95_ms"),
            "gen_tps_cv_pct": rnd(cv, 2),
        },
        "hardware": hardware,
        "checks": checks,
        "warnings": warnings,
        "passed": all(c["passed"] is not False for c in checks),
    }


def print_report(s: dict, out: Path) -> None:
    env, hw, sm = s["env"], s["hardware"], s["summary"]
    print()
    print(f"Model   {Path(env['model_path']).name}  ({hw['model_file_mib']} MiB)")
    print(f"Engine  {env['llama_server_version'].splitlines()[0]}")
    print(f"GPU     {env['gpu']}  driver {env['gpu_driver']}")
    print(f"Load    {s['load_seconds']} s")
    print()
    print(f"{'run':<6}{'TTFT ms':>9}{'prompt t/s':>12}{'gen t/s srv':>13}{'gen t/s cli':>13}{'ITL p95 ms':>12}{'tokens':>8}")
    for r in s["runs"]:
        print(
            f"{r['run']:<6}{r['client_ttft_ms']:>9}{r['prompt_tps']:>12}{r['server_gen_tps']:>13}"
            f"{r['client_gen_tps']:>13}{r['itl_p95_ms']:>12}{r['generated_tokens']:>8}"
        )
    print()
    print(f"Generation speed  median {sm['server_gen_tps']['median']} tok/s, run-to-run variation {sm['gen_tps_cv_pct']}%")
    print(f"GPU               {hw['gpu_util_mean_pct']}% busy, {hw['gpu_power_mean_w']} W mean, "
          f"{hw['gpu_power_peak_w']} W peak, {hw['gpu_temp_max_c']} C max, {hw['gpu_clock_mean_mhz']} MHz")
    print(f"Energy            {hw['gpu_energy_j_per_token']} J per token (GPU board only)")
    print(f"VRAM              {hw['vram_idle_mib']} MiB idle -> {hw['vram_peak_mib']} MiB peak "
          f"(+{hw['vram_delta_mib']} MiB for a {hw['model_file_mib']} MiB model)")
    b = s.get("gpu_buffers_mib") or {}
    if any(b.values()):
        print(f"VRAM by server    model {b.get('model')} + KV cache {b.get('kv')} + compute {b.get('compute')} MiB")
    print(f"Server process    {hw['server_rss_peak_mib']} MiB RAM peak, {hw['server_cpu_mean_pct']}% CPU")
    print()
    for c in s["checks"]:
        mark = {True: "PASS", False: "FAIL", None: "N/A "}[c["passed"]]
        print(f"[{mark}] {c['check']}: {c['value']} (limit {c['limit']})")
    for w in s["warnings"]:
        print(f"[WARN] {w}")
    print()
    print(("Phase 0 checks passed." if s["passed"] else "Some Phase 0 checks failed.") + f"  Details: {out}")


# ---------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    for p in (args.server, args.model):
        if not p.exists():
            print(f"Not found: {p}")
            return 2

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    print("Checking the GPU is idle...")
    preflight = gpu_preflight(handle, args.baseline_seconds, args.interval)
    need_mib = args.model.stat().st_size / 2**20 + 1024  # model plus a working margin
    busy = preflight["gpu_util_mean_pct"] > args.busy_threshold
    if (busy or preflight["vram_free_mib"] < need_mib) and not args.allow_busy:
        print(
            f"Stopped: the GPU is {preflight['gpu_util_mean_pct']:.0f}% busy with "
            f"{preflight['vram_used_mib']:.0f} MiB of VRAM in use and {preflight['vram_free_mib']:.0f} MiB free "
            f"(this run needs about {need_mib:.0f} MiB free and under {args.busy_threshold:.0f}% busy).\n"
            "Close games, video players and other GPU-heavy apps, then run again. "
            "Use --allow-busy to run anyway; the results will say the GPU was busy."
        )
        pynvml.nvmlShutdown()
        return 3

    out = args.out or SPIKE_DIR / "results" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print("Recording environment...")
    env = env_snapshot(args, handle)

    smi = sampler = server = None
    phases: list[dict] = []
    runs: list[dict] = []
    server_info: dict = {}
    prompt_tokens = None
    try:
        smi = start_smi(out / "nvidia_smi.csv", args.interval)
        sampler = start_sampler(out / "nvml.jsonl", args.interval)

        t0 = time.time()
        time.sleep(args.baseline_seconds)
        phases.append({"name": "idle", "t0": t0, "t1": time.time()})

        port = free_port()
        base = f"http://127.0.0.1:{port}"
        cmd = server_command(args, port)
        log_path = out / "server.log"
        print("Loading the model...")
        t0, t_load = time.time(), time.perf_counter()
        server = subprocess.Popen(
            cmd, stdout=open(log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT, cwd=str(out)
        )
        sampler_send(sampler, f"pid {server.pid}")
        wait_healthy(base, server, log_path)
        load_seconds = time.perf_counter() - t_load
        phases.append({"name": "load", "t0": t0, "t1": time.time()})

        server_info = {
            "command": cmd,
            "load_seconds": load_seconds,
            **parse_server_log(log_path.read_text(encoding="utf-8", errors="replace")),
        }

        with httpx.Client() as client:
            tok = client.post(f"{base}/tokenize", json={"content": PROMPT})
            if tok.status_code == 200:
                prompt_tokens = len(tok.json().get("tokens", []))
            labels = [f"warm{i + 1}" for i in range(args.warmup)] + [f"run{i + 1}" for i in range(args.runs)]
            for label in labels:
                print(f"  {label}...")
                t0 = time.time()
                result = stream_completion(client, base, args)
                result["phase"] = label
                runs.append(result)
                phases.append({"name": label, "t0": t0, "t1": time.time()})
                time.sleep(1.0)  # let power and clocks settle between runs
    finally:
        stop_process(server)
        if sampler is not None and sampler.poll() is None:
            try:
                sampler_send(sampler, "stop")
                sampler.stdin.close()
                sampler.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                sampler.kill()
        stop_process(smi)
        pynvml.nvmlShutdown()

    (out / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    summary = analyze(
        env, preflight, server_info, runs, phases,
        load_samples(out / "nvml.jsonl"), load_smi(out / "nvidia_smi.csv"), prompt_tokens, args,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_report(summary, out)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
