"""Telemetry: the sampler process, GPU idle checks, and summaries over time windows."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pynvml

from .stats import mean

MIB = 2**20
SAMPLER_SCRIPT = Path(__file__).with_name("sampler.py")

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


# ---------------------------------------------------------------- sampler process


class Sampler:
    """Runs sampler.py in its own process and collects what it wrote."""

    def __init__(self, out_path: Path, interval: float, gpu_index: int = 0):
        self.out_path = Path(out_path)
        self.interval = interval
        self.gpu_index = gpu_index
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 15.0) -> "Sampler":
        cmd = [
            sys.executable, str(SAMPLER_SCRIPT),
            "--out", str(self.out_path),
            "--interval", str(self.interval),
            "--gpu-index", str(self.gpu_index),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        # Wait for the first sample, so the model load is always on record.
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the telemetry sampler exited with code {self.proc.returncode}")
            if self.out_path.exists() and self.out_path.stat().st_size > 0:
                return self
            time.sleep(0.05)
        raise RuntimeError("the telemetry sampler did not start in time")

    def watch(self, pid: int) -> None:
        """Also sample this process's CPU and memory."""
        self._send(f"pid {pid}")

    def stop(self) -> list[dict]:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._send("stop")
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=10)
        return load_samples(self.out_path)

    def _send(self, line: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()


def load_samples(path: Path) -> list[dict]:
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line cut short when the sampler was killed
    return rows


# ---------------------------------------------------------------- summaries


def window(samples: list[dict], t0: float, t1: float) -> list[dict]:
    return [s for s in samples if t0 <= s["t"] <= t1]


def _vals(samples: list[dict], key: str) -> list[float]:
    return [s[key] for s in samples if s.get(key) is not None]


def energy_j(samples: list[dict], t0: float, t1: float) -> float | None:
    """GPU board energy over a time window, from instantaneous power. None with too few samples."""
    points = [(s["t"], s["gpu_power_instant_w"]) for s in window(samples, t0, t1) if s.get("gpu_power_instant_w") is not None]
    if len(points) < 2:
        return None
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:]))


def gpu_load(util: float | None, clock_mhz: float | None, max_clock_mhz: float | None) -> float | None:
    """Utilization weighted by clock speed: roughly, the share of full-speed capacity in use.

    NVML utilization is the share of time any work runs, whatever the clock. An idle
    desktop at 345 MHz reads about 40% on an RTX 3060 Ti, which weighted by clock is
    about 7% of what the GPU can do at its 2,160 MHz maximum. A game at 99% and
    1,785 MHz is about 82%.
    """
    if util is None:
        return None
    if clock_mhz and max_clock_mhz:
        return util * min(1.0, clock_mhz / max_clock_mhz)
    return util


def decode_clock_events(masks) -> list[str]:
    seen = 0
    for m in masks:
        if m:
            seen |= int(m)
    return [name for bit, name in CLOCK_EVENT_REASONS.items() if seen & bit]


def aggregate(samples: list[dict]) -> dict:
    """Means and peaks over a set of samples. Missing sensors give None."""
    power = _vals(samples, "gpu_power_w")
    instant = _vals(samples, "gpu_power_instant_w")
    vram = _vals(samples, "vram_used_mib")
    temp = _vals(samples, "gpu_temp_c")
    rss = _vals(samples, "proc_rss_mib")
    private = _vals(samples, "proc_private_mib")
    cpu_total = [sum(c) / len(c) for c in (s.get("cpu_pct_per_core") for s in samples) if c]
    # CPU used by everything except the watched server, as a share of the whole CPU.
    # psutil counts a process's CPU against one core (100% = one core busy).
    cpu_other = [
        max(0.0, sum(c) / len(c) - s["proc_cpu_pct"] / len(c))
        for s in samples
        if (c := s.get("cpu_pct_per_core")) and s.get("proc_cpu_pct") is not None
    ]
    return {
        "n_samples": len(samples),
        "gpu_util_mean_pct": mean(_vals(samples, "gpu_util_pct")),
        "gpu_power_mean_w": mean(power),
        "gpu_power_max_w": max(power) if power else None,
        "gpu_power_instant_mean_w": mean(instant),
        "gpu_power_instant_max_w": max(instant) if instant else None,
        "gpu_temp_max_c": max(temp) if temp else None,
        "gpu_clock_mean_mhz": mean(_vals(samples, "gpu_clock_mhz")),
        "vram_used_max_mib": max(vram) if vram else None,
        "cpu_total_mean_pct": mean(cpu_total),
        "cpu_other_mean_pct": mean(cpu_other),
        "proc_cpu_mean_pct": mean(_vals(samples, "proc_cpu_pct")),
        "proc_rss_max_mib": max(rss) if rss else None,
        "proc_private_max_mib": max(private) if private else None,
        "clock_events": decode_clock_events(s.get("clock_event_mask") for s in samples),
    }


# ---------------------------------------------------------------- in-process GPU


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


class Gpu:
    """NVML in the harness process, for idle checks and host facts (not for sampling runs)."""

    def __init__(self, index: int = 0):
        pynvml.nvmlInit()
        self.index = index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.max_clock_mhz = self._try(pynvml.nvmlDeviceGetMaxClockInfo, pynvml.NVML_CLOCK_GRAPHICS)

    def _try(self, fn, *args):
        try:
            return fn(self.handle, *args)
        except pynvml.NVMLError:
            return None

    def read(self) -> dict:
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle, pynvml.nvmlMemory_v2)
        util = self._try(pynvml.nvmlDeviceGetUtilizationRates)
        return {
            "util_pct": util.gpu if util else None,
            "clock_mhz": self._try(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_GRAPHICS),
            "used_mib": mem.used / MIB,
            "free_mib": mem.free / MIB,
            "total_mib": mem.total / MIB,
            "temp_c": self._try(pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU),
        }

    def info(self) -> dict:
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle, pynvml.nvmlMemory_v2)
        limit_mw = self._try(pynvml.nvmlDeviceGetEnforcedPowerLimit)
        try:
            cuda = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda = f"{cuda // 1000}.{cuda % 1000 // 10}"
        except pynvml.NVMLError:
            cuda = None
        return {
            "index": self.index,
            "name": _text(pynvml.nvmlDeviceGetName(self.handle)),
            "driver": _text(pynvml.nvmlSystemGetDriverVersion()),
            "cuda_driver": cuda,
            "vram_total_mib": round(mem.total / MIB),
            "max_clock_mhz": self.max_clock_mhz,
            "power_limit_w": limit_mw / 1000 if limit_mw else None,
            "pcie_gen_max": self._try(pynvml.nvmlDeviceGetMaxPcieLinkGeneration),
            "pcie_width_max": self._try(pynvml.nvmlDeviceGetMaxPcieLinkWidth),
        }

    def close(self) -> None:
        pynvml.nvmlShutdown()


def open_gpu(index: int = 0) -> Gpu | None:
    """The GPU, or None when no NVIDIA GPU is readable through NVML."""
    try:
        return Gpu(index)
    except pynvml.NVMLError:
        return None


def measure_idle(gpu, seconds: float, interval: float = 0.2) -> dict:
    """GPU load and memory over a short window, before anything of ours is running."""
    reads = []
    end = time.perf_counter() + seconds
    while True:
        reads.append(gpu.read())
        if time.perf_counter() >= end:
            break
        time.sleep(interval)
    max_clock = getattr(gpu, "max_clock_mhz", None)
    utils = [r["util_pct"] for r in reads if r["util_pct"] is not None]
    loads = [gpu_load(r["util_pct"], r.get("clock_mhz"), max_clock) for r in reads if r["util_pct"] is not None]
    last = reads[-1]
    return {
        "seconds": seconds,
        "gpu_load_mean_pct": mean(loads),
        "gpu_util_mean_pct": mean(utils),
        "gpu_util_max_pct": max(utils) if utils else None,
        "gpu_clock_mean_mhz": mean(r.get("clock_mhz") for r in reads),
        "gpu_max_clock_mhz": max_clock,
        "vram_used_mib": mean(r["used_mib"] for r in reads),
        "vram_free_mib": last["free_mib"],
        "vram_total_mib": last["total_mib"],
        "gpu_temp_c": last["temp_c"],
    }
