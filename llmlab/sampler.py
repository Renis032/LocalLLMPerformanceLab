"""Telemetry sampler, run as its own process by llmlab.telemetry.Sampler.

A separate process keeps sampling from competing with the benchmark client for
Python's GIL. Writes one JSON object per line to --out.

Control it through stdin, one command per line:
  pid <n>   also sample CPU and memory of process <n> (the inference server)
  stop      finish the current sample and exit
Closing stdin also stops it.

Without a readable NVIDIA GPU it still records CPU and memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import psutil
import pynvml


def _try(fn, *args):
    """NVML calls fail per field on some GPUs and drivers; record None instead."""
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return None


class GpuReader:
    def __init__(self, index: int) -> None:
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except pynvml.NVMLError:
            self.handle = None
        # Renamed from "throttle reasons" to "clocks event reasons" in newer NVML.
        self._clock_reasons = getattr(
            pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None
        ) or getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        self._instant_field = getattr(pynvml, "NVML_FI_DEV_POWER_INSTANT", None)

    def _instant_power_w(self):
        """Instantaneous board power. The usual power reading is a 1 s rolling average,
        which lags: it still read 140 W at 0% utilization 0.6 s after a run in Phase 0."""
        if self._instant_field is None:
            return None
        try:
            r = pynvml.nvmlDeviceGetFieldValues(self.handle, [self._instant_field])[0]
        except pynvml.NVMLError:
            return None
        return r.value.uiVal / 1000 if r.nvmlReturn == pynvml.NVML_SUCCESS else None

    def read(self) -> dict:
        h = self.handle
        if h is None:
            return {}
        util = _try(pynvml.nvmlDeviceGetUtilizationRates, h)
        # The v2 call separates driver-reserved memory from "used", matching nvidia-smi.
        # The default v1 call counts reserved memory as used (168 MiB too high in Phase 0).
        mem = _try(pynvml.nvmlDeviceGetMemoryInfo, h, pynvml.nvmlMemory_v2)
        power_mw = _try(pynvml.nvmlDeviceGetPowerUsage, h)
        return {
            "gpu_util_pct": util.gpu if util else None,
            "gpu_mem_util_pct": util.memory if util else None,
            "vram_used_mib": mem.used / 2**20 if mem else None,
            "gpu_power_w": power_mw / 1000 if power_mw is not None else None,
            "gpu_power_instant_w": self._instant_power_w(),
            "gpu_temp_c": _try(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_clock_mhz": _try(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_GRAPHICS),
            "mem_clock_mhz": _try(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_MEM),
            "pstate": _try(pynvml.nvmlDeviceGetPerformanceState, h),
            "clock_event_mask": _try(self._clock_reasons, h) if self._clock_reasons else None,
        }

    def close(self) -> None:
        if self.handle is not None:
            pynvml.nvmlShutdown()


class ProcessReader:
    def __init__(self, pid: int) -> None:
        self.proc = psutil.Process(pid)
        self.proc.cpu_percent(None)  # first call only primes the counter

    def read(self) -> dict:
        try:
            mi = self.proc.memory_info()
            private = getattr(mi, "private", None)  # Windows private bytes
            return {
                "proc_cpu_pct": self.proc.cpu_percent(None),
                "proc_rss_mib": mi.rss / 2**20,
                "proc_private_mib": private / 2**20 if private is not None else None,
            }
        except psutil.Error:
            return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    gpu = GpuReader(args.gpu_index)
    psutil.cpu_percent(percpu=True)  # first call only primes the counter
    state: dict = {"proc": None}
    stop = threading.Event()

    def control() -> None:
        for line in sys.stdin:
            cmd = line.split()
            if not cmd:
                continue
            if cmd[0] == "pid":
                try:
                    state["proc"] = ProcessReader(int(cmd[1]))
                except (psutil.Error, ValueError, IndexError):
                    pass
            elif cmd[0] == "stop":
                break
        stop.set()

    threading.Thread(target=control, daemon=True).start()

    with open(args.out, "w", encoding="utf-8") as f:
        next_tick = time.perf_counter()
        while not stop.is_set():
            vm = psutil.virtual_memory()
            sample = {
                "t": time.time(),
                **gpu.read(),
                "cpu_pct_per_core": psutil.cpu_percent(percpu=True),
                "ram_used_mib": vm.used / 2**20,
            }
            if state["proc"] is not None:
                sample.update(state["proc"].read())
            f.write(json.dumps(sample) + "\n")
            f.flush()
            next_tick += args.interval
            stop.wait(max(0.0, next_tick - time.perf_counter()))

    gpu.close()


if __name__ == "__main__":
    main()
