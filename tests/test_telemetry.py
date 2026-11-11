"""Purpose: hardware telemetry is collected and summarized correctly.

Runs the real sampler process, and checks the summaries the checks rely on:
clock-weighted GPU load, the CPU used by other programs, and missing sensors
staying missing.
"""

import os
import time

from llmlab import telemetry

from conftest import FakeGpu


def test_real_sampler_records_cpu_memory_and_a_watched_process(tmp_path):
    s = telemetry.Sampler(tmp_path / "samples.jsonl", interval=0.1).start()
    s.watch(os.getpid())
    time.sleep(0.6)
    rows = s.stop()
    assert s.proc.poll() is not None
    assert len(rows) >= 3
    assert all("t" in r and r["cpu_pct_per_core"] for r in rows)
    assert any("proc_rss_mib" in r for r in rows)
    # GPU fields are present when NVML is, and simply absent otherwise.
    if "gpu_util_pct" in rows[-1]:
        assert "gpu_power_instant_w" in rows[-1]


def test_aggregate_ignores_missing_sensors_and_decodes_clock_events():
    samples = [
        {"t": 1.0, "gpu_util_pct": 90, "gpu_power_w": 150.0, "gpu_power_instant_w": 190.0,
         "vram_used_mib": 4000.0, "clock_event_mask": 0x4, "cpu_pct_per_core": [10.0, 30.0]},
        {"t": 1.2, "gpu_util_pct": 80, "gpu_power_w": None, "gpu_power_instant_w": 170.0,
         "vram_used_mib": 4100.0, "clock_event_mask": 0x1, "cpu_pct_per_core": [20.0, 20.0]},
    ]
    a = telemetry.aggregate(samples)
    assert a["gpu_util_mean_pct"] == 85
    assert a["gpu_power_mean_w"] == 150.0
    assert a["gpu_power_instant_mean_w"] == 180.0
    assert a["vram_used_max_mib"] == 4100.0
    assert a["cpu_total_mean_pct"] == 20.0
    assert a["proc_cpu_mean_pct"] is None
    assert a["clock_events"] == ["gpu_idle", "sw_power_cap"]
    assert telemetry.window(samples, 1.1, 2.0) == samples[1:]
    assert telemetry.aggregate([])["gpu_util_mean_pct"] is None


def test_other_programs_cpu_excludes_the_server():
    # 12 logical cores, average 12.7% busy; the server keeps one core busy (100%),
    # which is 8.3% of the whole CPU, leaving 4.4% for everything else.
    samples = [{"t": 1.0, "cpu_pct_per_core": [100.0] + [4.8] * 11, "proc_cpu_pct": 100.0}]
    assert round(telemetry.aggregate(samples)["cpu_other_mean_pct"], 1) == 4.4
    # Without the server's own figure there's nothing to subtract, so no value.
    assert telemetry.aggregate([{"t": 1.0, "cpu_pct_per_core": [50.0]}])["cpu_other_mean_pct"] is None


def test_measure_idle_weights_utilization_by_clock():
    m = telemetry.measure_idle(FakeGpu(util=40, clock=540, max_clock=2160), seconds=0.0)
    assert m["gpu_util_mean_pct"] == 40 and m["gpu_load_mean_pct"] == 10.0
    assert m["gpu_clock_mean_mhz"] == 540 and m["vram_free_mib"] == 7000.0


def test_gpu_load():
    assert telemetry.gpu_load(99, 1785, 2160) == 99 * 1785 / 2160
    assert telemetry.gpu_load(50, 2500, 2160) == 50  # boost above the reported max counts as full speed
    assert telemetry.gpu_load(50, None, 2160) == 50  # unknown clock: raw utilization
    assert telemetry.gpu_load(None, 300, 2160) is None
