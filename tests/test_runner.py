"""Purpose: a whole experiment runs end to end, and every trust check fires when it should.

Runs complete experiments against a stand-in llama-server and a fake GPU. It checks
that a failing configuration doesn't stop the others, that a busy GPU stops the run
before anything is recorded, and that each check (valid tokens, settings applied,
steady clocks, quiet CPU and GPU) passes on good data and fails on the bad data seen
on real hardware.
"""

import pytest

from llmlab import db, report, runner
from llmlab.runner import GpuBusyError, config_checks, run_experiment, trial_problems

from conftest import FakeGpu, make_experiment


def _quiet(*args):
    pass


def test_end_to_end_with_a_sweep_and_a_failing_variant(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    exp = make_experiment(tmp_path, sweep={"server.ctx": [512, 999]})  # the stub fails to load at 999
    sid, results = run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)

    ok, failed = results
    assert ok["status"] == "ok", ok
    assert [(t["kind"], t["valid"]) for t in ok["trials"]] == [("warmup", 1)] + [("measured", 1)] * 3
    assert ok["summary"]["n"] == 3 and ok["summary"]["gen_tps"]["median"] > 0
    assert ok["prompt_tokens"] == 8
    assert {c["check"]: c["passed"] for c in ok["checks"]} == {
        "GPU idle before loading": True,
        "Layers on the GPU as configured": True,
        "Server used ctx = 512": True,
        "Server used flash_attn = on": True,
        "GPU clock steady during runs": None,  # no telemetry with the fake sampler
        "CPU quiet during runs": None,
        "GPU quiet after the runs": None,
    }
    assert failed["status"] == "failed"
    assert "exited with code 1" in failed["error"]

    session = db.get_session(conn, sid)
    assert session["status"] == "completed" and session["finished_at"]
    assert session["host_json"]["gpu"]["name"] == "Fake GPU"
    stored = db.get_configs(conn, sid)
    assert [c["status"] for c in stored] == ["ok", "failed"]
    assert stored[0]["engine_version"] == "test b0 (0000000)"
    assert stored[0]["model_sha256"] and stored[0]["facts_json"]["n_ctx"] == 512
    assert "listening on" in stored[0]["server_log"]
    assert len(db.get_trials(conn, stored[0]["id"])) == 4

    # Reports rebuild the same summary from the database.
    again = report.result_from_db(stored[0], db.get_trials(conn, stored[0]["id"]))
    assert again["summary"]["gen_tps"] == ok["summary"]["gen_tps"]
    text = report.session_text(conn, sid)
    assert "ctx=512" in text and "FAILED" in text
    _, verdict = report.compare_text(conn, sid, sid, 5.0)
    assert verdict is True


def test_a_busy_gpu_stops_the_run_before_anything_is_written(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    with pytest.raises(GpuBusyError, match="90% load"):
        run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(util=90), echo=_quiet)
    assert db.list_sessions(conn) == []


def test_allow_busy_runs_but_marks_the_result(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    _, (res,) = run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(util=90), allow_busy=True, echo=_quiet)
    assert res["status"] == "warning"
    assert [c["passed"] for c in res["checks"] if c["check"] == "GPU idle before loading"] == [False]


def test_a_cpu_only_run_does_not_need_vram_for_the_model(tmp_path):
    # Found in Phase 5: F16 on the CPU was refused because the check wanted 6.9 GB of VRAM free.
    big = tmp_path / "big.gguf"
    with open(big, "wb") as f:
        f.truncate(4096 * runner.MIB)
    exp = make_experiment(tmp_path, model={"path": str(big)}, idle={"vram_margin_mib": 1024},
                          sweep={"server.gpu_layers": [0, 18, "all"]})
    cpu, partial, gpu = (runner.vram_needed_mib(v.config) for v in exp.variants)
    assert cpu == 1024
    assert partial == gpu == 4096 + 1024  # a partial offload still counts the whole file


def test_a_missing_model_fails_that_variant_only(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    exp = make_experiment(tmp_path, sweep={"model.path": [str(tmp_path / "model.gguf"), str(tmp_path / "gone.gguf")]})
    _, (ok, missing) = run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)
    assert ok["status"] == "ok"
    assert missing["status"] == "failed" and "model.path not found" in missing["error"]


def _trial(**overrides):
    t = {"gen_tokens": 512, "prompt_tokens": 521, "client_tokens": 512, "gen_tps": 150.0, "client_gen_tps": 150.5}
    t.update(overrides)
    return t


def test_trial_problems():
    assert trial_problems(_trial(), 512, 521) == []
    assert "instead of 512" in trial_problems(_trial(gen_tokens=500, client_tokens=500), 512, 521)[0]
    assert "cached prompt" in trial_problems(_trial(prompt_tokens=1), 512, 521)[0]
    assert "client received 511" in trial_problems(_trial(client_tokens=511), 512, 521)[0]
    assert "differ by 10.0%" in trial_problems(_trial(client_gen_tps=165.0), 512, 521)[0]
    assert trial_problems(_trial(gen_tokens=None), 512, 521) == ["the server reported no timings"]


def _cfg(**server):
    s = {"ctx": 4096, "gpu_layers": "all", "flash_attn": "on", "threads": None, "threads_batch": None,
         "batch": None, "ubatch": None, "cache_type_k": None, "cache_type_v": None}
    s.update(server)
    return {"server": s, "idle": {"max_gpu_load_pct": 15.0, "max_other_cpu_pct": 8.0}}


def test_config_checks_compare_settings_with_the_server_log():
    idle = {"ok": True, "gpu_load_mean_pct": 0.5, "gpu_util_mean_pct": 1.0, "gpu_clock_mean_mhz": 1000.0,
            "vram_free_mib": 7000.0}
    facts = {"offloaded_layers": 18, "total_layers": 37, "n_ctx": 4096, "flash_attn": "enabled", "n_threads": 6}

    steady = {"run": 2, "spread_pct": 2.3, "min_mhz": 1890, "max_mhz": 1935}
    calm = {"run": 1, "pct": 3.8}
    checks = {c["check"]: c["passed"]
              for c in config_checks(_cfg(gpu_layers=18, threads=6), facts, idle, 2.0, steady, calm)}
    assert checks == {
        "GPU idle before loading": True,
        "Layers on the GPU as configured": True,
        "Server used ctx = 4096": True,
        "Server used threads = 6": True,
        "Server used flash_attn = on": True,
        "GPU clock steady during runs": True,
        "CPU quiet during runs": True,
        "GPU quiet after the runs": True,
    }
    ramping = {"run": 1, "spread_pct": 35.8, "min_mhz": 780, "max_mhz": 1215}
    contended = {"run": 2, "pct": 18.9}
    checks = {c["check"]: c["passed"]
              for c in config_checks(_cfg(gpu_layers="all", ubatch=256), facts, idle, 40.0, ramping, contended)}
    assert checks["Layers on the GPU as configured"] is False  # 18/37, but all were asked for
    assert checks["Server used ubatch = 256"] is None  # not in this log
    assert checks["GPU clock steady during runs"] is False
    assert checks["CPU quiet during runs"] is False
    assert checks["GPU quiet after the runs"] is False

    # With half the layers on the CPU, the GPU idles between its share of the work and
    # its clock swings with the load. That's expected, so the check doesn't apply.
    waiting = {"run": 4, "spread_pct": 84.0, "min_mhz": 270, "max_mhz": 1695, "gpu_util_mean_pct": 33.0}
    (clock_check,) = [c for c in config_checks(_cfg(gpu_layers=18), facts, idle, 2.0, waiting, calm)
                      if c["check"] == "GPU clock steady during runs"]
    assert clock_check["passed"] is None and "33% busy" in clock_check["detail"]


def test_busiest_other_cpu_looks_at_measured_runs_only():
    rows = [
        {"kind": "warmup", "idx": 0, "cpu_other_mean_pct": 40.0},
        {"kind": "measured", "idx": 0, "cpu_other_mean_pct": 3.8},
        {"kind": "measured", "idx": 1, "cpu_other_mean_pct": 18.9},
        {"kind": "measured", "idx": 2, "cpu_other_mean_pct": None},
    ]
    assert runner.busiest_other_cpu(rows) == {"run": 2, "pct": 18.9}
    assert runner.busiest_other_cpu(rows[:1]) is None


def test_clock_variation_finds_runs_that_ramp_up():
    # Session 1 on the RTX 3060 Ti: each run started downclocked at 780 MHz and ramped to ~1,200.
    trials = [
        {"kind": "warmup", "idx": 0, "t_first": 0.0, "t_last": 3.0},
        {"kind": "measured", "idx": 0, "t_first": 4.0, "t_last": 9.0},
        {"kind": "measured", "idx": 1, "t_first": 10.0, "t_last": 15.0},
    ]
    samples = [{"t": t, "gpu_clock_mhz": c, "gpu_util_pct": 96} for t, c in [
        (1.0, 1700), (2.0, 1100), (2.5, 1770),  # warm-up: not counted
        (4.5, 1935), (6.0, 1950), (8.0, 1920),  # steady
        (10.5, 780), (12.0, 975), (14.5, 1215),  # ramping
    ]]
    worst = runner.clock_variation(trials, samples)
    assert worst["run"] == 2 and round(worst["spread_pct"], 1) == 35.8
    assert worst["gpu_util_mean_pct"] == 96
    assert runner.clock_variation(trials, []) is None


def test_window_load_weights_by_clock():
    samples = [
        {"t": 10.7, "gpu_util_pct": 0, "gpu_clock_mhz": 1900},
        {"t": 10.9, "gpu_util_pct": 8, "gpu_clock_mhz": 1000},
        {"t": 12.0, "gpu_util_pct": 99, "gpu_clock_mhz": 1900},  # outside the window
    ]
    assert runner.window_load(samples, 10.6, 11.6, 2000) == 2.0  # (0 + 8 * 1000/2000) / 2
    assert runner.window_load(samples, 10.6, 11.6, None) == 4.0  # no max clock: raw utilization
    assert runner.window_load(samples, 20.0, 21.0, 2000) is None


def test_an_idle_desktop_at_low_clocks_is_not_busy(fake_engine, tmp_path):
    # Measured on the RTX 3060 Ti with only the desktop running: 40% at 345 MHz, 18 W.
    conn = db.connect(":memory:")
    _, (res,) = run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(util=40, clock=345, max_clock=2160),
                               echo=_quiet)
    assert res["status"] == "ok"
    # A game at 99% and 1,785 MHz is still busy.
    with pytest.raises(GpuBusyError, match="82% load"):
        run_experiment(make_experiment(tmp_path), db.connect(":memory:"),
                       gpu=FakeGpu(util=99, clock=1785, max_clock=2160), echo=_quiet)
