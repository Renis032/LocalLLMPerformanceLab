"""Run an experiment.

For each variant: check the GPU is idle, start llama-server, warm up, measure,
check every trial and the configuration, and store it all in SQLite. A variant
that fails (say, out of memory) is recorded as failed and the next one runs.
Cloud variants (Claude through Claude Code) run in cloud.py instead.
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
import time
from pathlib import Path

from . import __version__, cloud, db, llamacpp, report, stats, telemetry
from .live import NoLiveLog, StopRequested, check_stop
from .config import Experiment, Variant
from .host import host_snapshot, sha256_file
from .quality import run as quality_run
from .quality.suite import load_suite

MIB = 2**20

# Client and server generation speed must agree this closely for a trial to count.
TOL_CLIENT_SERVER_PCT = 5.0

# After the last run, the server sits idle while telemetry keeps sampling, to catch
# other programs using the GPU. NVML utilization reads 94% for about 0.45 s after a
# run ends on the RTX 3060 Ti, so the first 0.6 s is skipped, then 1 s is sampled.
GAP_SETTLE_S = 0.6
QUIET_SAMPLE_S = 1.0

# A run's GPU clock may vary this much and still count as steady. Steady runs varied
# 0.8-3% on the RTX 3060 Ti; runs that started on a downclocked GPU and ramped up
# during generation varied 33-36%.
CLOCK_STEADY_TOL_PCT = 10.0
# Below this GPU utilization the GPU is mostly waiting on the CPU (as with half the
# layers on the CPU), so its clock rises and falls with the load and the check doesn't apply.
CLOCK_CHECK_MIN_UTIL_PCT = 50.0

AUTO = object()  # run_experiment(gpu=AUTO): open the GPU through NVML if there is one


class GpuBusyError(RuntimeError):
    """Something else is using the GPU, so measurements would be unreliable."""


def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run_experiment(
    exp: Experiment, conn, *, allow_busy: bool = False, gpu=AUTO, echo=print, live=None
) -> tuple[int, list[dict]]:
    """Run every variant. Returns the session id and one result per variant.

    live receives progress events for the dashboard (see live.py) and can ask the run
    to stop; by default nothing is watching.
    """
    live = live or NoLiveLog()
    owns_gpu = gpu is AUTO
    if owns_gpu:
        gpu = telemetry.open_gpu()
        if gpu is None:
            echo("No NVIDIA GPU is readable through NVML, so GPU checks and GPU telemetry are skipped.")
    try:
        host = host_snapshot(gpu)
        # Check the GPU before writing anything, so a busy GPU leaves no empty session.
        idle = wait_for_idle(gpu, exp.variants[0], echo)
        if idle["ok"] is False and not allow_busy:
            live.emit("busy", message=busy_message(idle, exp.variants[0]))
            raise GpuBusyError(busy_message(idle, exp.variants[0]))

        session_id = db.insert(conn, "sessions", {
            "experiment": exp.name,
            "description": exp.description,
            "source_path": str(exp.path) if exp.path else None,
            "source_text": exp.source_text,
            "source_sha256": exp.source_sha256,
            "started_at": _now_iso(),
            "status": "running",
            "harness_version": __version__,
            "host_json": host,
        })
        conn.commit()
        echo(f"Session {session_id}: {exp.name}, {len(exp.variants)} configuration(s)")
        live.emit(
            "session", session_id=session_id, experiment=exp.name, description=exp.description,
            file=str(exp.path) if exp.path else None, gpu=(host.get("gpu") or {}).get("name"),
            variants=[{"index": v.index, "label": v.label, "engine": v.config["engine"]["type"]} for v in exp.variants],
        )

        status, error, results = "completed", None, []
        versions: dict[str, str] = {}
        try:
            for v in exp.variants:
                check_stop(live)
                if v.index > 0:
                    idle = wait_for_idle(gpu, v, echo)
                    if idle["ok"] is False and not allow_busy:
                        live.emit("busy", message=busy_message(idle, v))
                        raise GpuBusyError(busy_message(idle, v))
                res = run_variant(conn, session_id, v, len(exp.variants), idle, versions, echo, live)
                results.append(res)
                live.emit("variant_done", **live_summary(v, res))
        except KeyboardInterrupt as stop:
            status, error = "aborted", ("stopped from the dashboard" if isinstance(stop, StopRequested) else "stopped by the user")
            raise
        except Exception as e:
            status, error = "failed", str(e) if isinstance(e, GpuBusyError) else f"{type(e).__name__}: {e}"
            raise
        finally:
            db.update(conn, "sessions", session_id, {
                "status": status, "error": error, "finished_at": _now_iso(),
            })
            conn.commit()
            live.emit("done", session_id=session_id, status=status, error=error)
        return session_id, results
    finally:
        if owns_gpu and gpu is not None:
            gpu.close()


def live_summary(v: Variant, res: dict) -> dict:
    """What the dashboard shows when a configuration finishes."""
    s = res["summary"] or {}
    q = res.get("quality")

    def med(key):
        d = s.get(key)
        return d["median"] if d else None

    return {
        "quality": {k: q[k] for k in ("status", "error", "score", "correct", "attempts", "consistency")} if q else None,
        "index": v.index,
        "label": v.label,
        "status": res["status"],
        "error": res["error"],
        "config_id": res["config_id"],
        "runs": s.get("n", 0),
        "gen_tps": med("gen_tps"),
        "ttft_ms": med("ttft_ms"),
        "cv_pct": (s.get("gen_tps") or {}).get("cv_pct"),
        "checks": res["checks"],
        "invalid": [
            {"run": report.trial_name(t), "problems": t["problems_json"]}
            for t in res["trials"] if not t["valid"]
        ],
    }


# ---------------------------------------------------------------- idle check


def vram_needed_mib(config: dict) -> float:
    """Free VRAM a configuration needs before it starts.

    The model file counts only when some layers go on the GPU: with gpu_layers 0 the
    weights stay in RAM and the GPU holds only small buffers, which the margin covers.
    A partial offload still counts the whole file, which errs on the safe side.
    """
    model = Path(config["model"]["path"])
    on_gpu = config["server"]["gpu_layers"] != 0
    weights = model.stat().st_size / MIB if on_gpu and model.is_file() else 0.0
    return weights + config["idle"]["vram_margin_mib"]


def wait_for_idle(gpu, v: Variant, echo) -> dict:
    """Measure GPU load and free VRAM; if busy, keep checking for up to idle.max_wait_s."""
    if v.config["engine"]["type"] != "llama.cpp":
        return dict(cloud.NOT_LOCAL)
    limits = v.config["idle"]
    need_free = vram_needed_mib(v.config)
    if gpu is None:
        return {"ok": None, "skipped": "no GPU telemetry"}
    deadline = time.perf_counter() + limits["max_wait_s"]
    told = False
    while True:
        m = telemetry.measure_idle(gpu, limits["seconds"])
        load = m["gpu_load_mean_pct"]
        m["need_free_mib"] = need_free
        m["max_gpu_load_pct"] = limits["max_gpu_load_pct"]
        m["ok"] = (load is None or load <= limits["max_gpu_load_pct"]) and m["vram_free_mib"] >= need_free
        if m["ok"] or time.perf_counter() >= deadline:
            return m
        if not told:
            echo(
                f"  The GPU is busy ({describe_load(m)}, {report.fmt(m['vram_free_mib'], 0)} MiB free); "
                f"waiting up to {limits['max_wait_s']:.0f} s for it to settle..."
            )
            told = True


def describe_load(m: dict) -> str:
    """For example '82% load (99% utilization at 1,785 MHz)'."""
    return (
        f"{report.fmt(m['gpu_load_mean_pct'], 0)}% load ({report.fmt(m['gpu_util_mean_pct'], 0)}% utilization "
        f"at {report.fmt(m.get('gpu_clock_mean_mhz'), 0)} MHz)"
    )


def busy_message(m: dict, v: Variant) -> str:
    return (
        f"The GPU is at {describe_load(m)} with {report.fmt(m['vram_used_mib'], 0)} MiB of VRAM in use and "
        f"{report.fmt(m['vram_free_mib'], 0)} MiB free. Configuration '{v.label}' needs it under "
        f"{m['max_gpu_load_pct']:.0f}% load with about {m['need_free_mib']:.0f} MiB free. "
        "Close games, video players and other GPU-heavy apps, then run again, or pass --allow-busy "
        "to run anyway (the results will be marked)."
    )


# ---------------------------------------------------------------- one variant


def run_variant(
    conn, session_id: int, v: Variant, total: int, idle: dict, versions: dict, echo, live=None
) -> dict:
    live = live or NoLiveLog()
    run = v.config["run"]
    live.emit(
        "variant", index=v.index, total=total, label=v.label, engine=v.config["engine"]["type"],
        config_hash=v.config_hash, warmup=run["warmup"], repeats=run["repeats"],
    )
    if v.config["engine"]["type"] == "claude-code":
        return cloud.run_variant(conn, session_id, v, total, versions, echo, live)
    cfg = v.config
    exe, model = cfg["engine"]["server"], cfg["model"]["path"]
    run, w = cfg["run"], cfg["workload"]
    echo(f"\n[{v.index + 1}/{total}] {v.label}")
    config_id = db.insert(conn, "configs", {
        "session_id": session_id,
        "idx": v.index,
        "label": v.label,
        "config_hash": v.config_hash,
        "config_json": cfg,
        "overrides_json": v.overrides,
        "status": "running",
        "engine": cfg["engine"]["type"],
        "model_path": model,
        "idle_json": idle,
        "vram_before_mib": idle.get("vram_used_mib"),
        "started_at": time.time(),
    })
    conn.commit()

    missing = [
        f"{what} not found: {p}"
        for p, what in ((exe, "engine.server"), (model, "model.path"))
        if not Path(p).is_file()
    ]
    if missing:
        return _finish(conn, config_id, v, None, None, None, [], [], [], idle, "", "; ".join(missing), echo)

    if exe not in versions:
        versions[exe] = llamacpp.engine_version(exe)
    sha = db.known_sha256(conn, model)
    if sha is None:
        echo("  hashing the model file (only when it is new or has changed)...")
        sha = sha256_file(model)
        db.store_sha256(conn, model, sha)
    db.update(conn, "configs", config_id, {
        "engine_version": versions[exe],
        "model_bytes": Path(model).stat().st_size,
        "model_sha256": sha,
    })
    conn.commit()

    work = Path(tempfile.mkdtemp(prefix="llmlab-"))
    sampler = telemetry.Sampler(work / "samples.jsonl", run["sample_interval_s"])
    server = llamacpp.LlamaServer(exe, model, cfg["server"], work / "server.log")
    trials: list[dict] = []
    phases: list[dict] = []
    load_s = prompt_tokens = error = None
    interrupted: KeyboardInterrupt | None = None
    outcome = None
    try:
        sampler.start()
        live.emit("samples", path=str(sampler.out_path))
        t0 = time.time()
        live.emit("phase", name="load")
        load_s = server.start(run["load_timeout_s"])
        phases.append({"name": "load", "t0": t0, "t1": time.time()})
        sampler.watch(server.pid)
        prompt_tokens = server.tokenize(v.prompt)
        echo(f"  loaded in {load_s:.2f} s; the prompt is {prompt_tokens} tokens")
        live.emit("loaded", load_s=load_s, prompt_tokens=prompt_tokens)
        schedule = [("warmup", i) for i in range(run["warmup"])]
        schedule += [("measured", i) for i in range(run["repeats"])]
        for kind, i in schedule:
            check_stop(live)
            # A short pause lets the server finish the previous request. A long one lets
            # the GPU drop to a lower power state, and the next run starts slow.
            time.sleep(run["cooldown_s"])
            r = server.complete(v.prompt, w["n_predict"], w["seed"])
            r["kind"], r["idx"] = kind, i
            trials.append(r)
            echo(report.trial_progress(r, run["repeats"]))
            live.emit("trial", **live_trial(v, r))
        t_end = trials[-1]["t_last"]
        live.emit("phase", name="quiet")
        time.sleep(max(0.0, t_end + GAP_SETTLE_S + QUIET_SAMPLE_S - time.time()))
        phases.append({"name": "quiet", "t0": t_end + GAP_SETTLE_S, "t1": time.time()})
        if cfg["quality"]["suite"]:
            outcome = quality_run.start(load_suite(cfg["quality"]["suite"]), quality_settings(cfg))
            answerer = quality_run.LocalAnswerer(server, outcome.settings, w["seed"])
            quality_run.run_suite(outcome, answerer, v.index, run["cooldown_s"], echo, live)
            if outcome.t0:
                phases.append({"name": "quality", "t0": outcome.t0, "t1": outcome.t1})
    except KeyboardInterrupt as stop:
        error, interrupted = ("stopped from the dashboard" if isinstance(stop, StopRequested) else "stopped by the user"), stop
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        server.stop()
        samples = sampler.stop()

    result = _finish(
        conn, config_id, v, server.command, load_s, prompt_tokens, trials, samples, phases,
        idle, server.log_text(), error, echo,
    )
    if outcome is not None:
        outcome.error = outcome.error or error
        result["quality"] = quality_run.finish(conn, config_id, outcome, samples, echo)
    if error is None:
        shutil.rmtree(work, ignore_errors=True)
    else:
        echo(f"  working files kept in {work}")
    if interrupted:
        raise interrupted
    return result


def quality_settings(cfg: dict) -> dict:
    q = cfg["quality"]
    return {k: q[k] for k in ("repeats", "max_tokens", "temperature", "thinking", "system_prompt")}


def live_trial(v: Variant, r: dict) -> dict:
    """One run's result, as the dashboard shows it while the experiment runs."""
    keys = ("kind", "idx", "t_send", "t_first", "t_last", "gen_tps", "prompt_tps", "ttft_ms", "gen_tokens",
            "prompt_tokens", "thinking_tokens", "first_content_ms")
    return {"index": v.index, **{k: r.get(k) for k in keys}}


def _finish(conn, config_id, v, command, load_s, prompt_tokens, trials, samples, phases, idle, log, error, echo) -> dict:
    cfg = v.config
    facts = llamacpp.parse_log(log)
    rows = [trial_row(r, samples, cfg["workload"]["n_predict"], prompt_tokens) for r in trials]
    whole = telemetry.aggregate(samples)
    quiet = next((p for p in phases if p["name"] == "quiet"), None)
    quiet_load = window_load(samples, quiet["t0"], quiet["t1"], idle.get("gpu_max_clock_mhz")) if quiet else None
    checks = (
        config_checks(cfg, facts, idle, quiet_load, clock_variation(trials, samples), busiest_other_cpu(rows))
        if trials else []
    )
    measured = [t for t in rows if t["kind"] == "measured"]
    valid = [t for t in measured if t["valid"]]
    if error is None and not valid:
        error = "no measured run passed its checks"
    if error is not None:
        status = "failed"
    elif len(valid) == len(measured) and all(c["passed"] is not False for c in checks):
        status = "ok"
    else:
        status = "warning"

    db.update(conn, "configs", config_id, {
        "status": status,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "command_json": command,
        "load_s": load_s,
        "vram_peak_mib": whole["vram_used_max_mib"],
        "server_rss_max_mib": whole["proc_rss_max_mib"],
        "server_private_max_mib": whole["proc_private_max_mib"],
        "facts_json": facts,
        "checks_json": checks,
        "phases_json": phases,
        "server_log": log,
        "finished_at": time.time(),
    })
    db.insert_many(conn, "trials", [{"config_id": config_id, **t} for t in rows])
    db.insert_many(conn, "samples", [sample_row(config_id, s) for s in samples])
    conn.commit()

    result = {
        "config_id": config_id,
        "label": v.label,
        "config_hash": v.config_hash,
        "engine": cfg["engine"]["type"],
        "status": status,
        "error": error,
        "load_s": load_s,
        "prompt_tokens": prompt_tokens,
        "facts": facts,
        "checks": checks,
        "trials": rows,
        "summary": stats.summarize(valid),
        "vram_before_mib": idle.get("vram_used_mib"),
        "vram_peak_mib": whole["vram_used_max_mib"],
        "server_rss_max_mib": whole["proc_rss_max_mib"],
        "server_private_max_mib": whole["proc_private_max_mib"],
    }
    for line in report.config_lines(result):
        echo(line)
    return result


# ---------------------------------------------------------------- checks


def trial_problems(r: dict, n_predict: int, prompt_tokens: int | None) -> list[str]:
    """Reasons a trial can't be trusted. An empty list means it counts."""
    if r.get("gen_tokens") is None:
        return ["the server reported no timings"]
    p = []
    if r["gen_tokens"] != n_predict:
        p.append(f"generated {r['gen_tokens']} tokens instead of {n_predict}")
    if prompt_tokens is not None and r["prompt_tokens"] != prompt_tokens:
        p.append(
            f"processed {r['prompt_tokens']} of {prompt_tokens} prompt tokens (was a cached prompt reused?)"
        )
    if r["client_tokens"] != r["gen_tokens"]:
        p.append(f"the client received {r['client_tokens']} tokens, the server reported {r['gen_tokens']}")
    d = stats.pct_diff(r["client_gen_tps"], r["gen_tps"])
    if d is None or abs(d) > TOL_CLIENT_SERVER_PCT:
        amount = "an unknown amount" if d is None else f"{abs(d):.1f}%"
        p.append(f"client and server generation speeds differ by {amount}")
    return p


def trial_row(r: dict, samples: list[dict], n_predict: int, prompt_tokens: int | None) -> dict:
    gen = telemetry.aggregate(telemetry.window(samples, r["t_first"], r["t_last"]))
    # Instantaneous power follows the run; the 1 s average lags behind it.
    power = gen["gpu_power_instant_mean_w"]
    if power is None:
        power = gen["gpu_power_mean_w"]
    problems = trial_problems(r, n_predict, prompt_tokens)
    return {
        "kind": r["kind"],
        "idx": r["idx"],
        "t_send": r["t_send"],
        "t_first": r["t_first"],
        "t_last": r["t_last"],
        "prompt_tokens": r["prompt_tokens"],
        "gen_tokens": r["gen_tokens"],
        "client_tokens": r["client_tokens"],
        "prompt_ms": r["prompt_ms"],
        "prompt_tps": r["prompt_tps"],
        "gen_tps": r["gen_tps"],
        "client_gen_tps": r["client_gen_tps"],
        "ttft_ms": r["ttft_ms"],
        "total_ms": r["total_ms"],
        "itl_p50_ms": r["itl_p50_ms"],
        "itl_p95_ms": r["itl_p95_ms"],
        "itl_max_ms": r["itl_max_ms"],
        "itl_json": r["itl_ms"],
        "gpu_util_mean_pct": gen["gpu_util_mean_pct"],
        "gpu_power_mean_w": gen["gpu_power_mean_w"],
        "gpu_power_max_w": gen["gpu_power_max_w"],
        "gpu_power_instant_mean_w": gen["gpu_power_instant_mean_w"],
        "gpu_temp_max_c": gen["gpu_temp_max_c"],
        "gpu_clock_mean_mhz": gen["gpu_clock_mean_mhz"],
        "vram_used_max_mib": gen["vram_used_max_mib"],
        "cpu_total_mean_pct": gen["cpu_total_mean_pct"],
        "cpu_other_mean_pct": gen["cpu_other_mean_pct"],
        "proc_cpu_mean_pct": gen["proc_cpu_mean_pct"],
        "proc_rss_max_mib": gen["proc_rss_max_mib"],
        # Joules per token = watts / (tokens per second). GPU board power only.
        "j_per_token": power / r["gen_tps"] if power is not None and r["gen_tps"] else None,
        "clock_events_json": gen["clock_events"],
        "stop_type": r["stop_type"],
        "valid": int(not problems),
        "problems_json": problems,
    }


def window_load(samples: list[dict], t0: float, t1: float, max_clock_mhz: float | None) -> float | None:
    """Mean clock-weighted GPU load over a time window."""
    return stats.mean(
        telemetry.gpu_load(s.get("gpu_util_pct"), s.get("gpu_clock_mhz"), max_clock_mhz)
        for s in telemetry.window(samples, t0, t1)
    )


def clock_variation(trials: list[dict], samples: list[dict]) -> dict | None:
    """The measured run whose GPU clock varied most during generation, or None without data."""
    worst = None
    utils: list[float] = []
    for t in trials:
        if t["kind"] != "measured":
            continue
        win = telemetry.window(samples, t["t_first"], t["t_last"])
        utils += [s["gpu_util_pct"] for s in win if s.get("gpu_util_pct") is not None]
        clocks = [s["gpu_clock_mhz"] for s in win if s.get("gpu_clock_mhz")]
        if len(clocks) < 3:
            continue
        spread = (max(clocks) - min(clocks)) / max(clocks) * 100
        if worst is None or spread > worst["spread_pct"]:
            worst = {"run": t["idx"] + 1, "spread_pct": spread, "min_mhz": min(clocks), "max_mhz": max(clocks)}
    if worst is not None:
        worst["gpu_util_mean_pct"] = stats.mean(utils)
    return worst


def busiest_other_cpu(rows: list[dict]) -> dict | None:
    """The measured run in which other programs used the most CPU, or None without data."""
    measured = [t for t in rows if t["kind"] == "measured" and t.get("cpu_other_mean_pct") is not None]
    if not measured:
        return None
    worst = max(measured, key=lambda t: t["cpu_other_mean_pct"])
    return {"run": worst["idx"] + 1, "pct": worst["cpu_other_mean_pct"]}


def config_checks(
    cfg: dict, facts: dict, idle: dict, quiet_load: float | None,
    clocks: dict | None = None, other_cpu: dict | None = None,
) -> list[dict]:
    """Did the configuration run the way the experiment file says, on a quiet GPU?"""
    s, limits = cfg["server"], cfg["idle"]
    checks: list[dict] = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": passed, "detail": detail})

    ok = idle.get("ok")
    add(
        "GPU idle before loading",
        ok,
        "no GPU telemetry" if ok is None else f"{describe_load(idle)}, {report.fmt(idle['vram_free_mib'], 0)} MiB free",
    )

    off, total = facts.get("offloaded_layers"), facts.get("total_layers")
    if off is None or total is None:
        add("Layers on the GPU as configured", None, "not reported in the server log")
    else:
        want = total if s["gpu_layers"] == "all" else min(s["gpu_layers"], total)
        add("Layers on the GPU as configured", off == want, f"{off}/{total} on the GPU, expected {want}")

    applied = (
        ("ctx", "n_ctx"), ("batch", "n_batch"), ("ubatch", "n_ubatch"),
        ("threads", "n_threads"), ("threads_batch", "n_threads_batch"),
        ("cache_type_k", "kv_type_k"), ("cache_type_v", "kv_type_v"),
    )
    for key, fact in applied:
        want = s[key]
        if want is None:
            continue
        got = facts.get(fact)
        add(
            f"Server used {key} = {want}",
            None if got is None else got == want,
            "not reported in the server log" if got is None else f"the server reports {got}",
        )
    if s["flash_attn"] in ("on", "off"):
        want = "enabled" if s["flash_attn"] == "on" else "disabled"
        got = facts.get("flash_attn")
        add(
            f"Server used flash_attn = {s['flash_attn']}",
            None if got is None else got == want,
            "not reported in the server log" if got is None else f"the server reports {got}",
        )

    busy = clocks.get("gpu_util_mean_pct") if clocks else None
    if clocks is None:
        add("GPU clock steady during runs", None, "no clock samples during the runs")
    elif busy is not None and busy < CLOCK_CHECK_MIN_UTIL_PCT:
        add(
            "GPU clock steady during runs",
            None,
            f"not applicable: the GPU was only {busy:.0f}% busy, mostly waiting on the CPU, "
            "so its clock follows the load",
        )
    else:
        steady = clocks["spread_pct"] <= CLOCK_STEADY_TOL_PCT
        add(
            "GPU clock steady during runs",
            steady,
            f"varied up to {clocks['spread_pct']:.0f}% within a run (run {clocks['run']}: "
            f"{clocks['min_mhz']:,.0f} to {clocks['max_mhz']:,.0f} MHz)"
            + ("" if steady else "; the GPU was changing speed, for example ramping up from a power-saving state"),
        )

    if other_cpu is None:
        add("CPU quiet during runs", None, "no CPU samples during the runs")
    else:
        calm = other_cpu["pct"] <= limits["max_other_cpu_pct"]
        add(
            "CPU quiet during runs",
            calm,
            f"other programs used up to {other_cpu['pct']:.1f}% of the CPU (run {other_cpu['run']})"
            + ("" if calm else "; they compete with the server's busy core and slow generation"),
        )

    if quiet_load is None:
        add("GPU quiet after the runs", None, "no samples after the runs")
    else:
        quiet = quiet_load <= limits["max_gpu_load_pct"]
        add(
            "GPU quiet after the runs",
            quiet,
            f"{quiet_load:.0f}% load with the server idle" + ("" if quiet else " (another program may be using the GPU)"),
        )
    return checks


def sample_row(config_id: int, s: dict) -> dict:
    cores = s.get("cpu_pct_per_core")
    return {
        "config_id": config_id,
        "t": s["t"],
        "gpu_util_pct": s.get("gpu_util_pct"),
        "gpu_mem_util_pct": s.get("gpu_mem_util_pct"),
        "vram_used_mib": s.get("vram_used_mib"),
        "gpu_power_w": s.get("gpu_power_w"),
        "gpu_power_instant_w": s.get("gpu_power_instant_w"),
        "gpu_temp_c": s.get("gpu_temp_c"),
        "gpu_clock_mhz": s.get("gpu_clock_mhz"),
        "mem_clock_mhz": s.get("mem_clock_mhz"),
        "pstate": s.get("pstate"),
        "clock_event_mask": s.get("clock_event_mask"),
        "cpu_pct": sum(cores) / len(cores) if cores else None,
        "cpu_per_core_json": cores,
        "ram_used_mib": s.get("ram_used_mib"),
        "proc_cpu_pct": s.get("proc_cpu_pct"),
        "proc_rss_mib": s.get("proc_rss_mib"),
        "proc_private_mib": s.get("proc_private_mib"),
    }
