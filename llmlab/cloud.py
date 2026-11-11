"""Run a cloud configuration: one Claude model, through Claude Code.

There is no server to start and no GPU to watch. Each run is one request, and the
checks confirm it measured the model and nothing else Claude Code can add: another
request, another model, hidden thinking, or a cached prompt.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from . import claude_code, db, report, stats
from .live import NoLiveLog, StopRequested, check_stop
from .quality import run as quality_run
from .quality.suite import load_suite

# Our time to the first streamed content and Claude Code's own time to first token must
# agree this closely. Ours starts when Claude Code says it is ready, its own a little
# earlier; they differed by 39-57 ms. Both count thinking, which comes before any text.
TTFT_AGREE_MS = 150.0
TTFT_AGREE_PCT = 15.0

NOT_LOCAL = {"ok": None, "skipped": "cloud model: the local GPU isn't used"}


def run_variant(conn, session_id: int, v, total: int, versions: dict, echo, live=None) -> dict:
    from .runner import live_trial  # runner imports this module

    live = live or NoLiveLog()
    cfg = v.config
    e, w, run = cfg["engine"], cfg["workload"], cfg["run"]
    model = cfg["model"]["name"]
    echo(f"\n[{v.index + 1}/{total}] {v.label}")
    config_id = db.insert(conn, "configs", {
        "session_id": session_id,
        "idx": v.index,
        "label": v.label,
        "config_hash": v.config_hash,
        "config_json": cfg,
        "overrides_json": v.overrides,
        "status": "running",
        "engine": e["type"],
        "model_path": model,
        "idle_json": NOT_LOCAL,
        "started_at": time.time(),
    })
    conn.commit()

    try:
        exe = claude_code.find_cli(e["cli"])
        if exe not in versions:
            versions[exe] = claude_code.cli_version(exe)
    except claude_code.ClaudeCodeError as err:
        return _finish(conn, config_id, v, None, [], str(err), echo)
    db.update(conn, "configs", config_id, {"engine_version": versions[exe]})
    conn.commit()

    cmd = claude_code.build_command(exe, model, e["effort"], w["system_prompt"])
    env = claude_code.build_env(e["thinking"])
    # An empty working folder, so Claude Code finds no project files to add.
    work = Path(tempfile.mkdtemp(prefix="llmlab-claude-"))
    trials: list[dict] = []
    error = None
    interrupted: KeyboardInterrupt | None = None
    outcome = None
    try:
        schedule = [("warmup", i) for i in range(run["warmup"])]
        schedule += [("measured", i) for i in range(run["repeats"])]
        for kind, i in schedule:
            check_stop(live)
            if trials:
                time.sleep(run["cooldown_s"])
            r = claude_code.complete(cmd, env, v.prompt, work, run["timeout_s"])
            r["kind"], r["idx"] = kind, i
            trials.append(r)
            echo(report.cloud_trial_progress(r, run["repeats"]))
            live.emit("trial", **live_trial(v, r))
        q = cfg["quality"]
        if q["suite"]:
            outcome = quality_run.start(load_suite(q["suite"]), quality_settings(cfg))
            answerer = quality_run.CloudAnswerer(cmd, env, work, cfg)
            quality_run.run_suite(outcome, answerer, v.index, run["cooldown_s"], echo, live)
    except KeyboardInterrupt as stop:
        error, interrupted = ("stopped from the dashboard" if isinstance(stop, StopRequested) else "stopped by the user"), stop
    except Exception as err:
        error = f"{type(err).__name__}: {err}"
    finally:
        shutil.rmtree(work, ignore_errors=True)

    result = _finish(conn, config_id, v, cmd, trials, error, echo)
    if outcome is not None:
        outcome.error = outcome.error or error
        result["quality"] = quality_run.finish(conn, config_id, outcome, [], echo)
    if interrupted:
        raise interrupted
    return result


def _finish(conn, config_id: int, v, cmd, trials: list[dict], error, echo) -> dict:
    cfg = v.config
    rows = [trial_row(r, cfg) for r in trials]
    checks = config_checks(rows) if rows else []
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
    last = trials[-1] if trials else {}
    facts = {"auth": last.get("auth"), "cli_version": last.get("cli_version")}
    tokens = stats.median(t["prompt_tokens"] for t in rows)
    db.update(conn, "configs", config_id, {
        "status": status,
        "error": error,
        "prompt_tokens": round(tokens) if tokens is not None else None,
        "command_json": cmd,
        "facts_json": facts,
        "checks_json": checks,
        "finished_at": time.time(),
    })
    db.insert_many(conn, "trials", [{"config_id": config_id, **t} for t in rows])
    conn.commit()

    result = {
        "config_id": config_id,
        "label": v.label,
        "config_hash": v.config_hash,
        "engine": cfg["engine"]["type"],
        "status": status,
        "error": error,
        "load_s": None,
        "prompt_tokens": tokens,
        "facts": facts,
        "checks": checks,
        "trials": rows,
        "summary": stats.summarize(valid),
        "vram_before_mib": None,
        "vram_peak_mib": None,
        "server_rss_max_mib": None,
        "server_private_max_mib": None,
    }
    for line in report.config_lines(result):
        echo(line)
    return result


# ---------------------------------------------------------------- checks


def quality_settings(cfg: dict) -> dict:
    """What shapes a Claude answer. Claude has no temperature or seed to set."""
    e = cfg["engine"]
    return {
        "repeats": cfg["quality"]["repeats"],
        "effort": e["effort"],
        "thinking": e["thinking"],
        "system_prompt": cfg["workload"]["system_prompt"],
    }


def request_problems(r: dict, cfg: dict) -> list[str]:
    """Reasons a request didn't reach the model it was meant for, alone and uncached."""
    p = []
    want = cfg["model"]["name"]
    if r["api_calls"] != 1:
        p.append(
            f"Claude Code made {r['api_calls']} API requests instead of 1 "
            "(it continues by itself when an answer reaches the output limit)"
        )
    if r["model_used"] != want:
        p.append(f"answered by {r['model_used']} instead of {want}")
    if cfg["engine"]["thinking"] == "off" and r["thinking_tokens"]:
        p.append(f"{r['thinking_tokens']} hidden thinking tokens although thinking is off")
    if r["cached_tokens"]:
        p.append(f"{r['cached_tokens']} prompt tokens were read from or written to the cache")
    return p


def trial_problems(r: dict, cfg: dict) -> list[str]:
    """Reasons a run didn't measure the model alone. An empty list means it counts."""
    p = request_problems(r, cfg)
    if r["stop_type"] != "end_turn":
        p.append(f"the answer stopped early ({r['stop_type']})")
    ours, theirs = r["first_content_ms"], r["reported_ttft_ms"]
    if theirs is not None and abs(ours - theirs) > max(TTFT_AGREE_MS, TTFT_AGREE_PCT / 100 * theirs):
        p.append(
            f"the first streamed content arrived after {ours:.0f} ms here, {theirs:.0f} ms by Claude Code's "
            "own timing; they should agree"
        )
    return p


def trial_row(r: dict, cfg: dict) -> dict:
    problems = trial_problems(r, cfg)
    return {
        "kind": r["kind"],
        "idx": r["idx"],
        "t_send": r["t_send"],
        "t_first": r["t_first"],
        "t_last": r["t_last"],
        "prompt_tokens": r["prompt_tokens"],
        "gen_tokens": r["gen_tokens"],
        "client_tokens": None,
        "gen_tps": r["gen_tps"],
        "client_gen_tps": r["client_gen_tps"],
        "ttft_ms": r["ttft_ms"],
        "total_ms": r["total_ms"],
        "itl_p50_ms": r["itl_p50_ms"],
        "itl_p95_ms": r["itl_p95_ms"],
        "itl_max_ms": r["itl_max_ms"],
        "itl_json": r["itl_ms"],
        "stop_type": r["stop_type"],
        "startup_ms": r["startup_ms"],
        "reported_ttft_ms": r["reported_ttft_ms"],
        "first_content_ms": r["first_content_ms"],
        "chunks": r["chunks"],
        "thinking_tokens": r["thinking_tokens"],
        "cached_tokens": r["cached_tokens"],
        "api_calls": r["api_calls"],
        "model_used": r["model_used"],
        "cost_usd": r["cost_usd"],
        "valid": int(not problems),
        "problems_json": problems,
    }


def config_checks(rows: list[dict]) -> list[dict]:
    """Did every run send the same thing?"""
    counts = sorted({t["prompt_tokens"] for t in rows if t["prompt_tokens"] is not None})
    if not counts:
        return [{"check": "Same input every run", "passed": None, "detail": "no token counts reported"}]
    same = len(counts) == 1
    detail = (
        f"{counts[0]:,} input tokens every run, including Claude Code's own context"
        if same else
        f"input tokens varied from {counts[0]:,} to {counts[-1]:,}; Claude Code added different context"
    )
    return [{"check": "Same input every run", "passed": same, "detail": detail}]
