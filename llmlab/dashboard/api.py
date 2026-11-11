"""Data for the dashboard, as plain dicts ready for JSON.

Every number comes from the database through the same code the terminal reports use
(stats.summarize), so the dashboard and `llmlab show` always agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .. import db, stats
from ..config import ConfigError, load_experiment
from ..quality import summary as quality_summary

# Trial columns the dashboard needs; the inter-token gaps stay in the database.
TRIAL_KEYS = (
    "kind", "idx", "valid", "problems_json", "t_send", "t_first", "t_last", "gen_tps", "client_gen_tps",
    "prompt_tps", "ttft_ms", "total_ms", "prompt_tokens", "gen_tokens", "itl_p50_ms", "itl_p95_ms",
    "gpu_util_mean_pct", "gpu_power_instant_mean_w", "gpu_clock_mean_mhz", "gpu_temp_max_c", "j_per_token",
    "cpu_other_mean_pct", "proc_cpu_mean_pct", "stop_type", "reported_ttft_ms", "first_content_ms",
    "thinking_tokens", "cached_tokens", "api_calls", "model_used", "cost_usd", "startup_ms", "chunks",
)

CONFIG_KEYS = (
    "id", "session_id", "idx", "label", "config_hash", "status", "error", "engine", "engine_version",
    "model_path", "model_bytes", "model_sha256", "prompt_tokens", "load_s", "vram_before_mib", "vram_peak_mib",
    "server_rss_max_mib", "server_private_max_mib", "started_at", "finished_at",
)


def _median(values) -> float | None:
    return stats.median(values)


def _model_name(c: dict) -> str:
    """A short name: the GGUF file name, or the Claude model ID."""
    path = c.get("model_path") or ""
    return Path(path).name if c.get("engine") == "llama.cpp" else path


def _headline(trials: list[dict]) -> dict:
    valid = [t for t in trials if t["kind"] == "measured" and t["valid"]]
    gen = [t["gen_tps"] for t in valid]
    return {
        "runs": len(valid),
        "gen_tps": _median(gen),
        "ttft_ms": _median(t["ttft_ms"] for t in valid),
        "cv_pct": stats.cv_pct(gen),
    }


def sessions(conn) -> list[dict]:
    """Every session, newest first, with each configuration's headline numbers."""
    rows = conn.execute(
        "SELECT id, experiment, started_at, finished_at, status, error, host_json FROM sessions ORDER BY id DESC"
    ).fetchall()
    trials_by_config: dict[int, list[dict]] = {}
    for t in conn.execute("SELECT config_id, kind, valid, gen_tps, ttft_ms FROM trials"):
        trials_by_config.setdefault(t["config_id"], []).append(dict(t))
    configs_by_session: dict[int, list[dict]] = {}
    for c in conn.execute(
        "SELECT id, session_id, idx, label, engine, status, model_path, config_hash FROM configs ORDER BY idx"
    ):
        c = dict(c)
        configs_by_session.setdefault(c["session_id"], []).append({
            "id": c["id"],
            "label": c["label"],
            "engine": c["engine"],
            "status": c["status"],
            "model": _model_name(c),
            "config_hash": c["config_hash"],
            **_headline(trials_by_config.get(c["id"], [])),
        })
    out = []
    for r in rows:
        s = db.to_dict(r)
        out.append({
            "id": s["id"],
            "experiment": s["experiment"],
            "started_at": s["started_at"],
            "finished_at": s["finished_at"],
            "status": s["status"],
            "error": s["error"],
            "gpu": ((s.get("host_json") or {}).get("gpu") or {}).get("name"),
            "configs": configs_by_session.get(s["id"], []),
        })
    return out


def session_detail(conn, session_id: int) -> dict | None:
    s = db.get_session(conn, session_id)
    if s is None:
        return None
    configs = []
    for c in db.get_configs(conn, session_id):
        trials = db.get_trials(conn, c["id"])
        valid = [t for t in trials if t["kind"] == "measured" and t["valid"]]
        summary = stats.summarize(valid)
        configs.append({
            **{k: c.get(k) for k in CONFIG_KEYS},
            "model": _model_name(c),
            "config": c.get("config_json"),
            "command": c.get("command_json"),
            "facts": c.get("facts_json") or {},
            "checks": c.get("checks_json") or [],
            "idle": c.get("idle_json") or {},
            "summary": summary,
            "trials": [{k: t.get(k) for k in TRIAL_KEYS} for t in trials],
            "has_log": bool(c.get("server_log")),
            "quality": _quality_brief(db.get_quality_run(conn, c["id"])),
        })
    return {
        "session": {
            "id": s["id"],
            "experiment": s["experiment"],
            "description": s["description"],
            "status": s["status"],
            "error": s["error"],
            "started_at": s["started_at"],
            "finished_at": s["finished_at"],
            "source_path": s["source_path"],
            "source_text": s["source_text"],
            "harness_version": s["harness_version"],
            "host": s.get("host_json") or {},
        },
        "configs": configs,
    }


def timeline(conn, config_id: int) -> dict | None:
    """Telemetry, phases and run windows for one configuration, in seconds from its start."""
    c = db.to_dict(conn.execute("SELECT * FROM configs WHERE id = ?", (config_id,)).fetchone())
    if c is None:
        return None
    trials = db.get_trials(conn, config_id)
    samples = db.get_samples(conn, config_id)
    phases = c.get("phases_json") or []
    starts = [x for x in (
        samples[0]["t"] if samples else None,
        min((t["t_send"] for t in trials if t["t_send"]), default=None),
        min((p["t0"] for p in phases), default=None),
    ) if x is not None]
    t0 = min(starts) if starts else 0.0
    first_measured = min((t["t_send"] for t in trials if t["kind"] == "measured" and t["t_send"]), default=None)
    return {
        "config_id": config_id,
        "session_id": c["session_id"],
        "label": c["label"],
        "engine": c["engine"],
        "model": _model_name(c),
        "t0": t0,
        "first_measured_s": first_measured - t0 if first_measured else None,
        "phases": [{"name": p["name"], "t0": p["t0"] - t0, "t1": p["t1"] - t0} for p in phases],
        "trials": [
            {
                "kind": t["kind"], "idx": t["idx"], "valid": t["valid"], "gen_tps": t["gen_tps"],
                "t_send": t["t_send"] - t0 if t["t_send"] else None,
                "t_first": t["t_first"] - t0 if t["t_first"] else None,
                "t_last": t["t_last"] - t0 if t["t_last"] else None,
            }
            for t in trials
        ],
        "samples": [compact_sample(s, t0) for s in samples],
    }


def compact_sample(s: dict, t0: float = 0.0) -> dict:
    """One telemetry sample, from the database or straight from the sampler's file."""
    cores = s.get("cpu_per_core_json", s.get("cpu_pct_per_core"))
    cpu = s.get("cpu_pct")
    if cpu is None and cores:
        cpu = sum(cores) / len(cores)
    return {
        "t": s["t"] - t0,
        "util": s.get("gpu_util_pct"),
        "clock": s.get("gpu_clock_mhz"),
        "power": s.get("gpu_power_instant_w") if s.get("gpu_power_instant_w") is not None else s.get("gpu_power_w"),
        "vram": s.get("vram_used_mib"),
        "temp": s.get("gpu_temp_c"),
        "cpu": cpu,
        "cores": cores,
        "server_cpu": s.get("proc_cpu_pct"),
    }


def server_log(conn, config_id: int) -> str | None:
    row = conn.execute("SELECT server_log FROM configs WHERE id = ?", (config_id,)).fetchone()
    return row["server_log"] if row else None


def latest_results(conn) -> list[dict]:
    """The most recent successful measurement of each distinct configuration."""
    rows = conn.execute(
        "SELECT c.*, s.started_at FROM configs c JOIN sessions s ON s.id = c.session_id "
        "WHERE c.status = 'ok' ORDER BY c.id DESC"
    ).fetchall()
    seen, out = set(), []
    for r in rows:
        c = db.to_dict(r)
        if c["config_hash"] in seen:
            continue
        seen.add(c["config_hash"])
        trials = db.get_trials(conn, c["id"])
        valid = [t for t in trials if t["kind"] == "measured" and t["valid"]]
        out.append({
            "config_id": c["id"],
            "config_hash": c["config_hash"],
            "session_id": c["session_id"],
            "label": c["label"],
            "engine": c["engine"],
            "model": _model_name(c),
            "started_at": c["started_at"],
            **_headline(trials),
            "thinking_tokens": _median(t["thinking_tokens"] for t in valid),
            "cost_usd": _median(t["cost_usd"] for t in valid),
            "j_per_token": _median(t["j_per_token"] for t in valid),
        })
    return out


def _clean_speed(conn, r: dict) -> dict:
    """Speed from this session if it passed its checks, or else from the latest one that did.

    A quality score doesn't depend on GPU clocks, so a slowed GPU leaves it valid; the
    speed beside it should still be one the checks trust.
    """
    if r["config_status"] != "ok":
        row = conn.execute(
            "SELECT id, session_id FROM configs WHERE config_hash = ? AND status = 'ok' ORDER BY id DESC LIMIT 1",
            (r["config_hash"],),
        ).fetchone()
        if row:
            return {**_headline(db.get_trials(conn, row["id"])), "speed_session_id": row["session_id"], "speed_clean": True}
    trials = db.get_trials(conn, r["config_id"])
    return {**_headline(trials), "speed_session_id": r["session_id"], "speed_clean": r["config_status"] == "ok"}


def _quality_row(conn, r: dict) -> dict:
    return {
        "quality_run_id": r["id"],
        "config_id": r["config_id"],
        "session_id": r["session_id"],
        "label": r["label"],
        "engine": r["engine"],
        "model": _model_name(r),
        "started_at": r["session_started_at"],
        "status": r["status"],
        "speed_status": r["config_status"],
        "suite": r["suite_name"],
        "suite_version": r["suite_version"],
        "suite_sha256": r["suite_sha256"],
        "quality_hash": r["quality_hash"],
        "config_hash": r["config_hash"],
        "settings": r.get("settings_json") or {},
        "score": r["score"],
        "score_lo": r["score_lo"],
        "score_hi": r["score_hi"],
        "consistency": r["consistency"],
        "attempts": r["attempts"],
        "correct": r["correct"],
        "seconds": r["seconds"],
        "gen_tokens": r["gen_tokens"],
        "energy_j": r["energy_j"],
        "cost_usd": r["cost_usd"],
        "cost": quality_summary.cost_per_correct(r),
        "summary": r.get("summary_json") or {},
        "checks": r.get("checks_json") or [],
        **_clean_speed(conn, r),
    }


def _quality_brief(q: dict | None) -> dict | None:
    """What a session page shows about a configuration's quality run."""
    if q is None:
        return None
    return {
        **{k: q[k] for k in ("status", "error", "score", "score_lo", "score_hi", "consistency", "attempts", "correct", "suite_name", "suite_version")},
        "categories": (q.get("summary_json") or {}).get("categories") or {},
        "checks": q.get("checks_json") or [],
        "cost": quality_summary.cost_per_correct(q),
    }


def quality_compare(conn, a: int, b: int) -> dict | None:
    """Two configurations' quality, paired by task. None unless both have a quality run."""
    qa, qb = db.get_quality_run(conn, a), db.get_quality_run(conn, b)
    if qa is None or qb is None:
        return None
    return {
        "a": _quality_brief(qa),
        "b": _quality_brief(qb),
        "same_suite": qa["suite_sha256"] == qb["suite_sha256"],
        "difference": quality_summary.compare(qa.get("summary_json") or {}, qb.get("summary_json") or {})
        if qa["suite_sha256"] == qb["suite_sha256"] else None,
    }


QUALITY_ROWS = (
    "SELECT q.*, c.session_id, c.label, c.engine, c.model_path, c.config_hash, c.status AS config_status, "
    "s.started_at AS session_started_at FROM quality_runs q JOIN configs c ON c.id = q.config_id "
    "JOIN sessions s ON s.id = c.session_id "
)


def quality_results(conn) -> list[dict]:
    """The latest quality run of each distinct configuration and suite, best first."""
    rows = conn.execute(QUALITY_ROWS + "WHERE q.status != 'failed' ORDER BY q.id DESC").fetchall()
    seen, out = set(), []
    for row in rows:
        r = db.to_dict(row)
        key = (r["config_hash"], r["quality_hash"])
        if key in seen:
            continue
        seen.add(key)
        out.append(_quality_row(conn, r))
    return sorted(out, key=lambda x: -(x["score"] or 0))


def quality_detail(conn, config_id: int) -> dict | None:
    """One configuration's quality run, with every graded answer."""
    row = conn.execute(QUALITY_ROWS + "WHERE q.config_id = ?", (config_id,)).fetchone()
    if row is None:
        return None
    r = db.to_dict(row)
    keys = ("task_id", "category", "repeat", "seed", "valid", "problems_json", "passed", "detail", "answer",
            "truncated", "gen_tokens", "ttft_ms", "total_ms", "gen_tps", "cost_usd")
    answers = [{k: a.get(k) for k in keys} for a in db.get_quality_attempts(conn, r["id"])]
    suite = yaml.safe_load(r.get("suite_text") or "") or {}
    prompts = {t["id"]: t["prompt"].strip() for t in suite.get("tasks") or []}
    return {**_quality_row(conn, r), "answers": answers, "prompts": prompts}


def overview(conn) -> dict:
    counts = conn.execute(
        "SELECT (SELECT COUNT(*) FROM sessions) AS sessions, (SELECT COUNT(*) FROM configs) AS configs, "
        "(SELECT COUNT(*) FROM trials WHERE kind = 'measured') AS runs, "
        "(SELECT COUNT(*) FROM trials WHERE kind = 'measured' AND valid = 0) AS invalid_runs"
    ).fetchone()
    checks_run = checks_failed = 0
    for (raw,) in conn.execute("SELECT checks_json FROM configs WHERE checks_json IS NOT NULL"):
        checks = json.loads(raw)
        checks_run += sum(c["passed"] is not None for c in checks)
        checks_failed += sum(c["passed"] is False for c in checks)
    return {
        "counts": {**dict(counts), "checks_run": checks_run, "checks_failed": checks_failed},
        "latest": latest_results(conn),
        "quality": {r["config_hash"]: r["score"] for r in reversed(quality_results(conn))},
    }


# ---------------------------------------------------------------- experiment files


def experiments(directory: str | Path) -> list[dict]:
    """Every experiment file in a folder, checked the same way `run --dry-run` checks it."""
    out = []
    for path in sorted(Path(directory).glob("*.yaml")):
        item: dict = {"file": path.name}
        try:
            exp = load_experiment(path)
        except ConfigError as e:
            out.append({**item, "error": str(e)})
            continue
        variants = []
        for v in exp.variants:
            c = v.config
            engine = c["engine"]["type"]
            if engine == "llama.cpp":
                detail = {
                    "model": Path(c["model"]["path"]).name,
                    "model_found": Path(c["model"]["path"]).is_file(),
                    "settings": {k: val for k, val in c["server"].items() if val not in (None, [])},
                    "n_predict": c["workload"]["n_predict"],
                }
            else:
                detail = {
                    "model": c["model"]["name"],
                    "settings": {"effort": c["engine"]["effort"], "thinking": c["engine"]["thinking"]},
                }
            q = c["quality"]
            variants.append({
                "label": v.label, "engine": engine, "config_hash": v.config_hash,
                "warmup": c["run"]["warmup"], "repeats": c["run"]["repeats"],
                "quality_requests": q["tasks"] * q["repeats"] if q["suite"] else 0, **detail,
            })
        prompt = exp.variants[0].prompt
        out.append({
            **item,
            "name": exp.name,
            "description": exp.description,
            "variants": variants,
            "requests": sum(v["warmup"] + v["repeats"] + v["quality_requests"] for v in variants),
            "prompt_chars": len(prompt),
            "prompt_preview": prompt[:600],
        })
    return out
