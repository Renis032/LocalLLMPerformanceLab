"""Text reports for the terminal: progress, configuration summaries, sessions, comparisons.

Missing values print as N/A, never as zero. Cloud configurations (Claude) have no
prompt speed, GPU or memory figures; they report tokens, latency and cost instead.
"""

from __future__ import annotations

from . import db, stats
from .quality import summary as quality_summary


def fmt(v, digits: int = 1) -> str:
    if v is None:
        return "N/A"
    if digits == 0:
        return f"{round(v):,}"
    return f"{v:,.{digits}f}"


def _signed(pct: float | None) -> str:
    return "N/A" if pct is None else f"{pct:+.1f}%"


def _med(d: dict | None) -> float | None:
    return d["median"] if d else None


def _max(d: dict | None) -> float | None:
    return d["max"] if d else None


def trial_name(t: dict, repeats: int | None = None) -> str:
    if t["kind"] == "warmup":
        return f"warm-up {t['idx'] + 1}"
    return f"run {t['idx'] + 1}" + (f"/{repeats}" if repeats else "")


def trial_progress(r: dict, repeats: int) -> str:
    return (
        f"  {trial_name(r, repeats):<11}gen {fmt(r['gen_tps'])} tok/s   "
        f"prompt {fmt(r['prompt_tps'], 0)} tok/s   TTFT {fmt(r['ttft_ms'])} ms"
    )


def cloud_trial_progress(r: dict, repeats: int) -> str:
    return (
        f"  {trial_name(r, repeats):<11}gen {fmt(r['gen_tps'])} tok/s   "
        f"TTFT {fmt(r['ttft_ms'], 0)} ms   {fmt(r['gen_tokens'], 0)} tokens out"
    )


def _is_cloud(engine: str | None) -> bool:
    return engine == "claude-code"


def _money(v: float | None) -> str:
    return "N/A" if v is None else f"${v:.4f}"


# ---------------------------------------------------------------- one configuration


def result_from_db(c: dict, trials: list[dict]) -> dict:
    """The same shape runner._finish returns, rebuilt from stored rows."""
    valid = [t for t in trials if t["kind"] == "measured" and t["valid"]]
    idle = c.get("idle_json") or {}
    return {
        "config_id": c["id"],
        "label": c["label"],
        "config_hash": c["config_hash"],
        "engine": c.get("engine"),
        "status": c["status"],
        "error": c["error"],
        "load_s": c["load_s"],
        "prompt_tokens": c["prompt_tokens"],
        "facts": c.get("facts_json") or {},
        "checks": c.get("checks_json") or [],
        "trials": trials,
        "summary": stats.summarize(valid),
        "vram_before_mib": idle.get("vram_used_mib", c.get("vram_before_mib")),
        "vram_peak_mib": c["vram_peak_mib"],
        "server_rss_max_mib": c["server_rss_max_mib"],
        "server_private_max_mib": c["server_private_max_mib"],
    }


def _generation_line(g: dict) -> str:
    return (
        f"  Generation  {fmt(g['median'], 2)} tok/s median (range {fmt(g['min'], 2)} to {fmt(g['max'], 2)}, "
        f"CV {fmt(g['cv_pct'], 2)}%, {g['n']} runs)"
    )


def _failed_lines(res: dict) -> list[str]:
    if res["status"] != "failed":
        return []
    return ["  FAILED: " + (res.get("error") or "unknown error").replace("\n", "\n          ")]


def _check_lines(res: dict) -> list[str]:
    lines: list[str] = []
    checks = res.get("checks") or []
    for c in checks:
        if c["passed"] is False:
            lines.append(f"  [FAIL] {c['check']}: {c['detail']}")
    for c in checks:
        if c["passed"] is None:
            lines.append(f"  [N/A]  {c['check']}: {c['detail']}")
    for t in res.get("trials") or []:
        if not t["valid"]:
            lines.append(f"  [INVALID] {trial_name(t)}: {'; '.join(t['problems_json'])}")
    if checks:
        passed = sum(c["passed"] is True for c in checks)
        lines.append(f"  Checks      {passed} of {len(checks)} passed; status {res['status']}")
    return lines


def cloud_config_lines(res: dict) -> list[str]:
    lines = _failed_lines(res)
    s, f = res["summary"], res.get("facts") or {}
    if s and s["n"]:
        itl = s["itl_pooled_ms"]
        lines.append(_generation_line(s["gen_tps"]))
        lines.append(
            f"  Latency     TTFT {fmt(_med(s['ttft_ms']), 0)} ms to the first visible text (median); first content, "
            f"thinking included, after {fmt(_med(s['first_content_ms']), 0)} ms (Claude Code measured "
            f"{fmt(_med(s['reported_ttft_ms']), 0)} ms)"
        )
        lines.append(
            f"  Stream      text arrives in chunks every {fmt(itl['p50'], 0)} ms (p50), {fmt(itl['p95'], 0)} ms (p95); "
            f"Claude Code took {fmt(_med(s['startup_ms']), 0)} ms to start before each request"
        )
        lines.append(
            f"  Tokens      {fmt(_med(s['prompt_tokens']), 0)} in, {fmt(_med(s['gen_tokens']), 0)} out per run "
            f"(median), {fmt(_med(s['thinking_tokens']), 0)} of them hidden thinking"
        )
        lines.append(
            f"  Cost        {_money(_med(s['cost_usd']))} per run at API list prices; "
            f"signed in with: {f.get('auth') or 'N/A'}"
        )
    return lines + _check_lines(res)


def config_lines(res: dict) -> list[str]:
    if _is_cloud(res.get("engine")):
        return cloud_config_lines(res)
    lines = _failed_lines(res)
    s, f = res["summary"], res.get("facts") or {}
    if s and s["n"]:
        itl = s["itl_pooled_ms"]
        lines.append(_generation_line(s["gen_tps"]))
        lines.append(
            f"  Prompt      {fmt(_med(s['prompt_tps']), 0)} tok/s   TTFT {fmt(_med(s['ttft_ms']))} ms   "
            f"inter-token p50/p95/p99 {fmt(itl['p50'], 2)}/{fmt(itl['p95'], 2)}/{fmt(itl['p99'], 2)} ms"
        )
        lines.append(
            f"  GPU         {fmt(_med(s['gpu_util_mean_pct']), 0)}% busy, "
            f"{fmt(_med(s['gpu_power_instant_mean_w']), 0)} W, {fmt(_med(s['gpu_clock_mean_mhz']), 0)} MHz, "
            f"{fmt(_max(s['gpu_temp_max_c']), 0)} C max; {fmt(_med(s['j_per_token']), 3)} J/token (GPU board only)"
        )
        lines.append(
            f"  CPU         server {fmt(_med(s['proc_cpu_mean_pct']), 0)}% (100% = one core); "
            f"other programs {fmt(_med(s['cpu_other_mean_pct']), 1)}% of the whole CPU"
        )
    before, peak = res.get("vram_before_mib"), res.get("vram_peak_mib")
    if peak is not None:
        added = peak - before if before is not None else None
        parts = [f"VRAM {fmt(before, 0)} -> {fmt(peak, 0)} MiB (+{fmt(added, 0)})"]
        if f.get("gpu_model_mib") is not None:
            parts.append(
                f"the server lists model {fmt(f['gpu_model_mib'], 0)} + KV {fmt(f.get('gpu_kv_mib'), 0)} "
                f"+ compute {fmt(f.get('gpu_compute_mib'), 0)} MiB"
            )
        parts.append(f"server RAM {fmt(res.get('server_rss_max_mib'), 0)} MiB")
        lines.append("  Memory      " + "; ".join(parts))
    if f.get("file_type") or res.get("load_s") is not None:
        off, total = f.get("offloaded_layers"), f.get("total_layers")
        layers = f"{off}/{total} layers on the GPU" if off is not None else "layers on the GPU: N/A"
        lines.append(
            f"  Model       {f.get('file_type') or 'N/A'}, {fmt(f.get('bpw'), 2)} bits per weight, "
            f"{f.get('model_params') or 'N/A'} parameters; {layers}; loaded in {fmt(res.get('load_s'), 2)} s"
        )
    return lines + _check_lines(res)


def trial_table(trials: list[dict], engine: str | None = "llama.cpp") -> list[str]:
    if _is_cloud(engine):
        return cloud_trial_table(trials)
    head = (
        f"  {'':<11}{'gen tok/s':>10}{'client':>9}{'prompt tok/s':>14}{'TTFT ms':>9}"
        f"{'ITL p95':>9}{'W':>6}{'J/token':>9}  valid"
    )
    lines = [head]
    for t in trials:
        lines.append(
            f"  {trial_name(t):<11}{fmt(t['gen_tps'], 2):>10}{fmt(t['client_gen_tps'], 2):>9}"
            f"{fmt(t['prompt_tps'], 0):>14}{fmt(t['ttft_ms']):>9}{fmt(t['itl_p95_ms'], 2):>9}"
            f"{fmt(t['gpu_power_instant_mean_w'], 0):>6}{fmt(t['j_per_token'], 3):>9}  "
            + ("yes" if t["valid"] else "no")
        )
    return lines


def cloud_trial_table(trials: list[dict]) -> list[str]:
    head = (
        f"  {'':<11}{'gen tok/s':>10}{'TTFT ms':>9}{'content':>9}{'its own':>9}{'tokens in':>11}{'out':>7}"
        f"{'thinking':>10}{'requests':>10}  valid"
    )
    lines = [head]
    for t in trials:
        lines.append(
            f"  {trial_name(t):<11}{fmt(t['gen_tps'], 2):>10}{fmt(t['ttft_ms'], 0):>9}"
            f"{fmt(t.get('first_content_ms'), 0):>9}{fmt(t['reported_ttft_ms'], 0):>9}"
            f"{fmt(t['prompt_tokens'], 0):>11}{fmt(t['gen_tokens'], 0):>7}"
            f"{fmt(t['thinking_tokens'], 0):>10}{fmt(t['api_calls'], 0):>10}  "
            + ("yes" if t["valid"] else "no")
        )
    return lines


# ---------------------------------------------------------------- quality


def pct(v: float | None, digits: int = 0) -> str:
    return "N/A" if v is None else f"{v * 100:.{digits}f}%"


def cost_text(c: dict) -> str:
    """For example '$0.0123 at API list prices'."""
    if c["usd"] is None:
        return "N/A"
    what = "at API list prices" if c["kind"] == "list" else "of GPU electricity (estimate)"
    return f"${c['usd']:.5f} {what}"


def quality_lines(q: dict | None) -> list[str]:
    if q is None:
        return []
    s = q.get("summary_json") or {}
    lines = []
    if q["error"]:
        lines.append(f"  Quality     FAILED: {q['error']}")
    if q["score"] is not None:
        lines.append(
            f"  Quality     {pct(q['score'])} (95% interval {pct(q['score_lo'])} to {pct(q['score_hi'])}), "
            f"{q['correct']} of {q['attempts']} answers correct; consistency {pct(q['consistency'])}; "
            f"suite {q['suite_name']} v{q['suite_version']}" + (", regraded" if s.get("regraded_at") else "")
        )
        cats = "   ".join(f"{name} {pct(c['score'])}" for name, c in (s.get("categories") or {}).items())
        lines.append(f"              {cats}")
        lines.append(f"              a correct answer costs {cost_text(quality_summary.cost_per_correct(q))}")
    for c in q.get("checks_json") or []:
        if not c["passed"]:
            lines.append(f"  [FAIL] {c['check']}: {c['detail']}")
    return lines


def quality_compare_lines(qa: dict, qb: dict, a_id: int, b_id: int) -> list[str]:
    head = f"  Quality     {pct(qa['score'])} in session {a_id}, {pct(qb['score'])} in session {b_id}"
    if qa["suite_sha256"] != qb["suite_sha256"]:
        return [head, "  note: the sessions used different versions of the suite, so scores don't compare"]
    d = quality_summary.compare(qa.get("summary_json") or {}, qb.get("summary_json") or {})
    if d is None:
        return [head]
    lines = [
        f"{head}: {d['diff'] * 100:+.1f} points (95% interval {d['lo'] * 100:+.1f} to {d['hi'] * 100:+.1f}), "
        f"{d['verdict']}"
    ]
    if qa["quality_hash"] != qb["quality_hash"]:
        lines.append("  note: the quality settings differ between the sessions")
    return lines


def quality_table(rows: list[dict]) -> str:
    """Speed, quality and cost side by side, one line per configuration."""
    if not rows:
        return "No quality runs recorded yet. Add a quality: section to an experiment file."
    lines = [f"{'configuration':<36}{'quality':>8}{'95% interval':>14}{'consistent':>12}{'gen tok/s':>11}  cost per correct answer"]
    for r in rows:
        model = r["model"].removesuffix(".gguf")
        name = model if r["label"] in r["model"] else f"{model}, {r['label']}"
        speed = fmt(r["gen_tps"], 1) + ("" if r["speed_clean"] else "*")
        lines.append(
            f"{name[:35]:<36}{pct(r['score']):>8}{pct(r['score_lo']) + ' to ' + pct(r['score_hi']):>14}"
            f"{pct(r['consistency']):>12}{speed:>11}  {cost_text(r['cost'])}"
        )
    if not all(r["speed_clean"] for r in rows):
        lines.append("* no speed run of this configuration has passed its checks yet")
    return "\n".join(lines)


# ---------------------------------------------------------------- sessions


def _gpu_name(host: dict | None) -> str:
    gpu = (host or {}).get("gpu") or {}
    return gpu.get("name") or "no GPU telemetry"


def sessions_text(rows: list[dict]) -> str:
    if not rows:
        return "No sessions recorded yet."
    lines = [f"{'id':>4}  {'started':<17}{'experiment':<26}{'status':<11}{'configs':<22}GPU"]
    for r in rows:
        counts = [f"{r[k] or 0} {name}" for k, name in (("n_ok", "ok"), ("n_warning", "warning"), ("n_failed", "failed")) if r[k]]
        lines.append(
            f"{r['id']:>4}  {r['started_at'][:16].replace('T', ' '):<17}{r['experiment'][:25]:<26}"
            f"{r['status']:<11}{', '.join(counts) or '0':<22}{_gpu_name(r.get('host_json'))}"
        )
    return "\n".join(lines)


ENGINE_NAMES = {"llama.cpp": "llama.cpp", "claude-code": "Claude Code"}


def session_text(conn, session_id: int) -> str:
    s = db.get_session(conn, session_id)
    if s is None:
        raise LookupError(f"there is no session {session_id}")
    host = s.get("host_json") or {}
    gpu = host.get("gpu") or {}
    configs = db.get_configs(conn, session_id)
    engines: dict[str, set] = {}
    for c in configs:
        name = ENGINE_NAMES.get(c["engine"], c["engine"] or "N/A")
        engines.setdefault(name, set())
        if c["engine_version"]:
            engines[name].add(c["engine_version"])
    engine_text = "; ".join(f"{name} {', '.join(sorted(v)) or 'N/A'}" for name, v in engines.items())
    lines = [
        f"Session {s['id']}: {s['experiment']} ({s['status']})",
        f"  Started   {s['started_at']}, finished {s['finished_at'] or 'N/A'}",
        f"  GPU       {gpu.get('name', 'N/A')}, driver {gpu.get('driver', 'N/A')}, CUDA {gpu.get('cuda_driver', 'N/A')}, "
        f"power limit {fmt(gpu.get('power_limit_w'), 0)} W",
        f"  CPU       {host.get('cpu', 'N/A')} ({host.get('cpu_cores_physical')} cores, "
        f"{host.get('cpu_cores_logical')} threads), {fmt((host.get('ram_total_mib') or 0) / 1024, 0)} GiB RAM, "
        f"power plan {host.get('power_plan') or 'N/A'}",
        f"  Engine    {engine_text or 'N/A'}",
        f"  File      {s['source_path'] or 'N/A'} (sha256 {(s['source_sha256'] or '')[:12]})",
    ]
    if s["error"]:
        lines.append(f"  Error     {s['error']}")
    for c in configs:
        trials = db.get_trials(conn, c["id"])
        lines.append("")
        lines.append(f"[{c['idx'] + 1}] {c['label']}  ({c['status']}, config {c['config_hash']})")
        if trials:
            lines += trial_table(trials, c["engine"])
        lines += config_lines(result_from_db(c, trials))
        lines += quality_lines(db.get_quality_run(conn, c["id"]))
    return "\n".join(lines)


def session_json(conn, session_id: int) -> dict:
    s = db.get_session(conn, session_id)
    if s is None:
        raise LookupError(f"there is no session {session_id}")
    out = {"session": s, "configs": []}
    for c in db.get_configs(conn, session_id):
        trials = db.get_trials(conn, c["id"])
        out["configs"].append({
            "config": c,
            "trials": trials,
            "summary": result_from_db(c, trials)["summary"],
            "quality": db.get_quality_run(conn, c["id"]),
        })
    return out


# ---------------------------------------------------------------- comparison

# (name, how to read it from a summary, decimals, counts toward pass/fail)
COMPARED = (
    ("Generation tok/s", lambda s: _med(s["gen_tps"]), 2, True),
    ("Prompt tok/s", lambda s: _med(s["prompt_tps"]), 0, True),
    ("TTFT ms", lambda s: _med(s["ttft_ms"]), 1, False),
    ("Inter-token p95 ms", lambda s: s["itl_pooled_ms"]["p95"], 2, False),
    ("GPU W", lambda s: _med(s["gpu_power_instant_mean_w"]), 0, False),
    ("J/token", lambda s: _med(s["j_per_token"]), 3, False),
)


def compare_text(conn, a_id: int, b_id: int, tolerance_pct: float) -> tuple[str, bool | None]:
    """Compare configurations present in both sessions. Returns the text and the verdict:
    True if every matched configuration is within tolerance, False if one isn't, None if
    nothing matched."""
    sessions = {}
    for sid in (a_id, b_id):
        s = db.get_session(conn, sid)
        if s is None:
            raise LookupError(f"there is no session {sid}")
        sessions[sid] = s
    usable = {}
    skipped = []
    for sid in (a_id, b_id):
        usable[sid] = {}
        for c in db.get_configs(conn, sid):
            if c["status"] == "failed":
                skipped.append(f"session {sid}: {c['label']} failed")
            else:
                usable[sid][c["config_hash"]] = c

    sa, sb = sessions[a_id], sessions[b_id]
    lines = [
        f"Session {a_id} ({sa['experiment']}, {sa['started_at'][:16].replace('T', ' ')}) vs "
        f"session {b_id} ({sb['experiment']}, {sb['started_at'][:16].replace('T', ' ')})",
        f"Pass: median generation and prompt speed within {tolerance_pct:g}% for every configuration in both "
        "(prompt speed only where it is measured: not for cloud models).",
    ]
    matched = [h for h in usable[a_id] if h in usable[b_id]]
    verdict = True
    for h in matched:
        ca, cb = usable[a_id][h], usable[b_id][h]
        ra = result_from_db(ca, db.get_trials(conn, ca["id"]))["summary"]
        rb = result_from_db(cb, db.get_trials(conn, cb["id"]))["summary"]
        lines.append("")
        lines.append(f"{ca['label']}  (config {h})")
        lines.append(f"  {'':<22}{'session ' + str(a_id):>14}{'session ' + str(b_id):>14}{'change':>9}")
        if not ra["n"] or not rb["n"]:
            lines.append("  no valid measured runs in one of the sessions")
            verdict = False
            continue
        for name, read, digits, gate in COMPARED:
            va, vb = read(ra), read(rb)
            d = stats.pct_diff(vb, va)
            mark = ""
            if gate and va is None and vb is None:
                mark = "N/A"  # not measured for this engine
            elif gate:
                ok = d is not None and abs(d) <= tolerance_pct
                mark = "PASS" if ok else "FAIL"
                verdict = verdict and ok
            lines.append(f"  {name:<22}{fmt(va, digits):>14}{fmt(vb, digits):>14}{_signed(d):>9}  {mark}")
        cv_a, cv_b = ra["gen_tps"]["cv_pct"], rb["gen_tps"]["cv_pct"]
        lines.append(f"  {'Generation CV %':<22}{fmt(cv_a, 2):>14}{fmt(cv_b, 2):>14}")
        qa, qb = db.get_quality_run(conn, ca["id"]), db.get_quality_run(conn, cb["id"])
        if qa and qb:
            lines += quality_compare_lines(qa, qb, a_id, b_id)
        for field, what in (("engine_version", "engine build"), ("model_sha256", "model file")):
            if ca[field] != cb[field]:
                lines.append(f"  note: the {what} differs ({ca[field]} vs {cb[field]})")
        for sid, c in ((a_id, ca), (b_id, cb)):
            if c["status"] != "ok":
                failed = [x["check"] for x in (c.get("checks_json") or []) if x["passed"] is False]
                why = f": {', '.join(failed)} failed" if failed else ""
                lines.append(f"  note: session {sid} measured this with warnings{why}, so a difference is expected")

    for sid, other in ((a_id, b_id), (b_id, a_id)):
        only = [c["label"] for h, c in usable[sid].items() if h not in usable[other]]
        if only:
            lines.append(f"\nOnly in session {sid}: {', '.join(only)}")
    if skipped:
        lines.append("\nLeft out: " + "; ".join(skipped))
    lines.append("")
    if not matched:
        lines.append("No configuration appears in both sessions, so there is nothing to compare.")
        return "\n".join(lines), None
    word = "PASS" if verdict else "FAIL"
    lines.append(f"{word}: {len(matched)} configuration(s) compared, tolerance {tolerance_pct:g}%.")
    return "\n".join(lines), verdict
