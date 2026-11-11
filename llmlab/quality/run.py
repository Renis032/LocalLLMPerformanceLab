"""Ask an engine every task in a suite, grade each answer, and store the results.

Speed runs measure raw generation. Here the model gets its normal chat format, stops
when it is done, and samples with a different seed on each repeat, because repeats that
were identical couldn't show whether a model is consistent. Attempts that didn't measure
what they should (a fallback model, a cached prompt) don't count toward the score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import yaml

from .. import cloud, claude_code, db, llamacpp, report, telemetry
from ..live import NoLiveLog, check_stop
from . import scorers, summary
from .suite import Suite, parse_suite

# Stop after this many engine errors in a row: the engine is down or a limit was reached.
MAX_CONSECUTIVE_ERRORS = 3
ANSWER_CHARS_KEPT = 20_000

# Stop reasons that mean the answer hit the token limit instead of ending by itself.
CUT_OFF = {"limit", "max_tokens", "length"}


@dataclass
class Outcome:
    """A suite in progress. Attempts accumulate, so a stopped run keeps what it finished."""

    suite: Suite
    settings: dict
    grader_problems: list[str]
    attempts: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    t0: float | None = None
    t1: float | None = None
    error: str | None = None


class LocalAnswerer:
    """llama-server, with the model's own chat format."""

    def __init__(self, server: llamacpp.LlamaServer, settings: dict, seed: int):
        self.server, self.settings, self.seed = server, settings, seed
        self.rendered: dict[str, str] = {}

    def answer(self, task, repeat: int) -> dict:
        q = self.settings
        prompt = self.rendered.get(task.id)
        if prompt is None:
            prompt = self.server.render_chat(q["system_prompt"], task.prompt, q["thinking"] == "on")
            self.rendered[task.id] = prompt
        seed = self.seed + repeat
        r = self.server.complete(
            prompt, q["max_tokens"], seed, ignore_eos=False, temperature=q["temperature"],
        )
        r["seed"], r["problems"] = seed, []
        return r


class CloudAnswerer:
    """Claude, through Claude Code. There is no seed or temperature to set."""

    def __init__(self, cmd: list[str], env: dict, work, cfg: dict):
        self.cmd, self.env, self.work, self.cfg = cmd, env, work, cfg

    def answer(self, task, repeat: int) -> dict:
        r = claude_code.complete(self.cmd, self.env, task.prompt, self.work, self.cfg["run"]["timeout_s"])
        r["seed"], r["problems"] = None, cloud.request_problems(r, self.cfg)
        return r


def start(suite: Suite, settings: dict) -> Outcome:
    return Outcome(suite, settings, scorers.verify_graders(suite))


def run_suite(outcome: Outcome, answerer, index: int, pause_s: float, echo, live=None) -> None:
    """Every task once per repeat, in order, so slow drift can't fall on one task."""
    live = live or NoLiveLog()
    suite, repeats = outcome.suite, outcome.settings["repeats"]
    total = len(suite.tasks) * repeats
    echo(f"  Quality: suite {suite.name} v{suite.version}, {len(suite.tasks)} tasks x {repeats} repeats")
    if outcome.grader_problems:
        outcome.error = "the suite's graders failed their own check: " + "; ".join(outcome.grader_problems)
        echo(f"  {outcome.error}")
        return
    live.emit("phase", name="quality")
    live.emit("quality_start", index=index, total=total, suite=suite.name)
    outcome.t0 = time.time()
    failures = 0
    try:
        for repeat in range(repeats):
            correct = 0
            for task in suite.tasks:
                check_stop(live)
                if outcome.attempts or outcome.errors:
                    time.sleep(pause_s)
                try:
                    r = answerer.answer(task, repeat)
                except (llamacpp.ServerError, claude_code.ClaudeCodeError) as e:
                    outcome.errors.append(f"{task.id}, repeat {repeat + 1}: {e}")
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_ERRORS:
                        outcome.error = f"{failures} requests in a row failed; the last said: {e}"
                        echo(f"  {outcome.error}")
                        return
                    continue
                failures = 0
                row = attempt_row(task, repeat, r)
                outcome.attempts.append(row)
                correct += row["passed"]
                live.emit(
                    "attempt", index=index, task_id=task.id, category=task.category, repeat=repeat,
                    passed=row["passed"], done=len(outcome.attempts) + len(outcome.errors), total=total,
                )
            echo(f"    repeat {repeat + 1}/{repeats}: {correct} of {len(suite.tasks)} correct")
    finally:
        outcome.t1 = time.time()


def attempt_row(task, repeat: int, r: dict) -> dict:
    truncated = r.get("stop_type") in CUT_OFF
    s = scorers.score(task, r["text"], truncated=truncated)
    return {
        "task_id": task.id,
        "category": task.category,
        "repeat": repeat,
        "seed": r["seed"],
        "valid": int(not r["problems"]),
        "problems_json": r["problems"],
        "passed": int(s.passed),
        "detail": s.detail,
        "signature": s.signature,
        "answer": r["text"][:ANSWER_CHARS_KEPT],
        "truncated": int(truncated),
        "stop_type": r.get("stop_type"),
        "prompt_tokens": r.get("prompt_tokens"),
        "gen_tokens": r.get("gen_tokens"),
        "thinking_tokens": r.get("thinking_tokens"),
        "ttft_ms": r.get("ttft_ms"),
        "total_ms": r.get("total_ms"),
        "gen_tps": r.get("gen_tps"),
        "t_send": r.get("t_send"),
        "t_last": r.get("t_last"),
        "cost_usd": r.get("cost_usd"),
    }


def checks(outcome: Outcome, attempts: list[dict]) -> list[dict]:
    out = []
    n = len(outcome.suite.tasks)
    expected = n * outcome.settings["repeats"]
    out.append({
        "check": "Graders verified on known answers",
        "passed": not outcome.grader_problems,
        "detail": (
            f"all {n} tasks accepted their reference answer and rejected their wrong one"
            if not outcome.grader_problems else "; ".join(outcome.grader_problems)
        ),
    })
    out.append({
        "check": "Every task answered",
        "passed": len(attempts) == expected,
        "detail": (
            f"{len(attempts)} of {expected} answers received"
            + (f"; first error: {outcome.errors[0]}" if outcome.errors else "")
        ),
    })
    cut = sum(a["truncated"] for a in attempts)
    out.append({
        "check": "Answers ended by themselves",
        "passed": cut == 0,
        "detail": (
            "no answer reached the token limit" if cut == 0
            else "1 answer was cut off and counts as wrong" if cut == 1
            else f"{cut} answers were cut off and count as wrong"
        ),
    })
    invalid = [a for a in attempts if not a["valid"]]
    out.append({
        "check": "Each answer measured once",
        "passed": not invalid,
        "detail": (
            "every answer came from the requested model in one uncached request" if not invalid
            else f"{len(invalid)} answers don't count; for example: {'; '.join(invalid[0]['problems_json'])}"
        ),
    })
    return out


def regrade(conn, config_id: int) -> tuple[dict, dict, int]:
    """Grade a run's stored answers again, with today's graders and the suite as it was.

    Answers are kept for this: fixing a grader needs no new requests. Returns the run
    before and after, and how many answers changed.
    """
    before = db.get_quality_run(conn, config_id)
    suite = parse_suite(yaml.safe_load(before["suite_text"]), before["suite_text"])
    problems = scorers.verify_graders(suite)
    if problems:
        raise ValueError("the graders failed their own check: " + "; ".join(problems))
    tasks = {t.id: t for t in suite.tasks}
    attempts = db.get_quality_attempts(conn, before["id"])
    changed = 0
    for a in attempts:
        s = scorers.score(tasks[a["task_id"]], a["answer"], truncated=bool(a["truncated"]))
        new = {"passed": int(s.passed), "detail": s.detail, "signature": s.signature}
        if any(a[k] != v for k, v in new.items()):
            changed += a["passed"] != new["passed"]
            db.update(conn, "quality_attempts", a["id"], new)
            a.update(new)
    s = {**summary.summarize(attempts), "regraded_at": time.time()}
    db.update(conn, "quality_runs", before["id"], {
        "score": s["score"], "score_lo": s["ci"][0] if s["ci"] else None, "score_hi": s["ci"][1] if s["ci"] else None,
        "consistency": s["consistency"], "attempts": s["attempts"], "correct": s["correct"], "summary_json": s,
    })
    conn.commit()
    return before, db.get_quality_run(conn, config_id), changed


def finish(conn, config_id: int, outcome: Outcome, samples: list[dict], echo) -> dict:
    """Score the attempts, store everything, and print the result."""
    attempts = outcome.attempts
    s = summary.summarize(attempts)
    results = checks(outcome, attempts)
    if outcome.error or s["score"] is None:
        status = "failed"
    else:
        status = "ok" if all(c["passed"] for c in results) else "warning"
    costs = [a["cost_usd"] for a in attempts if a["cost_usd"] is not None]
    energy = telemetry.energy_j(samples, outcome.t0, outcome.t1) if outcome.t0 and samples else None
    row = {
        "config_id": config_id,
        "suite_name": outcome.suite.name,
        "suite_version": outcome.suite.version,
        "suite_sha256": outcome.suite.sha256,
        "suite_text": outcome.suite.text,
        "settings_json": outcome.settings,
        "quality_hash": summary.quality_hash(outcome.suite.sha256, outcome.settings),
        "status": status,
        "error": outcome.error,
        "score": s["score"],
        "score_lo": s["ci"][0] if s["ci"] else None,
        "score_hi": s["ci"][1] if s["ci"] else None,
        "consistency": s["consistency"],
        "attempts": s["attempts"],
        "correct": s["correct"],
        "summary_json": s,
        "checks_json": results,
        "gen_tokens": sum(a["gen_tokens"] or 0 for a in attempts),
        "cost_usd": sum(costs) if costs else None,
        "energy_j": energy,
        "seconds": outcome.t1 - outcome.t0 if outcome.t0 and outcome.t1 else None,
        "started_at": outcome.t0,
        "finished_at": outcome.t1,
    }
    run_id = db.insert(conn, "quality_runs", row)
    db.insert_many(conn, "quality_attempts", [{"quality_run_id": run_id, **a} for a in attempts])
    conn.commit()
    stored = db.get_quality_run(conn, config_id)
    for line in report.quality_lines(stored):
        echo(line)
    return stored
