"""From graded answers to a score, with how much to trust it.

A task's score is the share of its repeats answered correctly. The suite's score is the
mean over tasks, and its uncertainty comes from resampling tasks, because the tasks are
the sample: a different 24 would give a somewhat different score. Consistency is how
often repeats of a task say the same thing, right or wrong.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

from .. import stats
from .suite import CATEGORIES

# GPU electricity is reported as money so it can sit beside API prices. An estimate:
# GPU board power only, at this price. Set LLMLAB_KWH_USD to use your own.
DEFAULT_KWH_USD = 0.20
J_PER_KWH = 3_600_000


def kwh_price() -> float:
    try:
        return float(os.environ.get("LLMLAB_KWH_USD", DEFAULT_KWH_USD))
    except ValueError:
        return DEFAULT_KWH_USD


def quality_hash(suite_sha256: str, settings: dict) -> str:
    """Identity of what was scored. Repeats change how carefully it was scored, not what."""
    identity = {"suite": suite_sha256, **{k: v for k, v in settings.items() if k != "repeats"}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def summarize(attempts: list[dict]) -> dict:
    """Score, uncertainty, consistency and per-category results from valid attempts."""
    by_task: dict[str, list[dict]] = {}
    for a in attempts:
        if a["valid"]:
            by_task.setdefault(a["task_id"], []).append(a)
    tasks = {}
    for tid, rows in by_task.items():
        signatures = Counter(r["signature"] for r in rows)
        tasks[tid] = {
            "category": rows[0]["category"],
            "n": len(rows),
            "correct": sum(r["passed"] for r in rows),
            "rate": sum(r["passed"] for r in rows) / len(rows),
            "consistency": signatures.most_common(1)[0][1] / len(rows),
        }
    if not tasks:
        return {"score": None, "ci": None, "consistency": None, "attempts": 0, "correct": 0, "categories": {}, "tasks": {}}
    rates = [t["rate"] for t in tasks.values()]
    categories = {}
    for c in CATEGORIES:
        rows = [t for t in tasks.values() if t["category"] == c]
        if rows:
            categories[c] = {
                "tasks": len(rows),
                "score": stats.mean(t["rate"] for t in rows),
                "correct": sum(t["correct"] for t in rows),
                "attempts": sum(t["n"] for t in rows),
            }
    ci = stats.bootstrap_mean_ci(rates)
    return {
        "score": stats.mean(rates),
        "ci": list(ci),
        "consistency": stats.mean(t["consistency"] for t in tasks.values()),
        "attempts": sum(t["n"] for t in tasks.values()),
        "correct": sum(t["correct"] for t in tasks.values()),
        "categories": categories,
        "tasks": tasks,
    }


def compare(a: dict, b: dict) -> dict | None:
    """Is B's score different from A's, beyond what the task sample can show?

    Pairs the two by task, then resamples tasks. Returns None unless they share tasks.
    """
    ta, tb = a.get("tasks") or {}, b.get("tasks") or {}
    shared = [t for t in ta if t in tb]
    if not shared:
        return None
    diffs = [tb[t]["rate"] - ta[t]["rate"] for t in shared]
    lo, hi = stats.bootstrap_mean_ci(diffs)
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "no detectable difference"
    return {"tasks": len(shared), "diff": stats.mean(diffs), "lo": lo, "hi": hi, "verdict": verdict}


def cost_per_correct(run: dict) -> dict:
    """What a correct answer costs: API list price for the cloud, GPU electricity locally."""
    correct = run.get("correct") or 0
    if run.get("cost_usd") is not None:
        total, kind = run["cost_usd"], "list"
    elif run.get("energy_j") is not None:
        total, kind = run["energy_j"] / J_PER_KWH * kwh_price(), "electricity"
    else:
        return {"usd": None, "kind": None}
    return {"usd": total / correct if correct else None, "kind": kind}
