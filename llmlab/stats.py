"""Statistics for repeated measurements.

Repeats are described by their median with min and max; the coefficient of
variation (CV) says how much they disagree. Inter-token latency percentiles are
pooled across repeats, because one run of a few hundred tokens is too few to
estimate a p99 on its own.
"""

from __future__ import annotations

import json
import random
import statistics


def _clean(values) -> list[float]:
    return [float(v) for v in values if v is not None]


def mean(values) -> float | None:
    vals = _clean(values)
    return statistics.fmean(vals) if vals else None


def median(values) -> float | None:
    vals = _clean(values)
    return statistics.median(vals) if vals else None


def percentile(values, p: float) -> float | None:
    """Linear interpolation between closest ranks (the common default, as in numpy)."""
    vals = sorted(_clean(values))
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def cv_pct(values) -> float | None:
    """Sample standard deviation as a percentage of the mean."""
    vals = _clean(values)
    if len(vals) < 2:
        return None
    m = statistics.fmean(vals)
    return statistics.stdev(vals) / m * 100 if m else None


def pct_diff(a: float | None, b: float | None) -> float | None:
    """How far a is from b, as a percentage of b."""
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100


def bootstrap_mean_ci(
    values, level: float = 95.0, resamples: int = 10_000, seed: int = 20260
) -> tuple[float, float] | None:
    """A confidence interval for the mean, by resampling the values with replacement.

    Quality scores are means over a small, fixed set of tasks, which rules out a normal
    approximation. The seed is fixed so the same results always print the same interval.
    """
    vals = _clean(values)
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(resamples))
    tail = (100 - level) / 2
    return percentile(means, tail), percentile(means, 100 - tail)


def describe(values) -> dict | None:
    vals = _clean(values)
    if not vals:
        return None
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else None,
        "cv_pct": cv_pct(vals),
    }


# Per-trial values summarized across repeats.
SUMMARY_METRICS = (
    "gen_tps",
    "prompt_tps",
    "ttft_ms",
    "total_ms",
    "itl_p95_ms",
    "j_per_token",
    "gpu_util_mean_pct",
    "gpu_power_mean_w",
    "gpu_power_instant_mean_w",
    "gpu_clock_mean_mhz",
    "gpu_temp_max_c",
    "cpu_total_mean_pct",
    "cpu_other_mean_pct",
    "proc_cpu_mean_pct",
    "prompt_tokens",
    "gen_tokens",
    # cloud runs
    "startup_ms",
    "reported_ttft_ms",
    "first_content_ms",
    "thinking_tokens",
    "cost_usd",
)


def summarize(trials: list[dict]) -> dict:
    """Summary of a configuration's measured, valid trials.

    Each trial is a dict holding the metric keys above, plus its inter-token gaps
    in itl_json (a JSON list, or a list).
    """
    out: dict = {"n": len(trials)}
    for key in SUMMARY_METRICS:
        out[key] = describe(t.get(key) for t in trials)
    gaps: list[float] = []
    for t in trials:
        raw = t.get("itl_json")
        if raw:
            gaps.extend(json.loads(raw) if isinstance(raw, str) else raw)
    out["itl_pooled_ms"] = {
        "n": len(gaps),
        "p50": percentile(gaps, 50),
        "p95": percentile(gaps, 95),
        "p99": percentile(gaps, 99),
        "max": max(gaps) if gaps else None,
    }
    return out
