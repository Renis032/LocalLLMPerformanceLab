"""Purpose: the statistics in every report are computed correctly.

Medians, percentiles and run-to-run variation are what results are judged on, so
they're checked against hand-computed values. A missing value must stay missing (N/A),
never turn into zero.
"""

import json

import pytest

from llmlab import stats


def test_percentile_interpolates():
    assert stats.percentile([1, 2, 3, 4], 50) == 2.5
    assert stats.percentile([1, 2, 3, 4], 0) == 1
    assert stats.percentile([1, 2, 3, 4], 100) == 4
    assert stats.percentile([10, 20], 95) == pytest.approx(19.5)
    assert stats.percentile([], 50) is None
    assert stats.percentile([None, 5], 50) == 5


def test_describe_and_cv():
    d = stats.describe([100.0, 102.0, 98.0])
    assert (d["n"], d["median"], d["min"], d["max"]) == (3, 100.0, 98.0, 102.0)
    assert d["cv_pct"] == pytest.approx(2.0)
    one = stats.describe([5])
    assert one["stdev"] is None and one["cv_pct"] is None
    assert stats.describe([None]) is None


def test_pct_diff():
    assert stats.pct_diff(105, 100) == pytest.approx(5.0)
    assert stats.pct_diff(1, 0) is None
    assert stats.pct_diff(None, 1) is None


def test_summarize_pools_inter_token_gaps():
    trials = [
        {"gen_tps": 150.0, "itl_json": json.dumps([6.0, 7.0])},
        {"gen_tps": 152.0, "itl_json": [6.5, 30.0]},
    ]
    s = stats.summarize(trials)
    assert s["n"] == 2
    assert s["gen_tps"]["median"] == 151.0
    assert s["itl_pooled_ms"]["n"] == 4
    assert s["itl_pooled_ms"]["max"] == 30.0
    assert s["itl_pooled_ms"]["p50"] == pytest.approx(6.75)
    assert s["prompt_tps"] is None  # missing everywhere: N/A, not zero
