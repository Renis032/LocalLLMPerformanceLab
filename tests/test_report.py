"""Purpose: `compare`, which decides whether two sessions agree, gives the right verdict.

The comparison is the harness's reproducibility test, so it must pass within the
tolerance, fail outside it, explain differences caused by runs with warnings, and
refuse to give a verdict when nothing is comparable.
"""

from llmlab import db, report


def _session_with(conn, gen_values, prompt_tps=5000.0, config_hash="h1", label="gpu_layers=all"):
    sid = db.insert(conn, "sessions", {"experiment": "e", "started_at": "2026-11-07T12:00:00+01:00",
                                       "status": "completed"})
    cid = db.insert(conn, "configs", {"session_id": sid, "idx": 0, "label": label, "config_hash": config_hash,
                                      "config_json": {}, "status": "ok", "engine_version": "b1",
                                      "model_sha256": "abc"})
    db.insert_many(conn, "trials", [
        {"config_id": cid, "kind": "measured", "idx": i, "valid": 1, "gen_tps": g, "prompt_tps": prompt_tps,
         "ttft_ms": 97.0, "itl_json": [6.5, 6.6, 7.0]}
        for i, g in enumerate(gen_values)
    ])
    conn.commit()
    return sid


def test_compare_passes_within_tolerance_and_fails_outside():
    conn = db.connect(":memory:")
    a = _session_with(conn, [150.0, 151.0, 149.0])
    b = _session_with(conn, [152.0, 153.0, 151.5])  # +1.3%
    c = _session_with(conn, [135.0, 134.0, 136.0])  # -10%
    text, verdict = report.compare_text(conn, a, b, 5.0)
    assert verdict is True and "+1.3%" in text and "PASS" in text
    text, verdict = report.compare_text(conn, a, c, 5.0)
    assert verdict is False and "FAIL" in text


def test_compare_explains_differences_from_runs_with_warnings():
    conn = db.connect(":memory:")
    a = _session_with(conn, [150.0, 151.0])
    b = _session_with(conn, [135.0, 136.0])
    db.update(conn, "configs", 2, {"status": "warning", "checks_json": [
        {"check": "CPU quiet during runs", "passed": False, "detail": "18.9%"}]})
    text, verdict = report.compare_text(conn, a, b, 5.0)
    assert verdict is False
    assert "session 2 measured this with warnings: CPU quiet during runs failed" in text


def test_compare_with_nothing_in_common_has_no_verdict():
    conn = db.connect(":memory:")
    a = _session_with(conn, [150.0], config_hash="h1")
    b = _session_with(conn, [150.0], config_hash="h2", label="ctx=8192")
    text, verdict = report.compare_text(conn, a, b, 5.0)
    assert verdict is None
    assert "Only in session" in text and "nothing to compare" in text


def test_fmt_says_na_for_missing_values():
    assert report.fmt(None) == "N/A"
    assert report.fmt(5450.4, 0) == "5,450"
    assert report.fmt(1.13456, 3) == "1.135"
