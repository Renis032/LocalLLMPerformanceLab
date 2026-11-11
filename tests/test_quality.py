"""Purpose: a quality score means what it claims.

Covers the real task suite (each grader accepts a right answer and rejects a wrong one,
and the answer key is checked independently), what each grader forgives and what it
doesn't, the guards around generated code, the statistics behind the score, and whole
runs against the stand-in engines, including how failures are recorded.
"""

import datetime
import itertools
from pathlib import Path

import pytest

from llmlab import config, db, llamacpp, report, stats
from llmlab.dashboard import api
from llmlab.live import LiveLog, StopRequested, read_events
from llmlab.quality import sandbox, scorers, summary
from llmlab.quality.suite import SuiteError, load_suite, parse_suite
from llmlab.runner import run_experiment

from conftest import FakeGpu, make_cloud_experiment, make_experiment

CORE = Path(__file__).resolve().parents[1] / "evals" / "core.yaml"
SUITE = load_suite(CORE)
TASKS = {t.id: t for t in SUITE.tasks}


def _quiet(*args):
    pass


# ---------------------------------------------------------------- the real suite


@pytest.mark.parametrize("task", SUITE.tasks, ids=lambda t: t.id)
def test_every_grader_accepts_the_right_answer_and_rejects_the_wrong_one(task):
    assert scorers.score(task, task.reference).passed
    assert not scorers.score(task, task.wrong).passed


def test_the_suite_is_balanced_across_categories():
    counts = {c: sum(t.category == c for t in SUITE.tasks) for c in ("exact", "json", "code", "reasoning")}
    assert set(counts.values()) == {6}


def test_the_answer_key_is_right():
    # Worked out here independently of the answers written in the suite.
    numbers = [14, 22, 31, 40, 7, 28, 19, 56, 20, 33, 64]
    assert TASKS["count-even"].spec["answers"] == [str(sum(n % 2 == 0 and n > 20 for n in numbers))]
    words = ["maple", "Birch", "oak", "Aspen", "cedar", "willow", "Elm"]
    assert TASKS["sort-words"].spec["answers"] == [", ".join(sorted(words, key=str.casefold))]
    assert TASKS["reverse-word"].spec["answers"] == ["benchmark"[::-1]]
    assert TASKS["change-owed"].spec["answer"] == 30 - (4 * 3 + 6 * 1.5)
    assert TASKS["percent-up-down"].spec["answer"] == pytest.approx(80 * 1.25 * 0.8)
    assert TASKS["divisible-count"].spec["answer"] == sum(
        (n % 3 == 0 or n % 5 == 0) and n % 15 != 0 for n in range(1, 101)
    )
    arrival = datetime.datetime(2026, 1, 1, 22, 40) + datetime.timedelta(hours=5, minutes=35)
    assert TASKS["arrival-time"].spec["answer"] == arrival.strftime("%H:%M")
    ben = [b for b in range(13, 24) if 24 == 2 * (b - (24 - b))]
    assert ben == [TASKS["age-puzzle"].spec["answer"]]
    orders = [
        p for p in itertools.permutations("ABCDE")
        if p.index("A") < p.index("C") < p.index("D") and p.index("B") == p.index("D") + 1 and p[-1] == "E"
    ]
    assert len(orders) == 1 and orders[0][1] == TASKS["finish-order"].spec["answer"]


@pytest.mark.parametrize("change, message", [
    ({"tasks": []}, "tasks"),
    ({"version": "one"}, "version"),
    ({"tasks": [{"id": "a", "category": "poetry", "prompt": "p", "reference": "r", "wrong": "w"}]}, "category"),
    ({"tasks": [{"id": "a", "category": "exact", "prompt": "p", "reference": "r", "wrong": "w"}]}, "answers"),
    ({"tasks": [{"id": "a", "category": "exact", "prompt": "p", "wrong": "w", "answers": ["r"]}]}, "reference"),
    ({"tasks": [{"id": "a", "category": "json", "prompt": "p", "reference": "r", "wrong": "w"}]}, "expect or schema"),
    ({"tasks": [{"id": "a", "category": "code", "prompt": "p", "reference": "r", "wrong": "w",
                 "function": "f", "tests": [{"call": "f(", "expect": 1}]}]}, "not a Python expression"),
    ({"tasks": [{"id": "a", "category": "exact", "prompt": "p", "reference": "r", "wrong": "w", "answers": ["r"], "typo": 1}]}, "unknown"),
])
def test_mistakes_in_a_suite_are_caught(change, message):
    raw = {"name": "s", "version": 1, "tasks": [
        {"id": "a", "category": "exact", "prompt": "p", "reference": "r", "wrong": "w", "answers": ["r"]},
    ]}
    with pytest.raises(SuiteError, match=message):
        parse_suite({**raw, **change})


def test_a_repeated_task_id_is_caught():
    task = {"id": "a", "category": "exact", "prompt": "p", "reference": "r", "wrong": "w", "answers": ["r"]}
    with pytest.raises(SuiteError, match="repeated"):
        parse_suite({"name": "s", "version": 1, "tasks": [task, task]})


# ---------------------------------------------------------------- grading rules


def test_a_thinking_section_is_not_part_of_the_answer():
    t = TASKS["count-even"]
    assert scorers.score(t, "<think>maybe 6?</think>\n5").passed
    assert not scorers.score(t, "<think>5").passed  # it never finished thinking


def test_exact_match_forgives_format_but_not_content():
    t = TASKS["classify-review"]
    for good in ("mixed", "Mixed.", "**mixed**", '"MIXED"', "  mixed\n"):
        assert scorers.score(t, good).passed, good
    for bad in ("negative", "It is mixed", "mixed or negative"):
        assert not scorers.score(t, bad).passed, bad


def test_json_is_strict_about_types_and_prose():
    t = TASKS["nulls-and-booleans"]
    good = '{"host": "db-7.internal", "port": 5433, "tls": false, "owner": null}'
    assert scorers.score(t, good).passed
    assert scorers.score(t, f"```json\n{good}\n```").passed
    assert not scorers.score(t, f"Here it is: {good}").passed
    assert not scorers.score(t, good.replace("false", "0")).passed  # 0 is not false
    assert not scorers.score(t, good.replace("5433", '"5433"')).passed
    assert not scorers.score(t, good.replace("null", "NaN")).passed


def test_the_schema_grader_catches_each_kind_of_mistake():
    schema = TASKS["invent-books"].spec["schema"]
    good = [{"title": "A", "year": 1950, "tags": ["x", "y"]}, {"title": "B", "year": 2000, "tags": ["x", "z"]},
            {"title": "C", "year": 1999, "tags": ["p", "q"]}]
    assert scorers.schema_errors(good, schema) == []
    assert scorers.schema_errors(good[:2], schema)
    assert scorers.schema_errors([{**good[0], "year": 1800}, *good[1:]], schema)
    assert scorers.schema_errors([{**good[0], "year": "1950"}, *good[1:]], schema)
    assert scorers.schema_errors([{**good[0], "extra": 1}, *good[1:]], schema)
    assert scorers.schema_errors([good[0], good[0], good[2]], schema)


def test_reasoning_reads_the_last_answer_line():
    t = TASKS["change-owed"]
    for good in ("ANSWER: 9", "answer: $9.00", "**ANSWER: 9**", "First ANSWER: 3\nno, wait\nANSWER: 9", "ANSWER: <9>"):
        assert scorers.score(t, good).passed, good
    for bad in ("ANSWER: 3", "The change is 9 dollars.", "ANSWER: nine"):
        assert not scorers.score(t, bad).passed, bad


def test_an_answer_cut_off_at_the_token_limit_never_counts():
    t = TASKS["count-even"]
    r = scorers.score(t, "5", truncated=True)
    assert not r.passed and "cut off" in r.detail


def test_code_is_taken_from_the_last_python_block():
    t = TASKS["dedupe-in-order"]
    wrong_then_right = (
        "First try:\n```python\ndef dedupe(items):\n    return []\n```\nBetter:\n"
        f"```python\n{t.reference}```\nDone."
    )
    assert scorers.score(t, wrong_then_right).passed


def test_consistency_signatures_ignore_formatting():
    t = TASKS["count-even"]
    assert scorers.score(t, "5.").signature == scorers.score(t, "**5**").signature


# ---------------------------------------------------------------- generated code


def test_a_correct_program_runs_and_returns_plain_values():
    r = sandbox.run_tests("def f(x):\n    return (x, {'a': [x]})\n", ["f(1)", "f(2)"])
    assert r.status == "ok" and r.results[0] == {"value": [1, {"a": [1]}]}


def test_ordinary_python_is_allowed():
    source = (
        "from collections import Counter\nimport math\nfrom fractions import Fraction\n"
        "class C:\n    def __init__(self):\n        self._n = 2\n"
        "def f():\n    return C()._n + math.floor(1.5) + Counter('aa')['a'] + int(Fraction(4, 2))\n"
        "if __name__ == '__main__':\n    raise SystemExit(f())\n"
    )
    r = sandbox.run_tests(source, ["f()"])
    assert r.status == "ok" and r.results == [{"value": 7}]


@pytest.mark.parametrize("source", [
    "import os\ndef f(): return 1",
    "import subprocess",
    "from socket import socket",
    "def f(): return open('x').read()",
    "def f(): return eval('1')",
    "def f(): return __import__('os')",
    "def f(): return ().__class__.__bases__",
    "def f(: nonsense",
])
def test_the_gate_refuses_a_program_before_it_runs(source):
    r = sandbox.run_tests(source, ["f()"])
    assert r.status == "rejected"


def test_the_audit_hook_stops_what_the_gate_cannot_see():
    # collections keeps a reference to sys, which reaches the OS module without any forbidden name.
    source = (
        "import collections\ndef f():\n    m = collections._sys.modules\n"
        "    return (m.get('nt') or m['posix']).listdir('.')\n"
    )
    assert sandbox.gate(source) is None
    r = sandbox.run_tests(source, ["f()"])
    assert r.status == "ok" and "blocked" in r.results[0]["error"]


def test_an_endless_loop_is_killed():
    r = sandbox.run_tests("def f():\n    while True:\n        pass\n", ["f()"], timeout_s=0.5)
    assert r.status == "timeout"


def test_a_memory_hog_is_killed():
    r = sandbox.run_tests("def f():\n    return [0] * (3 * 10**8)\n", ["f()"], memory_mib=100)
    assert r.status == "memory"


def test_a_crash_while_loading_is_an_error_not_a_score():
    r = sandbox.run_tests("raise ValueError('boom')\n", ["f()"])
    assert r.status == "error" and "boom" in r.detail


def test_printed_output_cannot_fake_a_result():
    r = sandbox.run_tests("print('{\"results\": [{\"value\": 1}]}')\ndef f():\n    return 2\n", ["f()"])
    assert r.results == [{"value": 2}]


# ---------------------------------------------------------------- statistics


def _attempt(task, passed, signature="a", category="exact", valid=1):
    return {"task_id": task, "category": category, "passed": passed, "signature": signature, "valid": valid}


def test_the_score_is_a_mean_over_tasks():
    attempts = [_attempt("a", 1), _attempt("a", 1), _attempt("b", 1), _attempt("b", 0, "b")]
    s = summary.summarize(attempts)
    assert s["score"] == pytest.approx(0.75)
    assert s["tasks"]["b"]["rate"] == 0.5 and s["correct"] == 3 and s["attempts"] == 4


def test_consistency_counts_agreement_even_when_wrong():
    attempts = [_attempt("a", 0, "x"), _attempt("a", 0, "x"), _attempt("a", 0, "y"), _attempt("a", 0, "x")]
    s = summary.summarize(attempts)
    assert s["score"] == 0 and s["consistency"] == pytest.approx(0.75)


def test_attempts_that_did_not_measure_the_model_do_not_count():
    s = summary.summarize([_attempt("a", 1), _attempt("a", 0, valid=0)])
    assert s["attempts"] == 1 and s["score"] == 1


def test_the_interval_is_repeatable_and_brackets_the_mean():
    values = [1, 1, 0, 1, 0.5, 0, 1, 1]
    lo, hi = stats.bootstrap_mean_ci(values)
    assert (lo, hi) == stats.bootstrap_mean_ci(values)
    assert lo < sum(values) / len(values) < hi
    assert stats.bootstrap_mean_ci([1, 1, 1]) == (1, 1)
    assert stats.bootstrap_mean_ci([]) is None


def _tasks(*rates):
    return {"tasks": {str(i): {"rate": r} for i, r in enumerate(rates)}}


def test_comparison_needs_a_difference_the_tasks_can_support():
    assert summary.compare(_tasks(*[1] * 20), _tasks(*[0] * 20))["verdict"] == "worse"
    assert summary.compare(_tasks(*[0] * 20), _tasks(*[1] * 20))["verdict"] == "better"
    one_flip = summary.compare(_tasks(*[1] * 20), _tasks(0, *[1] * 19))
    assert one_flip["verdict"] == "no detectable difference" and one_flip["diff"] == pytest.approx(-0.05)
    assert summary.compare(_tasks(1), {"tasks": {}}) is None


def test_cost_per_correct_answer():
    assert summary.cost_per_correct({"correct": 4, "cost_usd": 0.08, "energy_j": None}) == {"usd": 0.02, "kind": "list"}
    local = summary.cost_per_correct({"correct": 2, "cost_usd": None, "energy_j": 36000})
    assert local["kind"] == "electricity" and local["usd"] == pytest.approx(36000 / 3_600_000 * 0.20 / 2)
    assert summary.cost_per_correct({"correct": 0, "cost_usd": 1.0})["usd"] is None
    assert summary.cost_per_correct({"correct": 1, "cost_usd": None, "energy_j": None})["usd"] is None


# ---------------------------------------------------------------- chat template


def test_todays_date_is_taken_out_of_rendered_prompts():
    day = datetime.date(2026, 11, 8)
    pinned = llamacpp.pin_date("Today Date: 08 November 2026\nor 2026-11-08 or 08 Nov 2026", day)
    assert "2026-01-15" in pinned and "15 January 2026" in pinned and "15 Jan 2026" in pinned and "November" not in pinned
    assert pinned == llamacpp.pin_date(pinned, datetime.date(2027, 3, 1))


# ---------------------------------------------------------------- whole runs


def _write_suite(tmp_path: Path, tasks=None) -> str:
    """Five tasks the stand-in engines answer: four right, one wrong."""
    body = tasks or """
  - {id: right-exact, category: exact, prompt: "say: forty two", answers: ["forty two"], reference: "forty two", wrong: "no"}
  - {id: wrong-exact, category: exact, prompt: "say: nonsense", answers: ["right"], reference: "right", wrong: "no"}
  - id: right-json
    category: json
    prompt: 'say: {"a": 1}'
    expect: {a: 1}
    reference: '{"a": 1}'
    wrong: '{"a": 2}'
  - {id: right-reason, category: reasoning, prompt: "say: ANSWER: 7", answer: 7, reference: "ANSWER: 7", wrong: "ANSWER: 8"}
  - id: right-code
    category: code
    function: f
    prompt: |
      say: def f(): return 3
    tests: [{call: "f()", expect: 3}]
    reference: "def f(): return 3"
    wrong: "def f(): return 4"
"""
    path = tmp_path / "suite.yaml"
    path.write_text(f"name: mini\nversion: 1\ntasks:\n{body}", encoding="utf-8")
    return str(path)


def _local(tmp_path, **quality):
    return make_experiment(tmp_path, quality={"suite": _write_suite(tmp_path), "repeats": 2, **quality})


def test_a_local_run_scores_every_task_after_its_speed_runs(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    live = LiveLog(tmp_path / "live")
    _, (res,) = run_experiment(_local(tmp_path), conn, gpu=FakeGpu(), echo=_quiet, live=live)
    assert res["status"] == "ok" and len(res["trials"]) == 4  # the speed runs are unchanged
    q = res["quality"]
    assert q["status"] == "ok" and q["attempts"] == 10 and q["correct"] == 8
    assert q["score"] == pytest.approx(0.8) and q["consistency"] == 1
    assert {c["check"]: c["passed"] for c in q["checks_json"]} == {
        "Graders verified on known answers": True, "Every task answered": True,
        "Answers ended by themselves": True, "Each answer measured once": True,
    }
    attempts = db.get_quality_attempts(conn, q["id"])
    assert sorted({a["seed"] for a in attempts}) == [1, 2]  # workload seed + repeat
    assert next(a for a in attempts if a["task_id"] == "right-exact")["answer"] == "forty two"
    assert q["summary_json"]["categories"]["code"]["score"] == 1
    kinds = [e["type"] for e in read_events(live.path)]
    assert kinds.count("attempt") == 10 and kinds.index("quality_start") < kinds.index("variant_done")


def test_adding_a_suite_leaves_the_speed_hash_alone(tmp_path):
    assert _local(tmp_path).variants[0].config_hash == make_experiment(tmp_path).variants[0].config_hash


def test_a_cloud_run_scores_every_task_too(fake_claude, tmp_path):
    conn = db.connect(":memory:")
    exp = make_cloud_experiment(tmp_path, quality={"suite": _write_suite(tmp_path), "repeats": 1})
    _, (res,) = run_experiment(exp, conn, gpu=None, echo=_quiet)
    q = res["quality"]
    assert q["status"] == "ok" and q["correct"] == 4 and q["attempts"] == 5
    assert q["cost_usd"] == pytest.approx(0.005) and q["energy_j"] is None


def test_a_fallback_model_does_not_count_toward_the_score(fake_claude, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "fallback")
    exp = make_cloud_experiment(tmp_path, quality={"suite": _write_suite(tmp_path), "repeats": 1})
    _, (res,) = run_experiment(exp, db.connect(":memory:"), gpu=None, echo=_quiet)
    q = res["quality"]
    assert q["status"] == "failed" and q["score"] is None
    assert next(c for c in q["checks_json"] if c["check"] == "Each answer measured once")["passed"] is False


def test_repeated_engine_errors_stop_the_suite(fake_claude, tmp_path, monkeypatch):
    from llmlab import claude_code

    real = claude_code.complete
    calls = {"n": 0}

    def flaky(cmd, env, prompt, cwd, timeout_s):
        calls["n"] += 1
        if "say:" in prompt:
            raise claude_code.ClaudeCodeError("usage limit reached")
        return real(cmd, env, prompt, cwd, timeout_s)

    monkeypatch.setattr(claude_code, "complete", flaky)
    exp = make_cloud_experiment(tmp_path, quality={"suite": _write_suite(tmp_path), "repeats": 1})
    _, (res,) = run_experiment(exp, db.connect(":memory:"), gpu=None, echo=_quiet)
    q = res["quality"]
    assert q["status"] == "failed" and "usage limit" in q["error"] and calls["n"] == 3 + 3


def test_an_answer_that_reaches_the_limit_is_marked(fake_engine, tmp_path):
    tasks = """
  - {id: ramble, category: exact, prompt: "say: RAMBLE", answers: ["x"], reference: "x", wrong: "y"}
  - {id: fine, category: exact, prompt: "say: x", answers: ["x"], reference: "x", wrong: "y"}
"""
    exp = make_experiment(tmp_path, quality={"suite": _write_suite(tmp_path, tasks), "repeats": 1, "max_tokens": 8})
    _, (res,) = run_experiment(exp, db.connect(":memory:"), gpu=FakeGpu(), echo=_quiet)
    q = res["quality"]
    assert q["status"] == "warning" and q["correct"] == 1
    assert next(c for c in q["checks_json"] if c["check"] == "Answers ended by themselves")["passed"] is False


def test_a_broken_grader_stops_the_suite_before_any_request(fake_engine, tmp_path):
    tasks = '  - {id: bad, category: exact, prompt: "say: x", answers: ["x"], reference: "not x", wrong: "y"}\n'
    exp = make_experiment(tmp_path, quality={"suite": _write_suite(tmp_path, tasks), "repeats": 1})
    conn = db.connect(":memory:")
    _, (res,) = run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)
    q = res["quality"]
    assert q["status"] == "failed" and "reference answer was rejected" in q["error"]
    assert db.get_quality_attempts(conn, q["id"]) == []


class StopInSuite(LiveLog):
    def stop_requested(self) -> bool:
        return sum(e["type"] == "attempt" for e in read_events(self.path)) >= 3


def test_stopping_during_the_suite_keeps_what_finished(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    live = StopInSuite(tmp_path / "live")
    with pytest.raises(StopRequested):
        run_experiment(_local(tmp_path), conn, gpu=FakeGpu(), echo=_quiet, live=live)
    (config_row,) = db.get_configs(conn, 1)
    q = db.get_quality_run(conn, config_row["id"])
    assert q["attempts"] == 3 and q["status"] == "failed" and q["error"] == "stopped from the dashboard"


@pytest.mark.parametrize("change, message", [
    ({"suite": "missing.yaml"}, "not found"),
    ({"suite": "BAD"}, "version"),
    ({"repeats": 0}, "quality.repeats"),
    ({"temperature": 5}, "quality.temperature"),
    ({"thinking": "maybe"}, "quality.thinking"),
])
def test_mistakes_in_the_quality_section_are_caught(tmp_path, change, message):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: s\ntasks: [1]\n", encoding="utf-8")
    q = {"suite": _write_suite(tmp_path), **change}
    if q["suite"] == "BAD":
        q["suite"] = str(bad)
    with pytest.raises(config.ConfigError, match=message):
        make_experiment(tmp_path, quality=q)


def test_reports_show_quality_and_compare_it(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    exp = _local(tmp_path)
    a, _ = run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)
    b, _ = run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)
    shown = report.session_text(conn, a)
    assert "Quality     80%" in shown and "8 of 10 answers correct" in shown and "suite mini v1" in shown
    text, verdict = report.compare_text(conn, a, b, 5.0)
    assert "+0.0 points" in text and "no detectable difference" in text and verdict
    rows = api.quality_results(conn)
    assert len(rows) == 1 and rows[0]["session_id"] == b and rows[0]["gen_tps"] > 0  # latest of the two
    assert "80%" in report.quality_table(rows)
    detail = api.quality_detail(conn, rows[0]["config_id"])
    assert detail["attempts"] == 10 and len(detail["answers"]) == 10 and "answer" in detail["answers"][0]
    assert api.quality_detail(conn, 999) is None


def test_regrading_applies_todays_graders_to_stored_answers(fake_engine, tmp_path):
    from llmlab.quality.run import regrade

    conn = db.connect(":memory:")
    _, (res,) = run_experiment(_local(tmp_path), conn, gpu=FakeGpu(), echo=_quiet)
    q = res["quality"]
    attempts = db.get_quality_attempts(conn, q["id"])
    # As if a grader had been too strict: two right answers stored as wrong.
    wrong = [a for a in attempts if a["task_id"] == "right-exact"]
    for a in wrong:
        db.update(conn, "quality_attempts", a["id"], {"passed": 0, "detail": "too strict", "signature": "x"})
    db.update(conn, "quality_runs", q["id"], {"score": 0.5, "correct": 6})
    conn.commit()
    before, after, changed = regrade(conn, res["config_id"])
    assert before["correct"] == 6 and after["correct"] == 8 and changed == 2
    assert after["score"] == pytest.approx(0.8) and after["summary_json"]["regraded_at"]
    assert all(a["detail"] == "matches" for a in db.get_quality_attempts(conn, q["id"]) if a["task_id"] == "right-exact")
    assert "regraded" in report.session_text(conn, 1)


def test_the_quality_page_uses_the_latest_speed_that_passed_its_checks(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    exp = _local(tmp_path)
    clean, _ = run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)
    flagged, _ = run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)
    (later,) = db.get_configs(conn, flagged)
    db.update(conn, "configs", later["id"], {"status": "warning"})
    (row,) = api.quality_results(conn)
    assert row["session_id"] == flagged and row["speed_session_id"] == clean and row["speed_clean"]

    (first,) = db.get_configs(conn, clean)
    db.update(conn, "configs", first["id"], {"status": "warning"})
    (row,) = api.quality_results(conn)
    assert row["speed_session_id"] == flagged and not row["speed_clean"]  # nothing clean: shown, and marked


def test_the_telemetry_window_gives_energy():
    from llmlab import telemetry

    samples = [{"t": 0.0, "gpu_power_instant_w": 100.0}, {"t": 1.0, "gpu_power_instant_w": 100.0},
               {"t": 2.0, "gpu_power_instant_w": 200.0}, {"t": 9.0, "gpu_power_instant_w": 500.0}]
    assert telemetry.energy_j(samples, 0.0, 2.0) == pytest.approx(250.0)
    assert telemetry.energy_j(samples, 8.0, 9.5) is None
