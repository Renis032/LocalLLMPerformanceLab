"""Purpose: the Claude cloud baseline measures the model, and nothing Claude Code adds.

Parses a real recorded Claude Code stream, then runs whole cloud experiments against a
stand-in for Claude Code. Each check must catch its problem: an extra API request, a
different model answering, hidden thinking, a cached prompt, disagreeing timings, or a
failed request. The tests use no network and none of the plan's usage.
"""

import json
from pathlib import Path

import pytest

from llmlab import claude_code, cloud, db, report
from llmlab.config import ConfigError, parse_experiment
from llmlab.runner import run_experiment

from conftest import FIXTURES, FakeGpu, make_cloud_experiment

REAL_STREAM = FIXTURES / "claude-code-2.1.311-sonnet-stream.jsonl"


def _quiet(*args):
    pass


def _events(path: Path) -> list[tuple[float, dict]]:
    return [(d["t"], d["ev"]) for d in map(json.loads, path.read_text(encoding="utf-8").splitlines())]


# ---------------------------------------------------------------- the stream


def test_a_real_stream_is_timed_from_when_claude_code_is_ready():
    r = claude_code.parse_stream(_events(REAL_STREAM), t0_wall=1000.0)
    assert r["startup_ms"] == pytest.approx(578.8, abs=0.5)  # before the request
    assert r["ttft_ms"] == pytest.approx(836.5, abs=0.5)
    assert r["first_content_ms"] == pytest.approx(r["ttft_ms"], abs=1)  # no thinking before the text
    assert r["reported_ttft_ms"] == 885  # Claude Code's own figure, about 45 ms more
    assert r["chunks"] == 314 and r["gen_tokens"] == 1232 and r["prompt_tokens"] == 1354
    assert r["gen_tps"] == pytest.approx(117.5, abs=0.1)
    assert (r["api_calls"], r["model_used"], r["stop_type"]) == (1, "claude-sonnet-5-5", "end_turn")
    assert (r["thinking_tokens"], r["cached_tokens"]) == (0, 0)
    assert r["auth"] == "Claude plan login" and r["cli_version"] == "2.1.311"
    assert r["t_send"] == pytest.approx(1000.5788) and len(r["itl_ms"]) == 313


def test_a_failed_request_raises_with_claude_codes_message():
    error = [(0.1, {"type": "result", "subtype": "success", "is_error": True, "result": "API Error: 529 overloaded"})]
    with pytest.raises(claude_code.ClaudeCodeError, match="529 overloaded"):
        claude_code.parse_stream(error, 0.0)
    with pytest.raises(claude_code.ClaudeCodeError, match="without a result.*not signed in"):
        claude_code.parse_stream([], 0.0, stderr="not signed in", returncode=1)


# ---------------------------------------------------------------- the command


def test_the_command_turns_off_what_claude_code_adds():
    cmd = claude_code.build_command("claude.exe", "claude-haiku-5-5", "low", "Be brief.")
    joined = " ".join(cmd)
    for flag in ("-p", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
                 "--include-partial-messages"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--system-prompt Be brief." in joined and "--model claude-haiku-5-5" in joined


def test_the_environment_turns_off_caching_and_thinking_and_drops_a_parent_session():
    base = {"PATH": "x", "CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli", "MAX_THINKING_TOKENS": "9"}
    env = claude_code.build_env("off", base)
    assert env["DISABLE_PROMPT_CACHING"] == "1" and env["MAX_THINKING_TOKENS"] == "0"
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_ENTRYPOINT" not in env and env["PATH"] == "x"
    assert "MAX_THINKING_TOKENS" not in claude_code.build_env("auto", base)


def test_the_npm_shim_is_skipped_for_the_real_executable(tmp_path):
    shim = tmp_path / "claude.cmd"
    shim.write_text("@echo off")
    assert claude_code.find_cli(str(shim)) == str(shim)  # no executable behind it
    native = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"")
    assert claude_code.find_cli(str(shim)) == str(native)
    with pytest.raises(claude_code.ClaudeCodeError, match="not found"):
        claude_code.find_cli(str(tmp_path / "missing.exe"))


# ---------------------------------------------------------------- experiment files


def test_a_cloud_experiment_sweeps_models_and_labels_them_by_id(tmp_path):
    exp = make_cloud_experiment(tmp_path, engine={"thinking": False},
                                sweep={"model.name": ["claude-a-1", "claude-b-1"]})
    assert [v.label for v in exp.variants] == ["claude-a-1", "claude-b-1"]
    c = exp.variants[0].config
    assert set(c) == {"engine", "model", "workload", "run", "quality"}
    assert c["engine"]["thinking"] == "off" and c["workload"]["system_prompt"]
    # Where Claude Code is installed doesn't change what is measured.
    moved = make_cloud_experiment(tmp_path, engine={"cli": str(tmp_path / "other" / "claude.exe")})
    assert moved.variants[0].config_hash == make_cloud_experiment(tmp_path).variants[0].config_hash


@pytest.mark.parametrize(
    "sections, message",
    [
        ({"model": {"name": "sonnet"}}, "full model ID"),
        ({"engine": {"effort": "huge"}}, "engine.effort"),
        ({"engine": {"thinking": True}}, "engine.thinking: must be off or auto"),
        ({"server": {"ctx": 512}}, "server: unknown setting"),
        ({"sweep": {"server.ctx": [1, 2]}}, "for example model.name"),
    ],
)
def test_cloud_experiment_mistakes_are_caught(tmp_path, sections, message):
    raw = {"name": "x", "engine": {"type": "claude-code"}, "model": {"name": "claude-a-1"},
           "workload": {"prompt": "hi"}}
    for k, v in sections.items():
        raw[k] = {**raw.get(k, {}), **v}
    with pytest.raises(ConfigError, match=message):
        parse_experiment(raw, base_dir=tmp_path)


# ---------------------------------------------------------------- whole experiments


def test_a_cloud_experiment_runs_end_to_end_even_with_the_gpu_busy(fake_claude, tmp_path):
    conn = db.connect(":memory:")
    exp = make_cloud_experiment(tmp_path)
    sid, (res,) = run_experiment(exp, conn, gpu=FakeGpu(util=99.0), echo=_quiet)

    assert res["status"] == "ok", res
    assert [(t["kind"], t["valid"]) for t in res["trials"]] == [("warmup", 1), ("measured", 1), ("measured", 1)]
    s = res["summary"]
    assert s["n"] == 2 and s["gen_tps"]["median"] > 0 and s["ttft_ms"]["median"] >= 40
    assert s["prompt_tokens"]["median"] == 605 and s["gen_tokens"]["median"] == 40
    assert res["checks"] == [{"check": "Same input every run", "passed": True,
                              "detail": "605 input tokens every run, including Claude Code's own context"}]

    stored = db.get_configs(conn, sid)[0]
    assert stored["engine"] == "claude-code" and stored["engine_version"] == "9.9.9"
    assert stored["model_path"] == "claude-test-1" and stored["idle_json"]["ok"] is None
    assert stored["facts_json"]["auth"] == "Claude plan login"
    trial = db.get_trials(conn, stored["id"])[-1]
    assert trial["api_calls"] == 1 and trial["chunks"] == 20 and trial["cost_usd"] == 0.001

    text = report.session_text(conn, sid)
    assert "Engine    Claude Code 9.9.9" in text and "signed in with: Claude plan login" in text
    assert "J/token" not in text  # no GPU figures for a cloud model
    compared, verdict = report.compare_text(conn, sid, sid, 5.0)
    prompt_line = next(line for line in compared.splitlines() if "Prompt tok/s" in line)
    assert verdict is True and "FAIL" not in prompt_line  # no prompt speed for a cloud model: not a failure


@pytest.mark.parametrize(
    "mode, problem",
    [
        ("continue", "made 2 API requests instead of 1"),
        ("fallback", "answered by claude-other-1 instead of claude-test-1"),
        ("think", "30 hidden thinking tokens"),
        ("cache", "100 prompt tokens were read from or written to the cache"),
        ("mismatch", "they should agree"),
    ],
)
def test_each_check_catches_its_problem(fake_claude, tmp_path, monkeypatch, mode, problem):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    conn = db.connect(":memory:")
    _, (res,) = run_experiment(make_cloud_experiment(tmp_path), conn, gpu=None, echo=_quiet)
    assert res["status"] == "failed" and res["error"] == "no measured run passed its checks"
    assert all(not t["valid"] for t in res["trials"])
    assert any(problem in p for p in res["trials"][0]["problems_json"]), res["trials"][0]["problems_json"]


def test_thinking_counts_when_the_model_may_think(fake_claude, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "think")
    conn = db.connect(":memory:")
    exp = make_cloud_experiment(tmp_path, engine={"thinking": "auto"})
    _, (res,) = run_experiment(exp, conn, gpu=None, echo=_quiet)
    assert res["status"] == "ok", res["trials"][0]["problems_json"]
    t = res["trials"][-1]
    assert t["thinking_tokens"] == 30 and t["gen_tokens"] == 70
    # Thinking delays the text, and the timing cross-check counts it like Claude Code does.
    assert t["ttft_ms"] - t["first_content_ms"] >= 40
    assert "30 of them hidden thinking" in "\n".join(report.config_lines(res))


@pytest.mark.parametrize("mode, message", [("error", "429 rate limited"), ("crash", "crashed")])
def test_a_failed_request_fails_the_configuration(fake_claude, tmp_path, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    conn = db.connect(":memory:")
    sid, (res,) = run_experiment(make_cloud_experiment(tmp_path), conn, gpu=None, echo=_quiet)
    assert res["status"] == "failed" and message in res["error"]
    assert db.get_session(conn, sid)["status"] == "completed"  # one failed model doesn't stop the session


def test_changing_input_means_claude_code_added_something():
    rows = [{"prompt_tokens": n} for n in (1354, 1354, 1360)]
    (check,) = cloud.config_checks(rows)
    assert check["passed"] is False and "1,354 to 1,360" in check["detail"]
