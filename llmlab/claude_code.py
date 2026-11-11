"""Claude as a cloud baseline, through Claude Code's headless mode (`claude -p`).

Claude's weights aren't public, so it can't run on the local GPU. The lab sends each
request through the Claude Code command line, which signs in with the user's Claude
plan: no API key is needed. Claude Code streams every event as a JSON line, and the lab
timestamps each line as it arrives.

Claude Code is an agent, not a bare API client, so the lab switches off everything an
agent adds and checks what is left:

  * Its coding system prompt is replaced, and its tools, MCP servers, slash commands and
    settings files are turned off. It still adds a fixed context of its own (about 600
    tokens: environment details and account notes), the same on every run.
  * It continues automatically when an answer reaches the output limit, which would
    merge several requests into one run. So output isn't capped, and every run must be
    exactly one API request.
  * Prompt caching would make later runs faster than the first; it is turned off.
  * Hidden thinking delays the first visible text. With thinking off, Claude Code is
    asked not to think (MAX_THINKING_TOKENS=0), but Claude Haiku 5.5 thinks anyway, so
    the thinking tokens and the time to the first streamed content are recorded too.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .stats import percentile


class ClaudeCodeError(RuntimeError):
    """Claude Code is missing, or a request through it failed."""


# apiKeySource in Claude Code's first event: "none" means it used the plan login.
AUTH_NAMES = {"none": "Claude plan login", "ANTHROPIC_API_KEY": "API key (billed per token)"}


def find_cli(cli: str) -> str:
    """The Claude Code executable to run.

    npm installs a claude.cmd shim on Windows. Running the executable behind it saves
    starting cmd.exe on every run.
    """
    path = cli if ("/" in cli or "\\" in cli) else shutil.which(cli)
    if not path or not Path(path).is_file():
        raise ClaudeCodeError(
            f"Claude Code not found: {cli}. Install it and sign in, or set engine.cli to its path"
        )
    p = Path(path)
    if p.suffix.lower() in (".cmd", ".bat", ".ps1", ""):
        native = p.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.is_file():
            return str(native)
    return str(p)


def cli_version(exe: str) -> str:
    """For example '2.1.311'."""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise ClaudeCodeError(f"could not run {exe} --version: {e}") from e
    m = re.match(r"\s*(\d[\w.\-]*)", out)
    return m.group(1) if m else (out.strip() or "unknown")


def build_command(exe: str, model: str, effort: str, system_prompt: str) -> list[str]:
    return [
        exe, "-p",
        "--model", model,
        "--effort", effort,
        "--output-format", "stream-json", "--include-partial-messages", "--verbose",
        "--system-prompt", system_prompt,
        "--tools", "",                 # no tools: one question, one answer
        "--strict-mcp-config",         # no MCP servers
        "--disable-slash-commands",    # no skills
        "--setting-sources", "",       # no settings files, so no hooks or plugins from them
        "--no-session-persistence",    # nothing saved to the user's session history
    ]


def build_env(thinking: str, base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    # Variables a parent Claude Code session sets; the benchmark is its own session.
    for key in list(env):
        if key == "CLAUDECODE" or key.startswith("CLAUDE_CODE_"):
            del env[key]
    env["DISABLE_PROMPT_CACHING"] = "1"
    # No telemetry, error reports or update checks competing with the measurement.
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    if thinking == "off":
        env["MAX_THINKING_TOKENS"] = "0"
    else:  # auto: the model decides
        env.pop("MAX_THINKING_TOKENS", None)
    return env


def complete(cmd: list[str], env: dict, prompt: str, cwd: str | Path, timeout_s: float) -> dict:
    """Run one request and time every streamed event."""
    t0_wall, t0 = time.time(), time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(cwd), env=env,
        )
    except OSError as e:
        raise ClaudeCodeError(f"could not start Claude Code: {e}") from e
    stderr: list[bytes] = []
    reader = threading.Thread(target=lambda: stderr.append(proc.stderr.read()), daemon=True)
    reader.start()
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, kill)
    timer.start()
    events: list[tuple[float, dict]] = []
    try:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
        for raw in proc.stdout:
            t = time.perf_counter() - t0
            try:
                events.append((t, json.loads(raw)))
            except json.JSONDecodeError:
                continue  # not an event line
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        raise
    finally:
        timer.cancel()
        reader.join(timeout=5)
    if timed_out.is_set():
        raise ClaudeCodeError(f"no answer within {timeout_s:.0f} s")
    err = b"".join(stderr).decode("utf-8", "replace").strip()
    return parse_stream(events, t0_wall, err, proc.returncode)


def parse_stream(events: list[tuple[float, dict]], t0_wall: float, stderr: str = "", returncode: int | None = 0) -> dict:
    """Timings and token counts from Claude Code's stream-json events.

    events: (seconds since the process started, event). Times are measured from the
    `init` event, when Claude Code is ready and about to send the request; the time
    before it is Claude Code starting up.
    """
    t_init = t_content = None
    init: dict = {}
    result: dict | None = None
    models: list[str] = []
    deltas: list[tuple[float, str]] = []
    stop = None
    for t, ev in events:
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init" and t_init is None:
            t_init, init = t, ev
        elif kind == "stream_event":
            e = ev.get("event") or {}
            if e.get("type") == "message_start":
                models.append((e.get("message") or {}).get("model"))
            elif e.get("type") in ("content_block_start", "content_block_delta") and t_content is None:
                t_content = t  # the first content of any kind, thinking included
            elif e.get("type") == "content_block_delta" and (e.get("delta") or {}).get("type") == "text_delta":
                deltas.append((t, e["delta"].get("text") or ""))
            elif e.get("type") == "message_delta":
                stop = (e.get("delta") or {}).get("stop_reason") or stop
        elif kind == "result":
            result = ev

    if result is None:
        tail = stderr[-600:] or "no output"
        raise ClaudeCodeError(f"Claude Code ended without a result (exit code {returncode}): {tail}")
    if result.get("is_error") or result.get("subtype") != "success":
        why = result.get("result") or result.get("subtype") or "unknown error"
        raise ClaudeCodeError(f"Claude Code reported an error: {str(why)[:600]}")
    if t_init is None or not deltas:
        raise ClaudeCodeError("Claude Code streamed no text")

    usage = result.get("usage") or {}
    out_tokens = usage.get("output_tokens")
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0
    t_first, t_last = deltas[0][0], deltas[-1][0]
    chars = sum(len(text) for _, text in deltas)
    gaps_ms = [(b[0] - a[0]) * 1000 for a, b in zip(deltas, deltas[1:])]
    # Streamed chunks carry several tokens each, and the stream doesn't say how many.
    # Speed counts the tokens after the first chunk, assuming tokens follow characters.
    visible = out_tokens - thinking if out_tokens is not None else None
    gen_tps = None
    if visible and chars and t_last > t_first:
        gen_tps = visible * (1 - len(deltas[0][1]) / chars) / (t_last - t_first)
    return {
        "text": "".join(text for _, text in deltas),
        "t_start": t0_wall,
        "t_send": t0_wall + t_init,
        "t_first": t0_wall + t_first,
        "t_last": t0_wall + t_last,
        "startup_ms": t_init * 1000,
        "ttft_ms": (t_first - t_init) * 1000,
        "first_content_ms": ((t_content if t_content is not None else t_first) - t_init) * 1000,
        "total_ms": (t_last - t_init) * 1000,
        "reported_ttft_ms": result.get("ttft_stream_ms"),
        "gen_tps": gen_tps,
        "client_gen_tps": gen_tps,
        "chunks": len(deltas),
        "chars": chars,
        "itl_ms": [round(g, 3) for g in gaps_ms],
        "itl_p50_ms": percentile(gaps_ms, 50),
        "itl_p95_ms": percentile(gaps_ms, 95),
        "itl_max_ms": max(gaps_ms) if gaps_ms else None,
        "prompt_tokens": usage.get("input_tokens"),
        "cached_tokens": (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0),
        "gen_tokens": out_tokens,
        "thinking_tokens": thinking,
        "api_calls": len(models),
        "model_used": ", ".join(sorted({m for m in models if m})) or None,
        "stop_type": stop,
        "cost_usd": result.get("total_cost_usd"),
        "auth": AUTH_NAMES.get(init.get("apiKeySource"), init.get("apiKeySource")),
        "cli_version": init.get("claude_code_version"),
    }
