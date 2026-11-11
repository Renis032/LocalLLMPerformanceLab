"""A stand-in for Claude Code's headless mode, so the tests run offline and use no plan.

It takes the same command line as `claude -p --output-format stream-json
--include-partial-messages --verbose`, reads the prompt from stdin, and prints the same
kinds of events. A prompt containing "say: X" gets X as its answer. FAKE_CLAUDE_MODE picks
a misbehaviour seen, or possible, in the real thing:

  ok         one request, answered normally (the default)
  continue   two requests in one run, as when an answer reaches the output limit
  fallback   answered by a different model
  think      hidden thinking tokens
  cache      part of the prompt read from the cache
  mismatch   Claude Code's own time to first token disagrees with the stream's
  error      Claude Code reports an API error
  crash      exits without printing a result
"""

import json
import os
import re
import sys
import time

args = sys.argv[1:]
model = args[args.index("--model") + 1]
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def stream(kind: str, **fields) -> None:
    emit({"type": "stream_event", "event": {"type": kind, **fields}})


if mode == "crash":
    print("fake Claude Code crashed", file=sys.stderr)
    sys.exit(1)

time.sleep(0.02)  # starting up
emit({"type": "system", "subtype": "init", "model": model, "apiKeySource": "none",
      "claude_code_version": "9.9.9", "tools": [], "mcp_servers": []})
t_init = time.perf_counter()

if mode == "error":
    emit({"type": "result", "subtype": "success", "is_error": True, "result": "API Error: 429 rate limited"})
    sys.exit(1)

answered_by = "claude-other-1" if mode == "fallback" else model
said = re.search(r"say: (.*)", prompt, re.DOTALL)
words = re.findall(r"\s*\S+", said.group(1).strip()) if said else ["word "] * 20
chunks = len(words)
ttft = None
for call in range(2 if mode == "continue" else 1):
    time.sleep(0.05)  # the request, the queue and the prompt
    stream("message_start", message={"model": answered_by, "usage": {"input_tokens": 600}})
    if mode == "think":
        # Thinking streams first, with its text withheld, as Claude Code shows it.
        stream("content_block_start", index=0, content_block={"type": "thinking", "thinking": ""})
        ttft = ttft if ttft is not None else (time.perf_counter() - t_init) * 1000
        stream("content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": ""})
        time.sleep(0.05)
        stream("content_block_stop", index=0)
    stream("content_block_start", index=0, content_block={"type": "text", "text": ""})
    for i in range(chunks):
        if i:
            time.sleep(0.005)
        stream("content_block_delta", index=0, delta={"type": "text_delta", "text": words[i]})
        if ttft is None:
            ttft = (time.perf_counter() - t_init) * 1000
    stream("content_block_stop", index=0)
    stream("message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 2 * chunks})
    stream("message_stop")

calls = 2 if mode == "continue" else 1
emit({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "num_turns": 1,
    "ttft_stream_ms": round(ttft + (2000 if mode == "mismatch" else 40)),
    "total_cost_usd": 0.001 * calls,
    "usage": {
        "input_tokens": (600 + len(prompt.split())) * calls,
        "cache_read_input_tokens": 100 if mode == "cache" else 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 2 * chunks * calls + (30 if mode == "think" else 0),
        "output_tokens_details": {"thinking_tokens": 30 if mode == "think" else 0},
    },
})
