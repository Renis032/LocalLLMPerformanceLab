"""llama.cpp's llama-server: start it with pinned settings, stream completions, read its log."""

from __future__ import annotations

import datetime as dt
import json
import re
import socket
import subprocess
import time
from pathlib import Path

import httpx

from .stats import percentile

# Settings every benchmark uses. Each one closes a gap found in Phase 0.
FIXED_ARGS = [
    "-np", "1",  # one slot: the whole context belongs to the benchmark request
    "--fit", "off",  # never let the server change settings to fit memory
    "-lm", "none",  # no memory-mapping, so RAM figures are real
    "--no-cache-prompt",  # every request processes its full prompt
    "--cache-ram", "0",  # no prompt cache in system RAM either (default: up to 8 GiB)
    "--no-ui",
    "-lv", "4",  # without it, this build doesn't log layer offload or buffer sizes
]

# Experiment settings (the server section) and the llama-server flags they become.
FLAGS = {
    "ctx": "-c",
    "gpu_layers": "-ngl",
    "flash_attn": "-fa",
    "threads": "-t",
    "threads_batch": "-tb",
    "batch": "-b",
    "ubatch": "-ub",
    "cache_type_k": "-ctk",
    "cache_type_v": "-ctv",
}

# How long a streaming request may go quiet. Prompt processing on the CPU with a
# long context can take minutes before the first token arrives.
READ_TIMEOUT_S = 600.0


# Chat templates can stamp today's date into every prompt (SmolLM3's does), so the same
# request would differ from one day to the next. Rendered prompts get this date instead.
PINNED_DATE = dt.date(2026, 1, 15)
DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%Y-%m-%d")

# The server's sampling defaults, written out so a new engine build can't change them.
SAMPLING = {"top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.0}


class ServerError(RuntimeError):
    """llama-server failed to start, crashed, or answered with an error."""


def pin_date(prompt: str, today: dt.date | None = None) -> str:
    """Replace today's date, in the formats templates use, with a fixed one."""
    today = today or dt.date.today()
    for fmt in DATE_FORMATS:
        prompt = prompt.replace(today.strftime(fmt), PINNED_DATE.strftime(fmt))
    return prompt


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_command(exe: str, model: str, server: dict, port: int) -> list[str]:
    """The llama-server command line. extra_args go last, so they can override anything."""
    cmd = [exe, "-m", model]
    for key, flag in FLAGS.items():
        value = server.get(key)
        if value is not None:
            cmd += [flag, str(value)]
    cmd += FIXED_ARGS
    cmd += ["--host", "127.0.0.1", "--port", str(port)]
    cmd += list(server.get("extra_args") or [])
    return cmd


def engine_version(exe: str) -> str:
    """For example '0.9.0-dev b12110 (cd959bae9)', from `llama-server --version`."""
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    text = (out.stdout + out.stderr).strip()
    m = re.search(r"version:\s*(\S+)\s*\(build (\d+), commit ([0-9a-f]+)\)", text)
    if m:
        return f"{m.group(1)} b{m.group(2)} ({m.group(3)})"
    return text.splitlines()[0] if text else "unknown"


class LlamaServer:
    """One llama-server process, loaded with one model and one set of settings."""

    def __init__(self, exe: str, model: str, server: dict, log_path: Path):
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.command = build_command(exe, model, server, self.port)
        self.log_path = Path(log_path)
        self.proc: subprocess.Popen | None = None
        self._log = None
        self._client: httpx.Client | None = None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def start(self, timeout: float) -> float:
        """Start the server and wait until it has loaded the model. Returns load seconds."""
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            self.command, stdout=self._log, stderr=subprocess.STDOUT, cwd=str(self.log_path.parent)
        )
        self._client = httpx.Client(
            base_url=self.base_url, timeout=httpx.Timeout(10.0, read=READ_TIMEOUT_S)
        )
        while True:
            if self.proc.poll() is not None:
                raise ServerError(
                    f"llama-server exited with code {self.proc.returncode} while loading. "
                    f"Last lines of its log:\n{self.log_tail()}"
                )
            try:
                if self._client.get("/health", timeout=2).status_code == 200:
                    return time.perf_counter() - t0
            except httpx.HTTPError:
                pass
            if time.perf_counter() - t0 > timeout:
                raise ServerError(
                    f"llama-server did not finish loading within {timeout:.0f} s. "
                    f"Last lines of its log:\n{self.log_tail()}"
                )
            time.sleep(0.05)

    def tokenize(self, text: str) -> int:
        """Prompt length in tokens, counted the way /completion counts it."""
        r = self._client.post("/tokenize", json={"content": text, "add_special": True})
        r.raise_for_status()
        return len(r.json().get("tokens", []))

    def render_chat(self, system: str, user: str, thinking: bool) -> str:
        """A conversation as the model expects it, through the model's own chat template."""
        r = self._client.post("/apply-template", json={
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": thinking},
        })
        if r.status_code != 200:
            raise ServerError(f"/apply-template answered {r.status_code}: {r.text[:500]}")
        return pin_date(r.json()["prompt"])

    def complete(
        self, prompt: str, n_predict: int, seed: int, *, ignore_eos: bool = True, temperature: float = 0.0
    ) -> dict:
        """Stream one completion and time it from the client side as well.

        By default it generates exactly n_predict tokens, greedily, which is what speed
        runs need. Quality runs set ignore_eos=False so the model stops when it is done.
        """
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "ignore_eos": ignore_eos,
            "cache_prompt": False,
            "seed": seed,
            "temperature": temperature,
            "stream": True,
            "return_tokens": True,  # token ids per event, so counts don't rely on text
            **SAMPLING,
        }
        events: list[tuple[float, float, int]] = []  # (wall clock, perf counter, tokens)
        pieces: list[str] = []
        final: dict = {}
        t_send_wall, t_send = time.time(), time.perf_counter()
        try:
            with self._client.stream("POST", "/completion", json=body) as r:
                if r.status_code != 200:
                    r.read()
                    raise ServerError(f"/completion answered {r.status_code}: {r.text[:500]}")
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    now_perf, now_wall = time.perf_counter(), time.time()
                    msg = json.loads(payload)
                    n = len(msg.get("tokens") or []) or (1 if msg.get("content") else 0)
                    if n:
                        events.append((now_wall, now_perf, n))
                        pieces.append(msg.get("content") or "")
                    if msg.get("stop"):
                        final = msg
        except httpx.HTTPError as e:
            raise ServerError(f"request to llama-server failed: {e}") from e

        if not events:
            raise ServerError("llama-server streamed no tokens")
        first_wall, first_perf, first_n = events[0]
        last_wall, last_perf, _ = events[-1]
        client_tokens = sum(e[2] for e in events)
        gaps_ms = [(b[1] - a[1]) * 1000 for a, b in zip(events, events[1:])]
        t = final.get("timings") or {}
        return {
            "text": "".join(pieces),
            "t_send": t_send_wall,
            "t_first": first_wall,
            "t_last": last_wall,
            "ttft_ms": (first_perf - t_send) * 1000,
            "total_ms": (last_perf - t_send) * 1000,
            "client_tokens": client_tokens,
            "client_events": len(events),
            "client_gen_tps": (
                (client_tokens - first_n) / (last_perf - first_perf) if last_perf > first_perf else None
            ),
            "itl_ms": [round(g, 3) for g in gaps_ms],
            "itl_p50_ms": percentile(gaps_ms, 50),
            "itl_p95_ms": percentile(gaps_ms, 95),
            "itl_max_ms": max(gaps_ms) if gaps_ms else None,
            "prompt_tokens": t.get("prompt_n"),
            "prompt_ms": t.get("prompt_ms"),
            "prompt_tps": t.get("prompt_per_second"),
            "gen_tokens": t.get("predicted_n"),
            "gen_tps": t.get("predicted_per_second"),
            "stop_type": final.get("stop_type"),
        }

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_text(self) -> str:
        if self._log is not None:
            self._log.flush()
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def log_tail(self, lines: int = 25) -> str:
        return "\n".join(self.log_text().splitlines()[-lines:])


# ---------------------------------------------------------------- log parsing


def _first(pattern: str, text: str, cast=str):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        return cast(m.group(1).strip())
    except ValueError:
        return None


def parse_log(text: str) -> dict:
    """Facts llama-server reports about what it actually loaded and how.

    Recorded with every configuration, and used to check that the settings in the
    experiment file are the ones the server used. Missing facts are None.
    """
    facts: dict = {
        "build": _first(r"build (\d+ \([0-9a-f]+\))", text),
        "arch": _first(r"print_info: arch\s+= (\S+)", text),
        "file_type": _first(r"print_info: file type\s+= (.+?)\s*$", text),
        "file_size": _first(r"print_info: file size\s+= ([\d.]+ \w+)", text),
        "bpw": _first(r"print_info: file size\s+= .*?\(([\d.]+) BPW\)", text, float),
        "model_params": _first(r"print_info: model params\s+= (.+?)\s*$", text),
        "n_layer": _first(r"print_info: n_layer\s+= (\d+)", text, int),
        "n_ctx_train": _first(r"print_info: n_ctx_train\s+= (\d+)", text, int),
        "n_ctx": _first(r"llama_context: n_ctx\s+= (\d+)", text, int),
        "n_batch": _first(r"llama_context: n_batch\s+= (\d+)", text, int),
        "n_ubatch": _first(r"llama_context: n_ubatch\s+= (\d+)", text, int),
        "flash_attn": _first(r"llama_context: flash_attn\s+= (\w+)", text),
        "n_threads": _first(r"system_info: n_threads = (\d+)", text, int),
        "n_threads_batch": _first(r"system_info: n_threads = \d+ \(n_threads_batch = (\d+)\)", text, int),
        "graph_splits": _first(r"graph: nodes = \d+, splits = (\d+)", text, int),
        "offloaded_layers": None,
        "total_layers": None,
        "kv_type_k": None,
        "kv_type_v": None,
    }
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
    if m:
        facts["offloaded_layers"], facts["total_layers"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"llama_kv_cache: size = .*?K \((\w+)\).*?V \((\w+)\)", text)
    if m:
        facts["kv_type_k"], facts["kv_type_v"] = m.group(1), m.group(2)

    # Buffer sizes per device and kind, e.g. {"CUDA0": {"model": 1819.11, "KV": 288.0}}.
    buffers: dict[str, dict[str, float]] = {}
    for dev, kind, mib in re.findall(r"(\S+)\s+(model|KV|compute|output) buffer size =\s*([\d.]+) MiB", text):
        slot = buffers.setdefault(dev, {})
        slot[kind] = round(slot.get(kind, 0.0) + float(mib), 2)
    facts["buffers_mib"] = buffers
    gpu = buffers.get("CUDA0", {})
    facts["gpu_model_mib"] = gpu.get("model")
    facts["gpu_kv_mib"] = gpu.get("KV")
    facts["gpu_compute_mib"] = gpu.get("compute")
    return facts
