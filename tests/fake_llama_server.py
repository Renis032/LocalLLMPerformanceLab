"""A stand-in for llama-server, for tests that need no GPU and no model.

It accepts the flags the harness passes, prints log lines in the format the harness
parses, and serves /health, /tokenize, /apply-template and a streaming /completion. A
context size of 999 makes it fail while loading, to test how failures are handled.

For quality runs (ignore_eos off), a prompt containing "say: X" gets X as its answer.
"""

import argparse
import datetime
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN_DELAY_S = 0.005
PROMPT_DELAY_S = 0.01
TOTAL_LAYERS = 9


def log_lines(args) -> list[str]:
    offloaded = TOTAL_LAYERS if args.ngl == "all" else min(int(args.ngl), TOTAL_LAYERS)
    threads = args.t or 6
    fa = {"on": "enabled", "off": "disabled"}.get(args.fa, "auto")
    return [
        "0.00.001.000 I cmn  common_param: common_params_print_info: build 0 (0000000) with a test stub",
        f"0.00.002.000 I cmn  common_param: system_info: n_threads = {threads} "
        f"(n_threads_batch = {args.tb or threads}) / 12 | CPU : AVX2 = 1 |",
        "0.00.003.000 I print_info: file type   = Q4_K - Medium",
        "0.00.003.000 I print_info: file size   = 0.01 GiB (4.96 BPW) ",
        "0.00.003.000 I print_info: arch                  = fake",
        "0.00.003.000 I print_info: n_layer               = 8",
        "0.00.003.000 I print_info: model params          = 0.01 B",
        f"0.00.004.000 I load_tensors: offloaded {offloaded}/{TOTAL_LAYERS} layers to GPU",
        "0.00.004.000 I load_tensors:        CUDA0 model buffer size =    10.00 MiB",
        f"0.00.005.000 I llama_context: n_ctx                 = {args.c}",
        f"0.00.005.000 I llama_context: n_batch               = {args.b}",
        f"0.00.005.000 I llama_context: n_ubatch              = {args.ub}",
        f"0.00.005.000 I llama_context: flash_attn            = {fa}",
        "0.00.006.000 I llama_kv_cache:      CUDA0 KV buffer size =     4.00 MiB",
        f"0.00.006.000 I llama_kv_cache: size =    4.00 MiB (  {args.c} cells,   8 layers,  1/1 seqs), "
        f"K ({args.ctk}):    2.00 MiB, V ({args.ctv}):    2.00 MiB",
        "0.00.007.000 I sched_reserve:      CUDA0 compute buffer size =     1.00 MiB",
        "0.00.007.000 I sched_reserve: graph: nodes = 100, splits = 1, input objects = 1, input tensors = 1",
    ]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the server log to the lines above
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _event(self, obj) -> None:
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tokenize":
            self._json(200, {"tokens": list(range(len(body.get("content", "").split())))})
        elif self.path == "/apply-template":
            messages = body["messages"]
            # Like SmolLM3's template, this one stamps today's date into every prompt.
            today = datetime.date.today().strftime("%d %B %Y")
            mode = "/think" if body["chat_template_kwargs"]["enable_thinking"] else "/no_think"
            text = f"Today Date: {today} {mode}\n" + "\n".join(m["content"] for m in messages)
            self._json(200, {"prompt": text})
        elif self.path == "/completion":
            if body.get("ignore_eos", True):
                self._stream(body)
            else:
                self._answer(body)
        else:
            self._json(404, {"error": "not found"})

    def _answer(self, body: dict) -> None:
        """A chat answer: the words after "say: ", ending by itself."""
        n_prompt = len(body["prompt"].split())
        said = re.search(r"say: (.*)", body["prompt"], re.DOTALL)
        text = said.group(1).strip() if said else "no idea"
        if text == "RAMBLE":
            text = " ".join(["more"] * body["n_predict"])
        pieces = re.findall(r"\s*\S+|\s+$", text)[: body["n_predict"]]
        cut_off = len(text.split()) >= body["n_predict"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, piece in enumerate(pieces):
            self._event({"content": piece, "tokens": [i], "stop": False})
            time.sleep(0.001)
        self._event({
            "content": "",
            "tokens": [],
            "stop": True,
            "stop_type": "limit" if cut_off else "eos",
            "timings": {
                "prompt_n": n_prompt, "prompt_ms": 1.0, "prompt_per_second": 1000.0 * n_prompt,
                "predicted_n": len(pieces), "predicted_ms": 1.0, "predicted_per_second": 1000.0,
            },
        })

    def _stream(self, body: dict) -> None:
        n_prompt = len(body["prompt"].split())
        n = body["n_predict"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        t0 = time.perf_counter()
        time.sleep(PROMPT_DELAY_S)
        prompt_s = time.perf_counter() - t0
        first = last = None
        for i in range(n):
            if i:
                time.sleep(TOKEN_DELAY_S)
            last = time.perf_counter()
            first = first or last
            self._event({"content": " tok", "tokens": [i], "stop": False})
        gen_s = last - first
        self._event({
            "content": "",
            "tokens": [],
            "stop": True,
            "stop_type": "limit",
            "timings": {
                "prompt_n": n_prompt,
                "prompt_ms": prompt_s * 1000,
                "prompt_per_second": n_prompt / prompt_s,
                "predicted_n": n,
                "predicted_ms": gen_s * 1000,
                "predicted_per_second": (n - 1) / gen_s if gen_s else 0.0,
            },
        })


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-m")
    ap.add_argument("-c", type=int, default=4096)
    ap.add_argument("-ngl", default="all")
    ap.add_argument("-fa", default="auto")
    ap.add_argument("-t", type=int)
    ap.add_argument("-tb", type=int)
    ap.add_argument("-b", type=int, default=2048)
    ap.add_argument("-ub", type=int, default=512)
    ap.add_argument("-ctk", default="f16")
    ap.add_argument("-ctv", default="f16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args, _ = ap.parse_known_args()

    if args.c == 999:
        print("0.00.010.000 E llama_init_from_model: failed to allocate the KV cache (test stub)", flush=True)
        sys.exit(1)
    for line in log_lines(args):
        print(line, flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"0.00.300.000 I srv  server_on_li: listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
