"""Purpose: llama-server is started with the intended settings, and its log is read correctly.

The log parser is tested against a real b12110 log, because the harness relies on it
to confirm what the server actually loaded (layers on the GPU, context size, buffer
sizes). The server lifecycle is tested against a stand-in server, so no GPU is needed.
"""

import sys

import pytest

from llmlab import llamacpp
from llmlab.llamacpp import FIXED_ARGS, LlamaServer, ServerError, build_command, parse_log

from conftest import FIXTURES


def _server(**overrides):
    s = {"ctx": 4096, "gpu_layers": "all", "flash_attn": "on", "threads": None, "threads_batch": None,
         "batch": None, "ubatch": 256, "cache_type_k": None, "cache_type_v": None, "extra_args": []}
    s.update(overrides)
    return s


def test_command_maps_settings_and_pins_the_rest():
    cmd = build_command("llama-server.exe", "m.gguf", _server(extra_args=["--fit", "on"]), 8080)
    assert cmd[:3] == ["llama-server.exe", "-m", "m.gguf"]
    joined = " ".join(cmd)
    assert "-c 4096 -ngl all -fa on -ub 256" in joined
    assert "-t " not in joined and "-ctk" not in joined  # unset settings are left to the server
    assert " ".join(FIXED_ARGS) in joined
    assert "--host 127.0.0.1 --port 8080" in joined
    assert cmd[-2:] == ["--fit", "on"]  # extra_args go last, so they can override


def test_parse_log_reads_the_real_b12110_log():
    facts = parse_log((FIXTURES / "llama-server-b12110-lv4.log").read_text(encoding="utf-8"))
    assert facts["build"] == "12110 (cd959bae9)"
    assert (facts["offloaded_layers"], facts["total_layers"]) == (37, 37)
    assert (facts["gpu_model_mib"], facts["gpu_kv_mib"], facts["gpu_compute_mib"]) == (1819.11, 288.0, 84.51)
    assert facts["buffers_mib"]["CUDA_Host"] == {"model": 205.49, "output": 0.49, "compute": 20.02}
    assert facts["file_type"] == "Q4_K - Medium"
    assert facts["file_size"] == "1.78 GiB"
    assert facts["bpw"] == 4.96
    assert facts["model_params"] == "3.08 B"
    assert (facts["arch"], facts["n_layer"], facts["n_ctx_train"]) == ("smollm3", 36, 65536)
    assert (facts["n_ctx"], facts["n_batch"], facts["n_ubatch"]) == (4096, 2048, 512)
    assert facts["flash_attn"] == "enabled"
    assert (facts["n_threads"], facts["n_threads_batch"]) == (6, 6)
    assert (facts["kv_type_k"], facts["kv_type_v"]) == ("f16", "f16")
    assert facts["graph_splits"] == 2


def test_parse_log_of_nothing_gives_nothing():
    facts = parse_log("")
    assert facts["offloaded_layers"] is None and facts["n_ctx"] is None
    assert facts["buffers_mib"] == {}


def test_server_lifecycle_against_the_stub(fake_engine, tmp_path):
    server = LlamaServer(sys.executable, "m.gguf", _server(ctx=512), tmp_path / "server.log")
    try:
        load_s = server.start(timeout=30)
        assert 0 < load_s < 30
        assert server.tokenize("one two three") == 3
        r = server.complete("one two three", n_predict=16, seed=1)
    finally:
        server.stop()
    assert server.proc.poll() is not None
    assert r["client_tokens"] == r["gen_tokens"] == 16
    assert r["prompt_tokens"] == 3
    assert len(r["itl_ms"]) == 15
    assert r["t_send"] <= r["t_first"] <= r["t_last"]
    assert abs(r["client_gen_tps"] - r["gen_tps"]) / r["gen_tps"] < 0.2
    assert parse_log(server.log_text())["n_ctx"] == 512


def test_server_that_fails_to_load_reports_its_log(fake_engine, tmp_path):
    server = LlamaServer(sys.executable, "m.gguf", _server(ctx=999), tmp_path / "server.log")
    with pytest.raises(ServerError, match="failed to allocate the KV cache"):
        server.start(timeout=30)
    server.stop()


def test_engine_version_parses_the_banner(monkeypatch):
    class Done:
        stdout = "version: 0.9.0-dev (build 12110, commit cd959bae9)\nbuilt with Clang 20.1.8 for Windows x86_64\n"
        stderr = ""

    monkeypatch.setattr(llamacpp.subprocess, "run", lambda *a, **k: Done())
    assert llamacpp.engine_version("x") == "0.9.0-dev b12110 (cd959bae9)"
