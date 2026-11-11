"""Shared test helpers: a fake GPU, a fake sampler, and stand-ins for llama-server and Claude Code."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pytest

from llmlab import claude_code, config, llamacpp, telemetry

FAKE_SERVER = Path(__file__).with_name("fake_llama_server.py")
FAKE_CLAUDE = Path(__file__).with_name("fake_claude_code.py")
FIXTURES = Path(__file__).with_name("fixtures")


class FakeGpu:
    def __init__(self, util: float = 0.0, clock: float = 2000.0, max_clock: float = 2000.0):
        self.util = util
        self.clock = clock
        self.max_clock_mhz = max_clock

    def read(self) -> dict:
        return {"util_pct": self.util, "clock_mhz": self.clock, "used_mib": 500.0, "free_mib": 7000.0,
                "total_mib": 8192.0, "temp_c": 40}

    def info(self) -> dict:
        return {"index": 0, "name": "Fake GPU", "driver": "0", "cuda_driver": "0.0", "vram_total_mib": 8192,
                "power_limit_w": 100.0, "pcie_gen_max": 4, "pcie_width_max": 16}

    def close(self) -> None:
        pass


class FakeSampler:
    """No telemetry: keeps tests independent of whatever else the real GPU is doing."""

    def __init__(self, out_path, interval, gpu_index=0):
        self.out_path = out_path

    def start(self, timeout=15.0):
        return self

    def watch(self, pid):
        pass

    def stop(self):
        return []


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    """Route llama-server launches to the stand-in server, and skip real telemetry."""
    # Working folders of failed variants are kept for debugging; keep test ones in tmp_path.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    real_build = llamacpp.build_command

    def build(exe, model, server, port):
        return [sys.executable, str(FAKE_SERVER)] + real_build(exe, model, server, port)[1:]

    monkeypatch.setattr(llamacpp, "build_command", build)
    monkeypatch.setattr(llamacpp, "engine_version", lambda exe: "test b0 (0000000)")
    monkeypatch.setattr(telemetry, "Sampler", FakeSampler)


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    """Route Claude Code launches to the stand-in; FAKE_CLAUDE_MODE picks its behaviour."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("FAKE_CLAUDE_MODE", raising=False)
    real_build = claude_code.build_command

    def build(exe, *args):
        return [sys.executable, str(FAKE_CLAUDE)] + real_build(exe, *args)[1:]

    monkeypatch.setattr(claude_code, "build_command", build)
    monkeypatch.setattr(claude_code, "find_cli", lambda cli: cli)
    monkeypatch.setattr(claude_code, "cli_version", lambda exe: "9.9.9")


def make_cloud_experiment(tmp_path: Path, **sections) -> config.Experiment:
    raw = {
        "name": "fake-cloud",
        "engine": {"type": "claude-code"},
        "model": {"name": "claude-test-1"},
        "workload": {"prompt": "one two three four five"},
        "run": {"warmup": 1, "repeats": 2, "cooldown_s": 0},
    }
    return config.parse_experiment(_deep_update(raw, sections), base_dir=tmp_path)


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def make_raw(tmp_path: Path, **sections) -> dict:
    model = tmp_path / "model.gguf"
    if not model.exists():
        model.write_bytes(b"GGUF" + bytes(1000))
    raw = {
        "name": "fake",
        "engine": {"server": sys.executable},  # any existing file; launches are routed to the stub
        "model": {"path": str(model)},
        "server": {"ctx": 512},
        "workload": {"prompt": "one two three four five six seven eight", "n_predict": 32, "seed": 1},
        "run": {"warmup": 1, "repeats": 3, "cooldown_s": 0.05},
        "idle": {"seconds": 0.0, "max_wait_s": 0},
    }
    return _deep_update(copy.deepcopy(raw), sections)


def make_experiment(tmp_path: Path, **sections) -> config.Experiment:
    return config.parse_experiment(make_raw(tmp_path, **sections), base_dir=tmp_path)
