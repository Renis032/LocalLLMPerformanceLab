"""Purpose: experiment files are read exactly as written, and mistakes are caught before a run.

A typo in a YAML file should stop the run with the setting's name, not silently fall
back to a default and benchmark the wrong thing. Also checks that sweeps expand into
the right configurations and that the configuration hash, which matches results
across sessions, changes only when what is measured changes.
"""

from pathlib import Path

import pytest

from llmlab import config
from llmlab.config import ConfigError, load_experiment, parse_experiment

from conftest import make_raw


def test_defaults_fill_in(tmp_path):
    exp = parse_experiment(make_raw(tmp_path, run={}, server={}), base_dir=tmp_path)
    (v,) = exp.variants
    assert v.label == "base"
    assert v.config["server"]["gpu_layers"] == "all"
    assert v.config["server"]["flash_attn"] == "on"
    assert v.config["workload"]["seed"] == 1
    assert v.config["idle"]["max_gpu_load_pct"] == 15.0


def test_yaml_on_off_become_words(tmp_path):
    raw = make_raw(tmp_path)
    f = tmp_path / "exp.yaml"
    f.write_text(
        "name: yaml\n"
        f"engine: {{server: '{raw['engine']['server']}'}}\n"
        f"model: {{path: '{raw['model']['path']}'}}\n"
        "server:\n  flash_attn: off\n"
        "workload: {prompt: hello there}\n"
        "sweep:\n  server.flash_attn: [on, off]\n",
        encoding="utf-8",
    )
    exp = load_experiment(f)
    assert [v.config["server"]["flash_attn"] for v in exp.variants] == ["on", "off"]
    assert [v.label for v in exp.variants] == ["flash_attn=on", "flash_attn=off"]
    assert exp.source_text.startswith("name: yaml")


def test_sweep_is_a_cartesian_product(tmp_path):
    exp = parse_experiment(
        make_raw(tmp_path, sweep={"server.gpu_layers": ["all", 18], "server.ctx": [512, 1024]}),
        base_dir=tmp_path,
    )
    assert [v.label for v in exp.variants] == [
        "gpu_layers=all, ctx=512", "gpu_layers=all, ctx=1024",
        "gpu_layers=18, ctx=512", "gpu_layers=18, ctx=1024",
    ]
    assert len({v.config_hash for v in exp.variants}) == 4
    assert exp.variants[2].overrides == {"server.gpu_layers": 18, "server.ctx": 512}


def test_sweep_over_model_paths_labels_by_file_name(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"x")
    exp = parse_experiment(make_raw(tmp_path, sweep={"model.path": ["a.gguf", "b.gguf"]}), base_dir=tmp_path)
    assert [v.label for v in exp.variants] == ["path=a.gguf", "path=b.gguf"]
    assert Path(exp.variants[0].config["model"]["path"]) == (tmp_path / "a.gguf").resolve()


def test_a_repeated_prompt_is_a_longer_input(tmp_path):
    one = parse_experiment(make_raw(tmp_path), base_dir=tmp_path).variants[0]
    exp = parse_experiment(make_raw(tmp_path, sweep={"workload.prompt_repeat": [1, 3]}), base_dir=tmp_path)
    assert [v.label for v in exp.variants] == ["prompt_repeat=1", "prompt_repeat=3"]
    assert exp.variants[0].prompt == one.prompt and exp.variants[1].prompt.count(one.prompt) == 3
    # Repeating once sends the same prompt, so it stays comparable with sessions before the setting existed.
    assert exp.variants[0].config_hash == one.config_hash
    assert exp.variants[1].config_hash != one.config_hash
    with pytest.raises(ConfigError, match="prompt_repeat"):
        parse_experiment(make_raw(tmp_path, workload={"prompt_repeat": 0}), base_dir=tmp_path)


@pytest.mark.parametrize(
    "sections, message",
    [
        ({"server": {"ctxx": 1}}, "server.ctxx: unknown setting"),
        ({"server": {"gpu_layers": "most"}}, "server.gpu_layers"),
        ({"server": {"ctx": 0}}, "server.ctx"),
        ({"server": {"flash_attn": "maybe"}}, "server.flash_attn"),
        ({"server": {"cache_type_k": "q2"}}, "server.cache_type_k"),
        ({"run": {"repeats": 0}}, "run.repeats"),
        ({"run": {"warmup": True}}, "run.warmup"),
        ({"idle": {"max_gpu_load_pct": 150}}, "idle.max_gpu_load_pct"),
        ({"workload": {"prompt_file": "p.txt"}}, "set exactly one of prompt or prompt_file"),
        ({"engine": {"type": "vllm"}}, "engine.type"),
        ({"sweep": {"run.repeats": [1, 2]}}, "sweep.run.repeats: not a setting that can be swept"),
        ({"sweep": {"server.nope": [1]}}, "sweep.server.nope"),
        ({"sweep": {"server.ctx": []}}, "non-empty list"),
        ({"sweep": {"server.ctx": [512, 512]}}, "repeats a value"),
        ({"sweep": {"server.ctx": [512, -1]}}, "in the sweep variant server.ctx=-1"),
        ({"name": ""}, "name: required"),
    ],
)
def test_invalid_settings_are_named(tmp_path, sections, message):
    with pytest.raises(ConfigError, match=message.replace(".", r"\.").replace("(", r"\(")):
        parse_experiment(make_raw(tmp_path, **sections), base_dir=tmp_path)


def test_prompt_file_is_relative_to_the_experiment(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "p.txt").write_text("from a file\n", encoding="utf-8")
    raw = make_raw(tmp_path)
    raw["workload"] = {"prompt_file": "prompts/p.txt"}
    (v,) = parse_experiment(raw, base_dir=tmp_path).variants
    assert v.prompt == "from a file\n"
    assert v.config["workload"]["prompt_file"] == str((tmp_path / "prompts" / "p.txt").resolve())

    raw["workload"] = {"prompt_file": "prompts/missing.txt"}
    with pytest.raises(ConfigError, match="not found"):
        parse_experiment(raw, base_dir=tmp_path)


def test_lab_home_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMLAB_HOME", str(tmp_path))
    monkeypatch.delenv("LLMLAB_DB", raising=False)
    raw = make_raw(tmp_path, model={"path": "${LLMLAB_HOME}/models/m.gguf"})
    (v,) = parse_experiment(raw, base_dir=Path.cwd()).variants
    assert Path(v.config["model"]["path"]) == (tmp_path / "models" / "m.gguf").resolve()
    assert config.default_db_path() == tmp_path / "results" / "lab.sqlite"


def test_config_hash_tracks_what_is_measured(tmp_path):
    def h(**sections):
        return parse_experiment(make_raw(tmp_path, **sections), base_dir=tmp_path).variants[0].config_hash

    base = h()
    assert h(run={"repeats": 10, "cooldown_s": 2.0}) == base  # how carefully, not what
    assert h(idle={"seconds": 5.0}) == base
    assert h(server={"ctx": 1024}) != base
    assert h(workload={"n_predict": 64}) != base
    assert h(workload={"prompt": "something else"}) != base

    # The prompt counts by content, not by where it is stored.
    f = tmp_path / "same.txt"
    f.write_text(make_raw(tmp_path)["workload"]["prompt"], encoding="utf-8")
    raw = make_raw(tmp_path)
    raw["workload"] = {"prompt_file": str(f), "n_predict": 32, "seed": 1}
    assert parse_experiment(raw, base_dir=tmp_path).variants[0].config_hash == base


def test_the_shipped_experiment_is_valid():
    root = Path(__file__).resolve().parents[1]
    exp = load_experiment(root / "experiments" / "smollm3-baseline.yaml")
    assert [v.label for v in exp.variants] == ["gpu_layers=all", "gpu_layers=18"]
    assert exp.variants[0].prompt.endswith("Continue these notes with a practical checklist:\n")
