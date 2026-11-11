"""Experiment files: load a YAML experiment, check it, fill in defaults, expand sweeps.

An experiment describes one or more configurations ("variants") to benchmark. Each
variant is fully resolved before anything runs: absolute paths, the prompt text read
in, every default filled. The database then records exactly what ran.

engine.type picks the settings an experiment file may contain: llama.cpp (a local
model, the default) or claude-code (Claude, through Claude Code on the user's plan).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .quality.suite import SuiteError, load_suite


def lab_home() -> Path:
    """Where tools, engines and models live (outside the repository)."""
    return Path(os.environ.get("LLMLAB_HOME", r"D:\LLMLab"))


def default_db_path() -> Path:
    env = os.environ.get("LLMLAB_DB")
    return Path(env) if env else lab_home() / "results" / "lab.sqlite"


# Every setting a llama.cpp experiment file may contain, with its default.
DEFAULTS: dict = {
    "name": None,
    "description": "",
    "engine": {
        "type": "llama.cpp",
        "server": "${LLMLAB_HOME}/engines/llama-b12110-cuda-13.4/llama-server.exe",
    },
    "model": {
        "path": None,
    },
    "server": {
        "ctx": 4096,
        "gpu_layers": "all",
        "flash_attn": "on",
        "threads": None,
        "threads_batch": None,
        "batch": None,
        "ubatch": None,
        "cache_type_k": None,
        "cache_type_v": None,
        "extra_args": [],
    },
    "workload": {
        "prompt": None,
        "prompt_file": None,
        "prompt_repeat": 1,  # send the prompt this many times over, to test long inputs
        "n_predict": 512,
        "seed": 42,
    },
    "run": {
        "warmup": 1,
        "repeats": 5,
        "cooldown_s": 0.1,
        "sample_interval_s": 0.2,
        "load_timeout_s": 180,
    },
    "idle": {
        "seconds": 3.0,
        "max_gpu_load_pct": 15.0,
        "max_other_cpu_pct": 8.0,
        "vram_margin_mib": 1024,
        "max_wait_s": 30.0,
    },
    "quality": {
        "suite": None,  # a task suite to score after the speed runs
        "repeats": 3,
        "max_tokens": 1500,  # a limit on each answer, thinking included
        "temperature": 0.7,  # above zero, so repeats can show whether answers are consistent
        "thinking": "on",  # the model's own reasoning mode, where it has one
        "system_prompt": "You are a helpful assistant.",
    },
    "sweep": {},
}

# Every setting a claude-code experiment file may contain, with its default.
CLAUDE_DEFAULTS: dict = {
    "name": None,
    "description": "",
    "engine": {
        "type": "claude-code",
        "cli": "claude",  # the Claude Code command, or a path to it
        "effort": "low",
        "thinking": "off",  # off, or auto: the model decides (Claude Haiku 5.5 thinks either way)
    },
    "model": {
        "name": None,  # a full model ID, such as claude-haiku-5-5
    },
    "workload": {
        "prompt": None,
        "prompt_file": None,
        # Replaces Claude Code's own system prompt, which is written for coding.
        "system_prompt": "You are a helpful assistant.",
    },
    "run": {
        "warmup": 1,
        "repeats": 5,
        "cooldown_s": 1.0,
        "timeout_s": 300,
    },
    "quality": {
        "suite": None,  # a task suite to score after the speed runs
        "repeats": 3,
    },
    "sweep": {},
}

ENGINE_DEFAULTS = {"llama.cpp": DEFAULTS, "claude-code": CLAUDE_DEFAULTS}

# The sections each engine's variants carry, in order.
ENGINE_SECTIONS = {
    "llama.cpp": ("engine", "model", "server", "workload", "run", "idle", "quality"),
    "claude-code": ("engine", "model", "workload", "run", "quality"),
}

EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Sections that define what a variant measures. Two variants with the same values
# here measure the same thing, so they can be compared across sessions.
IDENTITY_SECTIONS = ("engine", "model", "server", "workload")

KV_CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")


class ConfigError(ValueError):
    """The experiment file is invalid. The message names the setting and the problem."""


@dataclass(frozen=True)
class Variant:
    index: int
    label: str  # e.g. "gpu_layers=18", or "base" when there is no sweep
    overrides: dict  # sweep values for this variant, keyed "section.setting"
    config: dict  # resolved engine, model, server, workload, run and idle sections
    config_hash: str  # identity of what is measured; equal hashes are comparable

    @property
    def prompt(self) -> str:
        return self.config["workload"]["prompt"]


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    path: Path | None
    source_text: str
    source_sha256: str
    variants: list[Variant]


# ---------------------------------------------------------------- loading


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"experiment file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name} is not valid YAML: {e}") from None
    return parse_experiment(raw, base_dir=path.parent, source_path=path, source_text=text)


def parse_experiment(
    raw, base_dir: Path, source_path: Path | None = None, source_text: str = ""
) -> Experiment:
    if raw is None:
        raise ConfigError("the experiment file is empty")
    engine = _engine_type(raw)
    defaults = ENGINE_DEFAULTS[engine]
    base = _merge(defaults, raw, "")
    axes = _sweep_axes(base.pop("sweep"), defaults)
    _check_header(base)

    keys = [key for key, _ in axes]
    combos = list(itertools.product(*(values for _, values in axes))) if axes else [()]
    variants = []
    for index, combo in enumerate(combos):
        cfg = copy.deepcopy(base)
        overrides = dict(zip(keys, combo))
        for key, value in overrides.items():
            section, setting = key.split(".")
            cfg[section][setting] = copy.deepcopy(value)
        try:
            _validate(cfg)
            _resolve(cfg, base_dir)
        except ConfigError as e:
            if overrides:
                raise ConfigError(f"{e} (in the sweep variant {_plain(overrides)})") from None
            raise
        sections = {k: cfg[k] for k in ENGINE_SECTIONS[engine]}
        variants.append(
            Variant(
                index=index,
                label=_label(overrides, sections),
                overrides=overrides,
                config=sections,
                config_hash=config_hash(sections),
            )
        )
    return Experiment(
        name=base["name"],
        description=base["description"] or "",
        path=source_path,
        source_text=source_text,
        source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        variants=variants,
    )


def config_hash(cfg: dict) -> str:
    """Hash of the settings that define what is measured.

    Run settings (repeats, cooldown, idle limits) are left out: changing them changes
    how carefully something is measured, not what is measured. So is the quality section,
    which has its own hash: adding a suite doesn't change what the speed runs measure.
    The prompt counts by content, so moving the prompt file doesn't change the hash.
    """
    w = cfg["workload"]
    prompt_sha = hashlib.sha256(w["prompt"].encode("utf-8")).hexdigest()
    if cfg["engine"]["type"] == "claude-code":
        identity = {
            # Where Claude Code is installed doesn't change what is measured.
            "engine": {k: v for k, v in cfg["engine"].items() if k != "cli"},
            "model": cfg["model"],
            "workload": {"prompt_sha256": prompt_sha, "system_prompt": w["system_prompt"]},
        }
        blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    identity = {
        "engine": cfg["engine"],
        "model": cfg["model"],
        "server": cfg["server"],
        "workload": {
            "prompt_sha256": prompt_sha,
            "n_predict": w["n_predict"],
            "seed": w["seed"],
        },
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- structure


def _engine_type(raw) -> str:
    engine = raw.get("engine") if isinstance(raw, dict) else None
    kind = engine.get("type", "llama.cpp") if isinstance(engine, dict) else "llama.cpp"
    if kind not in ENGINE_DEFAULTS:
        raise ConfigError(f"engine.type: must be one of {', '.join(ENGINE_DEFAULTS)}, got {kind!r}")
    return kind


def _merge(defaults: dict, user, where: str) -> dict:
    if not isinstance(user, dict):
        raise ConfigError(f"{where or 'the experiment file'}: expected a set of settings, got {user!r}")
    out = copy.deepcopy(defaults)
    for key, value in user.items():
        path = f"{where}.{key}" if where else str(key)
        if key not in defaults:
            raise ConfigError(f"{path}: unknown setting. Allowed here: {', '.join(defaults)}")
        if isinstance(defaults[key], dict) and key != "sweep":
            if value is not None:
                out[key] = _merge(defaults[key], value, path)
        else:
            out[key] = value
    return out


def _sweep_axes(sweep, defaults: dict) -> list[tuple[str, list]]:
    if sweep is None:
        return []
    if not isinstance(sweep, dict):
        raise ConfigError("sweep: expected settings mapped to lists of values")
    axes = []
    for key, values in sweep.items():
        parts = str(key).split(".")
        if (
            len(parts) != 2
            or parts[0] not in IDENTITY_SECTIONS
            or parts[0] not in defaults
            or parts[1] not in defaults[parts[0]]
            or key == "engine.type"
        ):
            example = "model.name" if defaults is CLAUDE_DEFAULTS else "server.gpu_layers"
            sections = ", ".join(s for s in IDENTITY_SECTIONS if s in defaults)
            raise ConfigError(
                f"sweep.{key}: not a setting that can be swept. "
                f"Use section.setting from the {sections} sections, for example {example}"
            )
        if not isinstance(values, list) or not values:
            raise ConfigError(f"sweep.{key}: expected a non-empty list of values")
        if len({json.dumps(v, sort_keys=True, default=str) for v in values}) != len(values):
            raise ConfigError(f"sweep.{key}: the list repeats a value")
        axes.append((str(key), values))
    return axes


def _check_header(cfg: dict) -> None:
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ConfigError("name: required, a short name for the experiment")
    cfg["name"] = cfg["name"].strip()
    if cfg["description"] is not None and not isinstance(cfg["description"], str):
        raise ConfigError("description: must be text")


# ---------------------------------------------------------------- values


def _int(v, path: str, minimum: int, what: str = "") -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise ConfigError(f"{path}: must be {what or f'a whole number >= {minimum}'}, got {v!r}")
    return v


def _num(v, path: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < minimum or (
        maximum is not None and v > maximum
    ):
        limit = f"between {minimum:g} and {maximum:g}" if maximum is not None else f">= {minimum:g}"
        raise ConfigError(f"{path}: must be a number {limit}, got {v!r}")
    return float(v)


def _text(v, path: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"{path}: required, got {v!r}")
    return v


def _validate_prompt(w: dict) -> None:
    if (w["prompt"] is None) == (w["prompt_file"] is None):
        raise ConfigError("workload: set exactly one of prompt or prompt_file")
    if w["prompt"] is not None:
        _text(w["prompt"], "workload.prompt")
    else:
        _text(w["prompt_file"], "workload.prompt_file")


def _validate_runs(r: dict) -> None:
    r["warmup"] = _int(r["warmup"], "run.warmup", 0)
    r["repeats"] = _int(r["repeats"], "run.repeats", 1)
    r["cooldown_s"] = _num(r["cooldown_s"], "run.cooldown_s", 0)


def _validate_quality(q: dict, local: bool) -> None:
    if q["suite"] is None:
        return
    _text(q["suite"], "quality.suite")
    q["repeats"] = _int(q["repeats"], "quality.repeats", 1)
    if not local:
        return
    q["max_tokens"] = _int(q["max_tokens"], "quality.max_tokens", 1)
    q["temperature"] = _num(q["temperature"], "quality.temperature", 0, 2)
    # YAML reads a bare on/off as true/false.
    thinking = {True: "on", False: "off"}.get(q["thinking"], q["thinking"])
    if thinking not in ("on", "off"):
        raise ConfigError(f"quality.thinking: must be on or off, got {q['thinking']!r}")
    q["thinking"] = thinking
    q["system_prompt"] = _text(q["system_prompt"], "quality.system_prompt")


def _validate(cfg: dict) -> None:
    """Check every value and normalize it in place."""
    _validate_quality(cfg["quality"], cfg["engine"]["type"] == "llama.cpp")
    if cfg["engine"]["type"] == "claude-code":
        _validate_claude(cfg)
        return
    e = cfg["engine"]
    e["server"] = _text(e["server"], "engine.server")
    cfg["model"]["path"] = _text(cfg["model"]["path"], "model.path")

    s = cfg["server"]
    s["ctx"] = _int(s["ctx"], "server.ctx", 1)
    if s["gpu_layers"] != "all":
        s["gpu_layers"] = _int(s["gpu_layers"], "server.gpu_layers", 0, "a whole number >= 0 or 'all'")
    # YAML reads a bare on/off as true/false.
    fa = {True: "on", False: "off"}.get(s["flash_attn"], s["flash_attn"])
    if fa not in ("on", "off", "auto"):
        raise ConfigError(f"server.flash_attn: must be on, off or auto, got {s['flash_attn']!r}")
    s["flash_attn"] = fa
    for key in ("threads", "threads_batch", "batch", "ubatch"):
        if s[key] is not None:
            s[key] = _int(s[key], f"server.{key}", 1)
    for key in ("cache_type_k", "cache_type_v"):
        if s[key] is not None and s[key] not in KV_CACHE_TYPES:
            raise ConfigError(f"server.{key}: must be one of {', '.join(KV_CACHE_TYPES)}, got {s[key]!r}")
    extra = s["extra_args"] or []
    if not isinstance(extra, list) or any(
        isinstance(a, bool) or not isinstance(a, (str, int, float)) for a in extra
    ):
        raise ConfigError("server.extra_args: must be a list of command-line arguments")
    s["extra_args"] = [str(a) for a in extra]

    w = cfg["workload"]
    _validate_prompt(w)
    w["prompt_repeat"] = _int(w["prompt_repeat"], "workload.prompt_repeat", 1)
    w["n_predict"] = _int(w["n_predict"], "workload.n_predict", 1)
    w["seed"] = _int(w["seed"], "workload.seed", 0)

    r = cfg["run"]
    _validate_runs(r)
    r["sample_interval_s"] = _num(r["sample_interval_s"], "run.sample_interval_s", 0.05)
    r["load_timeout_s"] = _num(r["load_timeout_s"], "run.load_timeout_s", 1)

    i = cfg["idle"]
    i["seconds"] = _num(i["seconds"], "idle.seconds", 0)
    i["max_gpu_load_pct"] = _num(i["max_gpu_load_pct"], "idle.max_gpu_load_pct", 0, 100)
    i["max_other_cpu_pct"] = _num(i["max_other_cpu_pct"], "idle.max_other_cpu_pct", 0, 100)
    i["vram_margin_mib"] = _num(i["vram_margin_mib"], "idle.vram_margin_mib", 0)
    i["max_wait_s"] = _num(i["max_wait_s"], "idle.max_wait_s", 0)


def _validate_claude(cfg: dict) -> None:
    e = cfg["engine"]
    e["cli"] = _text(e["cli"], "engine.cli")
    if e["effort"] not in EFFORTS:
        raise ConfigError(f"engine.effort: must be one of {', '.join(EFFORTS)}, got {e['effort']!r}")
    th = {False: "off"}.get(e["thinking"], e["thinking"])  # YAML reads a bare off as false
    if th not in ("off", "auto"):
        raise ConfigError(
            f"engine.thinking: must be off or auto (the model decides), got {e['thinking']!r}. "
            "Claude Code can't force thinking on"
        )
    e["thinking"] = th

    name = _text(cfg["model"]["name"], "model.name").strip()
    # Aliases such as "sonnet" point at newer models over time, so results under one
    # name wouldn't stay comparable.
    if not name.startswith("claude-"):
        raise ConfigError(
            f"model.name: use a full model ID such as claude-haiku-5-5, got {name!r}. "
            "Aliases like 'sonnet' change meaning when a new model comes out"
        )
    cfg["model"]["name"] = name

    w = cfg["workload"]
    _validate_prompt(w)
    w["system_prompt"] = _text(w["system_prompt"], "workload.system_prompt")

    r = cfg["run"]
    _validate_runs(r)
    r["timeout_s"] = _num(r["timeout_s"], "run.timeout_s", 1)


def _expand_path(p: str, base_dir: Path) -> str:
    home = str(lab_home())
    p = os.path.expandvars(p.replace("${LLMLAB_HOME}", home).replace("$LLMLAB_HOME", home))
    path = Path(p)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve(cfg: dict, base_dir: Path) -> None:
    q = cfg["quality"]
    if q["suite"] is not None:
        q["suite"] = _expand_path(q["suite"], base_dir)
        try:
            suite = load_suite(q["suite"])
        except SuiteError as e:
            raise ConfigError(f"quality.suite: {e}") from None
        q["tasks"] = len(suite.tasks)
    e = cfg["engine"]
    if e["type"] == "claude-code":
        # A bare command name is looked up on PATH when the experiment runs.
        if "/" in e["cli"] or "\\" in e["cli"]:
            e["cli"] = _expand_path(e["cli"], base_dir)
    else:
        e["server"] = _expand_path(e["server"], base_dir)
        cfg["model"]["path"] = _expand_path(cfg["model"]["path"], base_dir)
    w = cfg["workload"]
    if w["prompt_file"] is not None:
        f = Path(_expand_path(w["prompt_file"], base_dir))
        if not f.is_file():
            raise ConfigError(f"workload.prompt_file: not found: {f}")
        w["prompt_file"] = str(f)
        w["prompt"] = f.read_text(encoding="utf-8")
        if not w["prompt"].strip():
            raise ConfigError(f"workload.prompt_file: the file is empty: {f}")
    if w.get("prompt_repeat", 1) > 1:
        w["prompt"] = "\n\n".join([w["prompt"]] * w["prompt_repeat"])


def _short(v) -> str:
    if isinstance(v, str) and ("/" in v or "\\" in v):
        return Path(v).name
    return str(v)


def _label(overrides: dict, cfg: dict) -> str:
    if not overrides:
        return "base"
    settings = [k.split(".")[1] for k in overrides]
    parts = []
    for key in overrides:
        section, setting = key.split(".")
        if key == "model.name":  # a Claude model ID says it all
            parts.append(cfg[section][setting])
            continue
        name = setting if settings.count(setting) == 1 else key
        parts.append(f"{name}={_short(cfg[section][setting])}")
    return ", ".join(parts)


def _plain(overrides: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in overrides.items())
