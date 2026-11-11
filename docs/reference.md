# Reference

Everything about using the harness: commands, experiment files, the checks, statistics, and the database.

- [Commands](#commands)
- [Experiment files](#experiment-files)
- [Trust checks](#trust-checks)
- [Statistics](#statistics)
- [Database](#database)

## Commands

Run from the repository root (`uv` is at `D:\LLMLab\tools\uv\uv.exe` on the development PC).

| Command | Purpose |
|---|---|
| `uv run python -m llmlab run FILE.yaml` | Run an experiment and store everything in the database |
| `uv run python -m llmlab run FILE.yaml --dry-run` | Check the file and print each configuration's llama-server or Claude Code command, without running |
| `uv run python -m llmlab sessions` | List recorded sessions |
| `uv run python -m llmlab show [ID]` | Show a session's runs, medians, memory breakdown and checks (default: the latest). `--json` for machine-readable output |
| `uv run python -m llmlab compare A B` | Check that two sessions agree within 5% (`--tolerance`) on median generation and prompt speed |
| `uv run python -m llmlab quality` | Speed, answer quality and cost, one line per scored configuration |
| `uv run python -m llmlab regrade [ID]` | Grade a session's stored answers again with the current graders, after fixing one (no new requests) |
| `uv run python -m llmlab dashboard` | Open the [dashboard](dashboard.md): start runs and watch them live, browse history, overlay any two configurations |
| `uv run python -m llmlab demo` | Open the dashboard on the results recorded while this project was built (`demo/lab-demo.sqlite.gz`), with no GPU or model. Starting runs is off |
| `uv run python -m llmlab demo --rebuild` | Make `demo/lab-demo.sqlite.gz` from your own database (`--db`), with home-folder and checkout paths replaced. It refuses to write the file if your user name is still in it |

`run` writes a live log the dashboard follows (`--no-live` turns it off). It refuses to start a local model if the GPU stays busy for 30 seconds, and records nothing. `--allow-busy` runs anyway and marks the results. Cloud models don't use the GPU, so they skip this check.

`compare` matches configurations by a hash of what they measure. For a local model that is the engine, model, server settings, prompt content, `n_predict` and seed. For Claude it is the model, effort, thinking setting, prompt and system prompt. Claude has no prompt speed, so only generation speed decides its verdict. `compare` also notes any configuration measured with warnings, since a difference is then expected.

The database is `D:\LLMLab\results\lab.sqlite`. Set `LLMLAB_DB` or pass `--db` to use another file.

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 | A configuration failed, or a comparison didn't pass |
| 2 | A bad file or argument |
| 3 | The GPU was busy (local models only) |
| 130 | Stopped with Ctrl+C, or from the dashboard |

## Experiment files

An experiment is a YAML file. A `sweep` turns one file into several configurations, one for each combination of the listed values. `engine.type` decides which settings the file takes: `llama.cpp` (the default) for a local model, or `claude-code` for Claude.

### Local models (llama.cpp)

```yaml
name: smollm3-baseline
model:
  path: ${LLMLAB_HOME}/models/SmolLM3-3B/SmolLM3-Q4_K_M.gguf
server:
  ctx: 4096
  gpu_layers: all
  flash_attn: "on"
workload:
  prompt_file: ../prompts/field-notes.txt   # relative to the experiment file
  n_predict: 512
run:
  warmup: 1
  repeats: 5
sweep:
  server.gpu_layers: [all, 18]
```

| Section | Setting | Default | Purpose |
|---|---|---|---|
| `engine` | `server` | `${LLMLAB_HOME}/engines/llama-b12110-cuda-13.4/llama-server.exe` | The llama-server binary |
| `model` | `path` | (required) | The GGUF model file |
| `server` | `ctx` | 4096 | Context size (`-c`) |
| | `gpu_layers` | `all` | Layers on the GPU (`-ngl`): a number or `all` |
| | `flash_attn` | `on` | Flash Attention (`-fa`): `on`, `off` or `auto` |
| | `threads`, `threads_batch` | server default | CPU threads (`-t`, `-tb`) |
| | `batch`, `ubatch` | server default | Batch sizes (`-b`, `-ub`) |
| | `cache_type_k`, `cache_type_v` | server default (f16) | KV cache types (`-ctk`, `-ctv`), for example `q8_0` |
| | `extra_args` | `[]` | Extra flags, passed last so they can override anything |
| `workload` | `prompt` or `prompt_file` | (one is required) | The prompt, sent raw to `/completion` with no chat template |
| | `n_predict` | 512 | Tokens generated, exactly, in every run |
| | `seed` | 42 | Sampling seed; sampling is greedy (temperature 0) |
| | `prompt_repeat` | 1 | Send the prompt this many times, joined by blank lines, for longer inputs. 1 leaves the configuration hash unchanged |
| `run` | `warmup` | 1 | Runs before measuring; stored but left out of the statistics |
| | `repeats` | 5 | Measured runs |
| | `cooldown_s` | 0.1 | Pause before each run. Longer pauses let the GPU downclock ([finding 1](measurement-findings.md#1-pauses-between-runs-let-the-gpu-downclock)) |
| | `sample_interval_s` | 0.2 | Telemetry interval |
| | `load_timeout_s` | 180 | Longest wait for the model to load |
| `idle` | `seconds` | 3.0 | How long the idle check measures |
| | `max_gpu_load_pct` | 15 | Highest clock-weighted GPU load counted as idle ([finding 3](measurement-findings.md#3-gpu-utilization-overstates-load-at-idle-clocks)) |
| | `max_other_cpu_pct` | 8 | Highest CPU use by other programs during a run ([finding 2](measurement-findings.md#2-other-programs-cpu-use-slows-a-small-model)) |
| | `vram_margin_mib` | 1024 | Free VRAM needed beyond the model file's size. With `gpu_layers: 0` the file doesn't count, since the weights stay in RAM |
| | `max_wait_s` | 30 | How long to wait for a busy GPU before giving up |

Settings in the `engine`, `model`, `server` and `workload` sections can be swept. Unknown settings and invalid values are rejected with the setting's name. YAML reads a bare `on` as true, so the harness accepts both.

### Claude (claude-code)

Claude runs in Anthropic's cloud, so the lab reaches it through Claude Code's headless mode (`claude -p`), signed in with a Claude plan. No API key is needed, and each request uses the plan's usage limits. Claude Code must be installed and signed in.

```yaml
name: claude-baseline
engine:
  type: claude-code
  effort: low
  thinking: auto
model:
  name: claude-haiku-5-5
workload:
  prompt_file: ../prompts/field-notes.txt
sweep:
  model.name: [claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5]
```

| Section | Setting | Default | Purpose |
|---|---|---|---|
| `engine` | `cli` | `claude` | The Claude Code command, or a path to it |
| | `effort` | `low` | Claude's effort level: `low`, `medium`, `high`, `xhigh` or `max` |
| | `thinking` | `off` | `off` asks Claude Code not to think, and a run that thinks anyway doesn't count. `auto` lets the model decide and records its thinking ([finding 7](measurement-findings.md#7-claude-haiku-55-thinks-even-when-told-not-to)) |
| `model` | `name` | (required) | A full model ID such as `claude-sonnet-5-5`. Aliases such as `sonnet` are refused, because they point at newer models over time |
| `workload` | `prompt` or `prompt_file` | (one is required) | The prompt, sent as the user's message |
| | `system_prompt` | `You are a helpful assistant.` | Replaces Claude Code's own system prompt, which is written for coding |
| `run` | `warmup`, `repeats` | 1, 5 | As for local models |
| | `cooldown_s` | 1.0 | Pause between requests |
| | `timeout_s` | 300 | Longest wait for one answer |

Output length can't be fixed: Claude Code continues by itself when an answer reaches a cap ([finding 8](measurement-findings.md#8-capping-claudes-output-turns-one-run-into-several-requests)).

### Quality scoring

Add a `quality` section to either kind of experiment, and after its speed runs the model answers every task in a suite. The speed runs and their configuration hash don't change.

```yaml
quality:
  suite: ../evals/core.yaml
  repeats: 3
```

| Setting | Default | Purpose |
|---|---|---|
| `suite` | none | The task file, relative to the experiment file. Without one, nothing is scored |
| `repeats` | 3 | Times each task is asked. Local models use `workload.seed` + the repeat number as the seed |
| `max_tokens` | 1500 | Local only. The limit on each answer, thinking included. An answer that reaches it counts as wrong |
| `temperature` | 0.7 | Local only. Above zero, so repeats can show whether answers are consistent. Claude's can't be set |
| `thinking` | `on` | Local only. `on` or `off`, passed to the model's own chat template |
| `system_prompt` | `You are a helpful assistant.` | Local only. Claude uses `workload.system_prompt` |

Quality answers use the model's chat template, with today's date replaced by a fixed one ([finding 11](measurement-findings.md#11-a-chat-template-puts-todays-date-in-every-prompt)).

A suite is a YAML file of tasks. Each has an `id`, a `category`, a `prompt`, a `reference` answer its grader must accept, a `wrong` answer its grader must reject, and what its category needs:

| Category | Needs | Passes when |
|---|---|---|
| `exact` | `answers`: accepted answers | The whole reply matches one, ignoring case, spacing, markdown, wrapping quotes and a final period |
| `json` | `expect` or `schema` | The reply is only JSON (a code fence is fine), and equals `expect` or fits `schema`, a subset of JSON Schema. Types are strict: `"34"` is not `34`, and `0` is not `false` |
| `code` | `function`, `tests` | The last Python block passes every test. It runs in a guarded subprocess: only pure standard-library imports, no files, network or processes, 5 s and 512 MiB. This contains mistakes; it isn't a security boundary against code written to escape |
| `reasoning` | `answer` | The last `ANSWER:` line has the known value |

A `<think>` section is removed before grading.

## Trust checks

### Local models

Each **run** counts only if:

| Check | Purpose |
|---|---|
| Exactly `n_predict` tokens were generated | Speeds are only comparable at the same output length |
| The whole prompt was processed | A reused cached prompt would fake prompt speed |
| The client received as many tokens as the server reports | Inter-token latency really is per token |
| Client and server generation speed agree within 5% | The reported speed is what a user actually receives |

Each **configuration** also checks:

| Check | Purpose |
|---|---|
| GPU idle before loading | Another program on the GPU would share its time with the benchmark |
| Layers on the GPU, context size, Flash Attention, and any thread, batch and KV-cache settings match the server log | The server can change settings on its own; results must describe what ran |
| GPU clock steady during runs | A run started on a downclocked GPU measures power management, not the model |
| CPU quiet during runs | Other programs slow the CPU-side work of generation |
| GPU quiet after the runs | Catches another program using the GPU while the benchmark ran |

### Claude

Each **run** counts only if:

| Check | Purpose |
|---|---|
| Claude Code made exactly one API request | It continues by itself after an output limit, which would merge several requests into one run |
| The requested model answered | A fallback to another model would measure the wrong model |
| No hidden thinking, when `thinking: off` | Thinking delays the first visible text ([finding 7](measurement-findings.md#7-claude-haiku-55-thinks-even-when-told-not-to)) |
| No prompt tokens read from or written to the cache | A cached prompt makes later runs faster than the first |
| The answer ended naturally | An answer cut short isn't the work the other runs did |
| The first streamed content arrived when Claude Code says it did (within 150 ms or 15%) | The harness's timing of the stream is independent of Claude Code's own, so the two confirm each other |

Each **configuration** also checks that every run sent the same number of input tokens, so Claude Code added the same context each time ([finding 9](measurement-findings.md#9-claude-code-adds-context-of-its-own)).

### Quality

Each quality run checks:

| Check | Purpose |
|---|---|
| Graders verified on known answers | Every task accepts its reference answer and rejects its wrong one. If not, the suite stops before any request |
| Every task answered | Engine errors leave gaps; three in a row stop the suite (usually a usage limit) |
| Answers ended by themselves | An answer cut off at the token limit counts as wrong, and this says how many there were |
| Each answer measured once | Claude only: an answer from a fallback model, a cached prompt or several API requests doesn't count toward the score |

### Status

A configuration is `ok` when everything passes. It's `warning` when a check fails or a run is invalid, and `failed` when it couldn't run (for example, out of memory). A failed configuration doesn't stop the others.

## Statistics

- **Repeats** are reported as median, minimum, maximum, and coefficient of variation (CV, the standard deviation as a percentage of the mean).
- **Inter-token latency** percentiles (p50, p95, p99) are pooled across all measured runs, since one run of 512 tokens is too few for a p99.
- **Energy per token** = mean instantaneous GPU board power during generation ÷ generation speed. It covers the GPU board only.
- **Claude's generation speed** = output tokens after the first streamed chunk ÷ the time from the first chunk to the last. Chunks carry several tokens each and the stream doesn't say how many, so the first chunk's share is estimated from its characters. Thinking tokens are left out.
- **Claude's time to first token** runs from Claude Code's `init` event, when it is ready to send the request, to the first visible text. It includes the network, queueing and any thinking; Claude Code's start-up is recorded separately.
- **Claude's stream** arrives in chunks, so its gaps are reported as chunk gaps, not inter-token latency.
- **Cost** for Claude is what the run would cost at API list prices, as Claude Code computes it. On a plan, runs count against usage limits instead.
- **Quality** is the share of a task's repeats answered correctly, averaged over tasks. Its 95% interval comes from 10,000 resamples of the tasks with a fixed seed, because the tasks are the sample: a different 24 would score somewhat differently.
- **Consistency** is the share of a task's repeats that give its most common answer, right or wrong, averaged over tasks. Answers are compared after grading reduces them to what matters: for code, which tests passed.
- **Comparing quality** pairs two runs by task and resamples the per-task differences. A difference counts only if its 95% interval excludes zero.
- **Cost per correct answer** is the whole suite's cost divided by its correct answers. For Claude that is the API list price. For a local model it is GPU board energy over the suite at $0.20/kWh (set `LLMLAB_KWH_USD` for your own); it leaves out the CPU, the rest of the PC and the price of the GPU.
- **Missing sensors** show as N/A, never as zero.

## Database

One SQLite file holds every session, so results stay comparable over time. Columns ending in `_json` hold JSON.

| Table | One row per | Holds |
|---|---|---|
| `sessions` | `run` invocation | The experiment file's full text and hash; the host (GPU, driver, CPU, RAM, OS, power plan; no hostname); status |
| `configs` | Configuration | Resolved settings, engine version, model SHA-256 (or Claude's model ID), server or Claude Code command, load time, VRAM before and at peak, server RAM, facts parsed from the server log, check results, the full server log |
| `trials` | Request | Token counts, speeds, TTFT, every inter-token gap, and GPU and CPU telemetry over its generation window. Claude's runs add thinking and cached tokens, API requests, the model that answered, Claude Code's own TTFT, start-up time and list-price cost |
| `samples` | Telemetry sample, 5 per second | GPU utilization, clocks, power, temperature, VRAM, P-state, clock-limit reasons; CPU per core; server CPU and memory |
| `quality_runs` | Scored configuration | The suite's text and hash, its settings, score and interval, consistency, per-category and per-task results, check results, cost or energy |
| `quality_attempts` | Answer | The task, repeat and seed, the full answer, whether it passed and why, tokens and timing |
| `file_hashes` | Model file | SHA-256, reused while the file's size and modification time are unchanged |

The schema is versioned. An older database is upgraded in place when opened, and a database from a newer version is refused.
