# Measurement findings

Building the lab meant finding out where measurements of LLM inference go wrong, on a local GPU and in the cloud. Each case below follows the same pattern: what we saw, the evidence, the cause, and what the harness does about it now.

Findings 1–6 are local: one PC with an RTX 3060 Ti 8 GB (driver 616.92), a Ryzen 5 5600X and 32 GB of RAM, on Windows 11, running SmolLM3 3B Q4_K_M on llama.cpp b12110. Findings 7–9 are about Claude, measured through Claude Code 2.1.311 on a Claude plan. Finding 10 is about the lab's own dashboard, and finding 11 about scoring answers.

| # | Finding | Effect if ignored | Fix |
|---|---|---|---|
| 1 | [Pauses between runs let the GPU downclock](#1-pauses-between-runs-let-the-gpu-downclock) | Speed under-reported by 37% | 0.1 s pause; clock-steadiness check |
| 2 | [Other programs' CPU use slows a small model](#2-other-programs-cpu-use-slows-a-small-model) | Run-to-run variation 24× higher | CPU-quiet check |
| 3 | [GPU utilization overstates load at idle clocks](#3-gpu-utilization-overstates-load-at-idle-clocks) | The idle check could never pass | Clock-weighted GPU load |
| 4 | [llama-server's defaults change what is measured](#4-llama-servers-defaults-change-what-is-measured) | Wrong settings, cached prompts, hidden RAM | Pinned server flags, verified from its log |
| 5 | [NVML's default memory reading includes reserved memory](#5-nvmls-default-memory-reading-includes-reserved-memory) | VRAM 168 MiB too high | NVML v2 memory call |
| 6 | [The power reading lags, and the energy counter is wrong](#6-the-power-reading-lags-and-the-energy-counter-is-wrong) | Energy per token under-reported | Instantaneous power |
| 7 | [Claude Haiku 5.5 thinks even when told not to](#7-claude-haiku-55-thinks-even-when-told-not-to) | First visible token 0.44 s later than it looks | Thinking tokens counted; timing cross-checked |
| 8 | [Capping Claude's output turns one run into several requests](#8-capping-claudes-output-turns-one-run-into-several-requests) | Three requests measured as one | No cap; one-request check |
| 9 | [Claude Code adds context of its own](#9-claude-code-adds-context-of-its-own) | About 600 hidden input tokens per request | Same-input check; start-up excluded from timing |
| 10 | [Watching a run live slowed it down](#10-watching-a-run-live-slowed-it-down) | Generation 3% slower; CPU-quiet check failed | Nothing animated; redraw only on new data |
| 11 | [A chat template puts today's date in every prompt](#11-a-chat-template-puts-todays-date-in-every-prompt) | The same request gives different answers on different days | The date is pinned before sending |

Three questions are still open; see [the end of this page](#open-questions).

---

## 1. Pauses between runs let the GPU downclock

**Symptom.** The same model on the same GPU generated 97 tok/s in one session and 152–154 tok/s in others. Every check passed in the slow session, and its five runs agreed closely with each other (1.3% variation), so nothing looked wrong.

![GPU clock during five runs with a 0.1 s and a 1 s pause](figures/clock-ramp.svg)

**Evidence.** The slow session paused 1 s between runs. In every pause, NVML showed the GPU dropping from performance state P2 to P3: 780 MHz core and 5,001 MHz memory. Each run then started on a slow GPU, and the clock climbed only about 100 MHz per second, from 780 to about 1,215 MHz, so no run ever reached full speed. Run back to back, the clock held near 1,950 MHz the whole time. The warm-up was fast in both sessions because loading the model had already raised the clocks, which made the problem easy to miss.

**Cause.** The NVIDIA driver's power management. Phase 0, run the same morning with the same 1 s pause, stayed at P2 throughout. The driver's behavior changes over time, so a harness can't assume either one.

**What the harness does now.**
- It pauses 0.1 s between runs. That's long enough for the server to finish the previous request, and too short for the GPU to downclock.
- A check, **"GPU clock steady during runs"**, fails any configuration where the clock varied more than 10% within a run. Steady runs vary 0.8–3%; ramping runs varied 33–36%.
- The check doesn't apply when the GPU is mostly waiting on the CPU (below 50% busy, as with half the layers on the CPU). There its clock follows the load by design.

**It can still happen.** Two sessions the next morning (16 and 17) fell to 96–99 tok/s after the first measured run, even with the 0.1 s pause, while eight sessions the night before held 148–150. The GPU flipped between P2 and P3 at about 1,000 MHz, and NVML gave its reason as "GPU idle" (`0x1`) at 96% utilization; the night before it was the power cap (`0x4`) at 1,950 MHz. The check flagged both sessions (clock varying 34–49% within a run). A harness can detect the driver's behaviour, but can't control it without changing system settings.

## 2. Other programs' CPU use slows a small model

**Symptom.** With 18 of 37 layers on the GPU, one session's runs varied 5.9%; the next session's varied 0.25%.

**Evidence.** The harness records the CPU used by everything except the server. During the noisy session, other programs used up to 14.8% of the whole CPU. During the quiet one they used 2.3%. The same pattern held with the model fully on the GPU: a run with 20.6% of the CPU taken by other programs was the slowest of its session (143.9 vs 147–148 tok/s).

**Cause.** Even fully on the GPU, a 3B model's generation is partly CPU-bound: one CPU core launches GPU work and samples each token. With half the layers on the CPU, six cores do the work directly. Background apps such as launchers, overlays and updaters take time from those cores.

**What the harness does now.** A check, **"CPU quiet during runs"**, fails any configuration where other programs used more than 8% of the CPU during a measured run (`idle.max_other_cpu_pct`). Clean runs measure 2–6%, slowed ones 10–19%. `compare` notes when a difference comes from runs measured with warnings.

## 3. GPU utilization overstates load at idle clocks

**Symptom.** With every game closed, the idle check refused to run: NVML reported the GPU 40% busy.

**Evidence.** At the same moment, the GPU ran at 330–375 MHz and drew 18 W, its idle power. Windows' own per-process GPU counters showed the busiest programs (the desktop compositor, a wallpaper app's settings window and the Claude app) using about 1% each.

**Cause.** NVML utilization is the share of time any work runs on the GPU, at whatever clock. At idle clocks, a trickle of desktop drawing keeps the GPU occupied for much of the time.

**What the harness does now.** The idle and quiet checks use **clock-weighted load**: utilization × current clock ÷ maximum clock. The idle desktop measures 6–7% (40% at 345 of 2,160 MHz). A game measures 82% (99% at 1,785 MHz) and is still refused.

## 4. llama-server's defaults change what is measured

| Default | What it does | Harness setting |
|---|---|---|
| `--fit on` | Silently changes settings, such as context size and layers, to fit memory | `--fit off`, and the log is checked against the experiment file |
| Memory-mapped model loading | The model's RAM shows up as file cache, not as the server's memory | `-lm none` |
| Prompt caching per request | A repeated prompt is partly skipped, which fakes prompt speed | `--no-cache-prompt` and `cache_prompt: false` |
| A prompt cache of up to 8 GiB in system RAM | A separate cache that stays on even with the above | `--cache-ram 0` |
| Automatic slot count | The context is split between slots | `-np 1` |
| Default log level | Layer offload and buffer sizes aren't logged | `-lv 4` |

The harness then reads the server's log to confirm what it actually did. That includes layers on the GPU, context size, batch sizes, Flash Attention, KV cache types and thread counts. Any mismatch fails a check.

## 5. NVML's default memory reading includes reserved memory

NVML's original memory call (`nvmlDeviceGetMemoryInfo`, v1) counts memory the driver reserves as used. It read 168 MiB higher than nvidia-smi. The v2 call reports reserved memory separately and matches nvidia-smi to the MiB, so the harness uses v2.

## 6. The power reading lags, and the energy counter is wrong

- **Lagging power reading:** NVML's usual power reading is a one-second rolling average. 0.6 s after a run ended it still read 140 W at 0% utilization, so averaging it over a 3.4 s run understates energy. The harness takes energy per token from NVML's **instantaneous** power field, sampled five times a second, and keeps the average too. This moved energy per token from 1.13 to about 1.30 J/token for the fully-on-GPU configuration.
- **Wrong energy counter:** NVML's cumulative energy counter would be the natural source, but on this GPU it's wrong. Under a steady 171–176 W load it rose at about 28 W, so the harness doesn't use it.

## 7. Claude Haiku 5.5 thinks even when told not to

**Symptom.** In the first cloud session (session 5), every Haiku run failed its checks while Sonnet and Opus passed. The harness's time to first token was 200–400 ms longer than the figure Claude Code reports for the same request.

**Evidence.** Each Haiku run used 60–102 hidden thinking tokens although Claude Code had been told not to think (`MAX_THINKING_TOKENS=0`). In 9 further test runs Haiku thought in 8, including all 3 with thinking also turned off in Claude Code's settings. The stream showed why the two timings disagreed. Claude Code times the first streamed content, which was a thinking block. The harness timed the first visible text, which arrived 0.4 s later. Sonnet and Opus at low effort didn't think in any of their 12 runs.

**Cause.** Claude Haiku 5.5 thinks by default. The API can turn that off, but Claude Code offers no setting that reaches it.

**What the harness does now.**
- Every run records its thinking tokens and the time to its first content of any kind, as well as to the first visible text.
- With `thinking: off`, a run that thinks doesn't count. With `thinking: auto`, the model decides and the thinking is reported. The baseline uses `auto`: Haiku's first visible text arrives after 1.64 s, of which about 0.44 s is thinking.
- The timing cross-check compares Claude Code's figure with the first content, thinking included. The two agree within 39–57 ms on every run so far.

## 8. Capping Claude's output turns one run into several requests

**Symptom.** To match the local runs' 512 generated tokens, an early test capped Claude's output at 512 tokens. One "run" then reported 5,859 input tokens instead of about 1,350.

**Evidence.** The stream held three API requests. Each time an answer reached the cap, Claude Code sent a new request on its own. It added a hidden instruction to resume where the answer was cut off.

**Cause.** Claude Code is an agent: it treats the output limit as an interruption to recover from, not as a stopping point.

**What the harness does now.** Claude's output isn't capped, so its answers vary in length (893 to 1,362 tokens on the baseline prompt). A check fails any run that made more than one API request, and another fails any answer that stopped before its natural end. Speeds are per token, so the different lengths still compare.

## 9. Claude Code adds context of its own

**Symptom.** A two-word prompt was billed as 600 input tokens.

**Evidence.** Claude Code adds its environment details and account notes to every request, even with its own system prompt replaced and its tools, MCP servers and settings files turned off. It also takes 0.5–0.6 s to start before each request.

**Cause.** Claude Code is built to give a coding agent context about its machine.

**What the harness does now.**
- It switches off everything Claude Code can drop: tools, MCP servers, slash commands, settings files, session history, prompt caching and background traffic. It also runs each request in an empty folder.
- A check fails a configuration whose runs didn't all send the same number of input tokens.
- Timing starts at Claude Code's `init` event, when it is ready to send the request, so start-up doesn't count toward time to first token. Start-up is recorded separately.

## 10. Watching a run live slowed it down

**Symptom.** A run watched in the dashboard failed the "CPU quiet during runs" check in both configurations, and the model fully on the GPU generated 2.6–3.1% slower than in the same run with nobody watching.

**Evidence.** Five sessions of the same experiment, back to back, recording every process's CPU time over each run:

| Watched by | Watching browser's CPU | Other programs during runs | CPU quiet | All layers on the GPU |
|---|---|---|---|---|
| Nobody (session 10) | none | 3.9–6.3% | Pass | 149.2 tok/s |
| The dashboard in a hidden browser pane (session 9) | No measurable change | 3.2–6.4% | Pass | 149.5 tok/s |
| A browser drawing the page (sessions 11, 12) | 2.9–3.0% of the CPU | 5.0–11.7% | Fail | 144.6, 145.4 tok/s |
| The same browser after the fix (session 13) | 0.29% | 3.2–7.4% | Pass | 148.9 tok/s |

The browser's CPU went half to its renderer and half to its compositor, and only while a run was live: the same page with the run finished used 0.16%. A hidden pane doesn't paint, which is why session 9 showed nothing. The watching browser was a headless Edge, which composites in software; a browser compositing on the GPU spends less CPU per frame, but draws just as many frames.

**Cause.** The "Running" badge had a pulsing dot, a CSS animation of its shadow that never stopped. A shadow can't be animated by the compositor alone, so the browser repainted the page every frame, about 60 times a second, for the whole run. That took about a third of a CPU core: the same kind of background load as [finding 2](#2-other-programs-cpu-use-slows-a-small-model).

**What the harness does now.**
- Nothing on the page animates, and the page redraws only when new data arrives: once a second during a run, with only the last two minutes of telemetry drawn.
- A test fails if the stylesheet gains an endless animation.
- The cost is measured, not assumed: about 0.3% of the CPU while watching, stated on the Run page next to the switch that turns the live charts off.

## 11. A chat template puts today's date in every prompt

**Symptom.** Scoring answers needs each model's real chat format, so the harness renders prompts with the model's own template. SmolLM3's rendered prompt began with a line that wasn't in the request: `Today Date: 08 November 2026`.

**Evidence.** The same 24 tasks, with the same seed and sampling settings, changing only that date:

| Date in the prompt | Answer text differs from the first | Final answer differs | Right or wrong flips | Tasks correct |
|---|---|---|---|---|
| 15 January 2026 | | | | 13 of 24 |
| 08 November 2026 | 11 of 24 | 3 | 1 | 14 of 24 |
| 20 March 2026 | 12 of 24 | 5 | 2 | 13 of 24 |

**Cause.** The template calls `strftime_now` when it renders. One changed token at the start of the prompt changes the sampler's path through everything after it.

**What the harness does now.** It replaces today's date in the rendered prompt with a fixed one, in the formats templates use, so a prompt is identical on any day.

---

## Open questions

- **Prompt speed with layers split between CPU and GPU isn't reproducible within 5%.** Within one session it ranged from 1,033 to 1,747 tok/s. During CPU-bound generation the GPU sits idle and downclocks, so each request's GPU-heavy prompt phase starts at a different clock, and the phase is too short (0.3–0.5 s) to ramp up. Generation speed in the same configuration is reproducible (+1.7% between sessions). Planned for Phase 5: measure prompt speed separately, with back-to-back prompt-only requests.
- **The split configuration drifted 5.8% between afternoon and midnight.** With 18 of 37 layers on the GPU, session 4 (18:14) generated 22.6 tok/s and session 10 (23:54) 21.3 tok/s, and both passed every check. The eight sessions around midnight (8–15) generated 21.1–21.6 tok/s, within 2.4% of each other. Other programs used more CPU at midnight (6–11% against 2–3%), but across the midnight sessions that range made no difference, so it isn't the whole explanation. The model fully on the GPU moved 3.0%, within the 5% bar. Next step: record the CPU's clock and temperature, which the harness doesn't measure yet.
- **Time to first token runs 15–25 ms above prompt time between back-to-back requests,** against about 2 ms on the first request. The server's log shows the extra time is outside prompt processing, which points at how the streaming HTTP connection is reused. It doesn't affect generation speed.
