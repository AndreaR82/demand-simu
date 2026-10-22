# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two single-file simulators of LLM load on vLLM, cross-linked from their sidebar headers:

- `index.html` — a **blocking groundedness guardrail** for a chatbot.
- `agent.html` — **engineers driving a Claude Code-style coding agent**.

Both exist to build intuition about load, queueing and GPU sizing (count, tensor parallel, quantisation) — not to be accurate about vLLM internals. No build step, no dependencies, no tests, no package manager. The two files deliberately share no code: helpers (knob builder, `spark`, tooltips, rng) are copied, because presets, engine and rendering differ.

## Running and verifying

```bash
open index.html                                   # just works, it is self-contained
open agent.html
python3 -m http.server 8777                       # only needed for browser automation (file:// is blocked there)
```

requestAnimationFrame is throttled when the tab is backgrounded, so **driving the UI from browser automation does not advance the clock**. Verify the engine headlessly instead, by stepping it directly in the page context:

```js
reset(); const C = cfg();
for (let i = 0; i < 60000; i++) step(0.01, C);   // 600 simulated seconds
S.turns / S.t                                     // ≈ sessions / (think + draft + service) at low load
```

That identity is the main regression check: in a closed loop, steady-state throughput must match the offered-load estimate printed in the Derived config panel. Under overload the expected signature is queue depth growing, `queue wait >> service time`, busy at 100%, and throughput pinned below `peakTok / (promptTok + outTok)` per replica.

For `agent.html` the checks are:

```js
reset(); const C = cfg();
for (let i = 0; i < 360000; i++) step(0.01, C);  // 1 h warm-up; tasks take minutes, so run long
const a = {...S.cnt}, t0 = S.t;
for (let i = 0; i < 720000; i++) step(0.01, C);  // 2 h measured
const n = S.cnt.tasks - a.tasks;
n / (S.t - t0) * (P.review + (S.cnt.wall - a.wall) / n)   // ≈ S.ses.length within ~3% (Little's Law on the closed loop)
```

- 1 session with `subP = 0` → `pct(S.slow, 50)` ≈ 1.00 (the idle baseline is exact).
- APC off → cache hit 0%. One replica → the three routers behave identically.
- Defaults (80 engineers) sit just below the KV cliff: TTFT p95 < 0.1 s, hit ≈ 95%, no queue. 100 engineers, APC off, `cont` 0.5, fp16 KV or keep-thinking each tip it into an eviction storm with TTFT p95 in seconds.
- Task slowdown p95 is already ~4x at the defaults with no queueing: the baseline decodes alone at batch 1 (~110 tok/s), loaded streams run at ~20–25. That is the batching trade-off, not a bug — which is why the target defaults to 5x (green at 80 engineers, red past the cliff) rather than something like 1.5x, which only holds at ~10 engineers.

After editing, reload and confirm the sliders still read their defaults — Chrome form restoration silently resurrects previous values, which is why every input carries `autocomplete="off"`.

## Architecture of `index.html`

Four layers, in file order: presets (`GPUS`, `MODELS`, `QUANTS`) → knob table (`KNOBS`) → sim (`cfg`, `reset`, `step`) → render (`drawFlow`, `renderUI`, `spark`). The loop in `frame()` runs the sim in fixed `dt = 0.01` steps (capped at 400 per frame) and renders once.

**`KNOBS` is the UI.** Each row is `[group, key, label, min, max, step, default, unit, tooltip]`; `buildKnobs()` generates the slider, the readout and the `?` tooltip, and writes into the global `P`. Adding a parameter means adding a row — never hand-write slider markup.

**`cfg()` is the whole performance model**, recomputed every frame so knobs take effect mid-run. It derives, from model × quantisation × GPU × TP: weight bytes per GPU, the leftover KV pool and therefore the KV-limited concurrency, single-stream prefill (compute-bound: `flops / 2·active_params`) and decode (bandwidth-bound: `bandwidth / weight_bytes`). Effective concurrency is `min(max_num_seqs, seqsByKv)` — the KV limit biting first at long context is a lesson the dashboard is meant to show, not a bug to clamp away.

**Batching** uses one saturating curve, per request: `rate(B) = single · (k+1) / (B + k)`. `kp` (prefill) is small because one long prompt already saturates compute; `kd` (decode) is large because the batch shares one weight read. Both facts — aggregate throughput rises with B, per-request speed falls — must survive any change here.

**`step()`** advances sessions (`think → draft → wait`) and, per replica, admits from the queue up to effective concurrency, then burns token budget: `prefill` until `rem <= 0`, then `decode`, then completion. Latency is recorded per call (enqueue → finish) and per turn (max over the turn's calls, since a fanned-out turn ends with its slowest call) — the SLO tile scores turns. The SLO (default 1 s, p95) budgets the **guardrail step only** — enqueue of the first call to finish of the last, queue wait included, chatbot draft and think time excluded — not the full user-facing response.

Model presets are **approximations**, deliberately overridable in the UI: `raw` is weights resident in HBM, `act` is parameters touched per token, `kvkb` is KV bytes per token at fp16. "Effective parameter" names (E2B/E4B) understate resident bytes, so never derive weight size from a model's name.

## Architecture of `agent.html`

Same layering and the same `KNOBS` convention (plus checkboxes for APC and keep-thinking, selects for router and KV dtype). Different workload and engine:

**Workload.** Each main session (`engineers × spe`) loops `review → task → review`. A task is N steps (lognormal); a step is one LLM call then a tool run, except the last call. Context = shared base + engineer prompt + every tool result + every tool call (+ thinking if kept); `cont` decides whether the next task continues it. Past `compactAt` a compaction call decodes a summary and the context restarts at base + summary. A task may fan out once to `subK` subagents (fresh context, own loop); the parent waits for the slowest.

**Engine (`runReplica`) runs one forward pass at a time per replica**: `T = overhead + max(memory, compute)`, memory = weights × `union(B)` + Σ context of decoding sequences × KV bytes, compute = tokens × 2·active / FLOPS. `union(n) = 1 − (1 − act/raw)^n` is the MoE expert union — the "3B active" advantage vanishes at batch. Decodes take 1 token each; prefills share `max_num_batched_tokens − decodes`, oldest first (chunked prefill, so long prefills stall every decode). Admission reserves the whole context + output, so there is no preemption.

**KV pool per replica** = running reservations (`kvRun`) + idle cached prefixes (`kvIdle`, one entry per session, evicted whole-session LRU) + the shared base, held once while APC is on. After a call a session's usable prefix is its prompt minus base, plus the output only if thinking is kept (stripped thinking breaks the prefix match). The router (`cache-aware` / `sticky` / `least-loaded`) decides whether the next call lands where that prefix lives.

**Slowdown baseline**: every call adds `callIdeal()` — the same call alone on an idle replica with a cache that never evicts (`s.warm` tracks that cache; `s.cTok` tracks the real one). Tool times are added as is; a fan-out adds its slowest child's ideal.

**Timeline**: sessions record `[kind, t0, t1]` segments via `setSeg` (30 min kept). The top rows are the first `WATCH` sessions at full height with subagent stripes; below, every session as a squeezed raster.

## Scope

Guardrail: decisions already taken and intentionally not revisited: queue abstraction rather than an iteration-level scheduler, closed-loop sessions rather than open-loop Poisson, flow animation as the hero panel, zero dependencies. Parked and not to be built unasked: paged-KV-block engine with prefill/decode contention, chatbot model co-resident on the same GPUs, two-stage escalation guardrail, headless GPU-sizing sweeps.

Coding agent: decided: forward-pass roofline engine without paged blocks, whole-session LRU prefix cache, cache-aware default router, task slowdown as headline with TTFT and per-stream decode as diagnostics, timeline as the hero panel, MiniMax-M2 fp8 / fp8 KV on 2 × TP4 H200 as default. Parked and not to be built unasked: workday arrival ramp, preemption/recompute, block-level (tail) eviction, speculative decoding, prefill/decode disaggregation, a small side model for harness chores, attention FLOPs growing with context, headless sweeps.
