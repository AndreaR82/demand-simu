# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file simulator (`index.html`) of a **blocking groundedness guardrail** for a chatbot, served by vLLM. It exists to build intuition about load, queueing and GPU sizing (count, tensor parallel, quantisation) — not to be accurate about vLLM internals. No build step, no dependencies, no tests, no package manager.

## Running and verifying

```bash
open index.html                                   # just works, it is self-contained
python3 -m http.server 8777                       # only needed for browser automation (file:// is blocked there)
```

requestAnimationFrame is throttled when the tab is backgrounded, so **driving the UI from browser automation does not advance the clock**. Verify the engine headlessly instead, by stepping it directly in the page context:

```js
reset(); const C = cfg();
for (let i = 0; i < 60000; i++) step(0.01, C);   // 600 simulated seconds
S.turns / S.t                                     // ≈ sessions / (think + draft + service) at low load
```

That identity is the main regression check: in a closed loop, steady-state throughput must match the offered-load estimate printed in the Derived config panel. Under overload the expected signature is queue depth growing, `queue wait >> service time`, busy at 100%, and throughput pinned below `peakTok / (promptTok + outTok)` per replica.

After editing, reload and confirm the sliders still read their defaults — Chrome form restoration silently resurrects previous values, which is why every input carries `autocomplete="off"`.

## Architecture of `index.html`

Four layers, in file order: presets (`GPUS`, `MODELS`, `QUANTS`) → knob table (`KNOBS`) → sim (`cfg`, `reset`, `step`) → render (`drawFlow`, `renderUI`, `spark`). The loop in `frame()` runs the sim in fixed `dt = 0.01` steps (capped at 400 per frame) and renders once.

**`KNOBS` is the UI.** Each row is `[group, key, label, min, max, step, default, unit, tooltip]`; `buildKnobs()` generates the slider, the readout and the `?` tooltip, and writes into the global `P`. Adding a parameter means adding a row — never hand-write slider markup.

**`cfg()` is the whole performance model**, recomputed every frame so knobs take effect mid-run. It derives, from model × quantisation × GPU × TP: weight bytes per GPU, the leftover KV pool and therefore the KV-limited concurrency, single-stream prefill (compute-bound: `flops / 2·active_params`) and decode (bandwidth-bound: `bandwidth / weight_bytes`). Effective concurrency is `min(max_num_seqs, seqsByKv)` — the KV limit biting first at long context is a lesson the dashboard is meant to show, not a bug to clamp away.

**Batching** uses one saturating curve, per request: `rate(B) = single · (k+1) / (B + k)`. `kp` (prefill) is small because one long prompt already saturates compute; `kd` (decode) is large because the batch shares one weight read. Both facts — aggregate throughput rises with B, per-request speed falls — must survive any change here.

**`step()`** advances sessions (`think → draft → wait`) and, per replica, admits from the queue up to effective concurrency, then burns token budget: `prefill` until `rem <= 0`, then `decode`, then completion. Latency is recorded per call (enqueue → finish) and per turn (max over the turn's calls, since a fanned-out turn ends with its slowest call) — the SLO tile scores turns. The SLO (default 1 s, p95) budgets the **guardrail step only** — enqueue of the first call to finish of the last, queue wait included, chatbot draft and think time excluded — not the full user-facing response.

Model presets are **approximations**, deliberately overridable in the UI: `raw` is weights resident in HBM, `act` is parameters touched per token, `kvkb` is KV bytes per token at fp16. "Effective parameter" names (E2B/E4B) understate resident bytes, so never derive weight size from a model's name.

## Scope

Decisions already taken and intentionally not revisited: queue abstraction rather than an iteration-level scheduler, closed-loop sessions rather than open-loop Poisson, flow animation as the hero panel, zero dependencies. Parked and not to be built unasked: paged-KV-block engine with prefill/decode contention, chatbot model co-resident on the same GPUs, two-stage escalation guardrail, headless GPU-sizing sweeps.
