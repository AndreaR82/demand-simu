# Load-testing vLLM for Claude Code with simulated engineers

How to size a vLLM deployment for engineers who use **Claude Code against a self-hosted model**. The
approach has four steps:

1. Run Claude Code headless on a set of tasks.
2. Let a second LLM play the engineer.
3. Record every model call.
4. Replay the recorded sessions at scale with guideLLM, and turn the result into a GPU count.

This is the agentic counterpart of `guidellm-guardrail-loadtest.md`, and it follows the same structure.

**Status: nothing has run against vLLM yet. The orchestrator has been dry-run against a mock.** Facts
were checked on 2026-10-21 against:
- **Claude Code 2.1.308**: flags from `claude --help`, env vars from the official docs.
- **vLLM**: its Claude Code integration page. The latest release on PyPI is 0.32.0.
- **guideLLM 0.8.1**: trace formats and replay options read from the 0.8.1 wheel's source.

Code in this directory:

| File | What it does |
|---|---|
| `simulate_engineers.py` | The orchestrator and simulated engineer (§4). |
| `mock_messages_server.py` | A scripted stand-in for vLLM's `/v1/messages`. It can also play the engineer. Use it for dry runs without a GPU or an API bill. |
| `example_tasks.jsonl` | Two tasks against this repo, showing the task format (§3). |

On 2026-10-22, real Claude Code 2.1.308 ran end to end against the mock with `--sandbox local`. Those
dry runs confirmed the facts marked *(dry run)* below. The vLLM side, the capture proxy, the trace
builder, the guideLLM replay and `--sandbox docker` are all **still unrun**. Run 0 is where they get
proven. Anything the docs couldn't confirm is marked *measure in Run 0*.

---

## 1. What we are simulating

An engineer gives Claude Code a request. Claude Code runs an agent loop: model call → tool call (read,
edit, bash, tests) → model call → … until the model ends its turn. The engineer then reads the result,
thinks, and replies, or stops. Every model call goes to vLLM.

| Property | Guardrail (other doc) | Claude Code session |
|---|---|---|
| Calls per user action | 1 (a small fan-out at most) | Tens per engineer turn. One published Claude Code conversation made 92 requests in ~17 min, with 4 subagents [A] |
| Input per call | 2k shared + 10k–55k unique | A shared tools + system prefix (~15k tokens in [A]; *measure yours in Run 0*) + the whole conversation so far, growing until compaction |
| Prefix reuse | ~11% | Very high: ~98% within a conversation, measured against the Anthropic API in [A] |
| Output per call | 16 tokens | Tool calls, code edits, explanations: tens to thousands of tokens |
| Gaps between calls | none | Tool runs (seconds to minutes) and engineer think time (minutes) |
| Calls in flight per user | 1 | 1 or more, because subagents run in parallel with their own contexts |
| What limits a replica | Prefill compute | KV memory holding long contexts, and decode bandwidth |

[A] = `agentic-coding-analysis`, see Sources. Its numbers come from Claude models on the Anthropic API.
An open model on vLLM will take different paths, so use them for orientation only.

### 1.1 Why the prefix cache decides the outcome

With prefix caching, each call only prefills what was appended since the previous call (a tool
result, a user message). The rest is read from KV blocks that are already cached. Those blocks
survive while the session waits for a tool or for the engineer, **but only until memory pressure
evicts them**. After that, the next call re-prefills the whole context, which can be 50–150k tokens.

That is the failure mode to look for. As concurrency rises, the hit rate collapses, prefill grows by
an order of magnitude, and latency hits a cliff. It is the KV-limited concurrency effect shown in
`index.html` (`seqsByKv`), made worse because idle sessions also compete for the cache.

Two client-side things can break the cache before load even matters (details in §5 and §10):
- **Claude Code's attribution block** at the start of the system prompt.
- **vLLM's handling of the system turns Claude Code inserts on every request.** This is vLLM issue
  #53393, still open; a comment on it says it still happens on v0.28.0.

### 1.2 What changes the traffic

- **The model.** A weaker or more heavily quantised model takes more steps, loops more, and writes more
  output. Record with the model you are sizing for.
- **Claude Code configuration.** Compaction window, max output tokens, subagents, background
  (Haiku-tier) calls, CLAUDE.md, hooks and MCP servers all change prompt size or call count (§5).
- **The task mix.** Bug fixes, features, refactors and read-only "explain this" sessions have different
  shapes.
- **Engineer behaviour.** How often they intervene, how long they pause, and when they give up.

---

## 2. Architecture

```
  tasks.jsonl ──► orchestrator ◄──────► engineer LLM
                       │                (separate endpoint, NOT the server under test)
                       │ one per session
           ┌───────────▼──────────────┐
           │ sandbox container        │
           │  repo @ base_commit      │
           │  claude -p (stream-json) │──► transcripts: $CLAUDE_CONFIG_DIR/projects/*.jsonl
           └───────────┬──────────────┘
                       │ POST /v1/messages
                ┌──────▼────────┐
                │ capture proxy │──► requests.db (every request/response + response time)
                └──────┬────────┘
                       │ POST /v1/messages
                ┌──────▼────────┐
                │ vLLM (SUT)    │──► /metrics
                └───────────────┘

  offline:  requests.db + transcripts ──► build_minimal_traces.py ──► WEKA trace
            WEKA trace ──► guideLLM 0.8.1 replay (kind=weka) ──► vLLM (SUT), at hundreds of sessions
```

| Component | Existing tool | Status |
|---|---|---|
| Orchestrator | `simulate_engineers.py` (this directory) | Starts sandboxes, drives engineer turns, applies the timing model (§4.3), runs each task's check. Dry-run against the mock. |
| Simulated engineer | `simulate_engineers.py` | A prompt and a loop (§4). The backend is any OpenAI-compatible endpoint, or Claude through the Anthropic SDK. |
| Sandbox | Docker. SWE-bench-style tasks come with prebuilt images. | `--sandbox docker`: one container per session, with network access only to the proxy. The image must have Claude Code installed. **Not yet run.** `--sandbox local` (a temp clone on the host, unsandboxed) is for dry runs only. |
| Capture proxy | `claude-code-proxy` (seifghazi), whose upstream is set by `ANTHROPIC_FORWARD_URL` | It is used in front of the Anthropic API. Its README doesn't say whether it works in front of vLLM. *Verify in Run 0.* |
| Trace builder | `agentic-coding-analysis`: `build_minimal_traces.py` | Reads `requests.db` plus Claude Code transcripts and writes WEKA traces. |
| Replayer | guideLLM 0.8.1, `--data kind=weka` with `--profile kind=replay` | Closed-loop per conversation, with prefix sharing and parallel subagents (§7, Run 3). |

Still to write: two small post-processing steps, the gap re-inflation (§6.3) and turn slowdown (§8).

---

## 3. Tasks

Each task is one JSONL line:

```json
{"id": "billing-042", "image": "registry/acme-billing:2026-10", "base_commit": "9f1c2e7",
 "setup": "make deps", "category": "bugfix",
 "intent": "Invoices for EU customers round VAT per line instead of per invoice. Fix rounding to per-invoice, keep the public API, add a regression test. The fix belongs in billing/vat.py; do not touch the PDF renderer.",
 "opener": "EU invoices have VAT off by a cent sometimes, can you look?",
 "check": "pytest tests/billing -q", "max_engineer_turns": 6}
```

| Field | Meaning |
|---|---|
| `image`, `workdir` | `--sandbox docker`: the image, and the repo's path inside it (default `/repo`). |
| `repo` | `--sandbox local`: a git URL, or a path relative to the tasks file. |
| `base_commit`, `setup` | Checked out, then run once before the session starts. Optional. |
| `intent` | The full spec, **hidden from the agent**. Only the simulated engineer sees it. |
| `opener` | The first message, as terse as a real one. If you leave it out, the engineer LLM writes it from `intent`. |
| `check` | Lets the orchestrator tell the engineer whether the tests pass. For capacity, its job is to make the session end the way a real one would, not to score the model. |
| `max_engineer_turns` | Hard cap, so a stuck session can't loop forever. |

**Sources.** SWE-bench Verified and SWE-Gym have hundreds of tasks with images. Closed issues from
your own repos are more representative, because **repo size drives context growth**: a monorepo
produces longer reads and bigger tool results.

**Mix.** Match what engineers actually do. Include read-only sessions ("how does X work?") and
multi-turn feature work, not just single-issue bug fixes. 50–100 tasks are enough for Run 1.

---

## 4. The simulated engineer

### 4.1 Loop

```
start `claude -p --input-format stream-json ...` in the sandbox            (§5.3)
send opener
loop:
    read stream-json events until the `result` message                     (agent turn done)
    if result.is_error: end the session as agent_error                     (retries exhausted, ...)
    shown = final assistant text + `git diff --stat` + check result (pass/fail, last lines)
    d = engineer_llm(ENGINEER_PROMPT, task.intent, visible conversation, shown)
    log modelled gap g = t_read + t_review + t_pause + t_type                (§4.3)
    if d.action != "reply" or turns == max_engineer_turns: close stdin; break
    sleep(g × ENGINEER_TIME_SCALE)                                          (0 while recording)
    send d.message
```

The engineer LLM runs on **its own endpoint**. That can be another vLLM, or a hosted API. If it shared
the server under test, its own calls would distort the measurements. `simulate_engineers.py` implements
this loop, and its `--help` lists every knob. Its outputs:

| Output | Contents |
|---|---|
| `sessions.jsonl` | One line per session: the outcome, engineer turns, calls, retries, agent time, modelled think time, the final check, the Claude Code version, and the context window Claude Code assumed. The outcome is `accepted`, `abandoned`, `max_turns`, `agent_error`, `turn_timeout`, `engineer_error` or `error`. |
| `<session>/engineer.jsonl` | One line per engineer turn. It holds the wall-clock `t_result` and `t_sent`; the agent turn's calls, retries, tokens and durations; lines changed; the check result; the decision; and the modelled think time, split into its parts and compared with `recorded_gap_s` (§6.3). |
| `<session>/stream.jsonl` | Claude Code's stream-json events, verbatim. |
| `<session>/cc/projects/` | Claude Code's transcripts, for the trace builder (§6.2). |

A dry run needs no GPU and no API bill. Run `mock_messages_server.py` and point both Claude Code
(`--base-url`) and the engineer (`--engineer openai --engineer-url <mock>/v1`) at it. The commands are
in the script's docstring.

### 4.2 Prompt (draft)

```text
You are a software engineer using an AI coding agent in your own repository.
You know exactly what you want; it is in <intent>. The agent does not see <intent>.

Rules:
- Reveal intent the way a busy engineer would: start vague, add detail only when the
  agent asks or gets it wrong. Never paste <intent> verbatim.
- Reply in at most 2-3 sentences. No praise, no summaries of what the agent did.
- If the change is wrong, say what is wrong as you would notice it (a failing test,
  a wrong file, a missed case), not how to fix it.
- Patience: after {patience} replies that don't move things forward, abandon.
- Accept as soon as the change does what <intent> asks and the check passes; don't
  ask for polish you wouldn't ask for in real life.

Output JSON only: {"action": "reply" | "accept" | "abandon", "message": "<text or empty>"}

<intent>{task.intent}</intent>
```

For an OpenAI-compatible engineer, fix the temperature and seed (`--engineer-temperature`,
`--engineer-seed`). Claude's API no longer accepts sampling parameters, so a Claude engineer varies from
run to run. Record enough sessions for that to average out. Spot-check 10 conversations by hand before
Run 1. The usual faults are a user who is too helpful, too wordy, or who leaks the spec.

### 4.3 Timing model

Engineer time never reaches the GPU, but it sets how many sessions are engaged at once. The figures
below are **illustrative defaults**; replace them with the calibration in §4.4.

| Component | Default | Notes |
|---|---|---|
| `t_read` | words of the final assistant text ÷ 4 words/s | Reading the agent's reply |
| `t_review` | 1 s per changed line, capped at 120 s | Reading the diff |
| `t_pause` | lognormal, median 60 s, p90 600 s | Context switch: the engineer is doing something else |
| `t_type` | words of the reply ÷ 0.7 words/s | ≈ 40 wpm |

**Record with `ENGINEER_TIME_SCALE = 0`** so Run 1 doesn't take hours of wall time. Log the modelled
gap `g` for each engineer turn, and add it back into the trace before replay (§6.3). Tool time is
real and is never compressed.

### 4.4 Calibrate from real transcripts

If the team already uses Claude Code, its transcripts hold the real distributions. They are in
`~/.claude/projects/<project>/<session>.jsonl`, and in 2.1.308 each line has a `type`, a
`timestamp` and a `sessionId`:

- **Human think time** is the `timestamp` of a `user` record that is a typed prompt (no
  `toolUseResult`, not `isMeta`) minus the `timestamp` of the `assistant` record just before it.
- **Calls per engineer turn** is the number of distinct `requestId`s on `assistant` records between
  two typed prompts. Records with `isSidechain` are subagents.
- **Context growth** is `message.usage.input_tokens + cache_read_input_tokens +
  cache_creation_input_tokens` per `requestId`.

These come from Claude models, so use them for **engineer** behaviour, not agent trajectories.
Transcripts contain code, so export aggregates only.

---

## 5. Configuring vLLM and Claude Code

### 5.1 vLLM (the system under test)

| Flag | Why it matters here |
|---|---|
| `--served-model-name` | Claude Code can't use a model name containing `/`, so serve it under a plain alias. |
| `--enable-auto-tool-choice --tool-call-parser <parser>` | Required: Claude Code relies on native tool calls. Pick the parser that matches the model (vLLM tool-calling docs). |
| `--max-model-len` | Must hold the compaction window plus max output tokens (§5.2). Otherwise requests near compaction fail with HTTP 400. |
| `--enable-prefix-caching` | On by default in V1. This whole workload depends on it (§1.1). |
| `--max-num-seqs`, `--max-num-batched-tokens` | Batch cap and chunked-prefill budget. At these context lengths, KV memory usually limits concurrency before `max-num-seqs` does. |
| `--tensor-parallel-size`, `--quantization` | The sizing knobs. Run the tests once per setting. |

**Version.** The vLLM docs say versions newer than 0.17.1 handle the per-request hash in Claude Code's
system prompt. Issue #53393 (inline system turns breaking the cache) is still open, so on any version,
check the cache in Run 0.

### 5.2 Claude Code environment (baked into the sandbox image)

The rule is that **every setting that changes the traffic must match what engineers will run.** The
test is only as good as that match. Set these in the image's `$CLAUDE_CONFIG_DIR/settings.json` `env`
block, or as container env vars.

| Variable | Value | Why |
|---|---|---|
| `ANTHROPIC_BASE_URL` | `http://<proxy>:3001` | The proxy, not vLLM directly, so every call is captured. |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` | `dummy` | The vLLM docs say Claude Code needs both set. |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`, `_SONNET_MODEL`, `_HAIKU_MODEL` | `$MODEL` | Maps every tier to the served model. The Haiku tier is also used for **background** work, so those calls hit vLLM too. *(dry run)* Against the mock, 2.1.308 in `-p` mode made no background calls: every call carried the full tool set. *Confirm with the real model in Run 0.* |
| `ANTHROPIC_MODEL` | `$MODEL` | **Required, although the vLLM docs leave it out.** *(dry run)* With only the tier defaults set, 2.1.308 runs `$MODEL[1m]`, assumes a **1M** window and ignores `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, so it never compacts before vLLM's limit. Setting `ANTHROPIC_MODEL` makes the cap apply. Without the cap, it assumes 200k. |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | `max_model_len` | Claude Code can't detect a smaller server limit. Check the assumed window in `sessions.jsonl` (`context_window`). |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | the production value (100k–1M) | When compaction fires, the prefix resets and the next call is a full prefill. |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | the production value | Defaults to 32,000 for unknown model IDs. Compaction window + this value must fit in `max_model_len`. |
| `CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT` | pin `0` or `1` | Chooses the full or the short system prompt. Pin it so experiments or server config can't change prompt size between runs. |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `0` (in production too) | Removes the attribution block at the start of the system prompt. It costs nothing and keeps the cached prefix stable. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` | No updates, telemetry or feature-flag fetches from the sandbox. |
| `CLAUDE_CONFIG_DIR` | e.g. `/cc` | Isolates settings and transcripts per sandbox. The trace builder reads `/cc/projects/`. |

Leave `CLAUDE_CODE_MAX_RETRIES` at its production default (10). Under overload, retries are real
load. Count them instead (§10).

### 5.3 Invocation

```bash
claude -p --input-format stream-json --output-format stream-json --verbose \
  --dangerously-skip-permissions \
  --disallowed-tools AskUserQuestion \
  --session-id "$(uuidgen)"
```

- **`--input-format stream-json`** keeps one process alive for the whole session, and the orchestrator
  writes each engineer message to stdin. That is closest to an interactive session. *(dry run)* It
  works as follows: each message on stdin produces a `system/init` event, the turn's events, and
  exactly one `result`, and the process exits when stdin closes. Restarting the process each turn with
  `--resume` isn't needed.
- **`--dangerously-skip-permissions`**: the sandbox has no network beyond the proxy, which is the
  setup the flag's own help text recommends. Without it, `-p` runs start in manual mode and deny
  anything that would prompt.
- **`--disallowed-tools AskUserQuestion`**: nobody can answer that tool headlessly. The agent asks in
  text instead, and the engineer answers in its next turn. This is a small deviation from interactive
  use.
- **Not `--bare`.** It skips CLAUDE.md, hooks, skills, MCP servers and system reminders, all of which
  are real prompt tokens and real calls. The docs say bare mode will become the default for `-p` in a
  future release, so pin the Claude Code version. In Run 0, check that the first call's input size
  matches an interactive session in the same repo.
- **No parent CLAUDE.md.** Claude Code also loads CLAUDE.md files from the directories above the repo.
  `--sandbox local` therefore clones into a temp directory, not under `--out`.

Reading the `result` event *(dry run, 2.1.308)*:
- A failed API call ends the turn with `is_error: true` and `terminal_reason: "api_error"`, but
  `subtype` still says `"success"`. Check `is_error`.
- `duration_api_ms` is cumulative over the process. `duration_ms`, `num_turns` and `usage` are per
  turn.
- Each retry is a `system/api_retry` event with `error_status` and `retry_delay_ms`. When the mock
  injected 529s, the counted retries matched the 529s it served.

---

## 6. Recording: capture and trace building

### 6.1 Capture proxy

Run `claude-code-proxy` next to vLLM:
- `ANTHROPIC_FORWARD_URL=http://<vllm>:8000`
- `DB_PATH=/data/requests.db`
- **`ANTHROPIC_MAX_RETRIES=0`**: the proxy retries 3 times by default, which would hide overload from
  the measurements.

Its README says nothing about vLLM upstreams. In Run 0, check that streaming passes through unchanged
and that `requests.db` has one row per call.

### 6.2 Build WEKA traces

Collect every sandbox's `$CLAUDE_CONFIG_DIR/projects/` into one `jsonl/` directory, then:

```bash
python3 build_message_index.py requests.db
python3 build_conversation_index.py requests.db --projects-path jsonl/
python3 build_minimal_traces.py requests.db --jsonl-dir jsonl/ --output-dir traces/ \
    --block-size 64 --min-requests 5 --include-subagents
```

Each output line is one conversation, with `id`, `block_size`, `tool_tokens`, `system_tokens` and a
`requests` array. Each request carries:

| Field | Meaning |
|---|---|
| `t` | Seconds from the start of the conversation |
| `in`, `out` | Input and output tokens |
| `hash_ids` | Chained IDs of 64-token blocks: the prefix-sharing structure |
| `input_types`, `stop` | What the client appended, and why the model stopped (`tool_use`, `end_turn`) |
| `api_time`, `ttft` | Recorded response time and time to first token |
| `think_time` | Client gap before this request |

Subagents appear as `type: "subagent"` entries with their own nested `requests`.

Caveats:
- **Token counts.** `hash_ids` are computed with tiktoken as an approximation, not the served model's
  tokenizer, so block counts and `in` won't line up exactly. guideLLM sizes each prompt to `in`.
- **TTFT.** `ttft` comes from a `Server-Timing` header that vLLM doesn't send, so expect it to be
  empty. guideLLM measures its own.
- **Built for the Anthropic API.** The builder was written against Anthropic API captures, and vLLM's
  `/v1/messages` usage block may lack the `cache_*` fields (issue #45079, closed as stale).
  *Check in Run 0 that the builder accepts vLLM's responses.*
- **Cache structure is client-side.** `hash_ids` describe the prefix as Claude Code sent it, **not as
  vLLM cached it**. If vLLM's frontend breaks the cache (§1.1), a replay built from these traces will
  be optimistic. Run 4 detects this.
- **Hash ID scope across conversations.** guideLLM keeps one token-block table for all conversations
  unless a row sets `"hash_id_scope": "local"`. The builder's docs disagree on whether its IDs are
  batch-global or restart in each conversation. *Check in Run 0.*
  - **If IDs restart per conversation:** unrelated sessions would get identical blocks wherever their
    IDs coincide, which is far more cross-session cache sharing than reality. Set
    `"hash_id_scope": "local"` on every line.
  - **If IDs are batch-global:** sessions share blocks only where the content really matched, such as
    the shared tools and system prompt, which is correct.

### 6.3 Put engineer time back

The recording ran with `ENGINEER_TIME_SCALE = 0` (`--time-scale 0`), but the gap before each
engineer message was not zero. It includes running the check, the engineer LLM call, and the git
snapshot. For each engineer turn, `engineer.jsonl` logs both:
- `think.total`: the modelled gap `g_k`;
- `recorded_gap_s`: the gap that actually happened.

Before Run 3, shift every request from the next engineer message onwards by `g_k − recorded_gap_s`.
The first request after an engineer message is the one whose `input_types` is `["text"]`. Tool gaps
are left untouched. This is a ~30-line script, still to write.

---

## 7. The test plan

Run 0 → 4 against **one replica** of each candidate configuration (GPU × TP × quantisation).

```bash
export TARGET=http://<vllm-host>:8000     # replay goes straight to vLLM, not via the proxy
export MODEL=<served-model-name>
export TOKENIZER=<HF id of the served model>
```

### Run 0: smoke and cache check (1 task, ~15 min)

Run one session end to end, then check:

1. **No HTTP 400s** in `requests.db`, and Claude Code's stream has no `system/api_retry` events. In
   `sessions.jsonl`, `context_window` equals `max_model_len`, not 1,000,000 (§5.2).
2. **Fixed overhead.** `in` of the first call is the tools + system prompt + CLAUDE.md size. Compare
   it with an interactive session in the same repo (§5.3).
3. **Cache health.** On the 2nd and later calls, vLLM's
   `prefix_cache_hits / prefix_cache_queries` (from `/metrics`; see §8 for names) should be close
   to the hit rate implied by the trace's `hash_ids`.
   - If vLLM's rate is much lower, the frontend or the attribution block is breaking the prefix
     (§1.1, §5).
   - **Fix it before going on.** Otherwise every later number is prefill-dominated.
4. **Background calls.** Calls carrying Haiku-tier work count as load too; record what share of calls
   they are.
5. **Tooling.** The proxy captures streaming correctly and the trace builder produces one WEKA line.

### Run 1: record the reference set (≈ 50–100 tasks)

Run every task at **low concurrency** (e.g. 4 sessions at a time). Check that vLLM's
`num_requests_waiting` stays at 0, so the recordings carry **unloaded** latencies. Build the traces
(§6.2) and keep each conversation's per-turn wall time as its baseline. The output is
`traces/run1.weka.jsonl`.

### Run 2: real-agent concurrency ladder (≈ 2–4 h)

Run real sessions at 1 / 4 / 8 / 16 / 32 concurrent, keeping engineer time compressed. Stop at
whatever the sandbox host can run (§10). At each step, record:
- `api_time` p50/p95 per call;
- vLLM KV usage, preemptions, prefix hit rate, and running/waiting requests;
- task outcomes (they should not change with load).

This is the ground truth that Run 4 checks the replay against. It also gives a direct, if expensive,
answer at small scale.

### Run 3: replay at scale (one guideLLM run per concurrency level)

```bash
for S in 8 16 32 64 128; do
guidellm run \
  --backend kind=openai_http,target=$TARGET,model=$MODEL,timeout=600 \
  --tokenizer kind=huggingface_auto,model=$TOKENIZER \
  --data "{\"kind\":\"weka\",\"source\":{\"kind\":\"json_file\",\"path\":\"traces/run1.weka.jsonl\"},
           \"duration_column\":\"api_time\",
           \"min_concurrent_sessions\":$S,\"copies\":2}" \
  --profile '{"kind":"replay","schedule_turn":"idle_gap"}' \
  --metrics '{"kind":"generative","slo":{"ttft_ms":3000,"tpot_ms":50}}' \
  --constraint kind=max_duration,seconds=1800 \
  --constraint kind=max_error_rate,rate=0.05,window=20 \
  --output kind=json,path=results/run3_replay_s$S.json \
  --label sessions=$S
done
```

These options were read from the 0.8.1 source, not run. Run the command against guideLLM's mock server
before using GPU time.

| Option | Meaning |
|---|---|
| `kind=weka` | Turns each trace line into a conversation graph. Requests sharing `hash_ids` get identical synthetic token blocks, so vLLM sees the recorded prefix sharing. Subagents run as parallel child chains, and their parent waits for all of them. |
| `duration_column=api_time` | The WEKA field holding each call's recorded duration. guideLLM's default column name is `duration`, so this must be set. |
| `schedule_turn=idle_gap` | **Closed loop per session**: each call starts the recorded idle gap (tool time, engineer time) after its predecessor *actually* finished. A slow server stretches sessions instead of piling up calls, as real Claude Code would. |
| `min_concurrent_sessions=S` | Packs sessions so at least S overlap. This is the concurrency knob. Sessions overlap more when the server slows them down, so S is a floor. |
| `copies` | Replays the set again with freshly salted token blocks, so one copy never hits another's cache. Use it when a pass is shorter than `max_duration`. |
| `max_context_len` | **Leave it unset.** In 0.8.1 it caps the *sum* of `in + out` over all turns of a conversation, not the size of one call. Agent traces re-send the full context on every call, so a cap of `max_model_len` would cut most sessions after a few calls. |
| `max_wait` | *(optional)* Caps long gaps, e.g. lunch breaks in calibrated engineer time, so a run isn't mostly idle. |

Requests go to `/v1/chat/completions` with synthetic text. That is fine for load, because KV and
compute depend on token counts and prefix structure, not on the words. It also means the replay
bypasses the `/v1/messages` frontend, which is why Run 4 exists.

### Run 4: validate the replay

Replay **Run 2's own traces**, with engineer time still compressed, at Run 2's top concurrency, and
compare with Run 2:
- `api_time` p50/p95 per call;
- KV usage;
- prefix hit rate;
- preemptions.

If they agree within ~15%, trust Run 3 at higher scale. If the replay looks much better than reality,
the usual cause is cache breakage on the `/v1/messages` path that synthetic prompts don't reproduce.
Fix the server, or size from Run 2 extrapolated.

---

## 8. Metrics and SLO

An agent turn is a chain of calls, so no single per-call number describes what the engineer sees.
Track three levels:

| Level | Metric | Where |
|---|---|---|
| Per call | **TTFT**: just the appended tokens when cached, the whole context when evicted | guideLLM JSON, `ttft` percentiles |
| Per call | **TPOT**: decode speed, which matters for long edits | guideLLM JSON, `tpot`/ITL percentiles |
| Per engineer turn | **Turn slowdown**: turn wall time under load ÷ the same turn's Run 1 baseline. A turn runs from the first call after an engineer message to `end_turn`. | Offline: join guideLLM's per-request records (conversation, timings) to the trace. A small script to write. |
| Server | `vllm:kv_cache_usage_perc`, `vllm:num_preemptions`, `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`, `vllm:num_requests_running` / `_waiting` | vLLM `/metrics`. Names change between versions and counters may have a `_total` suffix, so check yours. |

**Example SLO (illustrative):** p95 TTFT ≤ 3 s, p95 TPOT ≤ 50 ms (≥ 20 tok/s per stream), and **p95
turn slowdown ≤ 1.5×**. guideLLM can score the first two directly (`slo` takes `ttft_ms` and
`tpot_ms` together). Turn slowdown is the headline, because it is what the engineer feels.

---

## 9. From results to GPUs

1. **S\_max** is the largest `min_concurrent_sessions` in Run 3 that meets the SLO: the number of
   engaged sessions one replica can carry. The sessions include realistic engineer gaps (§6.3), so the
   duty cycle is already in the number.
2. **Peak engaged sessions** = engineers × the fraction with a session in active use during the busiest
   hour. Take that fraction from usage data, or bracket it.
3. **Replicas** = ⌈ peak engaged / S\_max ⌉. Add one if a replica may be down. **GPUs** = replicas × TP.
4. **Little's law cross-check.** Offered calls/s per replica ≈ S × calls per session ÷ mean session
   duration. Calls in flight ≈ that rate × mean `api_time`. This should match
   `vllm:num_requests_running` in Run 3.

**Worked example (illustrative numbers):** 300 engineers × 40% engaged at peak = 120 sessions.
- With S\_max = 24, that is ⌈120 / 24⌉ = 5 replicas, plus 1 spare = 6, and at TP = 2, **12 GPUs**.
- Cross-check: a session of 60 calls over 30 min gives 60 / 1,800 s = 0.033 calls/s. With 24 sessions
  that is 0.8 calls/s, and at a mean `api_time` of 8 s it means ≈ 6.4 calls in flight per replica.

Repeat for each GPU / TP / quantisation candidate and compare cost at equal SLO.

`index.html` is only an intuition aid here. Its closed-loop sessions (think → draft → wait) resemble an
engineer and agent, but it models a single short call per turn, not a context that grows across tens of
calls.

---

## 10. Pitfalls checklist

- **Fake engineer too helpful.** It accepts too early, leaks `intent`, never gets impatient. Sessions
  come out shorter and cleaner than real ones. Spot-check transcripts (§4.2).
- **Engineer LLM on the server under test.** Its calls become part of the measured load.
- **Recording with the wrong model.** Trajectories depend on the model. A test recorded with a strong
  model under-sizes for a weaker one, which takes more steps.
- **Prefix cache silently broken.** Causes: the attribution block (§5.2), inline system turns (vLLM
  #53393), or compaction. Run 0 and Run 4 exist to catch this.
- **Replay shares too much cache.** Trace IDs that restart per conversation, replayed with guideLLM's
  default global block table, make unrelated sessions share prefixes (§6.2). The replay then looks far
  better than reality.
- **Context overflow.** Claude Code assumes 200k for unknown models, or 1M when only the tier defaults
  name the model. Then sessions near compaction get HTTP 400s. Set `ANTHROPIC_MODEL` **and**
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (§5.2).
- **Sandbox bottleneck.** Builds and tests in 32 containers can saturate the host CPU. Tool time then
  stretches and the GPU looks under-loaded. Watch host CPU during Run 2.
- **Retries hide overload.** Claude Code retries up to 10 times and the proxy 3 times by default. Set
  the proxy to 0 and count `system/api_retry` events (`api_retries` in `sessions.jsonl`). A retried
  call is an SLO miss, not a success. A turn that runs out of retries ends its session as
  `agent_error`.
- **Long-tail sessions.** One runaway session can dominate a 30-minute replay step. Cap
  `max_engineer_turns` during recording, and don't use guideLLM's `max_context_len` to trim sessions
  (§7, Run 3).
- **Version drift.** Claude Code changes prompt size, background calls and `-p` defaults (bare mode)
  between releases. vLLM changes the `/v1/messages` frontend. Pin both and re-run Run 0 after any
  upgrade.

---

## Sources

- vLLM, Claude Code integration: https://docs.vllm.ai/en/stable/serving/integrations/claude_code/
- vLLM issue #53393, Anthropic system-turn handling breaks prefix caching: https://github.com/vllm-project/vllm/issues/53393
- vLLM issue #45079, cache fields missing from `/v1/messages` usage: https://github.com/vllm-project/vllm/issues/45079
- vLLM production metrics: https://docs.vllm.ai/en/stable/usage/metrics/
- Claude Code env vars: https://code.claude.com/docs/en/env-vars
- Claude Code headless mode: https://code.claude.com/docs/en/headless
- Claude Code model configuration (gateway context windows, compaction): https://code.claude.com/docs/en/model-config
- [A] agentic-coding-analysis (trace builder, WEKA format): https://github.com/callanjfox/agentic-coding-analysis and its `docs/TRACE_FORMAT.md`
- claude-code-proxy (capture): https://github.com/seifghazi/claude-code-proxy
- guideLLM 0.8.1 (`guidellm/data/deserializers/trace_weka.py`, `guidellm/scheduler/strategies.py`): https://github.com/vllm-project/guidellm
