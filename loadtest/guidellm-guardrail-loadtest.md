# Load-testing the groundedness guardrail with guideLLM

How to use [guideLLM](https://github.com/vllm-project/guidellm) to generate traffic shaped like the
groundedness guardrail against a vLLM endpoint, and how to turn the results
into a GPU count.

All commands were checked against **guideLLM 0.8.1** (the latest release on 2026-10-18) using its
bundled mock server. The docs on guideLLM's `main` branch are ahead of that release: several
options they describe, such as `prefetch` on profiles and `num_workers` on the data loader, either
don't exist in 0.8.1 or break the run. Pin the version.

---

## 1. What we are simulating

| Property | Value | Where it comes from |
|---|---|---|
| System prompt | 2,048 tokens, **identical on every call** | the guardrail's prompt |
| User query | heavy-tailed: p50 = 10k, p95 = 55k tokens | observed traffic |
| Output | a short verdict, assumed **16 tokens** | assumption (same default as `index.html`) |
| Arrival process | real message timestamps × invocation probability | traffic analysis |
| SLO | **p95 ≤ 1 s** for the guardrail call (queue wait included) | the 1 s latency the concurrency study assumes |

### 1.1 Query length: fitted distribution

guideLLM's built-in synthetic generator only offers a *clipped normal*. A normal distribution can't
have a p95 that is 5.5× its median, so a lognormal is fitted to the two observed percentiles:

```
mu    = ln(10,000)              = 9.210
sigma = ln(55,000/10,000)/1.645 = 1.036
```

| Percentile | Query tokens | + system prompt |
|---|---|---|
| p50 | 10,000 | 12,048 |
| p75 | 20,100 | 22,150 |
| p90 | 37,700 | 39,800 |
| p95 | 55,000 | 57,050 |
| p99 | 111,000 | 113,500 |
| mean | ~17,100 | ~19,150 |

The p99 tail exceeds most models' context window, so the generator caps queries at
`max_model_len − 2,048 − 16 − ~100` (template slack). With a 128k model that's about 126k tokens,
which clips ~0.7% of requests and brings the mean to ~16.7k query tokens (~18.7k per request).

> **The tail dominates the load.** The *mean* request (~18.7k tokens) is ~1.55× the *median*
> (~12k), and the top 5% of requests carry ~23% of all prefill tokens. A test run only at p50 will flatter the
> GPU badly.

### 1.2 Traffic: rates and concurrency

The figures below are **illustrative round numbers**, chosen to have the shape of real
chatbot traffic. Replace them with your own measured demand before sizing anything.

Request **rates** at 100% invocation:

| Window | Statistic | Rate (req/s) | Prompt tokens/s (≈ rate × 18.7k) |
|---|---|---|---|
| day | mean | 2 | ~37k |
| 1 h | P99 | 4 | ~75k |
| 1 h | max | 6.5 | ~120k |
| 60 s | P99 | 4.5 | ~85k |
| 60 s | max | 9.5 | ~180k |
| 1 s | P99 | 8 | ~150k (instantaneous) |

To get other traffic scales, multiply the rates by the invocation probability (×0.10 or ×0.05).

Daily-peak **concurrency** (calls in flight), *assuming 1 s service time*:

| Invocation | P50 | P95 | P99 | Max |
|---|---|---|---|---|
| 100% | 15 | 30 | 80 | 110 |
| 10% | 5 | 6 | 9 | 10 |
| 5% | 4 | 5 | 5 | 8 |

Two things to keep in mind:

1. **Those concurrencies assume W = 1 s.** By Little's law (L = λW), if the real guardrail takes
   2.5 s at p95, offered concurrency is roughly 2.5× higher. The load test measures W, so feed the
   measured value back in (§5).
2. **Real traffic is burstier than Poisson.** The busiest second can carry tens of times the mean
   rate, and many messages share their second with another one. guideLLM's `poisson` profile
   doesn't reproduce that. Only trace replay (Run 5) does.

### 1.3 Can the SLO be met at all?

At concurrency 1, a 57k-token prefill on a mid-size model can take more than a second on its own,
before any queueing. **Run 1 answers this first.** If p95 at zero load is already above 1 s, adding
GPUs won't help. The levers are then tensor parallelism, a smaller or quantised model, or a
different SLO for long inputs.

---

## 2. Setup

### 2.1 vLLM server (the system under test)

The vLLM flags that change the result of this test:

| Flag | Why it matters here |
|---|---|
| `--max-model-len` | Must fit 2k + the longest query + 16. It also sets the generator's `--max-query-tokens`. |
| `--enable-prefix-caching` | Default in vLLM V1. The shared 2k system prompt is computed once and reused, which removes about 11% of prefill tokens. Keep it on, because production will have it. |
| `--max-num-batched-tokens` | Chunked-prefill budget per step. Long prompts are split into chunks of this size, which shapes the latency of a long prompt arriving behind other long prompts. |
| `--max-num-seqs` | Batch cap. At this context length, KV memory usually limits concurrency before this does. |
| `--tensor-parallel-size`, `--quantization` | The sizing knobs you're comparing. One guideLLM run per setting. |
| `--served-model-name` | The name guideLLM sends as `model`. |

Test **one replica** first (Runs 1–4), then the full deployment behind its load balancer (Run 5).

### 2.2 Load generator

```bash
python -m venv .venv && source .venv/bin/activate
pip install "guidellm==0.8.1"          # also installs transformers + faker, used by the scripts here
guidellm --version                      # guidellm version: 0.8.1
```

Run it on a machine with a fast CPU and a low-latency network path to the server. guideLLM
tokenises every prompt on the client, and at 20–60k tokens per prompt the client itself can become
the bottleneck (see §6).

### 2.3 Build the dataset (recommended route)

`make_guardrail_dataset.py` writes a JSONL file. Each row has the same `system_prompt`, a unique
lognormal-length `prompt`, and the token counts:

```bash
python loadtest/make_guardrail_dataset.py \
    --tokenizer <HF id or path of the SERVED model> \
    --max-query-tokens 126000 \
    --rows 3000 \
    --out guardrail.jsonl
# wrote 3000 rows to guardrail.jsonl
# query tokens  p50=…  p95=…  p99=…  mean=…  capped=…%
```

| Argument | Meaning |
|---|---|
| `--tokenizer` | Must be the served model's tokenizer, otherwise "10k tokens" means a different number on the server. |
| `--max-query-tokens` | Context cap from §1.1. |
| `--rows` | Size of the pool. guideLLM **cycles** through a file dataset (it reports "inf unique requests"), and a repeated row is a full prefix-cache hit. Make `--rows` at least the number of requests in one run (§6). 3,000 rows ≈ 300 MB. |
| `--p50`, `--p95`, `--system-tokens`, `--output-tokens` | Defaults are 10,000 / 55,000 / 2,048 / 16. |

Every query starts with a unique `[case …]` tag. Because vLLM's prefix-hash chain starts at the
first token, the tag keeps two queries from sharing cache blocks beyond the system prompt.

**Quick alternative, without a dataset file.** `synthetic_text` supports a shared prefix but not a
skewed length. Use it to *bracket* the load, with one run at the median length and one at p95:

```bash
--data kind=synthetic_text,prompt_tokens=10000,prefix_tokens=2048,prefix_count=1,output_tokens=16
--data kind=synthetic_text,prompt_tokens=55000,prefix_tokens=2048,prefix_count=1,output_tokens=16
```

---

## 3. The parameters, one by one

Every guideLLM option uses the same pattern: `--option kind=<type>,key=value,...`. Use JSON
(`--option '{"kind":...}'`) when a value is a list or nested. You can't mix the two forms within one
option.

To keep the commands in §4 readable, set the shared parts once:

```bash
export TARGET=http://<vllm-host>:8000
export MODEL=<served-model-name>
export TOKENIZER=<same HF id as the dataset>

COMMON=(
  --backend kind=openai_http,target=$TARGET,model=$MODEL,timeout=120
  --data kind=json_file,path=guardrail.jsonl
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt","prefix_column":"system_prompt","prompt_tokens_count_column":"prompt_tokens_count","output_tokens_count_column":"output_tokens_count"}}'
  --tokenizer kind=huggingface_auto,model=$TOKENIZER
  --metrics '{"kind":"generative","slo":{"e2el_ms":1000}}'
  --seed kind=static,value=42
  --disable-console-interactive
)
```

### `--backend`: where requests go and how

| Key | Value here | Explanation |
|---|---|---|
| `kind` | `openai_http` | Talks to vLLM's OpenAI-compatible HTTP API. |
| `target` | `$TARGET` | Base URL. For Run 5, use the load balancer address instead of one replica. |
| `model` | `$MODEL` | Sent as the `model` field. If omitted, guideLLM takes the first model from `/v1/models`. |
| `request_format` | *(default)* `/v1/chat/completions` | Chat format. The system prompt is sent as a `system` message, as production would send it. |
| `stream` | *(default)* `true` | Streaming gives TTFT and inter-token latency as well as end-to-end latency. Keep it on. |
| `timeout` | `120` | Per-request timeout in seconds. With no timeout, overloaded runs hang instead of recording errors. |
| `max_tokens` | *(leave unset)* | Output length comes from the dataset's `output_tokens_count`. guideLLM then sets `max_tokens=16` **and `ignore_eos=true`**, so every verdict is exactly 16 tokens. |

### `--data`: what each request contains

| Key | Value | Explanation |
|---|---|---|
| `kind` | `json_file` | Loads the JSONL from §2.3. |
| `path` | `guardrail.jsonl` | One exact file (globs aren't expanded). |

### `--data-column-mapper`: which column is which

| Mapping | Column | Effect |
|---|---|---|
| `text_column` | `prompt` | The user message. |
| `prefix_column` | `system_prompt` | Sent as the system message: the shared, cacheable prefix. |
| `prompt_tokens_count_column` | `prompt_tokens_count` | Known token counts, used for the stats. |
| `output_tokens_count_column` | `output_tokens_count` | Drives `max_tokens` / `ignore_eos`. |

These happen to be guideLLM's default column names, so the mapper is technically optional. It is
spelled out so that nothing changes silently if the file is regenerated with other names.

### `--tokenizer`

`kind=huggingface_auto,model=$TOKENIZER`. guideLLM counts tokens on the client. Use the served
model's tokenizer, the same one the dataset was built with.

### `--profile`: the traffic pattern (the key choice)

| Profile | Loop | Use in this plan | Key parameters |
|---|---|---|---|
| `synchronous` | one request at a time | **Run 1**: latency floor, no batching | none |
| `concurrent` | closed loop, N requests always in flight | **Run 2**: latency vs concurrency, using the §1.2 concurrency points | `streams` (one value or a list), `rampup_duration` |
| `goodput` | concurrent, searched | **Run 3**: highest concurrency that still meets the SLO | `target_attainment`, `initial_streams`, `max_streams`, `tolerance`, `max_probes` |
| `poisson` | open loop, random arrivals at a rate | **Run 4**: queueing under realistic-ish arrivals | `rate` (one value or a list), `max_concurrency` |
| `replay` | open loop, timestamps from a trace | **Run 5**: real traffic bursts | `time_scale` |
| `constant` | open loop, evenly spaced arrivals | not used, because real traffic isn't evenly spaced | `rate` |
| `throughput` / `sweep` | flood / auto-ladder | not used: they find maximum throughput, but the question here is latency at a load level | — |

All profiles also accept `warmup` and `cooldown`. A value below 1 is a fraction of the run, and a
value of 1 or more is seconds. Requests in those windows are excluded from the statistics. Use a
warm-up here, because the first seconds include the client loading 20–60k-token prompts.

**Closed loop vs open loop.** The §1.2 concurrency numbers describe an *open* system: users
don't wait for the guardrail before sending their next message. `concurrent` and `goodput` are
*closed* loops (a new request only starts when one finishes), so they can't build up a backlog.
They answer "how many calls can one replica run in parallel within the SLO?". `poisson` and
`replay` answer "what happens to queue wait when real arrivals hit it?". You need both.

### `--metrics`: the SLO

`'{"kind":"generative","slo":{"e2el_ms":1000}}'`

- `e2el_ms` is the end-to-end request latency objective. It is the right objective here, because a
  guardrail is blocking: the chatbot waits for the whole verdict, not for the first token.
  (`ttft_ms` and `tpot_ms` also exist. With a 16-token output, TTFT makes up most of the latency
  anyway.)
- guideLLM then reports **attainment** (the share of requests within 1 s) and **goodput**
  (conforming requests per second). Attainment ≥ 95% is the same as p95 ≤ 1 s.
- Errors count against attainment, which is correct: a timeout is an SLO miss.

### `--constraint`: when each step stops (repeatable)

| Constraint | Use |
|---|---|
| `kind=max_duration,seconds=N` | Main stop condition per step. Use ≥ 180 s so the p95 rests on hundreds of samples. |
| `kind=max_requests,count=N` | For the synchronous run (a fixed sample size). |
| `kind=over_saturation,mode=enforce,min_seconds=60` | Open-loop runs: stops a rate step once the queue clearly diverges, so time isn't wasted on a backlog. |
| `kind=max_error_rate,rate=0.05,window=20` | Safety net: aborts a step if more than 5% of the last 20 requests fail. |

Constraints apply **per step**: `streams=[1,2,4]` with `max_duration,seconds=300` runs 3 × 5 minutes.

### `--seed`

`kind=static,value=42`. Makes Poisson arrival times reproducible, so two GPU configurations see the
same arrival sequence.

### `--output` (repeatable)

`kind=json,path=…`, `kind=csv,path=…`, `kind=html,path=…`. If you give no `--output`, guideLLM
writes `benchmarks.json` and `benchmarks.csv`. The JSON is the one to keep. It has every request,
`benchmarks[].metrics.request_latency.successful.percentiles.p95`,
`benchmarks[].metrics.slo_attainment`, `benchmarks[].warnings`, and for the goodput profile
`conclusions[0].best_passing_streams`.

### `--label` (optional, repeatable)

`--label gpu=H100 --label tp=1 --label quant=fp8`. Stored in the report, so later comparisons know
which configuration produced it.

---

## 4. The test plan

Run 0 → 4 against **one replica** of each candidate configuration (GPU × TP × quantisation).
Run 5 against the full deployment.

### Run 0: smoke test (1 min)

```bash
guidellm run "${COMMON[@]}" \
  --profile kind=concurrent,streams=1 \
  --constraint kind=max_requests,count=20 \
  --output kind=json,path=results/run0_smoke.json
```

Check that it ran with 0 errors, that input tokens per request in the token table are ≈ 2k + query
(median ≈ 12k), and that output tokens per request are 16.

### Run 1: latency floor (≈ 10 min)

```bash
guidellm run "${COMMON[@]}" \
  --profile kind=synchronous \
  --constraint kind=max_requests,count=300 \
  --output kind=json,path=results/run1_sync.json
```

Read the p50 and p95 request latency. This is the best that one replica can do. **If p95 > 1 s here,
stop and change the configuration (§1.3).** The ratio p95/p50 also shows how much of the tail comes
from long inputs alone.

### Run 2: concurrency ladder (≈ 1 h)

The streams are the §1.2 concurrency points (5%/10%/100% quantiles), plus a few fill-ins:

```bash
guidellm run "${COMMON[@]}" \
  --profile '{"kind":"concurrent","streams":[1,2,4,6,10,15,30,50,80,110],"warmup":30}' \
  --constraint kind=max_duration,seconds=300 \
  --constraint kind=max_error_rate,rate=0.05,window=20 \
  --output kind=json,path=results/run2_ladder.json \
  --output kind=html,path=results/run2_ladder.html
```

For each step, plot p95 latency, attainment, and prompt tokens/s against concurrency. You should see
aggregate tokens/s rise and then flatten while latency climbs (batching shares the GPU). Where the
attainment column drops below 95% is roughly the per-replica concurrency limit. Run 3 pins it down.

### Run 3: SLO-bound capacity per replica (≈ 30–45 min)

```bash
guidellm run "${COMMON[@]}" \
  --profile '{"kind":"goodput","target_attainment":0.95,"initial_streams":4,"max_streams":128,"tolerance":0.1,"warmup":30}' \
  --constraint kind=max_duration,seconds=240 \
  --output kind=json,path=results/run3_goodput.json
```

| Parameter | Meaning |
|---|---|
| `target_attainment=0.95` | Pass if ≥ 95% of requests finish within `e2el_ms` (= p95 ≤ 1 s). Use 0.99 for a p99 SLO. |
| `initial_streams=4` | First probe. The search doubles (4 → 8 → 16 …) until a level fails, then bisects. |
| `max_streams=128` | Ceiling, above the §1.2 maximum of 110. |
| `tolerance=0.1` | Stop when the answer is known within 10%. |
| `max_probes` | *(default 15)* Hard cap on the number of steps. |

The result is `conclusions[0].best_passing_streams` in the JSON. Call it **C\_max**: the most guardrail
calls one replica can hold in flight while still meeting p95 ≤ 1 s. If guideLLM warns that probes
"collected too few requests", raise `max_duration`.

### Run 4: open-loop check at real rates (≈ 30 min per scale)

Rates are the §1.2 rates × invocation probability. For 10% invocation:

```bash
guidellm run "${COMMON[@]}" \
  --profile '{"kind":"poisson","rate":[0.2,0.4,0.65,0.95],"warmup":30}' \
  --constraint kind=max_duration,seconds=420 \
  --constraint kind=over_saturation,mode=enforce,min_seconds=60 \
  --output kind=json,path=results/run4_poisson_p10.json
```

(For 100% invocation: `[2,4,6.5,9.5]`, which are the mean, busy-hour P99, busy-hour max and
60 s max.) List rates in ascending order, so that an over-saturation stop only skips harder steps.

Signature of a replica that keeps up: mean concurrency ≈ rate × mean latency (Little's law, which
you can check directly in the output), and attainment ≥ 95%. Signature of overload: concurrency
and latency keep growing during the step, and `over_saturation` ends it.

### Run 5: replay real bursts against the full deployment (optional, the most faithful)

`make_replay_trace.py` turns a file of real message timestamps into a trace. It uses the same method as the concurrency analysis
(jitter within the second, invocation probability) and the same lognormal lengths:

```bash
python loadtest/make_replay_trace.py --in message_timestamps.csv --column <timestamp column> \
    --day <YYYY-MM-DD> --from-hour <start> --to-hour <end> \
    --invocation-prob 0.10 --max-query-tokens 126000 --out trace_peak_p10.jsonl

guidellm run \
  --backend kind=openai_http,target=$LB_TARGET,model=$MODEL,timeout=120 \
  --tokenizer kind=huggingface_auto,model=$TOKENIZER \
  --data kind=trace_synthetic,source.kind=json_file,source.path=trace_peak_p10.jsonl \
  --profile kind=replay,time_scale=1.0 \
  --metrics '{"kind":"generative","slo":{"e2el_ms":1000}}' \
  --constraint kind=max_duration,seconds=10800 \
  --output kind=json,path=results/run5_replay_peak_p10.json
```

- Choose one of the busiest days in your traffic history, and a few hours around its busiest
  hour.
- `time_scale` < 1 compresses time (0.5 = twice as fast), which acts as a rough what-if for traffic
  growth. It isn't the same as more users, because bursts get sharper too.
- Limitation: `trace_synthetic` has no shared-prefix field, so the system prompt is added to
  `input_length` and never cached. This makes the result slightly pessimistic (~11% extra prefill).

---

## 5. From results to GPUs

1. **Measured latency.** Take the mean and p95 latency at the operating point from Run 3/4. Call
   the mean W.
2. **Offered concurrency at the real W.** The §1.2 table assumes W = 1 s. Scale it:
   C\_offered ≈ C\_1s × W / 1 s. This is exact for the mean and an approximation for the peaks.
   Pick the quantile you size for: P99 accepts brief queueing on event days, and max covers them.
3. **Replicas** = ⌈ C\_offered / C\_max ⌉ (C\_max from Run 3). Add headroom for one replica being
   down if that's a requirement.
4. **GPUs** = replicas × tensor-parallel size.
5. **Cross-check with tokens.** Per-replica prompt tokens/s at C\_max (Run 2/3) × replicas should be
   above the §1.2 token demand for your traffic scale.
6. **Confirm with Run 5** on the sized deployment. The replay has the real burst structure,
   which the closed-loop numbers can't capture.

Repeat 1–5 for each GPU / TP / quantisation candidate and compare cost at equal SLO.

The simulator in this repo (`index.html`) is a quick sanity check before spending GPU hours.
Set the same prompt/output tokens and SLO there, and compare its predicted knee with Run 2.

---

## 6. Pitfalls checklist

- **Version drift.** These commands are for 0.8.1. The docs on guideLLM's `main` branch describe
  `prefetch` and other newer options that 0.8.1 rejects or that break the run.
- **Accidental cache hits.** File datasets are cycled, and a repeated row hits vLLM's prefix cache
  for the *whole* query, which makes it far faster than reality. Check vLLM's `/metrics` during
  a run: `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` should be about **0.11**
  (2k shared out of ~18.7k per request). If it's much higher, rows are repeating. Generate more
  rows, or restart vLLM between runs.
- **Client bottleneck.** Warnings such as "mean generation delay was X% of mean time to first
  token…" and "p95 root dispatch delay … conversations were loaded late" mean the load generator,
  not the server, is limiting the load. Some of this at start-up is expected and is what `warmup`
  removes. If it keeps happening, use a faster client machine or fewer rows. Check
  `benchmarks[].warnings` in the JSON before trusting a step.
- **Wrong tokenizer.** Token counts silently differ and the length distribution shifts.
- **Context overflow.** Queries above `max_model_len` come back as HTTP 400 errors and count
  against attainment. Set `--max-query-tokens` from the real limit.
- **Forced verdict length.** `ignore_eos=true` makes every output exactly 16 tokens. If the real
  guardrail emits reasoning or a longer JSON, regenerate the dataset with `--output-tokens`
  set to the real length.
- **Short steps.** A p95 needs hundreds of samples. At concurrency 1 and ~1 s per request, 300 s
  gives ~300 samples. guideLLM marks low-sample p95 values with `*` and reports a confidence
  interval on attainment.
- **Single-replica tests vs the load balancer.** Runs 1–4 should hit one replica directly. Run 5
  goes through the real routing, because uneven routing of long prompts is part of production
  latency.
