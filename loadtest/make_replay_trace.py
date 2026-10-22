"""Turn real customer-message timestamps into a guideLLM trace_synthetic file.

Mirrors the concurrency analysis: each message second gets uniform jitter,
each message triggers a guardrail call with probability --invocation-prob.
Token lengths are drawn from the same lognormal as make_guardrail_dataset.py.
trace_synthetic has no shared-prefix field, so the system prompt is folded into
input_length (no prefix-cache benefit: slightly pessimistic).

    python make_replay_trace.py --in message_timestamps.csv --column timestamp \
        --day YYYY-MM-DD --invocation-prob 0.10 --max-query-tokens 126000 \
        --out trace_peak_p10.jsonl
"""

import argparse
import csv
import json
import math
import random
from datetime import datetime

Z95 = 1.6449


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="CSV with one row per customer message")
    ap.add_argument("--column", default="timestamp", help="ISO-8601 wall-clock column")
    ap.add_argument("--day", help="keep only this date (YYYY-MM-DD)")
    ap.add_argument("--from-hour", type=int, default=0)
    ap.add_argument("--to-hour", type=int, default=24)
    ap.add_argument("--invocation-prob", type=float, default=1.0)
    ap.add_argument("--system-tokens", type=int, default=2048)
    ap.add_argument("--p50", type=int, default=10_000)
    ap.add_argument("--p95", type=int, default=55_000)
    ap.add_argument("--min-query-tokens", type=int, default=256)
    ap.add_argument("--max-query-tokens", type=int, required=True)
    ap.add_argument("--output-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="trace.jsonl")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    mu, sigma = math.log(a.p50), math.log(a.p95 / a.p50) / Z95

    ts = []
    with open(a.src, newline="") as f:
        for row in csv.DictReader(f):
            t = datetime.fromisoformat(row[a.column])
            if a.day and t.date().isoformat() != a.day:
                continue
            if not a.from_hour <= t.hour < a.to_hour:
                continue
            if rng.random() < a.invocation_prob:
                ts.append(t.timestamp() + rng.random())  # jitter within the second
    ts.sort()

    with open(a.out, "w") as f:
        for t in ts:
            n = int(rng.lognormvariate(mu, sigma))
            n = max(a.min_query_tokens, min(a.max_query_tokens, n))
            f.write(json.dumps({
                "timestamp": round(t, 3),
                "input_length": a.system_tokens + n,
                "output_length": a.output_tokens,
            }) + "\n")

    span = (ts[-1] - ts[0]) if len(ts) > 1 else 0
    print(f"wrote {len(ts)} calls over {span / 3600:.2f} h "
          f"({len(ts) / span if span else 0:.2f} calls/s mean) to {a.out}")


if __name__ == "__main__":
    main()
