"""Generate a guideLLM json_file dataset shaped like groundedness-guardrail traffic.

Every row carries the same system prompt (so vLLM's prefix cache sees one shared
prefix, as in production) and a unique user query whose length follows a
lognormal fitted to the observed p50/p95. guideLLM's own synthetic_text only
offers a clipped normal, which cannot reproduce a p95 that is 5.5x the median.

    python make_guardrail_dataset.py --tokenizer <hf-id-of-served-model> \
        --rows 3000 --max-query-tokens 126000 --out guardrail.jsonl
"""

import argparse
import json
import math
import random

from faker import Faker
from transformers import AutoTokenizer

Z95 = 1.6449  # standard normal quantile at 0.95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF id or path of the served model")
    ap.add_argument("--out", default="guardrail.jsonl")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--system-tokens", type=int, default=2048)
    ap.add_argument("--p50", type=int, default=10_000, help="median user-query tokens")
    ap.add_argument("--p95", type=int, default=55_000, help="p95 user-query tokens")
    ap.add_argument("--min-query-tokens", type=int, default=256)
    ap.add_argument("--max-query-tokens", type=int, required=True,
                    help="max_model_len - system_tokens - output_tokens - ~100 template slack")
    ap.add_argument("--output-tokens", type=int, default=16, help="verdict length")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    fake = Faker()
    Faker.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    # One long token pool; each query is a slice of it. Slices overlap, but the
    # unique tag in front breaks vLLM's prefix-hash chain, so the only cache hit
    # a request can get is the shared system prompt.
    pool = []
    while len(pool) < a.max_query_tokens + a.system_tokens + 1000:
        pool += tok.encode(fake.text(max_nb_chars=200_000), add_special_tokens=False)

    system_prompt = tok.decode(pool[: a.system_tokens])
    mu = math.log(a.p50)
    sigma = math.log(a.p95 / a.p50) / Z95

    lengths = []
    with open(a.out, "w") as f:
        for i in range(a.rows):
            n = int(rng.lognormvariate(mu, sigma))
            n = max(a.min_query_tokens, min(a.max_query_tokens, n))
            start = rng.randrange(a.system_tokens, len(pool) - n)
            query = f"[case {i:06d}-{rng.getrandbits(32):08x}] " + tok.decode(pool[start : start + n])
            n = len(tok.encode(query, add_special_tokens=False))  # what the server will see
            lengths.append(n)
            f.write(json.dumps({
                "system_prompt": system_prompt,
                "prompt": query,
                "prompt_tokens_count": n,
                "output_tokens_count": a.output_tokens,
            }) + "\n")

    lengths.sort()
    q = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))]
    capped = sum(n >= a.max_query_tokens for n in lengths) / len(lengths)
    print(f"wrote {a.rows} rows to {a.out}")
    print(f"query tokens  p50={q(.50)}  p95={q(.95)}  p99={q(.99)}  "
          f"mean={sum(lengths) / len(lengths):.0f}  capped={capped:.1%}")


if __name__ == "__main__":
    main()
