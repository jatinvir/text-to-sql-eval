# gpt-5.6-luna - 20260915-004208_gpt-5.6-luna

model `gpt-5.6-luna` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 10 scored / 10 with output / 10 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 80.0% [40-100] |
| strict accuracy (truncated/unparseable = fail) | 80.0% [40-100] |
| lenient accuracy (extra columns allowed) | 80.0% [40-100] |
| easy (n=2) | 100.0% |
| medium (n=2) | 100.0% |
| hard (n=2) | 0.0% |
| extra (n=4) | 100.0% [100-100] |
| paraphrase flip rate (4 pairs) | 0.0% [0-0] |
| cost | $0.0034 total · $0.00034/case |
| latency | p50 3.01s · p95 5.28s |

Statuses: {'ok': 10} · rows needing retries: 0

Failure reasons:

- `duplicates` 2 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)

Tokens: input_tokens=6,968, output_tokens=1,635, cache_read_input_tokens=1,650, cache_creation_input_tokens=0, reasoning_tokens=1,085
