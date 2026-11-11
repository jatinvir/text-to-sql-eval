# haiku-4.5 - 20260915-002940_haiku-4.5

model `claude-haiku-4-5` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 10 scored / 10 with output / 10 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 70.0% [30-100] |
| strict accuracy (truncated/unparseable = fail) | 70.0% [30-100] |
| lenient accuracy (extra columns allowed) | 70.0% [30-100] |
| easy (n=2) | 100.0% |
| medium (n=2) | 100.0% |
| hard (n=2) | 0.0% |
| extra (n=4) | 75.0% [0-100] |
| paraphrase flip rate (4 pairs) | 0.0% [0-0] |
| cost | $0.0134 total · $0.00134/case |
| latency | p50 2.60s · p95 3.48s |

Statuses: {'ok': 10} · rows needing retries: 0

Failure reasons:

- `duplicates` 2 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `pred_sql_error` 1 - predicted SQL failed to execute

Tokens: input_tokens=11,242, output_tokens=436, cache_read_input_tokens=0, cache_creation_input_tokens=0
