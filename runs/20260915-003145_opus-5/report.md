# opus-5 - 20260915-003145_opus-5

model `claude-opus-5` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

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
| cost | $0.1050 total · $0.01050/case |
| latency | p50 3.21s · p95 5.24s |

Statuses: {'ok': 10} · rows needing retries: 0

Failure reasons:

- `duplicates` 2 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)

Tokens: input_tokens=16,044, output_tokens=991, cache_read_input_tokens=0, cache_creation_input_tokens=0
