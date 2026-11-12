# qwen3-8b - 20261112-201730_qwen3-8b

model `qwen3:8b` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 150 scored / 150 with output / 150 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 74.7% [66-83] |
| strict accuracy (truncated/unparseable = fail) | 74.7% [66-83] |
| lenient accuracy (extra columns allowed) | 76.0% [67-85] |
| easy (n=38) | 92.1% [83-100] |
| medium (n=38) | 89.5% [76-100] |
| hard (n=37) | 73.0% [56-88] |
| extra (n=37) | 43.2% [24-63] |
| paraphrase flip rate (50 pairs) | 12.0% [4-22] |
| cost | $0.0000 total · $0.00000/case |
| latency | p50 32.77s · p95 101.07s |

Statuses: {'ok': 150} · rows needing retries: 0

Failure reasons:

- `wrong_rows` 24 - different rows
- `extra_columns` 5 - more columns than gold (see lenient_passed)
- `duplicates` 4 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `pred_sql_error` 4 - predicted SQL failed to execute
- `missing_columns` 1 - fewer columns than gold

Tokens: input_tokens=134,397, output_tokens=136,149, cache_read_input_tokens=0, cache_creation_input_tokens=0
