# haiku-4.5 - 20260915-003451_haiku-4.5

model `claude-haiku-4-5` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 150 scored / 150 with output / 150 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 78.7% [71-86] |
| strict accuracy (truncated/unparseable = fail) | 78.7% [71-86] |
| lenient accuracy (extra columns allowed) | 79.3% [71-87] |
| easy (n=38) | 89.5% [79-97] |
| medium (n=38) | 92.1% [79-100] |
| hard (n=37) | 70.3% [52-87] |
| extra (n=37) | 62.2% [45-79] |
| paraphrase flip rate (50 pairs) | 16.0% [6-26] |
| cost | $0.1960 total · $0.00131/case |
| latency | p50 0.95s · p95 1.88s |

Statuses: {'ok': 150} · rows needing retries: 0

Failure reasons:

- `wrong_rows` 24 - different rows
- `duplicates` 3 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `extra_columns` 2 - more columns than gold (see lenient_passed)
- `wrong_order` 1 - right rows, wrong order (gold has ORDER BY)
- `pred_sql_error` 1 - predicted SQL failed to execute
- `missing_columns` 1 - fewer columns than gold

Tokens: input_tokens=163,491, output_tokens=6,505, cache_read_input_tokens=0, cache_creation_input_tokens=0
