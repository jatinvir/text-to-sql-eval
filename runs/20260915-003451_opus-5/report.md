# opus-5 - 20260915-003451_opus-5

model `claude-opus-5` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 150 scored / 150 with output / 150 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 89.3% [83-95] |
| strict accuracy (truncated/unparseable = fail) | 89.3% [83-95] |
| lenient accuracy (extra columns allowed) | 89.3% [83-95] |
| easy (n=38) | 92.1% [83-100] |
| medium (n=38) | 97.4% [92-100] |
| hard (n=37) | 89.2% [76-100] |
| extra (n=37) | 78.4% [62-92] |
| paraphrase flip rate (50 pairs) | 10.0% [2-20] |
| cost | $1.6143 total · $0.01076/case |
| latency | p50 2.48s · p95 3.80s |

Statuses: {'ok': 150} · rows needing retries: 0

Failure reasons:

- `wrong_rows` 10 - different rows
- `duplicates` 3 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `missing_columns` 2 - fewer columns than gold
- `wrong_order` 1 - right rows, wrong order (gold has ORDER BY)

Tokens: input_tokens=232,524, output_tokens=18,069, cache_read_input_tokens=0, cache_creation_input_tokens=0
