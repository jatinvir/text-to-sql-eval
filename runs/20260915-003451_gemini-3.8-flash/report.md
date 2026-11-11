# gemini-3.8-flash - 20260915-003451_gemini-3.8-flash

model `gemini-3.8-flash` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 150 scored / 150 with output / 150 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 88.0% [81-94] |
| strict accuracy (truncated/unparseable = fail) | 88.0% [81-94] |
| lenient accuracy (extra columns allowed) | 88.0% [81-94] |
| easy (n=38) | 97.4% [91-100] |
| medium (n=38) | 94.7% [84-100] |
| hard (n=37) | 89.2% [75-100] |
| extra (n=37) | 70.3% [51-88] |
| paraphrase flip rate (50 pairs) | 6.0% [0-14] |
| cost | $1.0747 total · $0.00716/case |
| latency | p50 4.54s · p95 23.82s |

Statuses: {'ok': 150} · rows needing retries: 0

Failure reasons:

- `wrong_rows` 13 - different rows
- `duplicates` 3 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `extra_columns` 1 - more columns than gold (see lenient_passed)
- `missing_columns` 1 - fewer columns than gold

Tokens: input_tokens=145,418, output_tokens=257,490, cache_read_input_tokens=0, cache_creation_input_tokens=0, reasoning_tokens=250,936
