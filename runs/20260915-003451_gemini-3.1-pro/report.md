# gemini-3.1-pro - 20260915-003451_gemini-3.1-pro

model `gemini-3.1-pro-preview` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 150 scored / 150 with output / 150 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 86.0% [79-92] |
| strict accuracy (truncated/unparseable = fail) | 86.0% [79-92] |
| lenient accuracy (extra columns allowed) | 86.0% [79-92] |
| easy (n=38) | 94.7% [87-100] |
| medium (n=38) | 92.1% [80-100] |
| hard (n=37) | 86.5% [72-97] |
| extra (n=37) | 70.3% [51-87] |
| paraphrase flip rate (50 pairs) | 8.0% [2-16] |
| cost | $1.1799 total · $0.00787/case |
| latency | p50 5.00s · p95 10.80s |

Statuses: {'ok': 150} · rows needing retries: 37

Failure reasons:

- `wrong_rows` 14 - different rows
- `duplicates` 4 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `extra_columns` 2 - more columns than gold (see lenient_passed)
- `missing_columns` 1 - fewer columns than gold

Tokens: input_tokens=145,418, output_tokens=74,088, cache_read_input_tokens=0, cache_creation_input_tokens=0, reasoning_tokens=67,918
