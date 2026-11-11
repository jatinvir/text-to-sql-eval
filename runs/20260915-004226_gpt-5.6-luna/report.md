# gpt-5.6-luna - 20260915-004226_gpt-5.6-luna

model `gpt-5.6-luna` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 150 scored / 150 with output / 150 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 82.7% [75-90] |
| strict accuracy (truncated/unparseable = fail) | 82.7% [75-90] |
| lenient accuracy (extra columns allowed) | 83.3% [75-91] |
| easy (n=38) | 86.8% [74-97] |
| medium (n=38) | 89.5% [74-100] |
| hard (n=37) | 78.4% [62-92] |
| extra (n=37) | 75.7% [57-90] |
| paraphrase flip rate (50 pairs) | 10.0% [2-18] |
| cost | $0.0507 total · $0.00034/case |
| latency | p50 2.41s · p95 5.83s |

Statuses: {'ok': 150} · rows needing retries: 9

Failure reasons:

- `wrong_rows` 21 - different rows
- `duplicates` 3 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `missing_columns` 1 - fewer columns than gold
- `extra_columns` 1 - more columns than gold (see lenient_passed)

Tokens: input_tokens=84,008, output_tokens=27,536, cache_read_input_tokens=40,385, cache_creation_input_tokens=0, reasoning_tokens=18,840
