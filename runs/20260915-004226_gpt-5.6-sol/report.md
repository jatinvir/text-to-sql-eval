# gpt-5.6-sol - 20260915-004226_gpt-5.6-sol

model `gpt-5.6-sol` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 150 scored / 150 with output / 150 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 80.7% [73-88] |
| strict accuracy (truncated/unparseable = fail) | 80.7% [73-88] |
| lenient accuracy (extra columns allowed) | 80.7% [73-88] |
| easy (n=38) | 89.5% [75-100] |
| medium (n=38) | 89.5% [75-100] |
| hard (n=37) | 75.7% [58-92] |
| extra (n=37) | 67.6% [49-84] |
| paraphrase flip rate (50 pairs) | 6.0% [0-12] |
| cost | $0.6605 total · $0.00440/case |
| latency | p50 3.86s · p95 9.79s |

Statuses: {'ok': 150} · rows needing retries: 10

Failure reasons:

- `wrong_rows` 25 - different rows
- `duplicates` 3 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)
- `missing_columns` 1 - fewer columns than gold

Tokens: input_tokens=85,703, output_tokens=15,109, cache_read_input_tokens=38,690, cache_creation_input_tokens=0, reasoning_tokens=7,509
