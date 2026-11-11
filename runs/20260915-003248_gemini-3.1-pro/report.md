# gemini-3.1-pro - 20260915-003248_gemini-3.1-pro

model `gemini-3.1-pro-preview` · test set `51989fb4b4f8` · prompt v1 · grader v2 · reps 1 · status **complete**

Coverage: 10 scored / 10 with output / 10 expected

| metric | value (95% CI) |
|---|---|
| accuracy | 90.0% [70-100] |
| strict accuracy (truncated/unparseable = fail) | 90.0% [70-100] |
| lenient accuracy (extra columns allowed) | 90.0% [70-100] |
| easy (n=2) | 50.0% |
| medium (n=2) | 100.0% |
| hard (n=2) | 100.0% |
| extra (n=4) | 100.0% [100-100] |
| paraphrase flip rate (4 pairs) | 25.0% [0-75] |
| cost | $0.0808 total · $0.00808/case |
| latency | p50 4.20s · p95 9.48s |

Statuses: {'ok': 10} · rows needing retries: 0

Failure reasons:

- `duplicates` 1 - same distinct rows as gold but different duplicate counts (DISTINCT mismatch)

Tokens: input_tokens=10,112, output_tokens=5,052, cache_read_input_tokens=0, cache_creation_input_tokens=0, reasoning_tokens=4,614
