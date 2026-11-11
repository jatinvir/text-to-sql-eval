# Decisions log

Newest at the bottom. Each entry: what was decided, and why.

## Test set source

**Spider dev split (`validation`), 1034 cases.** Not `train`.

- Dev is what published papers report on. The real test split was never released,
  so dev is the de facto benchmark and these numbers are comparable to published ones.
- Train is ~7000 cases across 140 databases, which is far more to download than this
  project needs.

**Known weakness:** Spider is from 2018 and partly saturated (frontier models score
~85–90%). Accepted deliberately. The harness is benchmark-agnostic, so swapping in
BIRD later only needs a new loader. Contamination is also likely (these questions have
been public for years), but it affects all models alike, and the comparison is between
models on identical inputs.

## Where the data came from

**`premai-io/spider` on HuggingFace.**

`xlangai/spider` contains only parquet files: the rows, but no SQLite databases. Without
the databases there is nothing to execute against, and execution is the entire
grading strategy. `premai-io/spider` has both (`validation.json` + `database/`).

Full repo is 919 MB, but most of that is `soccer_1` (~640 MB), a *train* database this
project never uses. Download selectively: read the distinct `db_id`s out of
`validation.json` first, then fetch only those ~20 database directories via
`allow_patterns`.

## Difficulty labels

Vendored Spider's own `eval_hardness` into `vendor/spider_eval/hardness.py`
(Apache 2.0, source + commit SHA recorded in that folder).

Used the official definition instead of writing my own, because:

1. The headline result is broken down by this metric. If it's Spider's own definition,
   unchanged, there is nothing to argue about. A homemade metric invites "why is that
   hard?"
2. Inventing the metric *and* reporting results broken down by it invites the
   question of whether the buckets were tuned to make the result look good.

Trimmed rather than vendored whole: `evaluation.py` imports `process_sql`, which
imports `nltk`, a large dependency for code the hardness path never calls.
Trimming forks the file, which is why the upstream commit SHA is recorded.

How it works: three counters, then thresholds.
- `count_component1`: surface structure. +1 each for WHERE/GROUP BY/ORDER BY/LIMIT,
  +1 per extra joined table, +1 per OR, +1 per LIKE. (AND is free; ORs are weighted
  because models get disjunctions wrong more often.)
- `count_component2`: number of nested subqueries.
- `count_others`: breadth. +1 for each of "more than one" aggregate / select column /
  where condition / group by.

Nesting is a **gate**: `comp2 == 0` guards easy and both medium clauses, so any
subquery at all means at least "hard".

Distribution across all 1034:

```
easy    248  24.0%
medium  446  43.1%
hard    174  16.8%
extra   166  16.1%
```

## Test set design

**150 questions = 50 pairs (100 questions) + 50 singles.  100 distinct gold queries.**
Stratified by difficulty, ~25 distinct queries per bucket. Format: YAML.

*Why stratified:* a random 150 gives whatever mix it happens to give (~16% extra-hard).
Stratifying lets me choose the cell sizes, so each difficulty bucket has enough cases to
say something about. The interesting finding is the interaction, e.g. "level on easy,
20 points apart on extra-hard, at 15x the cost", and that needs per-bucket comparisons.

*Pairs:* Spider dev is 564 distinct queries, 470 of which were paraphrased into two
questions each. Sampling both halves measures whether a model flips when the
question is reworded. This addresses the saturation problem: aggregate accuracy stops
discriminating at 88%, but *consistency* still does. Pairs must be sampled as a unit;
sampling questions independently produces orphaned halves. Both halves of a pair
always share a difficulty label (same SQL), so pairs never straddle buckets.

*Why ~50 pairs:* at a true flip rate of ~5%, 20 pairs would yield ~1 observed flip,
which can't distinguish 3% from 8%. The metric needs enough pairs to mean anything.

*Why 150 instead of 100:* API cost isn't the constraint (a full run is well under $1 per model).
The real constraint is **human adjudication time**. Every edge case (empty result sets,
row order, column order, float precision) is a judgment call I have to make, encode, and
then verify by eyeballing actual cases. ~150 cases is an evening's audit. 1000 would take
a week, so the audit would never get done and grader errors would go unnoticed.

*YAML over JSONL:* YAML supports comments, and I want to annotate cases where the gold
SQL is questionable ("gold uses ORDER BY without a tiebreak; ties are
non-deterministic"). Annotations make the ground truth easier to defend. YAML also
diffs per-field; JSONL shows a whole case as one changed line. Cost: YAML coerces
unquoted values, so quote consistently.

*Seed the sampler.* Hardcode `random.seed(...)`. Unseeded, re-running produces a
different 150 and breaks comparison with every past run. Generate once and commit the
file; the script documents how the file was made and isn't meant to be rerun.

*Version the test set.* Stamp every run with the case-list version, so a change like
"accuracy dropped 4 points" can always be traced to either the model or the cases.

## Pair grouping is case-insensitive

**Group by `(db_id, lowercased + whitespace-collapsed gold SQL)` → 551 groups, 483 pairs.**
Exact string match gives 564 groups, 470 pairs.

13 paraphrase pairs have gold SQL that differs only in letter case: one half is an
all-lowercase copy of the other. Exact grouping splits each into two "singles", and the
sampler could draw both halves as unrelated singles. That means a duplicated query in the
singles, and pair data thrown away.

Lowercasing isn't semantically safe, because SQLite `=` on text is case-sensitive, and
that turned up real gold bugs: `first_name = 'timmothy'` and `language = 'chinese'`
return nothing, where the other half's `'Timmothy'` / `'Chinese'` return the answer. Those
groups are excluded (next entry), so the normalization only ever merges queries that
behave identically.

## Gold-quality exclusions

**64 of 551 groups excluded before sampling.** Every gold query was executed first.

| reason | groups | why it can't stay |
|---|---|---|
| gold returns no rows | 26 | `SELECT NULL WHERE 0` passes it. Mostly `flight_2`, where stray spaces (`' APG'` in flights, `'Aberdeen '` in airports) make joins and filters match nothing; plus the lowercased-literal bugs above |
| gold returns only 0 / NULL | 14 | `count(*) … WHERE <typo>` returns the same 0 |
| `ORDER BY … LIMIT` cuts through a tie | 24 | gold is one arbitrary pick among tied rows; a correct query can return the other one |

Measured: on the full 1034, the empty-result baseline scores 43 passes. On the
sampled 150 it scores 0 (`python -m harness selfcheck`).

*Cost:* a slight shift in the difficulty mix, and `flight_2` is under-represented.
Accepted, because a case that can't tell a right query from a wrong one only adds noise.

*Detecting LIMIT ties:* re-run gold with its sort keys prepended to the select list and
the LIMIT removed, then check whether the row at the cutoff ties with the next one. Works
for all 231 top-level ORDER BY queries in dev (167 no ties, 20 ties inside the result,
44 at the cutoff).

## Grader

**Execution match.** Run gold and prediction against the database and compare results.
Exact match was ruled out (many right queries per question), and so was Spider's
component matching (it needs the question's parse, and rewards looking like gold rather
than being right).

The rules, each a judgment call:

- **Rows are a bag (multiset).** Duplicate counts matter, so a missing or extra
  `DISTINCT` fails, with its own reason (`duplicates`) so the audit can tell whether
  the question actually implied distinctness.
- **Column order is free; column count must match.** "Names and ages" doesn't define an
  order. The grader searches for a column permutation that reproduces gold's rows; matching
  each column's value multiset alone isn't enough when two columns hold the same values
  (tested). Extra columns fail strictly. `lenient_passed` records whether extra columns were
  the *only* problem, and the report shows lenient accuracy next to strict. Extra id columns
  are the most common "wrong but useful" LLM answer, so the gap between the two is worth
  knowing. The prompt asks for exactly the requested columns, so strict grading doesn't
  penalize a model for following the instructions it was given.
- **Order only matters if gold has a top-level ORDER BY**, detected with a
  literal-aware parser that agrees with Spider's own parse on 1034/1034 dev queries. Rows
  whose sort keys tie in gold may come in any order among themselves: gold's rows are split
  into blocks of equal keys, and the prediction must fill the blocks in sequence.
- **Right order by accident fails** (`order_by_accident`), via two checks on a prediction
  that matches an ordered case:
  - *Structural:* if gold's result spans more than one sort-key group, the prediction needs a
    top-level ORDER BY of its own. Without one, SQL leaves row order undefined, even when
    GROUP BY or an index happens to produce the right order in SQLite today.
  - *Scan order:* the prediction is re-run with SQLite's `reverse_unordered_selects` pragma.
    This catches the single-group case, e.g. an unordered `LIMIT 1` that picked the right
    row only because of storage order, while still passing a deterministic top-1 like
    `WHERE score = (SELECT max(score) …)`. The pragma doesn't affect queries with ORDER BY
    (verified), so a correct query is unaffected.

  (Grader v1 had only the pragma check, which missed order implied by GROUP BY. The code
  review found it; v2 added the structural check.)
- **Floats compare after rounding to 6 decimals; `3 == 3.0`.** `avg(x)` and
  `sum(x)/count(*)` differ in the 15th decimal. Rounding to 2 when the question didn't ask
  still fails.
- **Alternative gold answers.** `alt_gold_sql` in `cases.yaml`. This is how adjudication
  fixes an ambiguous question without editing Spider's gold, and `regrade` applies it to old
  runs without calling the model again.
- **Sandbox.** Model SQL is untrusted: read-only file handle, an authorizer that allows only
  reads (no writes, ATTACH, or PRAGMA), one statement, a 10s deadline, and a 100k-row cap.
  A timeout or row blowup is the model's failure (`pred_timeout`). Gold failing to run is a
  harness error and never scores a model.

**Validation.** Gold graded against itself passes on all 1034 dev queries. The oracle
scores 100% on the 150 and the empty baseline 0%. 80 tests, including the edge cases above
and each vendor's provider against a mock HTTP server through its real SDK.

### How lenient is execution match here?

`python -m analysis.grader_mutations` applies wrong-by-construction edits to each gold query
(`>`→`>=`, `=`→`!=`, `AND`→`OR`, `DESC`→`ASC`, `INTERSECT`→`UNION`, LIMIT n→n+1, …) and
grades the mutant.

**15% of wrong mutants pass (22/146).** Most are inequality boundaries: `>`→`>=` survives
9/15 and `<`→`<=` 4/4, because no row sits exactly on the boundary value.

The miss *rate* is roughly flat across difficulty: easy 3/20 (15%), medium 4/29 (14%),
hard 5/42 (12%), extra 10/55 (18%). Extra-hard has the most misses only because its longer
queries give more places to mutate. (An earlier version of this entry said the misses
"skew hard" from the raw counts alone; the per-tier rates don't support that.)

Consequence for the headline: execution match overstates every bucket's accuracy by a
similar margin, so it inflates absolute scores more than it distorts comparisons between
buckets. The fix is test-suite accuracy (Zhong et al., 2020): grade against several fuzzed
copies of each database so boundary values exist. That would extend the current grader,
not replace it.

Caveat on the 15%: the mutants are synthetic, not real model answers, and a few
"wrong-by-construction" edits are equivalent on inspection (e.g. `HAVING count(*) >= 1`
after an inner join). Treat it as an estimate of grader leniency, not a measured error
rate on model output.

(Separately, "add DISTINCT" and "JOIN → LEFT JOIN" pass ~90%, but those rewrites are
usually *correct* (`DISTINCT count(*)` is still the count), so they're reported apart.)

## Prompt (v1)

**CREATE TABLE statements verbatim + 3 example rows per table; JSON structured output
`{"sql": ...}`.** Averages ~800 input tokens.

- DDL keeps primary and foreign keys, which is how joins get found. Example rows show value
  formats the DDL hides (`'T'`/`'F'` flags, lowercase country names in `car_1`); without
  them, a filter can match nothing.
- Structured output removes "couldn't find the SQL in the prose" as a failure mode, so
  formatting can't affect the score. All three vendors' APIs support it for every roster
  model.
- Instructions ask for exactly the requested columns, in the order mentioned, and ORDER BY /
  LIMIT only when asked, so the prompt agrees with the grader.
- `PROMPT_VERSION` plus a content fingerprint go into every manifest. Scores from different
  prompts aren't comparable.
- No prompt caching: prompts are below the cacheable minimum, and warm-vs-cold caching
  would make cost depend on run order.

## Runs

**Plumbing failures never become model failures.** Rate limits, 5xx errors, and timeouts
are retried with jittered backoff (SDK retries off, so attempts are counted per row and
latency covers only the successful request). Anything still failing goes to
`errors.jsonl`, not `results.jsonl`. A resumed run retries it, and the report shows it as
missing coverage rather than as a wrong answer. Auth, 404, and 400 errors abort the run,
because they'd fail every case.

**Model output that isn't a usable answer has its own status.** `refusal` counts as not
passed and is listed separately. `truncated` (hit max_tokens) and `unparseable` are left out
of *accuracy* but counted in *strict accuracy*, and the report shows both.

**The model that answered must be the model I asked for.** The runner checks the served
model on every response. It accepts the exact ID, or the ID plus a snapshot-shaped suffix
(`-20251001`, `-2026-08-01`, `-001`, `-09-2025`). A plain prefix match isn't enough:
`claude-fable-5-1` starts with `claude-fable-5` but is a different model. Refusal fallbacks
are deliberately off, because a fallback means another model answered.

**Stop spending as soon as something is systematically wrong.**
- Ctrl-C stops a run immediately: queued cases are cancelled, and only requests already in
  flight finish. The manifest says `interrupted`.
- A response the harness can't interpret (an SDK validation error, an unexpected response
  shape, a served-model mismatch) goes to `errors.jsonl` like any infra error. Three in a
  row abort the run, because that pattern is a harness bug and every further case would be
  billed for nothing.
- An unexpected crash marks the manifest `crashed` instead of leaving it `running`.

**Manifest per run**: model config verbatim, test-set SHA-256, prompt fingerprint, grader
version, reps, case selection, git commit and dirty flag, SQLite/SDK versions. `--resume`
refuses to continue a run if any of those changed, and always resumes the saved case
selection, so resuming a 10-case smoke run can't accidentally turn it into a 150-case run.
Each run gets its own directory, even two started in the same second.

**Adjudication edits are validated, and they don't break the reproducibility check.**
`alt_gold_sql` must be a list of queries that run; a typo fails at load with the case ID,
before any regrade starts. `build_testset.py --check` (and its test) compare the
generated cases ignoring `notes` and `alt_gold_sql`, so the test suite stays green through
the review pass and still catches a change to the sampled cases themselves. Regenerating
refuses to overwrite a file that has adjudication edits.

**Roster**: see "Cross-vendor comparison" below. To change it, add a new roster name
instead of editing an existing one.

## Cross-vendor comparison

**Six models: one flagship-tier and one low-cost model each from Anthropic, OpenAI, and
Google.** Opus 5 / GPT-5.6 Sol / Gemini 3.1 Pro, and Haiku 4.5 / GPT-5.6 Luna / Gemini 3.8
Flash. Sonnet 5 is kept as an optional middle tier.

*Why not more:* API cost is small, but every model adds an `audit.md` to adjudicate, and
the test set was sized for one evening per audit. Two tiers per vendor is enough to ask
both "does the cheap model fall off on extra-hard?" and "does that hold across vendors?".

*Tiers are matched by role:* no two vendors price the same tier the same way
(flagship $5 / $4 / $2 per MTok input; low-cost $1 / $0.20 / $0.75). The report shows cost
per case next to accuracy, so price differences stay visible.
The most expensive models (GPT-6 Astra, Claude Fable 5.1, both $10/$50) are left out;
they can be added as extra roster names.

**What is held identical across vendors**
- The same system prompt, user message, and answer schema (`{"sql": string}`, strict JSON
  schema on all three APIs: Anthropic `output_config.format`, OpenAI `text.format` with
  `strict`, Gemini `response_json_schema`).
- The same output budget (16,000 tokens, reasoning included).
- The same grader, test set, and statistics. A provider class only translates the request
  and response; it has no say in scoring.

**Reasoning effort is "high" everywhere it can be set.** Vendor defaults differ: Opus 5
and Gemini 3.1 Pro default to high, GPT-5.6 and Gemini 3.8 Flash to medium. Comparing
defaults would mean comparing different amounts of thinking and calling it a model
difference. Haiku 4.5 has no effort control and doesn't think by default, so it runs
as-is, and that is noted wherever its numbers appear. Gemini can't turn thinking off.

**Usage normalized to one shape.** Every vendor reports tokens differently (OpenAI counts
cached tokens inside `input_tokens`; Gemini reports thinking tokens separately from
`candidates_token_count`). Each row stores `input_tokens` (full-price input),
`cache_read_input_tokens`, and `output_tokens` (including reasoning), plus
`reasoning_tokens` for information. Cost is always computed from those fields and that
vendor's price table.

**Plumbing rules apply per vendor**
- Pinned model IDs, never `-latest` aliases, with the served model checked on every
  response (OpenAI returns a dated snapshot; Gemini returns `modelVersion`).
- SDK-internal retries off (google-genai retries nothing by default); the shared retry loop
  treats 408/409/429/5xx and transport errors as transient and any other 4xx as fatal.
- Refusals and safety blocks (Anthropic `refusal`, OpenAI refusal content or
  `content_filter`, Gemini `SAFETY`/`RECITATION`/blocked prompt) map to `refusal`; hitting
  the token limit maps to `truncated` on all three.
- OpenAI requests use `store=False`, so no response is kept server-side.

**Known weaknesses**
- `gemini-3.1-pro-preview` is a preview model and can change or be retired on two weeks'
  notice. The served `modelVersion` in each row shows what actually answered.
- GPT-5.6 Sol's price is promotional through 2026-11-21, and Gemini 3.8 Flash's doubles on
  2027-01-01. Cost is recorded per row at run time, so old runs keep their prices.
- The OpenAI and Gemini providers are tested against mock HTTP servers through the real
  SDKs. The first real smoke run per vendor is the check that the live API accepts the
  request.
- Same-prompt fairness has a cost: the prompt was written without tuning for any vendor,
  so no vendor gets its preferred prompt style. That's deliberate; per-vendor tuning would be
  a separate experiment.

## Statistics

**95% CIs from a cluster bootstrap: resample pairs, not questions.** Both halves of a pair
share gold SQL, so they're correlated. Treating 150 questions as independent would make the
intervals too narrow. Each cluster is a pair (or a single), plus all its reps. Model
comparisons are paired, over the same cases.

**Noise floor, stated up front** (simulated on this test set's cluster structure):
CI half-width ≈ ±6 points on overall accuracy and ≈ ±12 per difficulty bucket (25 distinct
queries each). A *paired* 20-point gap within one bucket comes out around ±16, so
"20 points apart on extra-hard" is just detectable, and a 10-point gap isn't. The compare
table prints the interval, so this is visible in every result. Reps reduce sampling noise
*within* a question, but not the uncertainty from having only 25 queries per bucket. More
cases are the only fix for that.

---

## Open work

1. Adjudication: `python -m harness audit <run>` on each run, one checkbox per failure. If
   more than ~1 in 10 failures are grader or gold errors, fix those (`alt_gold_sql`, grader)
   and `regrade`. The results in `comparison.md` have not been adjudicated yet.
2. Test-suite accuracy (see "How lenient is execution match here?"), which matters most if
   an extra-hard gap is close.
