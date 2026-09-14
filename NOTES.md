# Decisions log

Newest at the bottom. Each entry: what was decided, and why — the why is the part
that matters in an interview.

## Test set source

**Spider dev split (`validation`), 1034 cases.** Not `train`.

- Dev is what published papers report on; the real test split was never released,
  so dev is the de facto benchmark and our numbers are comparable to published ones.
- Train is ~7000 cases across 140 databases — more data and more databases to
  download than we need.

**Known weakness:** Spider is from 2018 and partly saturated (frontier models score
~85–90%). Accepted deliberately. The harness is benchmark-agnostic — swapping in
BIRD later is a loader change, not a rewrite. Also, contamination is likely (these
questions have been public for years), but it hits all models equally and we are
comparing models against each other on identical inputs.

## Where the data came from

`premai-io/spider` on HuggingFace, **not** `xlangai/spider`.

`xlangai/spider` contains only parquet files — the rows, no SQLite databases. Without
the databases there is nothing to execute against, and execution is the entire
grading strategy. `premai-io/spider` has both (`validation.json` + `database/`).

Full repo is 919 MB, but most of that is `soccer_1` (~640 MB), a *train* database we
never touch. Download selectively: read the distinct `db_id`s out of `validation.json`
first, then fetch only those ~20 database directories via `allow_patterns`.

## Difficulty labels

Vendored Spider's own `eval_hardness` into `vendor/spider_eval/hardness.py`
(Apache 2.0, source + commit SHA recorded in that folder).

Used the official definition rather than writing our own, because:

1. We slice the headline result on this metric. "That's Spider's own hardness
   definition, unchanged" ends the conversation; a homemade metric invites
   "why is that hard?"
2. Inventing the metric *and* reporting results broken down by it invites the
   question of whether the buckets were tuned to make the result look good.

Trimmed rather than vendored whole: `evaluation.py` imports `process_sql`, which
imports `nltk` — a large dependency for code the hardness path never calls.
Trimming forks the file, which is why the upstream commit SHA is recorded.

How it works: three counters, then thresholds.
- `count_component1` — surface structure: +1 each for WHERE/GROUP BY/ORDER BY/LIMIT,
  +1 per extra joined table, +1 per OR, +1 per LIKE. (AND is free; ORs are weighted
  because models get disjunctions wrong more often.)
- `count_component2` — number of nested subqueries.
- `count_others` — breadth: +1 for each of "more than one" aggregate / select column /
  where condition / group by.

Nesting is a **gate, not points**: `comp2 == 0` guards easy and both medium clauses,
so any subquery at all means at least "hard".

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

*Stratified, not random:* a random 150 hands you whatever it hands you (~16% extra-hard).
Stratified means we choose the cell sizes, so per-difficulty comparisons have enough
cases in each bucket to say anything. The finding we want is the interaction —
"level on easy, 20 points apart on extra-hard, at 15x the cost" — not the main effect.

*Pairs:* Spider dev is 564 distinct queries, 470 of which were paraphrased into two
questions each. Sampling both halves lets us measure whether a model flips when the
question is reworded. This is the answer to the saturation problem — aggregate
accuracy stops discriminating at 88%, but *consistency* doesn't. Pairs must be
sampled as a unit; sampling questions independently produces orphaned halves.
Both halves of a pair always share a difficulty label (same SQL), so pairs never
straddle buckets.

*Why ~50 pairs and not 20:* at a true flip rate of ~5%, 20 pairs yields ~1 observed
flip. Cannot distinguish 3% from 8%. Need enough pairs for the metric to mean anything.

*Why 150 and not 100:* API cost is not the constraint — a full run is well under
$1 per model. The real constraint is **human adjudication time**. Every edge case
(empty result sets, row order, column order, float precision) is a judgment call we
have to make, encode, and then verify by eyeballing actual cases. ~150 cases is an
evening's audit. 1000 would be a week, so it would never get done and the grader
would quietly be wrong.

*YAML over JSONL:* YAML supports comments, and we will want to annotate cases where
the gold SQL is questionable ("gold uses ORDER BY without a tiebreak — ties are
non-deterministic"). Ground truth you cannot annotate is harder to defend. YAML also
diffs per-field; JSONL shows a whole case as one changed line. Cost: YAML coerces
unquoted values, so quote consistently.

*Seed the sampler.* Hardcode `random.seed(...)`. Unseeded, re-running produces a
different 150 and silently invalidates comparison with every past run. Generate once,
commit the file; the script is a record of how it was made, not a live dependency.

*Version the test set.* Stamp every run with the case-list version, so "accuracy
dropped 4 points" is always attributable to the model or to the cases, never
ambiguously both.

## Pair grouping is case-insensitive

**Group by `(db_id, lowercased + whitespace-collapsed gold SQL)` → 551 groups, 483 pairs.**
Not exact string match (564 groups, 470 pairs).

13 paraphrase pairs have gold SQL that differs only in letter case — one half is an
all-lowercase copy of the other. Exact grouping splits each into two "singles", and the sampler
could draw both halves as unrelated singles: a duplicated query in the singles, and pair
data we threw away.

Lowercasing is *not* semantically safe — SQLite `=` on text is case-sensitive — and
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

Measured, not argued: on the full 1034, the empty-result baseline scores 43 passes. On the
sampled 150 it scores 0 (`python -m harness selfcheck`).

*Cost:* a slight shift in the difficulty mix, and `flight_2` is under-represented. Accepted
— a case that can't tell a right query from a wrong one adds noise, not signal.

*Detecting LIMIT ties:* re-run gold with its sort keys prepended to the select list and
the LIMIT removed, then check whether the row at the cutoff ties with the next one. Works
for all 231 top-level ORDER BY queries in dev (167 no ties, 20 ties inside the result,
44 at the cutoff).

## Grader

**Execution match.** Run gold and prediction against the database and compare results.
Not exact-match (many right queries per question) and not Spider's component matching
(needs the question's parse, and rewards looking like gold rather than being right).

The rules, each a judgment call:

- **Rows are a bag (multiset).** Duplicate counts matter, so a missing or extra
  `DISTINCT` fails — with its own reason, `duplicates`, so the audit can tell whether
  the question actually implied distinctness.
- **Column order is free, column count is not.** "Names and ages" doesn't define an order.
  The grader searches for a column permutation that reproduces gold's rows; matching each
  column's value multiset alone isn't enough when two columns hold the same values (tested).
  Extra columns fail strictly. `lenient_passed` records whether extra columns were the
  *only* problem, and the report shows lenient accuracy next to strict. Extra id columns are
  the most common "wrong but useful" LLM answer, so the gap between the two is worth knowing.
  The prompt asks for exactly the requested columns, so strict grading doesn't penalize a
  model for following the instructions it was given.
- **Order only matters if gold has a top-level ORDER BY**, detected with a
  literal-aware parser that agrees with Spider's own parse on 1034/1034 dev queries. Rows
  whose sort keys tie in gold may come in any order among themselves: gold's rows are split
  into blocks of equal keys, and the prediction must fill the blocks in sequence.
- **Right order by accident fails.** Predictions that pass an ordered case are re-run with
  SQLite's `reverse_unordered_selects` pragma. A query with no ORDER BY of its own that
  only matched because rows happen to be stored in order comes back reversed and fails
  (`order_by_accident`). The pragma doesn't affect queries that have ORDER BY (verified),
  so a correct query is unaffected.
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
scores 100% on the 150 and the empty baseline 0%. 52 tests, including the edge cases above
and the Anthropic provider against a mock HTTP transport through the real SDK.

### How lenient is execution match here?

`python -m analysis.grader_mutations` applies wrong-by-construction edits to each gold query
(`>`→`>=`, `=`→`!=`, `AND`→`OR`, `DESC`→`ASC`, `INTERSECT`→`UNION`, LIMIT n→n+1, …) and
grades the mutant.

**15% of wrong mutants pass (22/146).** Almost all are inequality boundaries: `>`→`>=`
survives 9/15 and `<`→`<=` 4/4, because no row sits exactly on the boundary value. The
survivors skew hard: easy 3, medium 4, hard 5, **extra 10**.

Consequence for the headline: execution match overstates accuracy most on extra-hard,
which is exactly the bucket the comparison leans on. The fix is test-suite accuracy
(Zhong et al., 2020) — grade against several fuzzed copies of each database so boundary
values exist. It's a grader extension, not a rewrite, and is the first thing to build if
extra-hard differences turn out to be close.

(Separately, "add DISTINCT" and "JOIN → LEFT JOIN" pass ~90%, but those rewrites are
usually *correct* — `DISTINCT count(*)` is still the count — so they're reported apart.)

## Prompt (v1)

**CREATE TABLE statements verbatim + 3 example rows per table; JSON structured output
`{"sql": ...}`.** Averages ~800 input tokens.

- DDL keeps primary and foreign keys, which is how joins get found. Example rows show value
  formats the DDL hides (`'T'`/`'F'` flags, lowercase country names in `car_1`); without
  them, a filter can silently match nothing.
- Structured output removes "couldn't find the SQL in the prose" as a failure mode, so the
  score measures SQL, not formatting. All three vendors' APIs support it for every roster model.
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

**The model that answered must be the model we asked for.** The runner checks the served
model on every response, allowing only alias → dated snapshot. Refusal fallbacks are
deliberately off: a fallback means another model answered.

**Manifest per run**: model config verbatim, test-set SHA-256, prompt fingerprint, grader
version, reps, git commit and dirty flag, SQLite/SDK versions. `--resume` refuses to continue
a run if any of those changed.

**Roster**: see "Cross-vendor comparison" below. Change it by adding a new roster name,
not by editing an existing one.

## Cross-vendor comparison

**Six models: one flagship-tier and one low-cost model each from Anthropic, OpenAI, and
Google.** Opus 5 / GPT-5.6 Sol / Gemini 3.1 Pro, and Haiku 4.5 / GPT-5.6 Luna / Gemini 3.8
Flash. Sonnet 5 is kept as an optional middle tier.

*Why not more:* API cost is small, but every model adds an `audit.md` to adjudicate, and
the test set was sized for one evening per audit. Two tiers per vendor is enough to ask
both "does the cheap model fall off on extra-hard?" and "does that hold across vendors?".

*Tiers matched by role, not exact price:* no two vendors price the same tier the same way
(flagship $5 / $4 / $2 per MTok input; low-cost $1 / $0.20 / $0.75). The report shows cost
per case next to accuracy, so price differences are visible rather than assumed away.
The most expensive models (GPT-6 Astra, Claude Fable 5.1, both $10/$50) are left out;
they can be added as extra roster names.

**What is held identical across vendors**
- The same system prompt, user message, and answer schema (`{"sql": string}`, strict JSON
  schema on all three APIs: Anthropic `output_config.format`, OpenAI `text.format` with
  `strict`, Gemini `response_json_schema`).
- The same output budget (16,000 tokens, reasoning included).
- The same grader, test set, and statistics. A provider class only translates the request
  and response; it has no say in scoring.

**Reasoning effort: "high" everywhere it can be set, not vendor defaults.** Defaults
differ: Opus 5 and Gemini 3.1 Pro default to high, GPT-5.6 and Gemini 3.8 Flash to medium.
"Defaults vs defaults" would mean comparing different amounts of thinking and calling it
a model difference. Haiku 4.5 has no effort control and doesn't think by default, so it
runs as-is, and that is noted wherever its numbers appear. Gemini can't turn thinking off.

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

## Next step

1. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, then smoke-test each model:
   `python -m harness run <model> --limit 10` (checks auth, the model ID, the request format
   against the live API, the served-model check, and real cost per case)
2. Full runs of the six roster models; `python -m harness compare …`
3. The adjudication evening: `python -m harness audit <run>` on each run, one checkbox per failure.
   If more than ~1 in 10 failures are grader or gold errors, fix those (`alt_gold_sql`, grader)
   and `regrade` before reading any comparison
4. If the extra-hard gap is close: test-suite accuracy (see "How lenient is execution match here?")
