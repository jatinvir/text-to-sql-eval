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

---

## Next step

Write `build_testset.py`'s sampler:

1. Group all 1034 records by `(db_id, gold_sql)` → 564 groups, 470 of size 2
2. Label each group with `eval_hardness`
3. Per difficulty bucket: pick ~12–13 pair-groups (take both questions) and
   ~12–13 further groups (take one question)
4. Write `cases.yaml` — `case_id`, `db_id`, `question`, `gold_sql`, `difficulty`,
   and a field marking which pair a question belongs to
5. Seeded

Then: download the ~20 dev databases. Then the grader, which is the actual project.
