"""How lenient is execution-match grading on this test set?

Applies small, plausible-but-wrong edits to each gold query (the kind of mistake a model
makes) and grades the mutant against gold. A mutant that still passes is a wrong query
the grader cannot catch, because the database contents don't distinguish it from gold.
That is the ceiling on how far "accuracy" can be trusted as "correctness".

    python -m analysis.grader_mutations
"""

import re
from collections import Counter, defaultdict

from harness import testset
from harness.grader import grade
from harness.sqlexec import execute


def _sub_first(pattern, repl, flags=re.I):
    return lambda sql: re.sub(pattern, repl, sql, count=1, flags=flags)


def _after_where(pattern, repl):
    def mutate(sql):
        m = re.search(r"\bwhere\b", sql, re.I)
        if not m:
            return sql
        return sql[: m.end()] + re.sub(pattern, repl, sql[m.end():], count=1, flags=re.I)
    return mutate


MUTATIONS = {
    "flip >= to >": _sub_first(r">=", ">"),
    "flip > to >=": _sub_first(r">(?!=)", ">="),
    "flip <= to <": _sub_first(r"<=", "<"),
    "flip < to <=": _sub_first(r"<(?![=>])", "<="),
    "where = to !=": _after_where(r"(?<![<>!])=", "!="),
    "AND to OR": _sub_first(r"\band\b", "OR"),
    "DESC to ASC": _sub_first(r"\bdesc\b", "ASC"),
    "ASC/default to DESC": lambda s: re.sub(r"\border\s+by\s+([^\s,]+)(?!\s+desc)(\s+asc)?", r"ORDER BY \1 DESC", s, count=1, flags=re.I),
    "drop DISTINCT": _sub_first(r"\bdistinct\s+", ""),
    "add DISTINCT": _sub_first(r"^\s*select\s+(?!distinct)", "SELECT DISTINCT "),
    "LIMIT n to n+1": lambda s: re.sub(r"\blimit\s+(\d+)", lambda m: f"LIMIT {int(m.group(1)) + 1}", s, count=1, flags=re.I),
    "max to min": _sub_first(r"\bmax\s*\(", "min("),
    "min to max": _sub_first(r"\bmin\s*\(", "max("),
    "avg to sum": _sub_first(r"\bavg\s*\(", "sum("),
    "JOIN to LEFT JOIN": _sub_first(r"(?<!left )\bjoin\b", "LEFT JOIN"),
    "INTERSECT to UNION": _sub_first(r"\bintersect\b", "UNION"),
    "EXCEPT to UNION": _sub_first(r"\bexcept\b", "UNION"),
}

# Rewrites that are frequently *correct* answers (DISTINCT around COUNT(*) or unique values;
# LEFT JOIN when the WHERE clause filters on the right table). A pass there is usually the
# grader being right, so they are reported separately from wrong-by-construction edits.
OFTEN_EQUIVALENT = {"add DISTINCT", "JOIN to LEFT JOIN"}


def main():
    cases = testset.load().cases
    seen_gold = set()
    tally: dict[str, Counter] = defaultdict(Counter)
    survivors = []
    for case in cases:
        if (case.db_id, case.gold_sql) in seen_gold:
            continue  # both halves of a pair share gold
        seen_gold.add((case.db_id, case.gold_sql))
        for name, mutate in MUTATIONS.items():
            mutant = mutate(case.gold_sql)
            if mutant == case.gold_sql:
                continue
            res = execute(case.db_path, mutant)
            if not res.ok:
                tally[name]["errors"] += 1
                continue
            passed = grade(case.db_path, case.gold_sqls, mutant).passed
            tally[name]["applied"] += 1
            tally[name]["passed"] += passed
            if passed:
                survivors.append((name, case.case_id, case.difficulty, mutant))

    for title, names in (
        ("Wrong-by-construction edits (a pass is a grader miss)", [n for n in tally if n not in OFTEN_EQUIVALENT]),
        ("Often-equivalent rewrites (a pass is usually correct)", [n for n in tally if n in OFTEN_EQUIVALENT]),
    ):
        print(f"\n{title}\n{'mutation':34s} {'applied':>7s} {'passed':>7s}  rate")
        total = Counter()
        for name in sorted(names, key=lambda n: -tally[n]["applied"]):
            t = tally[name]
            total.update(t)
            print(f"{name:34s} {t['applied']:7d} {t['passed']:7d}  {100 * t['passed'] / t['applied']:4.0f}%")
        print(f"{'TOTAL':34s} {total['applied']:7d} {total['passed']:7d}  {100 * total['passed'] / total['applied']:4.0f}%")

    misses = [s for s in survivors if s[0] not in OFTEN_EQUIVALENT]
    print("\nGrader misses by difficulty:", dict(Counter(d for _, _, d, _ in misses)))
    for name, cid, diff, sql in misses:
        print(f"  {cid} [{diff}] {name}: {sql}")


if __name__ == "__main__":
    main()
