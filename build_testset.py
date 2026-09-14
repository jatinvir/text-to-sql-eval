"""Build cases.yaml: 150 Spider dev questions, stratified by difficulty.

50 paraphrase pairs + 50 singles = 100 distinct gold queries, 25 per difficulty bucket.
Seeded; re-running reproduces cases.yaml byte-for-byte. This script is a record of how
the test set was made, not a live dependency - the committed cases.yaml is the test set.

    python build_testset.py            # write cases.yaml
    python build_testset.py --check    # verify cases.yaml matches what the script produces
"""

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field

import yaml

from harness.grader import grade
from harness.paths import CASES_YAML, VALIDATION_JSON, db_path
from harness.sqlexec import execute
from vendor.spider_eval.hardness import eval_hardness

SEED = 20260815
TESTSET_VERSION = 1
BUCKETS = ["easy", "medium", "hard", "extra"]
PAIRS_PER_BUCKET = {"easy": 13, "medium": 13, "hard": 12, "extra": 12}
SINGLES_PER_BUCKET = {"easy": 12, "medium": 12, "hard": 13, "extra": 13}

EXCLUSION_REASONS = {
    "gold_empty": "gold returns no rows - an empty prediction passes, so the case can't discriminate",
    "gold_degenerate": "gold returns only 0/NULL - a wrong filter returns the same thing",
    "gold_limit_tie": "ORDER BY ... LIMIT cuts through tied rows - gold is one arbitrary pick",
    "halves_disagree": "the two paraphrases' gold queries return different results - one gold is wrong",
}


@dataclass
class Group:
    key: tuple[str, str]
    members: list[int] = field(default_factory=list)
    difficulty: str = ""
    exclusion: str | None = None


def normalize_sql(sql: str) -> str:
    # Case-insensitive on purpose: 13 paraphrase pairs have gold SQL that differs only in
    # letter case. Where that case change altered a string literal, the halves return
    # different results and the group is excluded as "halves_disagree".
    return " ".join(sql.strip().rstrip(";").lower().split())


def group_records(records: list[dict]) -> list[Group]:
    groups: dict[tuple[str, str], Group] = {}
    for idx, rec in enumerate(records):
        key = (rec["db_id"], normalize_sql(rec["query"]))
        groups.setdefault(key, Group(key=key)).members.append(idx)
    for g in groups.values():
        levels = {eval_hardness(records[i]["sql"]) for i in g.members}
        assert len(levels) == 1, f"hardness differs within group {g.key}: {levels}"
        g.difficulty = levels.pop()
    return sorted(groups.values(), key=lambda g: g.key)


def exclusion_for(group: Group, records: list[dict]) -> str | None:
    for i in group.members:
        rec = records[i]
        g = grade(db_path(rec["db_id"]), [rec["query"]], rec["query"])
        if "gold_empty" in g.flags:
            return "gold_empty"
        if "gold_limit_tie" in g.flags:
            return "gold_limit_tie"
    first = records[group.members[0]]
    for i in group.members[1:]:
        if not grade(db_path(first["db_id"]), [first["query"]], records[i]["query"]).passed:
            return "halves_disagree"
    rows = execute(db_path(first["db_id"]), first["query"]).rows
    if all(v in (0, None) for row in rows for v in row):
        return "gold_degenerate"
    return None


def sample(groups: list[Group], records: list[dict]) -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    cases: list[dict] = []
    pair_counter = 0
    stats = {}
    for bucket in BUCKETS:
        eligible = [g for g in groups if g.difficulty == bucket and g.exclusion is None]
        pair_pool = [g for g in eligible if len(g.members) == 2]
        pairs = rng.sample(pair_pool, PAIRS_PER_BUCKET[bucket])
        chosen = {g.key for g in pairs}
        rest = [g for g in eligible if g.key not in chosen]
        singles = rng.sample(rest, SINGLES_PER_BUCKET[bucket])
        stats[bucket] = {"eligible_groups": len(eligible), "eligible_pair_groups": len(pair_pool)}

        for g in sorted(pairs, key=lambda g: g.members[0]):
            pair_counter += 1
            for i in g.members:
                cases.append(_case(records, i, bucket, pair_id=f"p{pair_counter:02d}"))
        single_members = sorted(rng.choice(g.members) for g in singles)
        for i in single_members:
            cases.append(_case(records, i, bucket, pair_id=None))

    for n, case in enumerate(cases, start=1):
        case["case_id"] = f"c{n:03d}"
    return cases, stats


def _case(records, idx, difficulty, pair_id):
    rec = records[idx]
    return {
        "case_id": None,
        "pair_id": pair_id,
        "difficulty": difficulty,
        "db_id": rec["db_id"],
        "source_index": idx,
        "question": rec["question"],
        "gold_sql": rec["query"],
        "alt_gold_sql": [],
        "notes": "",
    }


def _q(s: str) -> str:
    # JSON string syntax is valid YAML double-quoted syntax; quoting every string avoids
    # YAML's implicit typing (a db_id or question that looks like a bool/number/null).
    return json.dumps(s, ensure_ascii=False)


def render_yaml(cases: list[dict], exclusions: Counter, stats: dict, n_records: int, n_groups: int) -> str:
    lines = [
        "# Spider dev text-to-SQL test set. Generated by build_testset.py; see NOTES.md.",
        "# Regenerating with the same seed reproduces this file exactly.",
        "# Hand edits are expected only in `notes` and `alt_gold_sql` (adjudication).",
        "# Any edit changes the file hash, which every run records.",
        f"version: {TESTSET_VERSION}",
        f"seed: {SEED}",
        f"source: {_q(f'premai-io/spider validation.json (Spider dev, {n_records} questions, {n_groups} distinct gold queries)')}",
        "excluded_groups:",
    ]
    for reason in EXCLUSION_REASONS:
        lines.append(f"  {reason}: {exclusions.get(reason, 0)}  # {EXCLUSION_REASONS[reason]}")
    lines.append("eligible_groups:")
    for bucket in BUCKETS:
        s = stats[bucket]
        lines.append(f"  {bucket}: {{groups: {s['eligible_groups']}, pair_groups: {s['eligible_pair_groups']}}}")
    lines.append("cases:")
    for c in cases:
        flags = grade(db_path(c["db_id"]), [c["gold_sql"]], c["gold_sql"]).flags
        if "gold_order_ties" in flags:
            lines.append("  # gold ORDER BY has tied sort keys; order is only checked between tie groups")
        lines.append(f"  - case_id: {_q(c['case_id'])}")
        lines.append(f"    pair_id: {_q(c['pair_id']) if c['pair_id'] else 'null'}")
        lines.append(f"    difficulty: {_q(c['difficulty'])}")
        lines.append(f"    db_id: {_q(c['db_id'])}")
        lines.append(f"    source_index: {c['source_index']}")
        lines.append(f"    question: {_q(c['question'])}")
        lines.append(f"    gold_sql: {_q(c['gold_sql'])}")
        lines.append("    alt_gold_sql: []")
        lines.append(f"    notes: {_q(c['notes'])}")
    return "\n".join(lines) + "\n"


def build() -> tuple[str, list[dict]]:
    with open(VALIDATION_JSON) as f:
        records = json.load(f)
    groups = group_records(records)
    for g in groups:
        g.exclusion = exclusion_for(g, records)
    exclusions = Counter(g.exclusion for g in groups if g.exclusion)
    cases, stats = sample(groups, records)
    text = render_yaml(cases, exclusions, stats, len(records), len(groups))
    parsed = yaml.safe_load(text)
    assert parsed["cases"] == cases, "YAML round-trip changed the data"

    print(f"{len(records)} records -> {len(groups)} groups "
          f"({sum(len(g.members) == 2 for g in groups)} pairs); excluded: {dict(exclusions)}", file=sys.stderr)
    by_bucket = Counter((c["difficulty"], c["pair_id"] is not None) for c in cases)
    for bucket in BUCKETS:
        print(f"  {bucket:6s} pair questions={by_bucket[(bucket, True)]:3d} singles={by_bucket[(bucket, False)]:3d}"
              f"  eligible={stats[bucket]}", file=sys.stderr)
    return text, cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify cases.yaml is what this script generates")
    args = ap.parse_args()
    text, cases = build()
    if args.check:
        current = CASES_YAML.read_text() if CASES_YAML.exists() else ""
        if current != text:
            print("cases.yaml differs from generated output (hand annotations, or the sampler changed)", file=sys.stderr)
            sys.exit(1)
        print("cases.yaml matches generator output", file=sys.stderr)
        return
    CASES_YAML.write_text(text)
    print(f"wrote {CASES_YAML} ({len(cases)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
