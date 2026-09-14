import json
from collections import Counter

import build_testset
from harness import testset
from harness.grader import grade, top_level_order_by
from harness.paths import CASES_YAML, VALIDATION_JSON, db_path


def test_shape_matches_design():
    ts = testset.load()
    cases = ts.cases
    assert len(cases) == 150
    pairs = Counter(c.pair_id for c in cases if c.pair_id)
    assert len(pairs) == 50 and set(pairs.values()) == {2}
    assert sum(1 for c in cases if not c.pair_id) == 50
    clusters_per_bucket = Counter()
    for cluster in {(c.difficulty, c.cluster) for c in cases}:
        clusters_per_bucket[cluster[0]] += 1
    assert clusters_per_bucket == {"easy": 25, "medium": 25, "hard": 25, "extra": 25}


def test_every_case_is_gradeable_and_discriminating():
    for c in testset.load().cases:
        r = grade(c.db_path, c.gold_sqls, c.gold_sql)
        assert r.passed, c.case_id
        assert "gold_empty" not in r.flags and "gold_limit_tie" not in r.flags, c.case_id


def test_distinct_gold_per_single():
    cases = testset.load().cases
    keys = Counter((c.db_id, build_testset.normalize_sql(c.gold_sql)) for c in cases if not c.pair_id)
    assert max(keys.values()) == 1
    pair_keys = {(c.db_id, build_testset.normalize_sql(c.gold_sql)) for c in cases if c.pair_id}
    assert not pair_keys & set(keys)


def test_generator_reproduces_committed_file():
    text, _ = build_testset.build()
    assert text == CASES_YAML.read_text(), "cases.yaml was hand-edited or the sampler changed; see build_testset.py --check"


def test_order_by_detection_agrees_with_spider_parser():
    records = json.loads(VALIDATION_JSON.read_text())
    for rec in records:
        assert (top_level_order_by(rec["query"]) is not None) == bool(rec["sql"]["orderBy"]), rec["query"]


def test_all_dev_gold_queries_pass_against_themselves():
    records = json.loads(VALIDATION_JSON.read_text())
    for rec in records:
        assert grade(db_path(rec["db_id"]), [rec["query"]], rec["query"]).passed, rec["query"]
