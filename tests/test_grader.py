import pytest

from harness.grader import GoldExecutionError, grade, top_level_order_by


def g(db, gold, pred, alts=()):
    return grade(db, [gold, *alts], pred)


def test_identical_query_passes(people_db):
    r = g(people_db, "SELECT name FROM person", "SELECT name FROM person")
    assert r.passed and r.reason == "match"


def test_equivalent_query_passes(people_db):
    r = g(people_db, "SELECT name FROM person WHERE age > 28", "select p.name from person as p where not p.age <= 28;")
    assert r.passed


def test_row_order_ignored_without_gold_order_by(people_db):
    r = g(people_db, "SELECT name FROM person", "SELECT name FROM person ORDER BY name DESC")
    assert r.passed


def test_column_order_is_free(people_db):
    r = g(people_db, "SELECT name, age FROM person", "SELECT age, name FROM person")
    assert r.passed


def test_column_permutation_with_identical_value_sets(people_db):
    # Both columns hold the same multiset of values, so column contents alone can't tell
    # them apart; only the swapped mapping lines the rows up.
    join = "FROM person a JOIN person b ON b.id = a.id % 5 + 1"
    r = g(people_db, f"SELECT a.age, b.age {join}", f"SELECT b.age, a.age {join}")
    assert r.passed


def test_extra_columns_fail_strict_but_pass_lenient(people_db):
    r = g(people_db, "SELECT name FROM person", "SELECT id, name FROM person")
    assert not r.passed and r.reason == "extra_columns" and r.lenient_passed


def test_extra_columns_with_wrong_rows_fail_lenient_too(people_db):
    r = g(people_db, "SELECT name FROM person", "SELECT id, name FROM person WHERE age > 26")
    assert not r.lenient_passed


def test_missing_columns(people_db):
    r = g(people_db, "SELECT name, age FROM person", "SELECT name FROM person")
    assert r.reason == "missing_columns" and not r.lenient_passed


def test_duplicates_matter(people_db):
    r = g(people_db, "SELECT DISTINCT city FROM person", "SELECT city FROM person")
    assert not r.passed and r.reason == "duplicates"


def test_wrong_rows(people_db):
    r = g(people_db, "SELECT name FROM person WHERE age > 26", "SELECT name FROM person WHERE age >= 30 AND city = 'paris'")
    assert r.reason == "wrong_rows"


def test_gold_order_by_enforces_order(people_db):
    r = g(people_db, "SELECT name FROM person ORDER BY name", "SELECT name FROM person ORDER BY name DESC")
    assert not r.passed and r.reason == "wrong_order"


def test_ties_in_gold_order_may_appear_in_any_order(people_db):
    gold = "SELECT name FROM person ORDER BY age"
    r = g(people_db, gold, "SELECT name FROM person ORDER BY age, name DESC")
    assert r.passed
    assert "gold_order_ties" in r.flags
    # ...but the tie groups themselves must still be in order.
    assert g(people_db, gold, "SELECT name FROM person ORDER BY age DESC").reason == "wrong_order"


def test_order_matching_by_accident_fails(people_db):
    # Rows are stored in id order, so an unordered scan happens to match ORDER BY id.
    r = g(people_db, "SELECT name FROM person ORDER BY id", "SELECT name FROM person")
    assert not r.passed and r.reason == "order_by_accident"
    assert g(people_db, "SELECT name FROM person ORDER BY id", "SELECT name FROM person ORDER BY id ASC").passed


def test_limit_tie_is_flagged(people_db):
    r = g(people_db, "SELECT name FROM person ORDER BY age LIMIT 1", "SELECT name FROM person ORDER BY age LIMIT 1")
    assert r.passed and "gold_limit_tie" in r.flags


def test_order_by_on_column_not_selected(people_db):
    gold = "SELECT name FROM person ORDER BY score DESC LIMIT 2"
    assert g(people_db, gold, "SELECT name FROM person ORDER BY score DESC LIMIT 2").passed
    assert g(people_db, gold, "SELECT name FROM person ORDER BY score ASC LIMIT 2").reason == "wrong_rows"


def test_float_tolerance_and_int_float_equality(people_db):
    assert g(people_db, "SELECT avg(score) FROM person", "SELECT sum(score) / count(*) FROM person").passed
    assert g(people_db, "SELECT count(*) FROM person", "SELECT 5.0").passed
    assert not g(people_db, "SELECT avg(score) FROM person", "SELECT round(avg(score), 2) FROM person").passed


def test_pred_errors(people_db):
    assert g(people_db, "SELECT name FROM person", "SELECT nme FROM person").reason == "pred_sql_error"
    assert g(people_db, "SELECT name FROM person", None).reason == "no_sql"
    assert g(people_db, "SELECT name FROM person", "DELETE FROM person").reason == "pred_sql_error"


def test_alternative_gold(people_db):
    r = g(people_db, "SELECT name FROM person WHERE age > 30", "SELECT name FROM person WHERE age >= 30",
          alts=["SELECT name FROM person WHERE age >= 30"])
    assert r.passed and r.matched_gold == 1


def test_broken_gold_raises(people_db):
    with pytest.raises(GoldExecutionError):
        g(people_db, "SELECT nope FROM person", "SELECT name FROM person")


def test_empty_gold_is_flagged(people_db):
    r = g(people_db, "SELECT name FROM person WHERE age > 100", "SELECT name FROM person WHERE 0")
    assert r.passed and "gold_empty" in r.flags


def test_top_level_order_by_parsing():
    ob = top_level_order_by("SELECT a FROM t ORDER BY count(*) DESC, b LIMIT 3 OFFSET 1;")
    assert ob.key_terms == ["count(*)", "b"] and ob.limit == 3 and ob.offset == 1
    assert top_level_order_by("SELECT a FROM t WHERE x = (SELECT y FROM u ORDER BY z LIMIT 1)") is None
    assert top_level_order_by("SELECT a FROM t WHERE b = 'order by me'") is None
    assert top_level_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a) ORDER BY a DESC").key_terms == ["a"]
