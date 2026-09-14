import sqlite3

from harness.sqlexec import clean_sql, execute


def test_select_returns_rows_and_columns(people_db):
    res = execute(people_db, "SELECT name, age FROM person WHERE city = 'rome' ORDER BY name")
    assert res.ok
    assert res.columns == ["name", "age"]
    assert res.rows == [("bob", 25), ("eve", 25)]


def test_writes_are_rejected_and_database_unchanged(people_db):
    for sql in (
        "DELETE FROM person",
        "UPDATE person SET age = 0",
        "INSERT INTO person (name) VALUES ('x')",
        "DROP TABLE person",
        "CREATE TABLE evil (x)",
    ):
        res = execute(people_db, sql)
        assert res.error_kind == "sql_error", sql
    assert sqlite3.connect(people_db).execute("SELECT count(*) FROM person").fetchone() == (5,)


def test_attach_and_pragma_are_rejected(people_db, tmp_path):
    assert execute(people_db, f"ATTACH DATABASE '{tmp_path / 'other.db'}' AS other").error_kind == "sql_error"
    assert execute(people_db, "PRAGMA table_info(person)").error_kind == "sql_error"
    assert not (tmp_path / "other.db").exists()


def test_multiple_statements_are_rejected(people_db):
    assert execute(people_db, "SELECT 1; SELECT 2").error_kind == "sql_error"


def test_runaway_query_times_out(people_db):
    sql = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"
    res = execute(people_db, sql, timeout_s=0.3)
    assert res.error_kind == "timeout"


def test_row_cap(people_db):
    res = execute(people_db, "SELECT * FROM person a, person b, person c", max_rows=100)
    assert res.error_kind == "too_many_rows"


def test_empty_sql():
    assert execute("/nonexistent", "   ;  ").error_kind == "empty_sql"


def test_clean_sql_strips_fences_and_semicolons():
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert clean_sql("  SELECT 1 ;;  ") == "SELECT 1"
    assert clean_sql(None) == ""


def test_floats_are_rounded(people_db):
    res = execute(people_db, "SELECT score FROM person WHERE name = 'cat'")
    assert res.rows == [(3.333333,)]


def test_reverse_scan_reverses_unordered_results_only(people_db):
    forward = execute(people_db, "SELECT name FROM person").rows
    backward = execute(people_db, "SELECT name FROM person", reverse_scan=True).rows
    assert backward == list(reversed(forward))
    ordered = "SELECT name FROM person ORDER BY name"
    assert execute(people_db, ordered, reverse_scan=True).rows == execute(people_db, ordered).rows
