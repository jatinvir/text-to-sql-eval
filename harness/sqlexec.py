"""Sandboxed SQL execution against a Spider SQLite database.

Model-written SQL is untrusted, so every query runs with three layers of protection:
  1. the file is opened read-only (``mode=ro``),
  2. an authorizer allows only reads - no writes, ATTACH, or PRAGMA from the query,
  3. a wall-clock deadline (progress handler) and a row cap stop runaway queries
     such as accidental cartesian products.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ROWS = 100_000
FLOAT_DECIMALS = 6

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


@dataclass
class ExecResult:
    rows: list[tuple] | None
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    # None on success; otherwise "sql_error" | "timeout" | "too_many_rows" | "empty_sql"
    error_kind: str | None = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error_kind is None


def clean_sql(sql: str | None) -> str:
    """Strip whitespace, markdown fences and trailing semicolons."""
    if not sql:
        return ""
    s = sql.strip()
    fence = re.fullmatch(r"```(?:sql|sqlite)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    if fence:
        s = fence.group(1)
    return s.strip().rstrip(";").strip()


def normalize_value(v):
    # Floats are rounded so AVG()-style results that differ only past the 6th decimal
    # (e.g. from a different but equivalent expression) still compare equal.
    # int 3 and float 3.0 already compare and hash equal in Python.
    if isinstance(v, float):
        return round(v, FLOAT_DECIMALS)
    return v


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    # as_uri() percent-encodes the path, so directories containing ?, # or % still work.
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    # Some Spider databases contain non-UTF-8 bytes in TEXT columns.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def _authorizer(action, *_args):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def execute(
    db_path: str | Path,
    sql: str | None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    reverse_scan: bool = False,
) -> ExecResult:
    """Run one query. Never raises for problems caused by the SQL itself.

    ``reverse_scan`` turns on SQLite's ``reverse_unordered_selects`` pragma, which makes
    queries without ORDER BY return rows in the opposite order. The grader uses it to
    detect predictions whose row order matched gold only by accident.
    """
    sql = clean_sql(sql)
    if not sql:
        return ExecResult(rows=None, error="empty SQL", error_kind="empty_sql")

    path = Path(db_path)
    if not path.exists():
        # A missing database is a harness problem, not a model problem.
        raise FileNotFoundError(path)

    conn = connect_readonly(path)
    start = time.monotonic()
    deadline = start + timeout_s
    timed_out = False

    def progress():
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # non-zero aborts the statement
        return 0

    try:
        if reverse_scan:
            conn.execute("PRAGMA reverse_unordered_selects = ON")
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(progress, 1000)
        cur = conn.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        rows: list[tuple] = []
        while True:
            batch = cur.fetchmany(1000)
            if not batch:
                break
            rows.extend(tuple(normalize_value(v) for v in row) for row in batch)
            if len(rows) > max_rows:
                return ExecResult(
                    rows=None,
                    columns=columns,
                    error=f"more than {max_rows} rows",
                    error_kind="too_many_rows",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )
        return ExecResult(rows=rows, columns=columns, elapsed_ms=(time.monotonic() - start) * 1000)
    except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError) as e:
        elapsed = (time.monotonic() - start) * 1000
        if timed_out:
            return ExecResult(rows=None, error=f"exceeded {timeout_s}s", error_kind="timeout", elapsed_ms=elapsed)
        return ExecResult(rows=None, error=f"{type(e).__name__}: {e}", error_kind="sql_error", elapsed_ms=elapsed)
    finally:
        conn.close()
