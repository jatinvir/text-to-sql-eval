"""Execution-match grading: does the predicted query return the same answer as gold?

Rules (the reasoning behind each is in NOTES.md, "Grader"):
  * Rows are compared as a multiset (bag): duplicates count, row order ignored...
  * ...unless gold has a top-level ORDER BY. Then order is checked, but rows whose sort
    keys tie in gold may appear in any order among themselves.
  * If order matters and the prediction passes, it is re-run with SQLite's
    reverse_unordered_selects pragma. A prediction with no ORDER BY of its own that only
    matched gold's order by accident of scan order fails that re-run.
  * Column order is free (any permutation of the predicted columns may match gold's),
    but the column count must match. Extra columns fail strictly; ``lenient_passed``
    records whether the prediction would pass if extra columns were allowed.
  * Floats compare after rounding to 6 decimals; 3 == 3.0.
  * A case may list alternative gold queries; matching any one passes.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .sqlexec import ExecResult, clean_sql, execute

GRADER_VERSION = "1"

REASONS = {
    "match": "result matches gold",
    "no_sql": "model produced no SQL (refusal, unparseable, or empty)",
    "pred_sql_error": "predicted SQL failed to execute",
    "pred_timeout": "predicted SQL exceeded the execution time limit",
    "pred_too_many_rows": "predicted SQL returned more rows than the cap",
    "missing_columns": "fewer columns than gold",
    "extra_columns": "more columns than gold (see lenient_passed)",
    "wrong_rows": "different rows",
    "duplicates": "same distinct rows as gold but different duplicate counts (DISTINCT mismatch)",
    "wrong_order": "right rows, wrong order (gold has ORDER BY)",
    "order_by_accident": "order matched gold only by accident of scan order (no ORDER BY)",
}

PREVIEW_ROWS = 5
_MAX_PERMUTATION_TRIES = 20_000


class GoldExecutionError(RuntimeError):
    """Gold SQL failed to run. This is a harness/data problem, never a model failure."""


@dataclass
class Grade:
    passed: bool
    lenient_passed: bool
    reason: str
    detail: str = ""
    matched_gold: int | None = None
    flags: list[str] = field(default_factory=list)
    pred_row_count: int | None = None
    gold_row_count: int | None = None
    pred_columns: list[str] = field(default_factory=list)
    gold_columns: list[str] = field(default_factory=list)
    pred_preview: list = field(default_factory=list)
    gold_preview: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# ORDER BY analysis
# ---------------------------------------------------------------------------

_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.I)
_LIMIT_RE = re.compile(r"\blimit\s+(\d+)(?:\s+offset\s+(\d+))?\s*$", re.I)
_SET_OP_RE = re.compile(r"\b(union|intersect|except)\b", re.I)
_DIRECTION_RE = re.compile(r"\s+(asc|desc)(\s+nulls\s+(first|last))?\s*$", re.I)
_SELECT_HEAD_RE = re.compile(r"^\s*select\s+(distinct\s+|all\s+)?", re.I)


def _mask_literals(sql: str) -> str:
    """Replace string literal contents with spaces so keywords inside them are ignored."""
    return _LITERAL_RE.sub(lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[-1], sql)


def _split_top_level(text: str, masked: str, sep: str = ",") -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


@dataclass
class OrderBy:
    prefix: str  # the query up to (not including) ORDER BY
    key_terms: list[str]  # sort expressions, direction stripped
    key_text: str  # the ORDER BY clause body as written, directions included
    limit: int | None
    offset: int


def top_level_order_by(sql: str) -> OrderBy | None:
    """Parse the outermost ORDER BY [LIMIT n [OFFSET m]] of a query, if any."""
    s = clean_sql(sql)
    masked = _mask_literals(s)
    matches = list(_ORDER_BY_RE.finditer(masked))
    if not matches:
        return None
    # The top-level ORDER BY, if present, is the last one: only LIMIT/OFFSET may follow it.
    m = matches[-1]
    if masked[: m.start()].count("(") != masked[: m.start()].count(")"):
        return None  # inside a subquery or window
    tail, tail_masked = s[m.end():], masked[m.end():]
    if tail_masked.count("(") != tail_masked.count(")"):
        return None
    limit, offset = None, 0
    lm = _LIMIT_RE.search(tail_masked)
    if lm:
        limit = int(lm.group(1))
        offset = int(lm.group(2) or 0)
        tail, tail_masked = tail[: lm.start()], tail_masked[: lm.start()]
    raw_terms = _split_top_level(tail, tail_masked)
    terms = [_DIRECTION_RE.sub("", t).strip() for t in raw_terms]
    return OrderBy(prefix=s[: m.start()].rstrip(), key_terms=terms, key_text=tail.strip(), limit=limit, offset=offset)


@dataclass
class OrderSpec:
    """How to check row order against gold.

    ``blocks`` is gold's rows split into runs of equal sort keys; a prediction's rows
    must fill the blocks in sequence, in any order within a block. ``None`` means the
    sort keys could not be recovered and order is compared exactly.
    """

    gold_rows: list[tuple]
    blocks: list[Counter] | None
    limit_tie: bool = False

    def check(self, rows: list[tuple]) -> bool:
        if self.blocks is None:
            return rows == self.gold_rows
        pos = 0
        for block in self.blocks:
            n = sum(block.values())
            if Counter(rows[pos: pos + n]) != block:
                return False
            pos += n
        return pos == len(rows)


def _order_spec(db: str, gold_sql: str, gold: ExecResult) -> tuple[OrderSpec | None, list[str]]:
    """Build the order check for a gold query. Returns (spec or None, flags)."""
    ob = top_level_order_by(gold_sql)
    if ob is None:
        return None, []
    exact = OrderSpec(gold_rows=gold.rows, blocks=None)
    ob_masked = _mask_literals(ob.prefix)
    if (
        _SET_OP_RE.search(ob_masked)
        or not _SELECT_HEAD_RE.match(ob.prefix)
        or any(re.fullmatch(r"\d+", t) for t in ob.key_terms)  # positional ORDER BY 2
    ):
        return exact, ["order_exact"]

    # Re-run gold with the sort keys prepended to the select list and no LIMIT, so we can
    # see which adjacent rows tie on the keys (including ties that straddle the LIMIT).
    k = len(ob.key_terms)
    select_keys = ", ".join(f"({t}) AS __order_key_{i}" for i, t in enumerate(ob.key_terms))
    probe_sql = _SELECT_HEAD_RE.sub(lambda m: "SELECT " + (m.group(1) or "") + select_keys + ", ", ob.prefix, count=1)
    probe_sql += " ORDER BY " + ob.key_text
    probe = _cached_execute(db, probe_sql, False)
    if not probe.ok:
        return exact, ["order_exact"]
    start = ob.offset
    end = len(probe.rows) if ob.limit is None else min(len(probe.rows), start + ob.limit)
    window = probe.rows[start:end]
    if [r[k:] for r in window] != gold.rows:
        # Injecting the keys changed the result (e.g. under DISTINCT); don't trust it.
        return exact, ["order_exact"]

    blocks: list[Counter] = []
    prev_key = object()
    for r in window:
        key = r[:k]
        if key != prev_key:
            blocks.append(Counter())
            prev_key = key
        blocks[-1][r[k:]] += 1
    flags = []
    limit_tie = bool(window) and end < len(probe.rows) and probe.rows[end][:k] == window[-1][:k]
    if limit_tie:
        flags.append("gold_limit_tie")
    if any(sum(b.values()) > 1 for b in blocks):
        flags.append("gold_order_ties")
    return OrderSpec(gold_rows=gold.rows, blocks=blocks, limit_tie=limit_tie), flags


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------


def _column_mapping(pred_rows, gold_rows, n_pred, n_gold, *, bag: bool, order: OrderSpec | None = None):
    """Find an injective map gold column j -> pred column perm[j] under which the
    projected prediction equals gold (as a bag or a set, and in order if given)."""
    agg = Counter if bag else set
    if bag and len(pred_rows) != len(gold_rows):
        return None
    pred_cols = [agg(r[i] for r in pred_rows) for i in range(n_pred)]
    gold_cols = [agg(r[j] for r in gold_rows) for j in range(n_gold)]
    candidates = []
    for j in range(n_gold):
        c = [i for i in range(n_pred) if pred_cols[i] == gold_cols[j]]
        if not c:
            return None
        c.sort(key=lambda i: i != j)  # try the identity mapping first
        candidates.append(c)
    target = agg(gold_rows)
    tries = 0

    def search(j, perm, used):
        nonlocal tries
        if j == n_gold:
            tries += 1
            projected = [tuple(r[i] for i in perm) for r in pred_rows]
            if agg(projected) != target:
                return None
            if order is not None and not order.check(projected):
                return None
            return list(perm)
        for i in candidates[j]:
            if i in used or tries >= _MAX_PERMUTATION_TRIES:
                continue
            perm.append(i)
            used.add(i)
            found = search(j + 1, perm, used)
            if found is not None:
                return found
            perm.pop()
            used.discard(i)
        return None

    return search(0, [], set())


def _compare(pred: ExecResult, gold: ExecResult, order: OrderSpec | None) -> tuple[str, bool, list[int] | None]:
    """Returns (reason, lenient_ok, column mapping if matched)."""
    n_pred, n_gold = len(pred.columns), len(gold.columns)
    if n_pred < n_gold:
        return "missing_columns", False, None
    if n_pred > n_gold:
        lenient = _column_mapping(pred.rows, gold.rows, n_pred, n_gold, bag=True, order=order) is not None
        return "extra_columns", lenient, None
    perm = _column_mapping(pred.rows, gold.rows, n_pred, n_gold, bag=True, order=order)
    if perm is not None:
        return "match", True, perm
    if order is not None and _column_mapping(pred.rows, gold.rows, n_pred, n_gold, bag=True) is not None:
        return "wrong_order", False, None
    if _column_mapping(pred.rows, gold.rows, n_pred, n_gold, bag=False) is not None:
        return "duplicates", False, None
    return "wrong_rows", False, None


@lru_cache(maxsize=4096)
def _cached_execute(db: str, sql: str, reverse_scan: bool) -> ExecResult:
    return execute(db, sql, reverse_scan=reverse_scan)


@lru_cache(maxsize=4096)
def _gold(db: str, gold_sql: str) -> tuple[ExecResult, OrderSpec | None, tuple[str, ...]]:
    res = _cached_execute(db, gold_sql, False)
    if not res.ok:
        raise GoldExecutionError(f"gold SQL failed ({res.error_kind}: {res.error}): {gold_sql}")
    spec, flags = _order_spec(db, gold_sql, res)
    if not res.rows:
        flags = [*flags, "gold_empty"]
    return res, spec, tuple(flags)


def _preview(res: ExecResult | None) -> list:
    if res is None or res.rows is None:
        return []

    def cell(v):
        if isinstance(v, bytes):
            v = repr(v)
        if isinstance(v, str) and len(v) > 80:
            v = v[:77] + "..."
        return v

    return [[cell(v) for v in row] for row in res.rows[:PREVIEW_ROWS]]


def grade(db_path: str | Path, gold_sqls: list[str], pred_sql: str | None) -> Grade:
    db = str(db_path)
    golds = [_gold(db, g) for g in gold_sqls]
    primary, _, primary_flags = golds[0]
    base = dict(
        gold_row_count=len(primary.rows),
        gold_columns=primary.columns,
        gold_preview=_preview(primary),
        flags=list(primary_flags),
    )

    if not clean_sql(pred_sql):
        return Grade(passed=False, lenient_passed=False, reason="no_sql", **base)
    pred = execute(db, pred_sql)
    pred_info = dict(pred_row_count=None if pred.rows is None else len(pred.rows),
                     pred_columns=pred.columns, pred_preview=_preview(pred))
    if not pred.ok:
        reason = {"timeout": "pred_timeout", "too_many_rows": "pred_too_many_rows"}.get(pred.error_kind, "pred_sql_error")
        return Grade(passed=False, lenient_passed=False, reason=reason, detail=pred.error or "", **base, **pred_info)

    first_failure: tuple[str, bool] | None = None
    lenient_any = False
    for idx, (gold, order, flags) in enumerate(golds):
        reason, lenient, perm = _compare(pred, gold, order)
        lenient_any = lenient_any or lenient
        if reason == "match" and order is not None:
            reversed_pred = execute(db, pred_sql, reverse_scan=True)
            if not reversed_pred.ok or _compare(reversed_pred, gold, order)[0] != "match":
                reason = "order_by_accident"
                lenient = False
        if reason == "match":
            base["flags"] = list(flags)
            return Grade(passed=True, lenient_passed=True, reason="match", matched_gold=idx, **base, **pred_info)
        if first_failure is None:
            first_failure = (reason, lenient)

    reason, _ = first_failure
    return Grade(passed=False, lenient_passed=lenient_any, reason=reason, **base, **pred_info)
