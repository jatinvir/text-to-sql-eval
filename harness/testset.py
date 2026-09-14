"""Load and validate cases.yaml."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .paths import CASES_YAML, db_path

DIFFICULTIES = ("easy", "medium", "hard", "extra")


@dataclass(frozen=True)
class Case:
    case_id: str
    pair_id: str | None
    difficulty: str
    db_id: str
    source_index: int
    question: str
    gold_sql: str
    alt_gold_sql: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def gold_sqls(self) -> list[str]:
        return [self.gold_sql, *self.alt_gold_sql]

    @property
    def cluster(self) -> str:
        """Statistical unit: both halves of a pair share gold SQL, so they are not
        independent observations and are resampled together."""
        return self.pair_id or self.case_id

    @property
    def db_path(self) -> Path:
        return db_path(self.db_id)


@dataclass(frozen=True)
class TestSet:
    __test__ = False  # not a pytest test class

    path: Path
    version: int
    sha256: str
    cases: tuple[Case, ...]

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]

    def by_id(self) -> dict[str, Case]:
        return {c.case_id: c for c in self.cases}


def load(path: Path = CASES_YAML) -> TestSet:
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    cases = tuple(
        Case(
            case_id=c["case_id"],
            pair_id=c.get("pair_id"),
            difficulty=c["difficulty"],
            db_id=c["db_id"],
            source_index=c["source_index"],
            question=c["question"],
            gold_sql=c["gold_sql"],
            alt_gold_sql=tuple(c.get("alt_gold_sql") or ()),
            notes=c.get("notes") or "",
        )
        for c in doc["cases"]
    )
    _validate(cases)
    return TestSet(path=path, version=doc["version"], sha256=hashlib.sha256(raw).hexdigest(), cases=cases)


def _validate(cases: tuple[Case, ...]) -> None:
    ids = Counter(c.case_id for c in cases)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate case_id(s): {dupes}")
    for c in cases:
        if c.difficulty not in DIFFICULTIES:
            raise ValueError(f"{c.case_id}: unknown difficulty {c.difficulty!r}")
        if not c.db_path.exists():
            raise ValueError(f"{c.case_id}: database missing at {c.db_path} (run download_data.py)")
    pairs: dict[str, list[Case]] = {}
    for c in cases:
        if c.pair_id:
            pairs.setdefault(c.pair_id, []).append(c)
    for pid, members in pairs.items():
        if len(members) != 2:
            raise ValueError(f"pair {pid} has {len(members)} members; pairs must be sampled as a unit")
        a, b = members
        if (a.difficulty, a.db_id) != (b.difficulty, b.db_id):
            raise ValueError(f"pair {pid} halves differ in difficulty or database")
