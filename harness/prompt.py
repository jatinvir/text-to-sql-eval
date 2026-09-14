"""Prompt construction. Any change to the text or schema rendering must bump PROMPT_VERSION,
because scores from different prompts are not comparable."""

from __future__ import annotations

import hashlib
import sqlite3
from functools import lru_cache
from pathlib import Path

PROMPT_VERSION = "v1"
SAMPLE_ROWS = 3
MAX_CELL_CHARS = 60

INSTRUCTIONS = """\
You write SQLite queries that answer questions about a database.

Rules:
- Write exactly one SQLite SELECT statement.
- Use only tables and columns that exist in the schema below.
- Return exactly the columns the question asks for, in the order the question mentions them. \
Do not add extra columns (such as ids) unless the question asks for them.
- Use ORDER BY or LIMIT only when the question asks for an ordering or a top/bottom-N result.

Respond with a JSON object whose "sql" field holds the query."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}},
    "required": ["sql"],
    "additionalProperties": False,
}


@lru_cache(maxsize=64)
def render_schema(db_path: str | Path, sample_rows: int = SAMPLE_ROWS) -> str:
    """CREATE TABLE statements (with keys) plus a few example rows per table.

    Example rows show value formats the DDL doesn't (e.g. 'T'/'F' flags, country codes),
    which is the difference between a filter that matches and one that silently doesn't.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY rowid"
        ).fetchall()
        parts = []
        for name, ddl in tables:
            block = ddl.strip() + ";"
            if sample_rows:
                quoted = '"' + name.replace('"', '""') + '"'
                cur = conn.execute(f"SELECT * FROM {quoted} LIMIT {int(sample_rows)}")
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                if rows:
                    lines = ["\t".join(cols)] + ["\t".join(_cell(v) for v in r) for r in rows]
                    block += f"\n/* {len(rows)} example rows from {name}:\n" + "\n".join(lines) + "\n*/"
            parts.append(block)
        return "\n\n".join(parts)
    finally:
        conn.close()


def _cell(v) -> str:
    s = "NULL" if v is None else str(v)
    s = s.replace("\t", " ").replace("\n", " ")
    return s if len(s) <= MAX_CELL_CHARS else s[: MAX_CELL_CHARS - 3] + "..."


def build(db_path: str | Path, question: str) -> tuple[str, str]:
    """Returns (system, user)."""
    system = f"{INSTRUCTIONS}\n\nDatabase schema:\n\n{render_schema(str(db_path))}"
    return system, f"Question: {question}"


def prompt_fingerprint() -> str:
    """Hash of everything that shapes the prompt, recorded in each run manifest."""
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    h.update(INSTRUCTIONS.encode())
    h.update(repr(sorted(OUTPUT_SCHEMA.items())).encode())
    h.update(f"{SAMPLE_ROWS}|{MAX_CELL_CHARS}".encode())
    return h.hexdigest()[:12]
