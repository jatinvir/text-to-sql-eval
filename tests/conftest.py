import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def make_db(tmp_path):
    """Create a throwaway SQLite database from a SQL script; returns its path."""

    def _make(script: str, name: str = "t.sqlite") -> Path:
        path = tmp_path / name
        conn = sqlite3.connect(path)
        conn.executescript(script)
        conn.commit()
        conn.close()
        return path

    return _make


@pytest.fixture
def people_db(make_db):
    return make_db(
        """
        CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, age INT, city TEXT, score REAL);
        INSERT INTO person (name, age, city, score) VALUES
            ('ann', 30, 'paris', 1.0),
            ('bob', 25, 'rome',  2.5),
            ('cat', 30, 'paris', 3.333333333),
            ('dan', 40, 'oslo',  4.0),
            ('eve', 25, 'rome',  5.0);
        """
    )
