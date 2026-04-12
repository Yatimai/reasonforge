"""Shared pytest fixtures and path setup for ReasonForge tests."""

import sqlite3
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `from src.X import Y` works in tests.
# The project is not installed in editable mode (no pyproject.toml).
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Create a small SQLite database for SQL execution tests.

    Schema:
        users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER)

    Rows:
        (1, 'Alice', 30)
        (2, 'Bob', 25)
        (3, 'Carol', NULL)
    """
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?)",
        [(1, "Alice", 30), (2, "Bob", 25), (3, "Carol", None)],
    )
    conn.commit()
    conn.close()
    return db_path
