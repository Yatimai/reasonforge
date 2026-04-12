"""Tests for src/sql_utils.py — SQL execution, parsing, and result matching."""

from pathlib import Path

from src.sql_utils import (
    check_result_match,
    execute_sql,
    normalize_sql,
    normalize_sql_result,
    parse_response,
)


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_sql_block_format(self):
        response = "Some reasoning here.\n```sql\nSELECT * FROM users\n```"
        _reasoning, sql = parse_response(response)
        assert sql == "SELECT * FROM users"

    def test_sql_block_case_insensitive(self):
        response = "```SQL\nSELECT name FROM users\n```"
        _reasoning, sql = parse_response(response)
        assert sql == "SELECT name FROM users"

    def test_sql_prefix_format(self):
        response = "Reasoning: count users.\nSQL: SELECT COUNT(*) FROM users\n\n"
        _reasoning, sql = parse_response(response)
        assert sql.startswith("SELECT COUNT(*)")
        assert "users" in sql

    def test_fallback_select_only(self):
        response = "SELECT name FROM users WHERE age > 18;"
        _reasoning, sql = parse_response(response)
        assert sql == "SELECT name FROM users WHERE age > 18"

    def test_fallback_strips_trailing_semicolon(self):
        response = "SELECT * FROM users;"
        _reasoning, sql = parse_response(response)
        assert not sql.endswith(";")

    def test_empty_response(self):
        _reasoning, sql = parse_response("")
        assert sql == ""

    def test_no_sql_in_response(self):
        _reasoning, sql = parse_response("This is just some text with no query.")
        assert sql == ""


# ---------------------------------------------------------------------------
# normalize_sql_result
# ---------------------------------------------------------------------------


class TestNormalizeSqlResult:
    def test_empty_results_returns_empty_set(self):
        assert normalize_sql_result([]) == set()

    def test_none_returns_empty_set(self):
        assert normalize_sql_result(None) == set()

    def test_single_row(self):
        result = normalize_sql_result([("Alice", 30)])
        assert len(result) == 1
        # Values in the row are sorted: ("30", "alice") after lower+sort
        row = next(iter(result))
        assert "alice" in row
        assert "30" in row

    def test_multiple_rows(self):
        result = normalize_sql_result([("Alice", 30), ("Bob", 25)])
        assert len(result) == 2

    def test_null_distinct_from_empty_string(self):
        with_null = normalize_sql_result([(None, 1)])
        with_empty = normalize_sql_result([("", 1)])
        assert with_null != with_empty

    def test_column_order_ignored(self):
        # Spider's official metric: column order should not matter
        # (because rows are sorted internally)
        a = normalize_sql_result([("Alice", 30)])
        b = normalize_sql_result([(30, "Alice")])
        assert a == b

    def test_case_insensitive(self):
        a = normalize_sql_result([("Alice",)])
        b = normalize_sql_result([("ALICE",)])
        assert a == b

    def test_strips_whitespace(self):
        a = normalize_sql_result([("Alice",)])
        b = normalize_sql_result([("  Alice  ",)])
        assert a == b


# ---------------------------------------------------------------------------
# check_result_match
# ---------------------------------------------------------------------------


class TestCheckResultMatch:
    def test_exact_match(self):
        pred = [("Alice", 30), ("Bob", 25)]
        gold = [("Alice", 30), ("Bob", 25)]
        assert check_result_match(pred, gold) is True

    def test_mismatch_different_values(self):
        pred = [("Alice", 30)]
        gold = [("Bob", 25)]
        assert check_result_match(pred, gold) is False

    def test_mismatch_different_row_count(self):
        pred = [("Alice", 30)]
        gold = [("Alice", 30), ("Bob", 25)]
        assert check_result_match(pred, gold) is False

    def test_row_order_ignored(self):
        # Results are sets, order doesn't matter
        pred = [("Alice", 30), ("Bob", 25)]
        gold = [("Bob", 25), ("Alice", 30)]
        assert check_result_match(pred, gold) is True

    def test_column_order_ignored(self):
        pred = [("Alice", 30)]
        gold = [(30, "Alice")]
        assert check_result_match(pred, gold) is True

    def test_both_empty_match(self):
        assert check_result_match([], []) is True

    def test_null_vs_empty_string_no_match(self):
        # Critical: NULL must be distinguishable from ''
        pred = [(None,)]
        gold = [("",)]
        assert check_result_match(pred, gold) is False


# ---------------------------------------------------------------------------
# normalize_sql
# ---------------------------------------------------------------------------


class TestNormalizeSql:
    def test_lowercases(self):
        assert normalize_sql("SELECT * FROM Users") == "select * from users"

    def test_collapses_whitespace(self):
        assert normalize_sql("SELECT   *\n\tFROM    users") == "select * from users"

    def test_strips(self):
        assert normalize_sql("   SELECT * FROM users   ") == "select * from users"


# ---------------------------------------------------------------------------
# execute_sql
# ---------------------------------------------------------------------------


class TestExecuteSql:
    def test_select_success(self, tmp_db: Path):
        success, results, error = execute_sql("SELECT name FROM users ORDER BY id", tmp_db)
        assert success is True
        assert error is None
        assert results is not None
        assert len(results) == 3
        names = [row[0] for row in results]
        assert names == ["Alice", "Bob", "Carol"]

    def test_select_with_where(self, tmp_db: Path):
        success, results, _error = execute_sql("SELECT name FROM users WHERE age > 25", tmp_db)
        assert success is True
        assert results == [("Alice",)]

    def test_count_query(self, tmp_db: Path):
        success, results, _error = execute_sql("SELECT COUNT(*) FROM users", tmp_db)
        assert success is True
        assert results == [(3,)]

    def test_null_value_returned(self, tmp_db: Path):
        success, results, _error = execute_sql("SELECT age FROM users WHERE id = 3", tmp_db)
        assert success is True
        assert results == [(None,)]

    def test_empty_sql_rejected(self, tmp_db: Path):
        success, results, error = execute_sql("", tmp_db)
        assert success is False
        assert results is None
        assert "Empty SQL" in error

    def test_whitespace_only_sql_rejected(self, tmp_db: Path):
        success, _results, error = execute_sql("   \n\t  ", tmp_db)
        assert success is False
        assert "Empty SQL" in error

    def test_syntax_error(self, tmp_db: Path):
        success, results, error = execute_sql("SELECT FROM WHERE", tmp_db)
        assert success is False
        assert results is None
        assert error is not None

    def test_unknown_table(self, tmp_db: Path):
        success, _results, error = execute_sql("SELECT * FROM nonexistent_table", tmp_db)
        assert success is False
        assert "no such table" in error.lower()
