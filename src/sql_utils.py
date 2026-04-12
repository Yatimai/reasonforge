#!/usr/bin/env python3
"""
Centralized SQL utilities for STaR-SQL.

This module provides consistent SQL execution, parsing, and evaluation
across all files (training, evaluation, inference).

Usage:
    from sql_utils import execute_sql, parse_response, check_result_match
"""

import re
import sqlite3
import time
from pathlib import Path
from typing import List, Tuple, Optional


# =============================================================================
# SQL EXECUTION
# =============================================================================

# Maximum VM operations before interrupting (prevents runaway cartesian products)
MAX_SQL_OPERATIONS = 1_000_000

# Maximum result rows to prevent memory issues
MAX_RESULT_ROWS = 50_000


class SQLTimeout(Exception):
    """Raised when SQL query exceeds operation limit."""
    pass


def execute_sql(sql: str, db_path: Path, timeout: int = 5, max_ops: int = MAX_SQL_OPERATIONS) -> Tuple[bool, Optional[List], Optional[str]]:
    """
    Execute SQL query and return results with proper timeout.

    Uses sqlite3.set_progress_handler() to actually interrupt queries,
    unlike ThreadPoolExecutor which only stops waiting but leaves threads running.

    Args:
        sql: SQL query to execute
        db_path: Path to SQLite database
        timeout: Timeout in seconds (default: 5)
        max_ops: Max SQLite VM operations before interrupt (default: 1M)

    Returns:
        Tuple of (success, results, error_message)
    """
    # CRITICAL: Reject empty SQL - prevents false positives when gold returns empty results
    if not sql or not sql.strip():
        return False, None, "Empty SQL query"

    start_time = time.time()
    conn = None

    def progress_callback():
        """Called every N SQLite VM operations. Raises to interrupt."""
        if time.time() - start_time > timeout:
            raise SQLTimeout(f"Query timed out after {timeout}s")
        return 0  # Return non-zero to abort

    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.text_factory = str
        # Set progress handler to check every 10K operations
        conn.set_progress_handler(progress_callback, 10000)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchmany(MAX_RESULT_ROWS)

        # Check if there are more results (truncated)
        if cursor.fetchone() is not None:
            return True, results, f"Results truncated to {MAX_RESULT_ROWS} rows"

        return True, results, None

    except SQLTimeout as e:
        return False, None, str(e)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            return False, None, f"Query interrupted after {time.time() - start_time:.1f}s"
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_response(response: str) -> Tuple[str, str]:
    """
    Parse model response to extract SQL (and optionally reasoning).

    Supports multiple formats:
    - CV prompt (raw SQL): SELECT name FROM users
    - SQL block: ```sql SELECT ... ```
    - With prefix: SQL: SELECT ...

    Args:
        response: Raw model response

    Returns:
        Tuple of (reasoning, sql) - reasoning is empty for CV prompt
    """
    reasoning = ""
    sql = ""

    # Try to find SQL block (format fine-tuned)
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # Try with "SQL:" prefix
        sql_match = re.search(r'SQL:\s*(SELECT.*?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1).strip()

    # Fallback: chercher n'importe quel SELECT/INSERT/UPDATE/DELETE (pour baseline)
    if not sql:
        sql_match = re.search(r'(SELECT\s+.+?)(?:;|```|\n\n|$)', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1).strip()
            # Nettoyer le point-virgule final si present
            if sql.endswith(';'):
                sql = sql[:-1].strip()

    # Extract reasoning (everything before SQL:)
    sql_start = response.lower().find('sql:')
    if sql_start > 0:
        reasoning = response[:sql_start].strip()
    else:
        reasoning = response.split('```')[0].strip() if '```' in response else response

    # Clean reasoning
    if reasoning.lower().startswith('reasoning:'):
        reasoning = reasoning[10:].strip()

    return reasoning, sql


# =============================================================================
# RESULT MATCHING (Spider official metric)
# =============================================================================

def normalize_sql_result(results: List) -> set:
    """
    Normalize SQL results for comparison.

    Converts all values to lowercase strings and creates a set of tuples.
    Sorts values within each row to ignore column order differences.
    """
    if results is None:
        return set()

    normalized = set()
    for row in results:
        # Convert row to tuple of strings, sorted to ignore column order
        # This fixes false negatives when SELECT order differs from gold
        # Use '<NULL>' sentinel to distinguish NULL from empty string ''
        row_values = [str(v).lower().strip() if v is not None else '<NULL>' for v in row]
        normalized.add(tuple(sorted(row_values)))

    return normalized


def check_result_match(predicted_results: List, gold_results: List) -> bool:
    """
    Check if predicted results match gold results.

    This is the Spider official metric (Execution Accuracy).

    Args:
        predicted_results: Results from predicted SQL
        gold_results: Results from gold SQL

    Returns:
        True if results match, False otherwise
    """
    pred_normalized = normalize_sql_result(predicted_results)
    gold_normalized = normalize_sql_result(gold_results)
    return pred_normalized == gold_normalized


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for string comparison.

    Used for exact match metric (not recommended, use result match instead).
    """
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.strip()
    return sql


# =============================================================================
# DATA LOADING
# =============================================================================

def get_cell_values(db_path: Path, table: str, column: str, limit: int = 3) -> List[str]:
    """Get sample distinct values from a column (CV prompt)."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL LIMIT {limit}")
        values = [str(row[0])[:20] for row in cursor.fetchall()]
        conn.close()
        return values
    except:
        return []


def load_spider_data(spider_dir: Path, with_cell_values: bool = True) -> Tuple[List, List, dict]:
    """
    Load Spider train and dev data.

    Args:
        spider_dir: Path to Spider data directory
        with_cell_values: If True, include sample cell values in schema (CV prompt)

    Returns:
        Tuple of (train_data, dev_data, schemas)
    """
    import json

    train_path = spider_dir / "train_spider.json"
    dev_path = spider_dir / "dev.json"
    tables_path = spider_dir / "tables.json"

    with open(train_path) as f:
        train_data = json.load(f)

    with open(dev_path) as f:
        dev_data = json.load(f)

    with open(tables_path) as f:
        tables_data = json.load(f)

    # Build schema dict
    schemas = {}
    for db in tables_data:
        db_id = db['db_id']
        db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        schema_parts = []

        for i, table in enumerate(db['table_names_original']):
            columns = [
                db['column_names_original'][j][1]
                for j in range(len(db['column_names_original']))
                if db['column_names_original'][j][0] == i
            ]
            table_def = f"CREATE TABLE {table} ({', '.join(columns)})"

            # CV prompt: Add sample cell values
            if with_cell_values and db_path.exists():
                value_parts = []
                for col in columns[:4]:  # First 4 columns
                    if '_id' in col.lower() or col.lower() == 'id':
                        continue
                    values = get_cell_values(db_path, table, col)
                    if values:
                        value_parts.append(f"{col}: {', '.join(values)}")
                if value_parts:
                    table_def += f"\n-- Sample values: {'; '.join(value_parts[:2])}"

            schema_parts.append(table_def)

        schemas[db_id] = '\n\n'.join(schema_parts)

    return train_data, dev_data, schemas
