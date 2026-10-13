import pytest

from querypilot.tools.errors import ToolError
from querypilot.tools.query_validator import validate_sql
from querypilot.tools.schema_inspector import inspect_schema
from querypilot.tools.sql_executor import execute_sql
from tests.conftest import count_rows


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "SELECT 1 UNION SELECT 2",
    ],
)
def test_validator_allows_read_only_queries(sql):
    result = validate_sql(sql)
    assert result.ok and result.read_only


@pytest.mark.parametrize(
    "sql",
    ["DELETE FROM orders WHERE id = 1", "UPDATE orders SET status = 'paid'", "INSERT INTO customers VALUES (1, 'a', 'b')"],
)
def test_validator_flags_writes_for_approval(sql):
    result = validate_sql(sql)
    assert result.ok and not result.read_only


@pytest.mark.parametrize(
    "sql",
    ["DROP TABLE orders", "SELECT 1; SELECT 2", "", "   ", "SELEC * FROM orders", "PRAGMA table_info(orders)"],
)
def test_validator_rejects_dangerous_or_invalid_sql(sql):
    result = validate_sql(sql)
    assert not result.ok and result.error


def test_inspect_schema_lists_tables_and_counts(db_path):
    schema = inspect_schema(db_path)
    assert "CREATE TABLE customers" in schema
    assert "CREATE TABLE orders" in schema
    assert "30 rows" in schema and "300 rows" in schema


def test_inspect_schema_missing_db(tmp_path):
    with pytest.raises(ToolError):
        inspect_schema(str(tmp_path / "nope.db"))


def test_execute_select_returns_rows(db_path):
    result = execute_sql(db_path, "SELECT COUNT(*) AS n FROM customers")
    assert result["columns"] == ["n"]
    assert result["rows"] == [[30]]


def test_execute_truncates_large_results(db_path):
    result = execute_sql(db_path, "SELECT * FROM orders", max_rows=10)
    assert result["row_count"] == 10 and result["truncated"] is True


def test_execute_bad_sql_raises_tool_error(db_path):
    with pytest.raises(ToolError):
        execute_sql(db_path, "SELECT * FROM missing_table")


def test_read_only_connection_blocks_writes_without_approval(db_path):
    with pytest.raises(ToolError):
        execute_sql(db_path, "DELETE FROM orders")
    assert count_rows(db_path, "orders") == 300


def test_write_allowed_with_flag(db_path):
    result = execute_sql(db_path, "DELETE FROM orders WHERE status = 'cancelled'", allow_write=True)
    assert result["rows_affected"] > 0
    assert count_rows(db_path, "orders", "status = 'cancelled'") == 0
