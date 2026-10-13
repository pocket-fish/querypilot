import pytest

from querypilot.tools.schema_inspector import read_schema_info
from querypilot.tools.sql_linter import lint_sql


@pytest.fixture
def schema(db_path):
    return read_schema_info(db_path)


def test_schema_info_reads_keys_from_the_database(schema):
    assert schema.primary_keys["customers"] == {"id"}
    assert ("orders", "customer_id", "customers", "id") in schema.foreign_keys


# --- queries a model legitimately writes: the linter must stay silent -------------------

GOOD_QUERIES = [
    "SELECT COUNT(*) FROM customers",
    "SELECT SUM(amount) FROM orders WHERE status = 'paid'",
    "SELECT AVG(amount) FROM orders WHERE status = 'paid'",
    "SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC LIMIT 3",
    "SELECT country, COUNT(*) FROM customers GROUP BY 1 ORDER BY 2 DESC LIMIT 3",
    # name/country depend on the grouped primary key c.id
    "SELECT c.id, c.name, c.country, SUM(o.amount) AS ltv FROM customers c "
    "JOIN orders o ON c.id = o.customer_id WHERE o.status = 'paid' GROUP BY c.id",
    "SELECT c.country, AVG(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id "
    "GROUP BY c.country",
    "SELECT substr(created_at, 1, 7) AS month, SUM(amount) FROM orders "
    "GROUP BY substr(created_at, 1, 7)",
    # GROUP BY an alias of a select expression (the real query that was falsely rejected)
    "SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS n FROM orders "
    "GROUP BY month ORDER BY month",
    "SELECT country, COUNT(*) FROM customers WHERE country IN ('Germany', 'France') "
    "GROUP BY country",
    "SELECT * FROM orders WHERE status <> 'cancelled' ORDER BY amount DESC LIMIT 5",
    "SELECT * FROM orders WHERE amount IS NOT NULL ORDER BY amount DESC LIMIT 5",
    "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id "
    "WHERE o.id IS NULL",
    "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders GROUP BY customer_id "
    "HAVING SUM(amount) > 1000)",
    "SELECT customer_id, SUM(amount) OVER (PARTITION BY customer_id) FROM orders",
]


@pytest.mark.parametrize("sql", GOOD_QUERIES)
def test_correct_queries_produce_no_findings(schema, sql):
    assert lint_sql(sql, schema) == []


# --- one broken query per rule ---------------------------------------------------------


def test_join_on_the_wrong_key_is_reported_with_the_right_one(schema):
    sql = "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id = o.id GROUP BY c.id"
    [issue] = lint_sql(sql, schema)
    assert "c.id = o.id" in issue
    assert "orders.customer_id = customers.id" in issue


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM customers, orders",
        "SELECT * FROM customers CROSS JOIN orders",
    ],
)
def test_cartesian_product_is_reported(schema, sql):
    [issue] = lint_sql(sql, schema)
    assert "cartesian" in issue


def test_bare_column_next_to_aggregate_without_group_by(schema):
    [issue] = lint_sql("SELECT country, COUNT(*) FROM customers", schema)
    assert "country" in issue and "arbitrary row" in issue


def test_bare_column_missing_from_group_by(schema):
    sql = (
        "SELECT c.id, o.created_at, SUM(o.amount) FROM customers c "
        "JOIN orders o ON c.id = o.customer_id GROUP BY c.id"
    )
    [issue] = lint_sql(sql, schema)
    assert "o.created_at" in issue


def test_schema_info_discovers_enum_like_columns(schema):
    assert schema.enum_values[("orders", "status")] == {"paid", "pending", "cancelled"}
    assert ("customers", "name") not in schema.enum_values  # too many distinct values
    assert ("orders", "created_at") not in schema.enum_values


def test_invented_status_value_is_reported_with_the_valid_ones(schema):
    [issue] = lint_sql("SELECT SUM(amount) FROM orders WHERE status = 'completed'", schema)
    assert "'completed' is not a value of orders.status" in issue
    assert "cancelled, paid, pending" in issue


def test_invented_value_inside_in_list_and_through_alias(schema):
    sql = (
        "SELECT SUM(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id "
        "WHERE o.status IN ('paid', 'delivered')"
    )
    [issue] = lint_sql(sql, schema)
    assert "'delivered'" in issue


def test_enum_values_are_case_sensitive_like_sqlite(schema):
    [issue] = lint_sql("SELECT COUNT(*) FROM customers WHERE country = 'germany'", schema)
    assert "'germany'" in issue and "Germany" in issue


def test_comparison_with_null_is_reported(schema):
    [issue] = lint_sql("SELECT * FROM orders WHERE status = NULL", schema)
    assert "IS NULL" in issue


def test_limit_without_order_by_is_reported(schema):
    [issue] = lint_sql("SELECT * FROM orders LIMIT 5", schema)
    assert "ORDER BY" in issue


def test_several_problems_are_all_listed(schema):
    sql = "SELECT * FROM customers c JOIN orders o ON c.id = o.id LIMIT 3"
    assert len(lint_sql(sql, schema)) == 2


def test_unparseable_sql_is_left_to_the_validator(schema):
    assert lint_sql("SELEC nonsense", schema) == []
