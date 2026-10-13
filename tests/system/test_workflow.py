from querypilot.llm import LLMError
from tests.conftest import count_rows

NEEDS_DATA = '{"needs_data": true}'
COUNT_SQL = "SELECT COUNT(*) FROM customers"
DELETE_SQL = "DELETE FROM orders WHERE status = 'cancelled'"


def test_happy_path_visits_nodes_in_order(make_harness):
    h = make_harness([NEEDS_DATA, COUNT_SQL, "There are 30 customers."])
    result = h.run("How many customers do we have?")

    assert result["status"] == "completed"
    assert result["final_answer"] == "There are 30 customers."
    assert result["visited_nodes"] == [
        "analyze_request",
        "tool_selection",
        "tool_execution",
        "tool_selection",
        "tool_execution",
        "generate_answer",
        "validate_answer",
        "finalize",
    ]
    assert result["tools_used"] == ["schema_inspector", "sql_executor"]
    assert result["tool_result"]["rows"] == [[30]]
    assert result["iterations"] == 0


def test_question_without_data_skips_tools(make_harness):
    h = make_harness(['{"needs_data": false}', "Hello! Ask me about orders."])
    result = h.run("Hi there")

    assert result["status"] == "completed"
    assert "tool_execution" not in result["visited_nodes"]
    assert result["tools_used"] == []


def test_failed_sql_is_self_corrected(make_harness):
    h = make_harness(
        [NEEDS_DATA, "SELECT COUNT(*) FROM clients", COUNT_SQL, "There are 30 customers."]
    )
    result = h.run("How many customers?")

    assert result["status"] == "completed"
    assert result["iterations"] == 1
    assert result["sql"] == COUNT_SQL
    assert "no such table" in h.llm.prompts[2]  # error was fed back to the model


def test_silently_wrong_all_null_result_triggers_rewrite(make_harness):
    bad = "SELECT id, NULL AS ltv FROM customers"
    good = "SELECT COUNT(*) FROM customers"
    h = make_harness([NEEDS_DATA, bad, good, "There are 30 customers."])
    result = h.run("Compute LTV")

    assert result["status"] == "completed"
    assert result["iterations"] == 1
    assert result["sql"] == good
    assert "NULL in every row" in h.llm.prompts[2]  # the warning reached the model


def test_all_null_result_that_never_gets_fixed_ends_as_max_iterations(make_harness):
    bad = "SELECT NULL AS ltv FROM customers"
    h = make_harness([NEEDS_DATA, bad, bad, "LTV could not be computed."], max_iterations=2)
    result = h.run("Compute LTV")

    assert result["status"] == "max_iterations"
    assert "unreliable" in h.llm.prompts[3]


def test_dangerous_sql_is_rejected_by_validator_and_retried(make_harness):
    h = make_harness([NEEDS_DATA, "DROP TABLE orders", COUNT_SQL, "There are 30 customers."])
    result = h.run("How many customers?")

    assert result["status"] == "completed"
    assert result["iterations"] == 1
    assert count_rows(h.db_path, "orders") == 300


def test_join_on_wrong_key_is_caught_before_execution_and_fixed(make_harness):
    wrong = (
        "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id = o.id "
        "WHERE o.status = 'paid' GROUP BY c.id"
    )
    right = wrong.replace("o.id", "o.customer_id")
    top = right + " ORDER BY SUM(o.amount) DESC LIMIT 1"
    h = make_harness([NEEDS_DATA, wrong, top, "Customer 02 spent 2955.05."])
    result = h.run("Who spent the most?")

    assert result["status"] == "completed"
    assert result["iterations"] == 1
    assert result["sql"] == top
    # the wrong query never ran: only the schema tool and the corrected query did
    assert result["tools_used"] == ["schema_inspector", "sql_executor"]
    retry_prompt = h.llm.prompts[2]
    assert "does not follow the foreign key" in retry_prompt
    assert "orders.customer_id = customers.id" in retry_prompt


def test_invented_filter_value_is_caught_and_the_model_sees_valid_values(make_harness):
    invented = "SELECT SUM(amount) FROM orders WHERE status = 'completed'"
    fixed = "SELECT SUM(amount) FROM orders WHERE status = 'paid'"
    h = make_harness([NEEDS_DATA, invented, fixed, "Paid revenue is 47926.08."])
    result = h.run("What is the revenue?")

    assert result["status"] == "completed"
    assert result["sql"] == fixed
    assert "orders.status values: 'cancelled', 'paid', 'pending'" in h.llm.prompts[1]  # in schema
    assert "valid values: cancelled, paid, pending" in h.llm.prompts[2]  # in the retry hint


def test_failure_answer_is_deterministic_and_localized_not_improvised(make_harness):
    bad = "SELECT * FROM customers, orders"
    h = make_harness([NEEDS_DATA, bad, bad], max_iterations=2)
    result = h.run("Сколько клиентов?")

    assert result["status"] == "max_iterations"
    assert result["final_answer"].startswith("Не удалось построить надёжный запрос")
    assert len(h.llm.prompts) == 3  # analyze + two SQL attempts, no LLM call for the answer


def test_cartesian_product_never_reaches_the_database(make_harness):
    bad = "SELECT COUNT(*) FROM customers, orders"
    h = make_harness([NEEDS_DATA, bad, bad, "I could not build a reliable query."], max_iterations=2)
    result = h.run("How many customers?")

    assert result["status"] == "max_iterations"
    assert result["tools_used"] == ["schema_inspector"]
    assert result["tool_result"] is None


def test_hallucinated_number_triggers_improve_cycle(make_harness):
    h = make_harness(
        [NEEDS_DATA, COUNT_SQL, "There are 99 customers.", "There are 30 customers."]
    )
    result = h.run("How many customers?")

    assert result["status"] == "completed"
    assert result["final_answer"] == "There are 30 customers."
    assert result["iterations"] == 1
    assert result["visited_nodes"].count("validate_answer") == 2
    assert "improve_answer" in result["visited_nodes"]


def test_answer_loop_stops_at_max_iterations(make_harness):
    bad = "There are 99 customers."
    h = make_harness([NEEDS_DATA, COUNT_SQL, bad, bad, bad], max_iterations=2)
    result = h.run("How many customers?")

    assert result["status"] == "max_iterations"
    assert result["iterations"] == 2
    assert "iteration limit" in result["final_answer"]
    assert result["visited_nodes"][-1] == "finalize"


def test_tool_error_loop_stops_at_max_iterations(make_harness):
    broken = "SELECT * FROM nowhere"
    h = make_harness([NEEDS_DATA, broken, broken, "I could not query the data."], max_iterations=2)
    result = h.run("Anything?")

    assert result["status"] == "max_iterations"
    assert result["iterations"] == 2
    assert result["tool_result"] is None
    assert result["error"]


def test_write_pauses_for_approval_then_runs_when_approved(make_harness):
    h = make_harness([NEEDS_DATA, DELETE_SQL, "Cancelled orders were deleted."])
    first = h.run("Delete all cancelled orders")

    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["sql"] == DELETE_SQL
    assert count_rows(h.db_path, "orders", "status = 'cancelled'") > 0  # nothing happened yet

    final = h.resume("approve")
    assert final["status"] == "completed"
    assert final["approval"] == "approved"
    assert count_rows(h.db_path, "orders", "status = 'cancelled'") == 0


def test_rejected_write_changes_nothing(make_harness):
    h = make_harness([NEEDS_DATA, DELETE_SQL])
    before = count_rows(h.db_path, "orders")
    h.run("Delete all cancelled orders")

    final = h.resume("reject")
    assert final["status"] == "rejected"
    assert "rejected" in final["final_answer"]
    assert final["visited_nodes"] == [
        "analyze_request",
        "tool_selection",
        "tool_execution",
        "tool_selection",
        "human_approval",
        "finalize",
    ]
    assert count_rows(h.db_path, "orders") == before


def test_llm_failure_ends_in_controlled_failed_state(make_harness):
    h = make_harness([LLMError("ollama is down")])
    result = h.run("How many customers?")

    assert result["status"] == "failed"
    assert "ollama is down" in result["final_answer"]
    assert result["visited_nodes"] == ["analyze_request", "finalize"]
