from querypilot import routers
from querypilot.services.answer_checker import check_answer
from querypilot.state import initial_state

RESULT = {"columns": ["n"], "rows": [[30]], "row_count": 1, "truncated": False}


def state(**overrides):
    return {**initial_state("q", max_iterations=3), **overrides}


def test_analyze_routes_by_needs_data():
    assert routers.route_after_analyze(state(needs_data=True)) == "tool_selection"
    assert routers.route_after_analyze(state(needs_data=False)) == "generate_answer"
    assert routers.route_after_analyze(state(status="failed")) == "finalize"


def test_selection_routes_read_only_call_to_execution():
    call = {"name": "sql_executor", "args": {}, "requires_approval": False}
    assert routers.route_after_selection(state(tool_call=call)) == "tool_execution"


def test_selection_routes_write_call_to_approval_until_approved():
    call = {"name": "sql_executor", "args": {}, "requires_approval": True}
    assert routers.route_after_selection(state(tool_call=call)) == "human_approval"
    assert routers.route_after_selection(state(tool_call=call, approval="approved")) == "tool_execution"


def test_selection_without_call_retries_until_limit():
    assert routers.route_after_selection(state(tool_call=None, iterations=1)) == "tool_selection"
    assert routers.route_after_selection(state(tool_call=None, iterations=3)) == "generate_answer"


def test_approval_rejection_finalizes():
    assert routers.route_after_approval(state(approval="rejected")) == "finalize"
    assert routers.route_after_approval(state(approval="approved")) == "tool_execution"


def test_execution_loops_back_on_error_until_limit():
    call = {"name": "sql_executor", "args": {}, "requires_approval": False}
    assert routers.route_after_execution(state(tool_call=call, error="boom", iterations=1)) == "tool_selection"
    assert routers.route_after_execution(state(tool_call=call, error="boom", iterations=3)) == "generate_answer"
    assert routers.route_after_execution(state(tool_call=call)) == "generate_answer"


def test_execution_after_schema_goes_back_to_selection():
    call = {"name": "schema_inspector", "args": {}, "requires_approval": False}
    assert routers.route_after_execution(state(tool_call=call)) == "tool_selection"


def test_validation_cycle_respects_iteration_limit():
    assert routers.route_after_validation(state(feedback="bad", iterations=2)) == "improve_answer"
    assert routers.route_after_validation(state(feedback="bad", iterations=3)) == "finalize"
    assert routers.route_after_validation(state(feedback=None)) == "finalize"


def test_checker_accepts_numbers_from_result():
    assert check_answer("There are 30 customers.", "How many customers?", RESULT) is None


def test_checker_flags_invented_numbers():
    feedback = check_answer("There are 99 customers.", "How many customers?", RESULT)
    assert feedback and "99" in feedback


def test_checker_names_the_closest_real_value_for_truncated_numbers():
    result = {"rows": [[27, 1149.58]], "row_count": 1}
    feedback = check_answer("Customer 27 has 1149", "LTV?", result)
    assert "1149.58" in feedback


def test_checker_allows_numbers_from_question():
    result = {"rows": [["a", 1]], "row_count": 1}
    assert check_answer("Top 5 requested, found a with 1", "Show top 5", result) is None


def test_checker_flags_empty_answer():
    assert check_answer("  ", "q", RESULT)
