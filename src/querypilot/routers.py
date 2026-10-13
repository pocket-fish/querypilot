"""Pure routing functions for conditional edges (no LLM, no I/O)."""

from querypilot.state import AgentState


def _exhausted(state: AgentState) -> bool:
    return state.get("iterations", 0) >= state.get("max_iterations", 3)


def route_after_analyze(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "finalize"
    return "tool_selection" if state.get("needs_data", True) else "generate_answer"


def route_after_selection(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "finalize"
    call = state.get("tool_call")
    if call is None:  # validator rejected the SQL: retry or give up
        return "generate_answer" if _exhausted(state) else "tool_selection"
    if call["requires_approval"] and state.get("approval") != "approved":
        return "human_approval"
    return "tool_execution"


def route_after_approval(state: AgentState) -> str:
    if state.get("status") == "failed" or state.get("approval") == "rejected":
        return "finalize"
    return "tool_execution"


def route_after_execution(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "finalize"
    call = state.get("tool_call")
    if call and call["name"] == "schema_inspector" and not state.get("error"):
        return "tool_selection"
    if state.get("error"):
        return "generate_answer" if _exhausted(state) else "tool_selection"
    return "generate_answer"


def route_after_generation(state: AgentState) -> str:
    return "finalize" if state.get("status") == "failed" else "validate_answer"


def route_after_validation(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "finalize"
    if state.get("feedback") and not _exhausted(state):
        return "improve_answer"
    return "finalize"
