import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Status = Literal["running", "completed", "rejected", "failed", "max_iterations"]


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    requires_approval: bool


class AgentState(TypedDict, total=False):
    # Required by the assignment
    messages: Annotated[list[AnyMessage], add_messages]
    user_question: str
    current_step: str
    tool_result: dict[str, Any] | None
    final_answer: str | None
    iterations: int

    # Extensions
    max_iterations: int
    needs_data: bool
    schema_text: str | None
    sql: str | None
    tool_call: ToolCall | None
    error: str | None
    approval: Literal["approved", "rejected"] | None
    feedback: str | None
    status: Status
    visited_nodes: Annotated[list[str], operator.add]
    tools_used: Annotated[list[str], operator.add]


def initial_state(question: str, max_iterations: int = 3) -> AgentState:
    return AgentState(
        messages=[],
        user_question=question,
        current_step="start",
        tool_result=None,
        final_answer=None,
        iterations=0,
        max_iterations=max_iterations,
        needs_data=True,
        schema_text=None,
        sql=None,
        tool_call=None,
        error=None,
        approval=None,
        feedback=None,
        status="running",
        visited_nodes=[],
        tools_used=[],
    )
