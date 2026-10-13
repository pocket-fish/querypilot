from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from querypilot import routers
from querypilot.llm import FakeLLM
from querypilot.nodes import Deps, build_nodes
from querypilot.state import AgentState


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None):
    nodes = build_nodes(deps)
    # ty does not yet accept any TypedDict as LangGraph's `TypedDictLike` state bound
    # (reproducible with a two-line TypedDict); this is the documented LangGraph usage.
    graph = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]
    for name, fn in nodes.items():
        graph.add_node(name, fn)  # ty: ignore[invalid-argument-type]

    graph.add_edge(START, "analyze_request")
    graph.add_conditional_edges(
        "analyze_request",
        routers.route_after_analyze,
        ["tool_selection", "generate_answer", "finalize"],
    )
    graph.add_conditional_edges(
        "tool_selection",
        routers.route_after_selection,
        ["tool_selection", "human_approval", "tool_execution", "generate_answer", "finalize"],
    )
    graph.add_conditional_edges(
        "human_approval", routers.route_after_approval, ["tool_execution", "finalize"]
    )
    graph.add_conditional_edges(
        "tool_execution",
        routers.route_after_execution,
        ["tool_selection", "generate_answer", "finalize"],
    )
    graph.add_conditional_edges(
        "generate_answer", routers.route_after_generation, ["validate_answer", "finalize"]
    )
    graph.add_conditional_edges(
        "validate_answer",
        routers.route_after_validation,
        ["improve_answer", "finalize"],
    )
    graph.add_edge("improve_answer", "validate_answer")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def render_mermaid() -> str:
    """Mermaid diagram of the graph; no node runs, so the dependencies are placeholders."""
    return build_graph(Deps(llm=FakeLLM([]), db_path="")).get_graph().draw_mermaid()
