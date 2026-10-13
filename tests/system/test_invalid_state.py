"""The graph must end in a controlled `failed` status, never crash, on a corrupt state."""

import pytest

from querypilot.graph import build_graph
from querypilot.llm import FakeLLM
from querypilot.nodes import Deps, build_nodes
from querypilot.state import initial_state


@pytest.fixture
def nodes(db_path):
    return build_nodes(Deps(llm=FakeLLM([]), db_path=db_path))


@pytest.mark.parametrize("node", ["tool_execution", "human_approval"])
def test_node_without_tool_call_fails_in_a_controlled_way(nodes, node):
    update = nodes[node](initial_state("q"))

    assert update["status"] == "failed"
    assert "without a tool_call" in update["error"]
    assert update["visited_nodes"] == [node]


def test_unknown_tool_is_rejected_instead_of_executed(nodes):
    state = initial_state("q")
    state["tool_call"] = {"name": "rm_rf", "args": {}, "requires_approval": False}

    update = nodes["tool_execution"](state)

    assert update["status"] == "failed"
    assert "unknown tool 'rm_rf'" in update["error"]


def test_graph_without_a_question_ends_failed_and_reaches_finalize(db_path):
    graph = build_graph(Deps(llm=FakeLLM([]), db_path=db_path))

    result = graph.invoke({"iterations": 0}, {"configurable": {"thread_id": "bad"}})

    assert result["status"] == "failed"
    assert result["visited_nodes"] == ["analyze_request", "finalize"]
    assert "could not finish" in result["final_answer"]


def test_failed_state_is_routed_to_finalize_from_every_router():
    from querypilot import routers

    failed = {**initial_state("q"), "status": "failed"}
    for route in (
        routers.route_after_analyze,
        routers.route_after_selection,
        routers.route_after_approval,
        routers.route_after_execution,
        routers.route_after_generation,
        routers.route_after_validation,
    ):
        assert route(failed) == "finalize", route.__name__
