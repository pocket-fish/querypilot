import sqlite3

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from querypilot.api.app import create_app
from querypilot.graph import build_graph
from querypilot.llm import FakeLLM
from querypilot.nodes import Deps
from querypilot.repositories.run_repository import RunRepository
from querypilot.services.agent_service import AgentService
from tests.conftest import count_rows

NEEDS_DATA = '{"needs_data": true}'
COUNT_SQL = "SELECT COUNT(*) FROM customers"
DELETE_SQL = "DELETE FROM orders WHERE status = 'cancelled'"


@pytest.fixture
def make_client(db_path, tmp_path):
    def factory(responses: list) -> TestClient:
        graph = build_graph(Deps(llm=FakeLLM(responses), db_path=db_path))
        service = AgentService(graph, RunRepository(str(tmp_path / "history.db")))
        return TestClient(create_app(service))

    return factory


def test_run_completes_and_history_records_everything(make_client):
    client = make_client([NEEDS_DATA, COUNT_SQL, "There are 30 customers."])

    created = client.post("/agent/run", json={"question": "How many customers?"})
    assert created.status_code == 200
    body = created.json()
    assert body["status"] == "completed"
    assert body["final_answer"] == "There are 30 customers."
    run_id = body["run_id"]

    status = client.get(f"/agent/status/{run_id}").json()
    assert status["status"] == "completed" and status["current_step"] == "finalize"

    history = client.get(f"/agent/history/{run_id}").json()
    assert history["question"] == "How many customers?"
    assert history["visited_nodes"][0] == "analyze_request"
    assert history["visited_nodes"][-1] == "finalize"
    assert history["tools_used"] == ["schema_inspector", "sql_executor"]
    assert history["sql"] == COUNT_SQL
    assert history["tool_result"]["rows"] == [[30]]
    assert history["iterations"] == 0
    assert history["created_at"] and history["updated_at"]


def test_write_waits_for_approval_then_approve_runs_it(make_client, db_path):
    client = make_client([NEEDS_DATA, DELETE_SQL, "Cancelled orders were deleted."])

    waiting = client.post("/agent/run", json={"question": "Delete cancelled orders"}).json()
    assert waiting["status"] == "waiting_for_approval"
    assert waiting["pending_approval"]["sql"] == DELETE_SQL
    assert waiting["final_answer"] is None
    assert count_rows(db_path, "orders", "status = 'cancelled'") > 0

    done = client.post(f"/agent/approve/{waiting['run_id']}").json()
    assert done["status"] == "completed"
    assert count_rows(db_path, "orders", "status = 'cancelled'") == 0


def test_reject_changes_nothing(make_client, db_path):
    client = make_client([NEEDS_DATA, DELETE_SQL])
    before = count_rows(db_path, "orders")
    run_id = client.post("/agent/run", json={"question": "Delete cancelled orders"}).json()["run_id"]

    done = client.post(f"/agent/reject/{run_id}").json()
    assert done["status"] == "rejected"
    assert count_rows(db_path, "orders") == before


def test_cannot_decide_twice_or_on_a_finished_run(make_client):
    client = make_client([NEEDS_DATA, DELETE_SQL, "Done."])
    run_id = client.post("/agent/run", json={"question": "Delete cancelled orders"}).json()["run_id"]
    assert client.post(f"/agent/approve/{run_id}").status_code == 200

    assert client.post(f"/agent/approve/{run_id}").status_code == 409
    assert client.post(f"/agent/reject/{run_id}").status_code == 409


def test_unknown_run_returns_404(make_client):
    client = make_client([])
    for path in ("status", "history"):
        assert client.get(f"/agent/{path}/nope").status_code == 404
    for path in ("approve", "reject"):
        assert client.post(f"/agent/{path}/nope").status_code == 404


def test_empty_question_is_rejected_by_validation(make_client):
    client = make_client([])
    assert client.post("/agent/run", json={"question": ""}).status_code == 422


def test_llm_failure_is_reported_as_failed_run_not_http_error(make_client):
    client = make_client([])  # FakeLLM raises LLMError immediately
    response = client.post("/agent/run", json={"question": "How many customers?"})

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_pending_approval_survives_a_server_restart(db_path, tmp_path):
    checkpoints = str(tmp_path / "checkpoints.db")
    history = str(tmp_path / "history.db")

    def boot(responses: list) -> TestClient:
        saver = SqliteSaver(sqlite3.connect(checkpoints, check_same_thread=False))
        graph = build_graph(Deps(llm=FakeLLM(responses), db_path=db_path), saver)
        return TestClient(create_app(AgentService(graph, RunRepository(history))))

    first = boot([NEEDS_DATA, DELETE_SQL])
    run_id = first.post("/agent/run", json={"question": "Delete cancelled orders"}).json()["run_id"]

    second = boot(["Cancelled orders were deleted."])  # new process, fresh graph and service
    assert second.get(f"/agent/status/{run_id}").json()["status"] == "waiting_for_approval"
    assert second.post(f"/agent/approve/{run_id}").json()["status"] == "completed"
    assert count_rows(db_path, "orders", "status = 'cancelled'") == 0
