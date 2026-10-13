from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from querypilot.api.app import create_app
from querypilot.graph import build_graph
from querypilot.llm import FakeLLM
from querypilot.nodes import Deps
from querypilot.repositories.run_repository import RunRepository
from querypilot.services.agent_service import AgentService
from tests.conftest import count_rows

NEEDS_DATA = '{"needs_data": true}'
DELETE_SQL = "DELETE FROM orders WHERE status = 'cancelled'"


class Clock:
    """Fake time: real now plus a manually advanced offset."""

    def __init__(self) -> None:
        self.offset = timedelta()

    def __call__(self) -> datetime:
        return datetime.now(UTC) + self.offset

    def advance(self, seconds: float) -> None:
        self.offset += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def make_client(db_path, tmp_path, clock):
    def factory(responses: list, timeout: float | None) -> TestClient:
        graph = build_graph(Deps(llm=FakeLLM(responses), db_path=db_path))
        service = AgentService(
            graph, RunRepository(str(tmp_path / "history.db")), approval_timeout=timeout, clock=clock
        )
        return TestClient(create_app(service))

    return factory


def start_pending_delete(client: TestClient) -> dict:
    body = client.post("/agent/run", json={"question": "Delete cancelled orders"}).json()
    assert body["status"] == "waiting_for_approval"
    return body


def test_pending_run_reports_its_deadline(make_client):
    client = make_client([NEEDS_DATA, DELETE_SQL], timeout=60)
    body = start_pending_delete(client)

    assert body["approval_expires_at"]
    status = client.get(f"/agent/status/{body['run_id']}").json()
    assert status["approval_expires_at"] == body["approval_expires_at"]


def test_run_expires_after_timeout_and_nothing_is_executed(make_client, clock, db_path):
    client = make_client([NEEDS_DATA, DELETE_SQL], timeout=60)
    before = count_rows(db_path, "orders")
    run_id = start_pending_delete(client)["run_id"]

    clock.advance(61)
    status = client.get(f"/agent/status/{run_id}").json()

    assert status["status"] == "expired"
    assert "timed out" in status["final_answer"]
    assert count_rows(db_path, "orders") == before


def test_approve_or_reject_after_expiry_returns_409(make_client, clock, db_path):
    client = make_client([NEEDS_DATA, DELETE_SQL], timeout=60)
    before = count_rows(db_path, "orders")
    run_id = start_pending_delete(client)["run_id"]

    clock.advance(61)
    approve = client.post(f"/agent/approve/{run_id}")

    assert approve.status_code == 409
    assert "expired" in approve.json()["detail"]
    assert client.post(f"/agent/reject/{run_id}").status_code == 409
    assert count_rows(db_path, "orders") == before


def test_approving_before_the_deadline_still_works(make_client, clock, db_path):
    client = make_client([NEEDS_DATA, DELETE_SQL, "Cancelled orders were deleted."], timeout=60)
    run_id = start_pending_delete(client)["run_id"]

    clock.advance(30)
    done = client.post(f"/agent/approve/{run_id}").json()

    assert done["status"] == "completed"
    assert count_rows(db_path, "orders", "status = 'cancelled'") == 0


def test_history_of_expired_run_keeps_the_pending_request(make_client, clock):
    client = make_client([NEEDS_DATA, DELETE_SQL], timeout=60)
    run_id = start_pending_delete(client)["run_id"]

    clock.advance(61)
    history = client.get(f"/agent/history/{run_id}").json()

    assert history["status"] == "expired"
    assert history["sql"] == DELETE_SQL


@pytest.mark.parametrize("timeout", [None, 0])
def test_no_timeout_means_waiting_forever(make_client, clock, timeout):
    client = make_client([NEEDS_DATA, DELETE_SQL], timeout=timeout)
    run_id = start_pending_delete(client)["run_id"]

    clock.advance(10**7)

    assert client.get(f"/agent/status/{run_id}").json()["status"] == "waiting_for_approval"
