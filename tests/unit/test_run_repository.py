import pytest

from querypilot.repositories.run_repository import RunRepository


@pytest.fixture
def repo(tmp_path) -> RunRepository:
    return RunRepository(str(tmp_path / "history.db"))


def test_create_and_get(repo):
    repo.create("r1", "How many customers?")
    record = repo.get("r1")
    assert record["question"] == "How many customers?"
    assert record["status"] == "running"
    assert record["visited_nodes"] == []
    assert record["created_at"]


def test_get_unknown_returns_none(repo):
    assert repo.get("missing") is None


def test_update_round_trips_json_fields(repo):
    repo.create("r1", "q")
    repo.update("r1", status="completed", visited_nodes=["a", "b"], tool_result={"rows": [[1]]})
    record = repo.get("r1")
    assert record["status"] == "completed"
    assert record["visited_nodes"] == ["a", "b"]
    assert record["tool_result"] == {"rows": [[1]]}


def test_update_rejects_unknown_fields(repo):
    repo.create("r1", "q")
    with pytest.raises(ValueError):
        repo.update("r1", id="hacked")


def test_claim_succeeds_once(repo):
    repo.create("r1", "q")
    repo.update("r1", status="waiting_for_approval")
    assert repo.claim("r1", "waiting_for_approval", "running") is True
    assert repo.claim("r1", "waiting_for_approval", "running") is False
