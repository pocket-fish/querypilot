import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.types import Command

from querypilot.repositories.run_repository import RunRepository
from querypilot.state import initial_state

logger = logging.getLogger(__name__)

WAITING = "waiting_for_approval"
EXPIRED = "expired"


class RunNotFoundError(Exception):
    pass


class InvalidRunStateError(Exception):
    pass


class AgentService:
    """Runs the graph and mirrors every run into the history repository.

    A run waiting for approval expires after `approval_timeout` seconds (None or 0
    disables this). Expiry is lazy: it is applied the next time the run is read.
    """

    def __init__(
        self,
        graph: Any,
        repository: RunRepository,
        max_iterations: int = 3,
        approval_timeout: float | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._graph = graph
        self._repo = repository
        self._max_iterations = max_iterations
        self._approval_timeout = approval_timeout if approval_timeout else None
        self._clock = clock

    def run(self, question: str) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        self._repo.create(run_id, question)
        logger.info("run=%s workflow started", run_id)
        return self._execute(run_id, initial_state(question, self._max_iterations))

    def approve(self, run_id: str) -> dict[str, Any]:
        return self._decide(run_id, "approve")

    def reject(self, run_id: str) -> dict[str, Any]:
        return self._decide(run_id, "reject")

    def get(self, run_id: str) -> dict[str, Any]:
        record = self._repo.get(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return self._apply_approval_deadline(record)

    def _apply_approval_deadline(self, record: dict[str, Any]) -> dict[str, Any]:
        if record["status"] != WAITING or self._approval_timeout is None:
            return record
        deadline = datetime.fromisoformat(record["updated_at"]) + timedelta(
            seconds=self._approval_timeout
        )
        if self._clock() < deadline:
            return {**record, "approval_expires_at": deadline.isoformat(timespec="seconds")}
        if self._repo.claim(record["id"], WAITING, EXPIRED):
            logger.warning("run=%s approval timed out, nothing was executed", record["id"])
            self._repo.update(
                record["id"],
                final_answer="Approval timed out. The action was not executed and nothing was changed.",
            )
        return self._repo.get(record["id"]) or record

    def _decide(self, run_id: str, decision: str) -> dict[str, Any]:
        record = self.get(run_id)
        if not self._repo.claim(run_id, WAITING, "running"):
            raise InvalidRunStateError(
                f"Run {run_id} is '{record['status']}', not waiting for approval"
            )
        logger.info("run=%s decision=%s", run_id, decision)
        return self._execute(run_id, Command(resume=decision))

    def _execute(self, run_id: str, graph_input: Any) -> dict[str, Any]:
        config = {"configurable": {"thread_id": run_id}}
        try:
            result = self._graph.invoke(graph_input, config)
            self._persist(run_id, config, result)
        except Exception as exc:  # the graph handles node errors itself; this is the last net
            logger.exception("run=%s crashed", run_id)
            self._repo.update(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        return self.get(run_id)

    def _persist(self, run_id: str, config: dict, result: dict[str, Any]) -> None:
        values = self._graph.get_state(config).values
        pending = result["__interrupt__"][0].value if "__interrupt__" in result else None
        status = WAITING if pending else values.get("status", "failed")
        logger.info(
            "run=%s status=%s iterations=%s", run_id, status, values.get("iterations", 0)
        )
        self._repo.update(
            run_id,
            status=status,
            current_step=values.get("current_step"),
            final_answer=None if pending else values.get("final_answer"),
            iterations=values.get("iterations", 0),
            visited_nodes=values.get("visited_nodes", []),
            tools_used=values.get("tools_used", []),
            sql=values.get("sql"),
            tool_result=values.get("tool_result"),
            pending_approval=pending,
            error=values.get("error"),
        )
