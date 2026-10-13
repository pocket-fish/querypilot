import sqlite3

import pytest

from querypilot.demo_db import create_demo_db
from querypilot.graph import build_graph
from querypilot.llm import FakeLLM
from querypilot.nodes import Deps
from querypilot.state import initial_state


@pytest.fixture
def db_path(tmp_path) -> str:
    path = tmp_path / "demo.db"
    create_demo_db(str(path))
    return str(path)


def count_rows(db_path: str, table: str, where: str = "1=1") -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


class Harness:
    """Runs the real graph with a scripted LLM against a temp database."""

    def __init__(self, db_path: str, responses: list, max_iterations: int = 3) -> None:
        self.db_path = db_path
        self.llm = FakeLLM(responses)
        self.graph = build_graph(Deps(llm=self.llm, db_path=db_path))
        self.config = {"configurable": {"thread_id": "t1"}}
        self.max_iterations = max_iterations

    def run(self, question: str) -> dict:
        return self.graph.invoke(initial_state(question, self.max_iterations), self.config)

    def resume(self, decision: str) -> dict:
        from langgraph.types import Command

        return self.graph.invoke(Command(resume=decision), self.config)


@pytest.fixture
def make_harness(db_path):
    def factory(responses: list, max_iterations: int = 3) -> Harness:
        return Harness(db_path, responses, max_iterations)

    return factory
