import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from querypilot.config import Settings, get_settings
from querypilot.graph import build_graph
from querypilot.llm import OllamaLLM
from querypilot.logging_config import setup_logging
from querypilot.nodes import Deps
from querypilot.repositories.run_repository import RunRepository
from querypilot.services.agent_service import AgentService


def build_default_service(settings: Settings | None = None) -> AgentService:
    """Wire the real stack: Ollama LLM, SQLite checkpoints (pauses survive restarts) and history."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False))
    deps = Deps(
        llm=OllamaLLM(settings.ollama_model, settings.ollama_base_url),
        db_path=settings.data_db_path,
        max_rows=settings.max_rows,
    )
    return AgentService(
        graph=build_graph(deps, checkpointer),
        repository=RunRepository(settings.history_db_path),
        max_iterations=settings.max_iterations,
    )


def app():
    """ASGI entry point: `uvicorn querypilot.bootstrap:app --factory`."""
    from querypilot.api.app import create_app

    return create_app()
