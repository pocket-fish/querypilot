import argparse
import uuid
from pathlib import Path

from langgraph.types import Command

from querypilot.config import get_settings
from querypilot.demo_db import create_demo_db
from querypilot.graph import build_graph, render_mermaid
from querypilot.llm import create_llm
from querypilot.logging_config import setup_logging
from querypilot.nodes import Deps
from querypilot.state import initial_state


def _deps() -> Deps:
    s = get_settings()
    return Deps(llm=create_llm(s), db_path=s.data_db_path, max_rows=s.max_rows)


def _ask(question: str) -> None:
    settings = get_settings()
    graph = build_graph(_deps())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(initial_state(question, settings.max_iterations), config)
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"\n[approval required] {request['message']}\n  {request['sql']}")
        answer = input("Approve? [y/N] ").strip().lower()
        result = graph.invoke(Command(resume="approve" if answer == "y" else "reject"), config)
    print(f"\n[{result['status']}] {result['final_answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="querypilot")
    sub = parser.add_subparsers(dest="command", required=True)
    seed = sub.add_parser("seed", help="create the demo database")
    seed.add_argument("--if-missing", action="store_true", help="do nothing if it already exists")
    sub.add_parser("graph", help="print the graph as a Mermaid diagram")
    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    ask = sub.add_parser("ask", help="ask a question in natural language")
    ask.add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    if args.command == "seed":
        if args.if_missing and Path(settings.data_db_path).exists():
            print(f"Demo database already exists at {settings.data_db_path}")
        else:
            create_demo_db(settings.data_db_path)
            print(f"Demo database created at {settings.data_db_path}")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("querypilot.bootstrap:app", factory=True, host=args.host, port=args.port)
    elif args.command == "graph":
        print(render_mermaid(Deps(llm=None, db_path=settings.data_db_path)))  # type: ignore[arg-type]
    else:
        _ask(args.question)
