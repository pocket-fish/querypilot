import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from querypilot.llm import LLM, LoggedLLM
from querypilot.services.answer_checker import check_answer
from querypilot.services.language import detect_language
from querypilot.services.result_checker import find_null_columns
from querypilot.state import AgentState, InvalidStateError, ToolCall
from querypilot.tools.errors import ToolError
from querypilot.tools.query_validator import validate_sql
from querypilot.tools.schema_inspector import inspect_schema, read_schema_info
from querypilot.tools.sql_executor import execute_sql
from querypilot.tools.sql_linter import lint_sql

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], dict[str, Any]]

def _answer_rules(question: str) -> str:
    return (
        "You are a data analyst assistant. Answer the user's question in one or two complete "
        f"sentences. Write the whole answer in {detect_language(question)} only, never in "
        "Chinese. Use only numbers that appear in the query result, copied exactly and written "
        "without thousands separators; never merge or invent numbers. For lists, put each item "
        "on its own line. For INSERT/UPDATE/DELETE, say how many rows were affected "
        "(rows_affected).\n\n"
    )


@dataclass
class Deps:
    llm: LLM
    db_path: str
    max_rows: int = 100


def _guarded(name: str) -> Callable[[NodeFn], NodeFn]:
    """Log node entry, record it in visited_nodes, turn unexpected errors into status=failed."""

    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        def wrapper(state: AgentState) -> dict[str, Any]:
            logger.info("node=%s iteration=%s", name, state.get("iterations", 0))
            try:
                update = fn(state)
            except GraphBubbleUp:  # interrupt() must propagate to the runtime
                raise
            except Exception as exc:
                logger.exception("node=%s failed", name)
                return {
                    "current_step": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "visited_nodes": [name],
                }
            update.setdefault("current_step", name)
            update["visited_nodes"] = [name]
            return update

        return wrapper

    return decorator


KNOWN_TOOLS = ("schema_inspector", "sql_executor")


def _require_tool_call(state: AgentState, node: str) -> ToolCall:
    call = state.get("tool_call")
    if call is None:
        raise InvalidStateError(f"{node} was reached without a tool_call")
    if call["name"] not in KNOWN_TOOLS:
        raise InvalidStateError(f"{node}: unknown tool '{call['name']}'")
    return call


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _clean_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip().rstrip(";").strip()


def build_nodes(deps: Deps) -> dict[str, NodeFn]:
    llm = LoggedLLM(deps.llm)

    @_guarded("analyze_request")
    def analyze_request(state: AgentState) -> dict[str, Any]:
        reply = llm.invoke(
            "Decide whether answering the question requires querying a SQL database "
            "with business data (orders, customers). Reply with JSON only: "
            '{"needs_data": true} or {"needs_data": false}.\n\n'
            f"Question: {state['user_question']}"
        )
        needs_data = bool(_extract_json(reply).get("needs_data", True))
        logger.info("analyze_request needs_data=%s", needs_data)
        return {
            "needs_data": needs_data,
            "messages": [HumanMessage(content=state["user_question"])],
        }

    @_guarded("tool_selection")
    def tool_selection(state: AgentState) -> dict[str, Any]:
        if not state.get("schema_text"):
            logger.info("tool selected: schema_inspector")
            return {
                "tool_call": {"name": "schema_inspector", "args": {}, "requires_approval": False}
            }

        retry_hint = ""
        if state.get("error"):
            retry_hint = (
                f"\nYour previous SQL was:\n{state.get('sql')}\n"
                f"It failed with: {state['error']}\nFix it."
            )
        reply = llm.invoke(
            "You are a SQLite expert. Write ONE SQL statement that answers the question. "
            "Reply with the SQL only.\n"
            "Rules: use only tables and columns from the schema; prefer the simplest query "
            "that works and avoid date functions unless the question needs them. If a business "
            "metric is not defined, use a simple standard definition (for example LTV = total "
            "amount of paid orders per customer). Add a filter on orders.status only for "
            "money questions (revenue, sales, LTV, average order value) or when the question "
            "names a status; counting orders counts all statuses.\n\n"
            f"Schema:\n{state['schema_text']}\n\nQuestion: {state['user_question']}{retry_hint}"
        )
        sql = _clean_sql(reply)
        verdict = validate_sql(sql)
        problem = verdict.error
        if verdict.ok and verdict.read_only:
            issues = lint_sql(sql, read_schema_info(deps.db_path))
            if issues:
                problem = "SQL review found problems: " + " ".join(issues)
        if problem:
            logger.warning("sql rejected: %s", problem)
            return {
                "sql": sql,
                "tool_call": None,
                "error": problem,
                "iterations": state.get("iterations", 0) + 1,
            }
        logger.info("tool selected: sql_executor read_only=%s", verdict.read_only)
        return {
            "sql": sql,
            "error": None,
            "tool_call": {
                "name": "sql_executor",
                "args": {"sql": sql},
                "requires_approval": not verdict.read_only,
            },
        }

    @_guarded("human_approval")
    def human_approval(state: AgentState) -> dict[str, Any]:
        call = _require_tool_call(state, "human_approval")
        decision = interrupt(
            {
                "type": "approval_required",
                "tool": call["name"],
                "sql": call["args"].get("sql"),
                "message": "This statement modifies data. Approve?",
            }
        )
        if str(decision).lower() in ("approve", "approved", "yes", "true"):
            logger.info("approval granted")
            return {"approval": "approved"}
        logger.info("approval rejected")
        return {
            "approval": "rejected",
            "status": "rejected",
            "final_answer": "The action was rejected by the user. Nothing was changed.",
        }

    @_guarded("tool_execution")
    def tool_execution(state: AgentState) -> dict[str, Any]:
        call = _require_tool_call(state, "tool_execution")
        name = call["name"]
        logger.info("executing tool=%s", name)
        try:
            if name == "schema_inspector":
                schema = inspect_schema(deps.db_path)
                logger.info("tool=%s ok", name)
                return {"schema_text": schema, "error": None, "tools_used": [name]}
            result = execute_sql(
                deps.db_path,
                call["args"]["sql"],
                allow_write=call["requires_approval"] and state.get("approval") == "approved",
                max_rows=deps.max_rows,
            )
        except ToolError as exc:
            logger.warning("tool=%s failed: %s", name, exc)
            return {
                "error": str(exc),
                "tool_result": None,
                "iterations": state.get("iterations", 0) + 1,
                "tools_used": [name],
            }
        logger.info("tool=%s ok rows=%s", name, result["row_count"])
        null_columns = find_null_columns(result)
        if null_columns:
            names = ", ".join(null_columns)
            logger.warning("suspicious result: all-NULL columns: %s", names)
            return {
                "tool_result": result,
                "error": (
                    f"Column(s) {names} are NULL in every row, so the expression is probably "
                    "wrong. Rewrite the query in a simpler way."
                ),
                "iterations": state.get("iterations", 0) + 1,
                "tools_used": [name],
            }
        return {"tool_result": result, "error": None, "tools_used": [name]}

    @_guarded("generate_answer")
    def generate_answer(state: AgentState) -> dict[str, Any]:
        question = state["user_question"]
        error = state.get("error")
        if error and not state.get("tool_result"):
            # No data to explain: never let the model improvise an answer here.
            prefix = (
                "Не удалось построить надёжный запрос. Последняя проблема: "
                if detect_language(question) == "Russian"
                else "I could not build a reliable query. Last problem: "
            )
            return {"final_answer": prefix + error, "feedback": None}
        if state.get("tool_result"):
            context = f"SQL: {state.get('sql')}\nResult: {json.dumps(state['tool_result'])}"
            if state.get("error"):
                context += f"\nWarning: {state['error']} Mention that the result may be unreliable."
        else:
            context = (
                "No database lookup was needed. If asked what you can do: you answer questions "
                "about customers and orders by writing SQL, and you can change data only after "
                "the user approves. Do not state any numbers."
            )
        answer = llm.invoke(_answer_rules(question) + f"Question: {question}\n{context}").strip()
        return {"final_answer": answer, "feedback": None}

    @_guarded("validate_answer")
    def validate_answer(state: AgentState) -> dict[str, Any]:
        feedback = check_answer(
            state.get("final_answer"), state["user_question"], state.get("tool_result")
        )
        logger.info("validate_answer feedback=%s", feedback)
        return {"feedback": feedback}

    @_guarded("improve_answer")
    def improve_answer(state: AgentState) -> dict[str, Any]:
        iterations = state.get("iterations", 0) + 1
        logger.info("improving answer, iteration=%s", iterations)
        improved = llm.invoke(
            _answer_rules(state["user_question"]) + "Rewrite the previous answer to fix the problem. "
            "Reply with the corrected answer only, no preface.\n\n"
            f"Question: {state['user_question']}\n"
            f"Query result: {json.dumps(state.get('tool_result'))}\n"
            f"Previous answer: {state.get('final_answer')}\n"
            f"Problem: {state.get('feedback')}"
        ).strip()
        return {"final_answer": improved, "iterations": iterations}

    @_guarded("finalize")
    def finalize(state: AgentState) -> dict[str, Any]:
        status = state.get("status", "running")
        if status == "running":
            exhausted = state.get("iterations", 0) >= state.get("max_iterations", 3)
            unresolved = bool(state.get("feedback") or state.get("error"))
            status = "max_iterations" if exhausted and unresolved else "completed"
        answer = state.get("final_answer")
        if not answer:
            answer = f"The agent could not finish: {state.get('error') or 'unknown error'}"
        if status == "max_iterations":
            answer += "\n\n(Stopped: iteration limit reached; the answer may be imperfect.)"
        logger.info("workflow finished status=%s iterations=%s", status, state.get("iterations"))
        return {
            "status": status,
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return {
        "analyze_request": analyze_request,
        "tool_selection": tool_selection,
        "human_approval": human_approval,
        "tool_execution": tool_execution,
        "generate_answer": generate_answer,
        "validate_answer": validate_answer,
        "improve_answer": improve_answer,
        "finalize": finalize,
    }
