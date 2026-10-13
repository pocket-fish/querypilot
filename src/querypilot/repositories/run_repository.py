import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_JSON_FIELDS = ("visited_nodes", "tools_used", "tool_result", "pending_approval")
_UPDATABLE = {
    "status",
    "current_step",
    "final_answer",
    "iterations",
    "sql",
    "error",
    *_JSON_FIELDS,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    current_step TEXT,
    final_answer TEXT,
    iterations INTEGER NOT NULL DEFAULT 0,
    visited_nodes TEXT NOT NULL DEFAULT '[]',
    tools_used TEXT NOT NULL DEFAULT '[]',
    sql TEXT,
    tool_result TEXT,
    pending_approval TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunRepository:
    """Stores the execution history of agent runs in SQLite."""

    def __init__(self, path: str) -> None:
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, run_id: str, question: str) -> None:
        now = _now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO runs (id, question, status, created_at, updated_at) "
                "VALUES (?, ?, 'running', ?, ?)",
                (run_id, question, now, now),
            )

    def update(self, run_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"Unknown run fields: {sorted(unknown)}")
        values = {
            k: json.dumps(v) if k in _JSON_FIELDS and v is not None else v
            for k, v in fields.items()
        }
        values["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in values)
        with closing(self._connect()) as conn, conn:
            conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*values.values(), run_id))

    def claim(self, run_id: str, from_status: str, to_status: str) -> bool:
        """Atomically move a run between statuses; False if it was not in from_status."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (to_status, _now(), run_id, from_status),
            )
            return cursor.rowcount == 1

    def get(self, run_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        for field in _JSON_FIELDS:
            if record[field] is not None:
                record[field] = json.loads(record[field])
        return record
