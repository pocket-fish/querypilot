import sqlite3
from pathlib import Path
from typing import Any

from querypilot.tools.errors import ToolError


def execute_sql(
    db_path: str, sql: str, *, allow_write: bool = False, max_rows: int = 100
) -> dict[str, Any]:
    """Run one statement. Read-only connection unless allow_write is set (after approval)."""
    if not Path(db_path).exists():
        raise ToolError(f"Database not found: {db_path}")
    mode = "rw" if allow_write else "ro"
    uri = Path(db_path).resolve().as_uri() + f"?mode={mode}"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.execute(sql)
        if cursor.description is None:
            conn.commit()
            return {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "rows_affected": cursor.rowcount,
            }
        columns = [c[0] for c in cursor.description]
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [list(r) for r in fetched[:max_rows]]
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "rows_affected": None,
        }
    except sqlite3.Error as exc:
        raise ToolError(f"SQL execution failed: {exc}") from exc
    finally:
        conn.close()
