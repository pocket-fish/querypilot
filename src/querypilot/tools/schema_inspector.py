import sqlite3
from pathlib import Path

from querypilot.tools.errors import ToolError


def inspect_schema(db_path: str) -> str:
    """Return the CREATE statements of all user tables, plus row counts."""
    if not Path(db_path).exists():
        raise ToolError(f"Database not found: {db_path}")
    try:
        with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
            tables = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            parts = []
            for name, ddl in tables:
                count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                parts.append(f"{ddl};  -- {count} rows")
            return "\n".join(parts)
    except sqlite3.Error as exc:
        raise ToolError(f"Schema inspection failed: {exc}") from exc
