import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from querypilot.tools.errors import ToolError

# A TEXT column with at most this many distinct values is treated as an enum
# (status, country...). Its values are shown to the model and used by the linter.
MAX_ENUM_VALUES = 10


@dataclass(frozen=True)
class SchemaInfo:
    """Keys and enum-like values of the user tables; table/column names are lower-cased."""

    primary_keys: dict[str, frozenset[str]] = field(default_factory=dict)
    # (table, column, referenced_table, referenced_column)
    foreign_keys: frozenset[tuple[str, str, str, str]] = frozenset()
    # (table, column) -> allowed values, original casing
    enum_values: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict)


def _connect(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise ToolError(f"Database not found: {db_path}")
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _enum_values(conn: sqlite3.Connection, table: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for _, name, col_type, *_rest in conn.execute(f'PRAGMA table_info("{table}")'):
        if "TEXT" not in (col_type or "").upper():
            continue
        values = [
            r[0]
            for r in conn.execute(
                f'SELECT DISTINCT "{name}" FROM "{table}" WHERE "{name}" IS NOT NULL '
                f"LIMIT {MAX_ENUM_VALUES + 1}"
            )
        ]
        if 0 < len(values) <= MAX_ENUM_VALUES:
            found[name] = sorted(values)
    return found


def read_schema_info(db_path: str) -> SchemaInfo:
    try:
        with _connect(db_path) as conn:
            primary_keys: dict[str, frozenset[str]] = {}
            foreign_keys: set[tuple[str, str, str, str]] = set()
            enums: dict[tuple[str, str], frozenset[str]] = {}
            for table in _user_tables(conn):
                columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                primary_keys[table.lower()] = frozenset(c[1].lower() for c in columns if c[5])
                for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
                    # fk: id, seq, table, from, to, ...
                    if fk[4]:
                        foreign_keys.add(
                            (table.lower(), fk[3].lower(), fk[2].lower(), fk[4].lower())
                        )
                for column, values in _enum_values(conn, table).items():
                    enums[(table.lower(), column.lower())] = frozenset(values)
            return SchemaInfo(primary_keys, frozenset(foreign_keys), enums)
    except sqlite3.Error as exc:
        raise ToolError(f"Schema inspection failed: {exc}") from exc


def inspect_schema(db_path: str) -> str:
    """CREATE statements with row counts, plus the allowed values of enum-like columns."""
    try:
        with _connect(db_path) as conn:
            parts = []
            for table in _user_tables(conn):
                ddl = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
                ).fetchone()[0]
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                block = [f"{ddl};  -- {count} rows"]
                for column, values in _enum_values(conn, table).items():
                    quoted = ", ".join(f"'{v}'" for v in values)
                    block.append(f"-- {table}.{column} values: {quoted}")
                parts.append("\n".join(block))
            return "\n".join(parts)
    except sqlite3.Error as exc:
        raise ToolError(f"Schema inspection failed: {exc}") from exc
