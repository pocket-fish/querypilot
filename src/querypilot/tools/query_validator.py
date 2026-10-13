from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

_WRITE_NODES = (exp.Insert, exp.Update, exp.Delete)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    read_only: bool = True
    error: str | None = None


def validate_sql(sql: str) -> ValidationResult:
    """Static safety check.

    SELECT queries are read-only and run freely. INSERT/UPDATE/DELETE are valid but
    flagged (read_only=False) so the graph asks a human first. Everything else
    (DROP, ALTER, CREATE, PRAGMA, multiple statements...) is rejected.
    """
    if not sql or not sql.strip():
        return ValidationResult(ok=False, error="Empty SQL")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as exc:
        return ValidationResult(ok=False, error=f"SQL syntax error: {exc}")
    if len(statements) != 1:
        return ValidationResult(ok=False, error="Exactly one SQL statement is allowed")

    stmt = statements[0]
    if isinstance(stmt, exp.Query):
        if stmt.find(*_WRITE_NODES):
            return ValidationResult(ok=False, error="Data-modifying statement inside a query")
        return ValidationResult(ok=True, read_only=True)
    if isinstance(stmt, _WRITE_NODES):
        return ValidationResult(ok=True, read_only=False)
    return ValidationResult(ok=False, error=f"Statement type not allowed: {type(stmt).__name__}")
