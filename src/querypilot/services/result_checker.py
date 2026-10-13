from typing import Any


def find_null_columns(result: dict[str, Any] | None) -> list[str]:
    """Names of columns that are NULL in every row.

    SQLite does not fail on many broken expressions (bad date modifiers, wrong
    aggregates); it silently returns NULL. An all-NULL column is a strong sign the
    query is wrong even though it executed.
    """
    if not result or not result.get("rows"):
        return []
    columns = result.get("columns", [])
    return [
        name
        for index, name in enumerate(columns)
        if all(row[index] is None for row in result["rows"])
    ]
