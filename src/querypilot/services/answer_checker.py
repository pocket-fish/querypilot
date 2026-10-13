import math
import re
from typing import Any

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", ".")) for m in _NUMBER.findall(text)]


def check_answer(
    answer: str | None, question: str, tool_result: dict[str, Any] | None
) -> str | None:
    """Deterministic fact check. Returns feedback for the model, or None if the answer is OK.

    Every number in the answer must come from the tool result (a cell value, the
    row count) or from the user's question. This catches hallucinated figures.
    """
    if not answer or not answer.strip():
        return "The answer is empty. Answer the question using the query result."
    if not tool_result or not tool_result.get("rows"):
        return None

    allowed = _numbers(question)
    allowed.append(float(tool_result.get("row_count", 0)))
    for row in tool_result["rows"]:
        for cell in row:
            allowed.extend(_numbers(str(cell)))

    unknown = [
        n
        for n in _numbers(answer)
        if not any(math.isclose(n, a, abs_tol=0.01) or round(a) == n for a in allowed)
    ]
    if unknown:
        shown = ", ".join(f"{n:g}" for n in dict.fromkeys(unknown))
        return (
            f"These numbers are not in the query result: {shown}. "
            "Use only numbers from the result, written without thousands separators."
        )
    return None
