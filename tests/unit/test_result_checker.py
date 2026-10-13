from querypilot.services.result_checker import find_null_columns


def result(columns, rows):
    return {"columns": columns, "rows": rows, "row_count": len(rows)}


def test_flags_column_that_is_null_in_every_row():
    data = result(["name", "ltv"], [["a", None], ["b", None]])
    assert find_null_columns(data) == ["ltv"]


def test_partially_null_column_is_fine():
    assert find_null_columns(result(["x"], [[None], [5]])) == []


def test_empty_or_missing_result_is_not_suspicious():
    assert find_null_columns(result(["x"], [])) == []
    assert find_null_columns(None) == []
