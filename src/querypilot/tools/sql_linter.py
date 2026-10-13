"""Structural review of a SELECT that is syntactically valid but probably wrong.

SQLite happily runs many broken queries (cartesian joins, joins on the wrong key,
bare columns next to aggregates) and returns plausible-looking rows. These checks
are deterministic and aim for high precision: they stay silent when they cannot be
sure, because every finding sends the agent back for another LLM call.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from querypilot.tools.schema_inspector import SchemaInfo


def lint_sql(sql: str, schema: SchemaInfo) -> list[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return []  # the validator reports syntax errors
    if not isinstance(tree, exp.Query):
        return []

    aliases = {
        t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table) if t.name
    }
    issues: list[str] = []
    for select in tree.find_all(exp.Select):
        scope_tables = _scope_tables(select)
        issues += _check_joins(select, aliases, scope_tables, schema)
        issues += _check_bare_columns(select, aliases, scope_tables, schema)
        issues += _check_limit(select)
        issues += _check_enum_literals(select, aliases, scope_tables, schema)
    issues += _check_null_comparison(tree)
    return list(dict.fromkeys(issues))


def _scope_tables(select: exp.Select) -> list[str]:
    return [
        t.name.lower()
        for t in select.find_all(exp.Table)
        if t.name and t.find_ancestor(exp.Select) is select
    ]


def _resolve(column: exp.Column, aliases: dict[str, str], scope_tables: list[str]) -> str | None:
    if column.table:
        return aliases.get(column.table.lower())
    return scope_tables[0] if len(set(scope_tables)) == 1 else None


def _check_joins(
    select: exp.Select, aliases: dict[str, str], scope_tables: list[str], schema: SchemaInfo
) -> list[str]:
    issues = []
    for join in select.args.get("joins") or []:
        target = join.this.name if isinstance(join.this, exp.Table) else "a subquery"
        on, using = join.args.get("on"), join.args.get("using")
        if not on and not using:
            issues.append(
                f"Join with {target} has no ON condition, which multiplies every row by every "
                "row (cartesian product). Join on the foreign key."
            )
            continue
        for eq in on.find_all(exp.EQ) if on else []:
            left, right = eq.left, eq.right
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            lt = _resolve(left, aliases, scope_tables)
            rt = _resolve(right, aliases, scope_tables)
            if not lt or not rt or lt == rt:
                continue
            pair = {(lt, left.name.lower(), rt, right.name.lower()), (rt, right.name.lower(), lt, left.name.lower())}
            if pair & schema.foreign_keys:
                continue
            declared = [
                f"{t}.{c} = {rt_}.{rc}"
                for (t, c, rt_, rc) in schema.foreign_keys
                if {t, rt_} == {lt, rt}
            ]
            if declared:
                issues.append(
                    f"Join condition {left.sql()} = {right.sql()} does not follow the foreign "
                    f"key ({'; '.join(declared)})."
                )
    return issues


def _is_aggregated(column: exp.Column) -> bool:
    return column.find_ancestor(exp.AggFunc, exp.Window) is not None


def _check_bare_columns(
    select: exp.Select, aliases: dict[str, str], scope_tables: list[str], schema: SchemaInfo
) -> list[str]:
    group = select.args.get("group")
    has_aggregate = any(
        agg.find_ancestor(exp.Select) is select and agg.find_ancestor(exp.Window) is None
        for proj in select.expressions
        for agg in proj.find_all(exp.AggFunc)
    )
    if not group and not has_aggregate:
        return []

    group_exprs = list(group.expressions) if group else []
    by_alias = {p.alias.lower(): p.unalias() for p in select.expressions if p.alias}

    def expand(g: exp.Expression) -> exp.Expression:
        if isinstance(g, exp.Literal) and g.is_int and 0 < int(g.name) <= len(select.expressions):
            return select.expressions[int(g.name) - 1]  # GROUP BY 1
        if isinstance(g, exp.Column) and not g.table and g.name.lower() in by_alias:
            return by_alias[g.name.lower()]  # GROUP BY month, where month is a select alias
        return g

    resolved_group = [expand(g) for g in group_exprs]
    group_sql = {g.unalias().sql(dialect="sqlite").lower() for g in resolved_group}
    group_keys = {
        (_resolve(c, aliases, scope_tables), c.name.lower())
        for g in resolved_group
        for c in g.unalias().find_all(exp.Column)
    }

    def in_group(table: str | None, name: str) -> bool:
        return any(n == name and (t == table or t is None or table is None) for t, n in group_keys)

    issues = []
    for proj in select.expressions:
        if proj.unalias().sql(dialect="sqlite").lower() in group_sql:
            continue
        for column in proj.find_all(exp.Column):
            if column.find_ancestor(exp.Select) is not select or _is_aggregated(column):
                continue
            table = _resolve(column, aliases, scope_tables)
            name = column.name.lower()
            if in_group(table, name):
                continue
            pk = schema.primary_keys.get(table or "", frozenset())
            if pk and all(in_group(table, k) for k in pk):
                continue  # functionally dependent on a grouped primary key
            issues.append(
                f"Column {column.sql()} is selected next to an aggregate but is neither in "
                "GROUP BY nor aggregated; SQLite would return a value from an arbitrary row. "
                "Add it to GROUP BY or wrap it in an aggregate."
            )
    return issues


def _check_enum_literals(
    select: exp.Select, aliases: dict[str, str], scope_tables: list[str], schema: SchemaInfo
) -> list[str]:
    """A filter like status = 'completed' on a column whose real values are known."""
    issues = []

    def check(column: exp.Expression, literals: list[exp.Expression]) -> None:
        if not isinstance(column, exp.Column):
            return
        table = _resolve(column, aliases, scope_tables)
        allowed = schema.enum_values.get((table or "", column.name.lower()))
        if not allowed:
            return
        for literal in literals:
            if isinstance(literal, exp.Literal) and literal.is_string and literal.name not in allowed:
                issues.append(
                    f"'{literal.name}' is not a value of {table}.{column.name.lower()}; "
                    f"valid values: {', '.join(sorted(allowed))}."
                )

    for node in select.find_all(exp.EQ, exp.NEQ, exp.In):
        if node.find_ancestor(exp.Select) is not select:
            continue
        if isinstance(node, exp.In):
            check(node.this, list(node.expressions))
        else:
            check(node.left, [node.right])
            check(node.right, [node.left])
    return issues


def _check_limit(select: exp.Select) -> list[str]:
    if select.args.get("limit") and not select.args.get("order"):
        return ["LIMIT without ORDER BY returns arbitrary rows. Add an ORDER BY."]
    return []


def _check_null_comparison(tree: exp.Expression) -> list[str]:
    for node in tree.find_all(exp.EQ, exp.NEQ):
        if isinstance(node.left, exp.Null) or isinstance(node.right, exp.Null):
            return ["Comparing with NULL using = or != is never true. Use IS NULL / IS NOT NULL."]
    return []
