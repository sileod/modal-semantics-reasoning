from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    atom: int | None = None

    @property
    def size(self) -> int:
        return 1 + sum(arg.size for arg in self.args)

    @property
    def depth(self) -> int:
        return 0 if not self.args else 1 + max(arg.depth for arg in self.args)

    def canonical(self) -> tuple:
        return _canonical(_rename_atoms(self))

    def atoms(self) -> list[int]:
        return sorted(
            {node.atom for node in _walk(self) if node.op == "atom" and node.atom is not None}
        )


def generate_expressions(limit: int) -> list[Expr]:
    """Enumerate a bounded context-sensitive grammar.

    Productions:
        expression -> atom
        expression -> not expression
        expression -> expression and expression
        expression -> expression or expression

    Context-sensitive constraints canonicalize atom renaming, remove
    commutative duplicates, and reject repeated siblings.
    """
    atoms = [Expr("atom", atom=index) for index in range(5)]
    seen: dict[tuple, Expr] = {}
    by_size: dict[int, list[Expr]] = {1: atoms}
    for expression in atoms:
        _add(seen, expression)

    for size in range(2, 10):
        current: dict[tuple, Expr] = {}
        for child in by_size.get(size - 1, []):
            expression = Expr("not", (child,))
            if expression.depth <= 4 and _readable(expression):
                current.setdefault(expression.canonical(), _rename_atoms(expression))
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            for left, right in product(
                by_size.get(left_size, []), by_size.get(right_size, [])
            ):
                for op in ("and", "or"):
                    if left == right:
                        continue
                    expression = Expr(op, (left, right))
                    if expression.depth <= 4 and _readable(expression):
                        current.setdefault(
                            expression.canonical(), _rename_atoms(expression)
                        )
        by_size[size] = list(current.values())[:2_000]
        for expression in by_size[size]:
            _add(seen, expression)
        if len(seen) >= limit and size >= 5:
            break

    ordered = sorted(
        seen.values(), key=lambda item: (item.size, item.depth, repr(item.canonical()))
    )
    return ordered[:limit]


def render_formula(expression: Expr, symbols: tuple[str, ...], variable: str | None = None) -> str:
    if expression.op == "atom":
        symbol = symbols[expression.atom or 0]
        return symbol if variable is None else f"{symbol}({variable})"
    child = render_formula(expression.args[0], symbols, variable)
    if expression.op == "not":
        return f"(~ ({child}))"
    left = render_formula(expression.args[0], symbols, variable)
    right = render_formula(expression.args[1], symbols, variable)
    connective = "&" if expression.op == "and" else "|"
    return f"(({left}) {connective} ({right}))"


def render_clause(expression: Expr) -> str:
    atoms = (
        "the signal is active",
        "the alarm is sounding",
        "the gate is open",
        "the indicator is lit",
        "the channel is available",
    )
    if expression.op == "atom":
        return atoms[expression.atom or 0]
    if expression.op == "not":
        return f"not ({render_clause(expression.args[0])})"
    left, right = (render_clause(arg) for arg in expression.args)
    connective = "and" if expression.op == "and" else "or"
    return f"({left} {connective} {right})"


def render_property(expression: Expr) -> str:
    atoms = ("registered", "approved", "verified", "listed", "certified")
    if expression.op == "atom":
        return atoms[expression.atom or 0]
    if expression.op == "not":
        return f"not ({render_property(expression.args[0])})"
    left, right = (render_property(arg) for arg in expression.args)
    connective = "and" if expression.op == "and" else "or"
    return f"({left} {connective} {right})"


def _add(seen: dict[tuple, Expr], expression: Expr) -> None:
    canonical = expression.canonical()
    seen.setdefault(canonical, _rename_atoms(expression))


def _rename_atoms(expression: Expr) -> Expr:
    mapping: dict[int, int] = {}

    def visit(node: Expr) -> Expr:
        if node.op == "atom":
            atom = node.atom or 0
            mapping.setdefault(atom, len(mapping))
            return Expr("atom", atom=mapping[atom])
        return Expr(node.op, tuple(visit(arg) for arg in node.args))

    return visit(expression)


def _walk(expression: Expr):
    yield expression
    for arg in expression.args:
        yield from _walk(arg)


def _readable(expression: Expr) -> bool:
    leaves = [node.atom for node in _walk(expression) if node.op == "atom"]
    if len(leaves) != len(set(leaves)):
        return False
    for node in _walk(expression):
        if node.op == "not" and node.args[0].op == "not":
            return False
        if node.op in {"and", "or"}:
            left, right = node.args
            if left == right:
                return False
            if _negates(left, right) or _negates(right, left):
                return False
    return True


def _negates(left: Expr, right: Expr) -> bool:
    return left.op == "not" and left.args[0] == right


def _canonical(expression: Expr) -> tuple:
    if expression.op in {"and", "or"}:
        children = sorted(
            (_canonical(arg) for arg in expression.args), key=repr
        )
        return (expression.op, *children)
    if expression.op == "not":
        return ("not", _canonical(expression.args[0]))
    return ("atom", expression.atom)
