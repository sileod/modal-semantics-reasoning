from paper_v2.grammar import (
    Expr,
    generate_expressions,
    render_clause,
    render_formula,
    render_property,
)


def test_variants_are_structurally_unique_and_readable():
    expressions = generate_expressions(30)
    assert len(expressions) == 30
    assert len({expression.canonical() for expression in expressions}) == 30
    for expression in expressions:
        assert render_formula(expression, ("p", "q", "r", "s", "t"))
        assert render_clause(expression)
        assert render_property(expression)


def test_atom_renaming_is_canonicalized():
    expressions = generate_expressions(3)
    assert expressions[0].canonical() == ("atom", 0)


def test_controlled_english_preserves_boolean_scope():
    expression = Expr(
        "or",
        (
            Expr("atom", atom=0),
            Expr("and", (Expr("atom", atom=1), Expr("atom", atom=2))),
        ),
    )
    assert render_clause(expression) == (
        "(the signal is active or "
        "(the alarm is sounding and the gate is open))"
    )
