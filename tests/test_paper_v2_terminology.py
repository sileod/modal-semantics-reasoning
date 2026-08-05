from paper_v2.terminology import _define_frame_properties, _tptp_prompt


def test_named_property_is_replaced_by_plain_definition():
    prompt = (
        "Semantic specification:\n"
        "- The accessibility relation is reflexive, transitive.\n\n"
        "Statement: p."
    )
    defined = _define_frame_properties(prompt, ["reflexive", "transitive"])
    assert "reflexive" not in defined
    assert "transitive" not in defined
    assert "Every world accesses itself." in defined
    assert "accesses a third" in defined


def test_symbolic_surface_contains_exact_modal_system_and_formula():
    pair = {
        "semantics_a": {"system": "S4"},
        "premises": [],
        "conjecture": {
            "ast": {"symbols": ["p"]},
            "tptp": "({$box} @ (p) => {$box} @ ({$box} @ (p)))",
        },
    }
    prompt = _tptp_prompt(pair, "a")
    assert "$modal_system_S4" in prompt
    assert "tff(conjecture, conjecture" in prompt
