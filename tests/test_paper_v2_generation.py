import pytest

from paper_v2.candidates import generate_candidates
from paper_v2.oracle import build_tptp
from paper_v2.render import render_prompt


@pytest.mark.parametrize(
    ("axis", "contrast"),
    [
        ("frame", ("K", "D")),
        ("domain", ("varying", "cumulative")),
        ("designation", ("rigid", "flexible")),
    ],
)
def test_generation_is_deterministic(axis, contrast):
    first = generate_candidates(axis, contrast, seed=7, n=3)
    second = generate_candidates(axis, contrast, seed=7, n=3)
    assert first == second


def test_frame_candidates_are_propositional():
    candidate = generate_candidates("frame", ("K", "T"), seed=1, n=1)[0]
    assert "$i" not in candidate.conjecture
    assert "![" not in candidate.conjecture


def test_domain_candidates_have_quantifier_modal_alternation():
    candidate = generate_candidates(
        "domain", ("varying", "cumulative"), seed=1, n=1
    )[0]
    assert candidate.alternation_pattern == "box-forall"
    assert "$box" in candidate.conjecture and "![" in candidate.conjecture
    assert "the_" not in candidate.conjecture


def test_generation_varies_grammar_structure_not_only_symbols():
    candidates = generate_candidates("frame", ("K", "T"), seed=1, n=20)
    structures = {tuple(candidate.ast["content"]) for candidate in candidates}
    assert len(structures) == 20


def test_designation_prompt_explains_flexibility_directly():
    candidate = generate_candidates(
        "designation", ("rigid", "flexible"), seed=1, n=1
    )[0]
    prompt = render_prompt(candidate, candidate.semantics_b)
    assert "may denote different objects" in prompt
    assert "local/global" not in prompt


def test_zero_premise_candidate_uses_validity_prompt():
    candidate = generate_candidates("frame", ("K", "D"), seed=1, n=1)[0]
    prompt = render_prompt(candidate, candidate.semantics_a)
    assert "Statement:" in prompt
    assert "Is the statement valid" in prompt
    assert "Premises:" not in prompt
    assert "(none)" not in prompt
    assert "is active is necessary" not in prompt


def test_premise_bearing_candidate_uses_nli_prompt():
    candidate = generate_candidates(
        "designation", ("rigid", "flexible"), seed=1, n=1
    )[0]
    prompt = render_prompt(candidate, candidate.semantics_a)
    assert "Premises:" in prompt
    assert "Conjecture:" in prompt
    assert "follow from the premises" in prompt


@pytest.mark.parametrize(
    ("axis", "contrast"),
    [
        ("frame", ("T", "S4")),
        ("domain", ("varying", "cumulative")),
    ],
)
def test_nli_candidates_have_readable_premises(axis, contrast):
    candidate = generate_candidates(axis, contrast, seed=1, n=1, mode="nli")[0]
    prompt = render_prompt(candidate, candidate.semantics_a)
    assert candidate.premises
    assert "Premises:" in prompt
    assert "claim_0" not in prompt
    assert "marked_0" not in prompt
    assert "(none)" not in prompt


def test_tptp_uses_single_brace_modal_connectives():
    candidate = generate_candidates("frame", ("K", "D"), seed=1, n=1)[0]
    source = build_tptp(candidate, candidate.semantics_a)
    assert "{$box}" in source and "{$dia}" in source
    assert "{{$box}}" not in source and "{{$dia}}" not in source
