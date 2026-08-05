from paper_v2.candidates import generate_candidates
from paper_v2.final_freeze import (
    _candidate_formula_hash,
    _cells,
    _validate_final,
)
from paper_v2.semantics import load_config


def test_final_quotas_sum_to_axis_targets():
    config = load_config()
    cells = _cells(config)
    for axis, target in config["final"]["target_pairs"].items():
        assert sum(quota for cell_axis, _, _, quota in cells if cell_axis == axis) == target


def test_final_validation_rejects_duplicate_formulas():
    config = {
        "final": {"target_pairs": {"frame": 1, "domain": 0}}
    }
    pairs = [
        {
            "pair_id": "frame-a",
            "canonical_formula_hash": "same",
            "axis": "frame",
            "label_a": True,
        },
        {
            "pair_id": "frame-b",
            "canonical_formula_hash": "same",
            "axis": "frame",
            "label_a": False,
        },
    ]
    try:
        _validate_final(pairs, config)
    except ValueError as error:
        assert "duplicate canonical formulas" in str(error)
    else:
        raise AssertionError("duplicate formulas were accepted")


def test_domain_cells_can_draw_disjoint_formula_reserves():
    seed = 20260716
    first = generate_candidates(
        "domain", ("varying", "decreasing"), seed, 300, mode="nli"
    )
    second = generate_candidates(
        "domain", ("cumulative", "constant"), seed, 300, mode="nli"
    )
    used = {_candidate_formula_hash(candidate) for candidate in first[:67]}
    available = [
        candidate for candidate in second if _candidate_formula_hash(candidate) not in used
    ]
    assert len(available) >= 66
