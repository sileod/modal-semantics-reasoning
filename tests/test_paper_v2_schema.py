from dataclasses import replace

import pytest

from paper_v2.schema import Pair
from paper_v2.semantics import Semantics


def frame_pair() -> Pair:
    return Pair(
        axis="frame",
        contrast=("K", "T"),
        premises=({"op": "box", "arg": {"pred": "p"}},),
        conjecture={"pred": "p"},
        semantics_a=Semantics.make("K"),
        semantics_b=Semantics.make("T"),
        label_a=False,
        label_b=True,
        oracle_a={"status": "CounterSatisfiable"},
        oracle_b={"status": "Theorem"},
        controlled_english={"premises": ["Necessarily p."], "conjecture": "p."},
        generator_seed=7,
        modal_depth=1,
        quantifier_depth=0,
        premise_status="premise_dependent",
    )


def test_hashes_and_identifier_are_stable():
    first = frame_pair()
    second = frame_pair()
    assert first.canonical_formula_hash == second.canonical_formula_hash
    assert first.pair_id == second.pair_id


def test_identifier_depends_on_canonical_content_not_seed():
    first = frame_pair()
    second = replace(first, generator_seed=999)
    assert first.pair_id == second.pair_id


def test_opposite_labels_are_required():
    pair = replace(frame_pair(), label_a=True)
    with pytest.raises(ValueError, match="opposite"):
        pair.validate()


def test_serialization_contains_canonical_fields():
    data = frame_pair().to_dict()
    assert data["pair_id"].startswith("frame-")
    assert data["canonical_formula_hash"]
    assert data["premise_dependent"] is True
