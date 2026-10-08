import json
from collections import Counter
from pathlib import Path

import numpy as np

from paper_v2 import diamond, kripke

DIAMOND = Path("data/paper_v2/frozen/diamond-1.0")
P = ("atom", 0)
X = ("pred", 0, "X")


def test_frame_checker_separates_b_and_s4():
    b_axiom = ("imp", P, ("box", ("dia", P)))
    four = ("imp", ("box", P), ("box", ("box", P)))
    assert kripke.prop_countermodel(b_axiom, "B") is None
    assert kripke.prop_countermodel(b_axiom, "S4") is not None
    assert kripke.prop_countermodel(four, "S4") is None
    assert kripke.prop_countermodel(four, "B") is not None


def test_domain_checker_separates_barcan_formulas():
    converse_barcan = ("imp", ("box", ("all", "X", X)), ("all", "X", ("box", X)))
    barcan = ("imp", ("all", "X", ("box", X)), ("box", ("all", "X", X)))
    assert kripke.fo_countermodel(converse_barcan, "cumulative", rounds=8) is None
    assert kripke.fo_countermodel(converse_barcan, "decreasing", rounds=8) is not None
    assert kripke.fo_countermodel(barcan, "decreasing", rounds=8) is None
    assert kripke.fo_countermodel(barcan, "cumulative", rounds=8) is not None


def test_countermodels_falsify_their_formula():
    for record in map(json.loads, (DIAMOND / "oracle.jsonl").read_text().splitlines()[:40]):
        pair_side = record["side_a"] if record["side_a"]["consensus"] is False else record["side_b"]
        cm = pair_side["countermodel"]
        assert cm["world"] < cm["worlds"]
        assert pair_side["resolution"] == "explicit_countermodel"
        assert all(p.get("szs_status") != "Theorem" for p in pair_side["provers"].values())


def test_rendering_brackets_binary_operands():
    f = ("imp", ("or", ("box", P), ("dia", P)), ("box", ("dia", P)))
    text = diamond.to_english(f)
    assert text.startswith("if [[in every accessible world, the signal is active] or ")
    assert text.endswith("then [in every accessible world, in some accessible world, the signal is active]")


def test_frozen_diamond_is_balanced():
    pairs = [json.loads(line) for line in (DIAMOND / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 160 and len({p["pair_id"] for p in pairs}) == 160
    assert Counter(p["metadata"]["formula_valid_under"] for p in pairs) == {
        "B": 40, "S4": 40, "cumulative": 40, "decreasing": 40}
    for p in pairs:
        assert p["label_a"] != p["label_b"]
        assert p["modal_depth"] >= (3 if p["axis"] == "frame" else 2)
        assert p["quantifier_depth"] == (0 if p["axis"] == "frame" else 2)
