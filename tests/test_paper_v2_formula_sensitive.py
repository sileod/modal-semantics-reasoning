import json
import re
from pathlib import Path

from paper_v2.formula_sensitive import (
    _candidate,
    _countermodel,
    _evaluate_witness,
    _validate,
)


def test_frozen_formula_sensitive_prompts_hide_conventional_labels():
    path = Path("data/paper_v2/frozen/formula-sensitive-1.0/prompts.jsonl")
    prompts = [json.loads(line)["prompt"] for line in path.read_text().splitlines()]
    assert len(prompts) == 320
    for prompt in prompts:
        assert not re.search(r"\b(?:B|S4|cumulative|decreasing)\b", prompt)


def _source(contrast):
    rows = [
        json.loads(line)
        for line in Path("data/paper_v2/frozen/scaled-0.2/pairs.jsonl").read_text().splitlines()
    ]
    return next(row for row in rows if row["contrast"] == contrast)


def test_non_nested_candidates_preserve_formula_and_change_conditions():
    source = _source(["T", "B"])
    candidate = _candidate(source, ("B", "S4"))
    assert candidate.conjecture == source["conjecture"]["tptp"]
    assert candidate.semantics_a.system == "B"
    assert candidate.semantics_b.system == "S4"


def test_explicit_countermodels_falsify_each_schema_family():
    cases = (
        (["T", "B"], ("B", "S4"), "b"),
        (["T", "S4"], ("B", "S4"), "a"),
        (["varying", "cumulative"], ("cumulative", "decreasing"), "b"),
        (["varying", "decreasing"], ("cumulative", "decreasing"), "a"),
    )
    for source_contrast, diagnostic_contrast, invalid_side in cases:
        candidate = _candidate(_source(source_contrast), diagnostic_contrast)
        semantics = getattr(candidate, f"semantics_{invalid_side}")
        witness = _countermodel(candidate, semantics)
        assert not _evaluate_witness(candidate, semantics, witness)


def test_formula_sensitive_validation_requires_label_balance():
    rows = []
    for axis, contrast in (("frame", ["B", "S4"]), ("domain", ["cumulative", "decreasing"])):
        for valid_under in contrast:
            for index in range(2):
                swapped = index % 2
                labels = [valid_under == contrast[0], valid_under == contrast[1]]
                semantics = (
                    [{"system": "B"}, {"system": "S4"}]
                    if axis == "frame"
                    else [{"domain": "cumulative"}, {"domain": "decreasing"}]
                )
                rows.append(
                    {
                        "pair_id": f"{axis}-{valid_under}-{index}",
                        "axis": axis,
                        "contrast": contrast,
                        "semantics_a": semantics[swapped],
                        "semantics_b": semantics[1 - swapped],
                        "label_a": labels[swapped],
                        "label_b": labels[1 - swapped],
                        "metadata": {"formula_valid_under": valid_under},
                    }
                )
    _validate(rows, per_family_mode=1)
