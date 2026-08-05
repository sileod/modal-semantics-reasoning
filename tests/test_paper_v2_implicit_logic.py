import json

import pytest

from paper_v2.implicit_logic import (
    SYSTEMS,
    _explicit_prediction,
    logic_prediction,
    score,
)


def test_modal_cube_predictions():
    expected = {
        "K": (False, False, False, False, False),
        "D": (True, False, False, False, False),
        "T": (True, True, False, False, False),
        "B": (True, True, True, False, False),
        "S4": (True, True, False, True, False),
        "S5": (True, True, True, True, True),
    }
    contrasts = (("K", "D"), ("K", "T"), ("T", "B"), ("T", "S4"), ("B", "S5"))
    assert {
        system: tuple(logic_prediction(system, contrast) for contrast in contrasts)
        for system in SYSTEMS
    } == expected


def test_scoring_recovers_s4(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    raw = tmp_path / "raw.jsonl"
    prompt_rows = []
    raw_rows = []
    for index, contrast in enumerate(
        (("K", "D"), ("K", "T"), ("T", "B"), ("T", "S4"), ("B", "S5"))
    ):
        source = f"p{index}"
        validity = logic_prediction("S4", contrast)
        for polarity, answer in (("valid", validity), ("fail", not validity)):
            pair_id = f"{source}-{polarity}"
            prompt_rows.append(
                {
                    "pair_id": pair_id,
                    "source_pair_id": source,
                    "prompt_hash": pair_id,
                    "contrast": contrast,
                    "query_polarity": polarity,
                }
            )
            raw_rows.append(
                {
                    "pair_id": pair_id,
                    "prompt_hash": pair_id,
                    "parsed_answer": answer,
                    "model": "test",
                }
            )
    prompts.write_text("".join(json.dumps(row) + "\n" for row in prompt_rows))
    raw.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    result = score(prompts, raw, bootstrap_samples=100)
    assert result["best_logics"] == ["S4"]
    assert result["logic_fits"]["S4"]["agreement"] == 1
    assert result["logic_fits"]["S4"]["balanced_agreement"] == 1
    assert result["property_affinity"]["4"]["rate"] == 1


def test_best_fit_macro_averages_contrasts(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    raw = tmp_path / "raw.jsonl"
    prompt_rows = []
    raw_rows = []
    examples = [("K", "D")] * 10 + [("K", "T"), ("T", "B"), ("T", "S4"), ("B", "S5")]
    for index, contrast in enumerate(examples):
        source = f"p{index}"
        validity = index < 4 or contrast == ("K", "T")
        for polarity, answer in (("valid", validity), ("fail", not validity)):
            pair_id = f"{source}-{polarity}"
            prompt_rows.append(
                {
                    "pair_id": pair_id,
                    "source_pair_id": source,
                    "prompt_hash": pair_id,
                    "contrast": contrast,
                    "query_polarity": polarity,
                }
            )
            raw_rows.append(
                {
                    "pair_id": pair_id,
                    "prompt_hash": pair_id,
                    "parsed_answer": answer,
                    "model": "test",
                }
            )
    prompts.write_text("".join(json.dumps(row) + "\n" for row in prompt_rows))
    raw.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))

    result = score(prompts, raw, bootstrap_samples=100)

    assert result["best_logics"] == ["T"]
    assert (
        result["logic_fits"]["K"]["agreement"] > result["logic_fits"]["T"]["agreement"]
    )
    assert result["logic_fits"]["T"]["balanced_agreement"] == pytest.approx(0.88)


def test_explicit_profile_effect_controls_gold_label(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    raw = tmp_path / "raw.jsonl"
    pair_rows = [
        {
            "pair_id": "positive",
            "contrast": ["K", "D"],
            "label_a": True,
            "label_b": False,
        },
        {
            "pair_id": "negative",
            "contrast": ["T", "B"],
            "label_a": True,
            "label_b": False,
        },
    ]
    raw_rows = [
        {"pair_id": "positive", "side": "a", "parsed_answer": True},
        {"pair_id": "positive", "side": "b", "parsed_answer": True},
        {"pair_id": "negative", "side": "a", "parsed_answer": False},
        {"pair_id": "negative", "side": "b", "parsed_answer": False},
    ]
    pairs.write_text("".join(json.dumps(row) + "\n" for row in pair_rows))
    raw.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))

    result = _explicit_prediction(
        "T", {"positive", "negative"}, pairs, raw, bootstrap_samples=100, seed=1
    )

    assert result["accuracy_by_gold"]["true"]["accuracy"] == 0.5
    assert result["accuracy_by_gold"]["false"]["accuracy"] == 0.5
    assert result["bias_controlled_profile_effect"]["true"]["delta_yes_rate"] == 1
    assert result["bias_controlled_profile_effect"]["false"]["delta_yes_rate"] == 1
    assert result["profile_effect_identifiable"] is True

    k_result = _explicit_prediction(
        "K", {"positive", "negative"}, pairs, raw, bootstrap_samples=100, seed=1
    )
    assert k_result["profile_effect_identifiable"] is False
