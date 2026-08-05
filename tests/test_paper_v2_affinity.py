import json

from paper_v2.affinity import _omit_axis_clause, score_affinity


def test_frame_omission_removes_empty_semantic_header():
    prompt = (
        "Semantic specification:\n"
        "- The accessibility relation is reflexive.\n\n"
        "Statement: p."
    )
    omitted = _omit_axis_clause(prompt, "frame")
    assert omitted == "Statement: p."


def test_domain_omission_retains_frame_clause():
    prompt = (
        "Semantic specification:\n"
        "- The accessibility relation is serial.\n"
        "- Objects may appear or disappear at accessible worlds.\n\n"
        "Statement: p."
    )
    omitted = _omit_axis_clause(prompt, "domain")
    assert "serial" in omitted
    assert "Objects may" not in omitted


def test_affinity_scoring_maps_answer_to_semantic_side(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    raw = tmp_path / "raw.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "pair_id": "p",
                "axis": "frame",
                "contrast": ["K", "T"],
                "prompt_hash": "h",
                "label_a": False,
                "label_b": True,
                "semantics_a": {"system": "K"},
                "semantics_b": {"system": "T"},
            }
        )
        + "\n"
    )
    raw.write_text(
        json.dumps(
            {
                "pair_id": "p",
                "prompt_hash": "h",
                "model": "test",
                "parsed_answer": True,
            }
        )
        + "\n"
    )
    result = score_affinity(prompts, raw)
    assert result["by_axis"]["frame"]["affinity_counts"] == {"T": 1}
    assert result["by_axis"]["frame"]["added_property_rate"] == 1
    assert result["by_axis"]["frame"]["added_property_ci95"][0] < 1
