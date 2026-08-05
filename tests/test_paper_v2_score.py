import json

from paper_v2.score import score


def test_scoring_handles_malformed_and_missing_answers(tmp_path):
    pairs = [
        {
            "pair_id": "p",
            "axis": "frame",
            "contrast": ["K", "T"],
            "label_a": False,
            "label_b": True,
            "premise_status": "conjecture_only",
            "metadata": {"mode": "validity"},
        }
    ]
    prompts = [
        {"pair_id": "p", "side": "a", "prompt_hash": "a"},
        {"pair_id": "p", "side": "b", "prompt_hash": "b"},
    ]
    raw = [
        {
            "pair_id": "p",
            "side": "a",
            "prompt_hash": "a",
            "model": "test",
            "parsed_answer": None,
            "parse_status": "malformed",
        },
        {
            "pair_id": "p",
            "side": "b",
            "prompt_hash": "b",
            "model": "test",
            "parsed_answer": True,
            "parse_status": "ok",
        },
    ]
    paths = []
    for name, rows in (("pairs", pairs), ("prompts", prompts), ("raw", raw)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
    result = score(*paths)
    assert result["metrics"]["strict_flip_accuracy"] == 0
    assert result["metrics"]["side_accuracy"] == 0.5
    assert result["metrics"]["answer_change_rate"] is None
    assert result["metrics"]["pair_outcomes"]["incorrect_correct"] == 1
    assert result["metrics"]["strict_flip_accuracy_parsed"] is None


def test_scoring_can_use_a_prompt_selected_pair_subset(tmp_path):
    pair = {
        "axis": "frame",
        "contrast": ["K", "T"],
        "label_a": False,
        "label_b": True,
        "premise_status": "conjecture_only",
        "metadata": {"mode": "validity"},
    }
    pairs = [
        {**pair, "pair_id": pair_id}
        for pair_id in ("selected-b", "unused", "selected-a")
    ]
    prompts = [
        {"pair_id": pair_id, "side": side, "prompt_hash": side}
        for pair_id in ("selected-a", "selected-b")
        for side in "ab"
    ]
    raw = [
        {
            "pair_id": pair_id,
            "side": side,
            "prompt_hash": side,
            "model": "test",
            "parsed_answer": answer,
            "parse_status": "ok",
        }
        for pair_id in ("selected-a", "selected-b")
        for side, answer in (("a", False), ("b", True))
    ]
    raw.append(
        {
            "pair_id": "unused",
            "side": "a",
            "prompt_hash": "a",
            "model": "test",
            "parsed_answer": False,
            "parse_status": "ok",
        }
    )
    paths = []
    for name, rows in (("pairs", pairs), ("prompts", prompts), ("raw", raw)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)

    result = score(*paths)

    assert result["pair_count"] == 2
    assert [row["pair_id"] for row in result["pairs"]] == ["selected-b", "selected-a"]
    assert result["metrics"]["strict_flip_accuracy"] == 1
