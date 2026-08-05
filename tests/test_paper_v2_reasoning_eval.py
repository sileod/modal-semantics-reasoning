import json

from paper_v2.reasoning_eval import _prompt_variant, _raw_summary


BASE = {
    "pair_id": "frame-test",
    "side": "named_a",
    "source_side": "a",
    "surface": "named",
    "prompt_hash": "old",
    "prompt": "Question: Is it valid?\nAnswer only YES or NO.",
    "label": True,
}


def test_direct_prompt_uses_title_case_labels():
    prompt = _prompt_variant(BASE, rationale=False)
    assert prompt["side"] == "a"
    assert prompt["prompt"].endswith("Answer only Yes or No.")


def test_rationale_prompt_requires_machine_parseable_final_line():
    prompt = _prompt_variant(BASE, rationale=True)
    assert "Reason step by step" in prompt["prompt"]
    assert "at most five sentences" in prompt["prompt"]
    assert prompt["prompt"].endswith("`Answer: Yes` or `Answer: No`.")


def test_raw_summary_is_limited_to_compared_pairs(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "pair_id": pair_id,
            "raw_response": "Yes",
            "request_parameters": {},
            "provider_model": "test",
            "response_cost_usd": 0.01,
        }
        for pair_id in ("keep", "exclude")
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = _raw_summary(raw, {"keep"})

    assert summary["responses"] == 1
    assert summary["cost_usd"] == 0.01
