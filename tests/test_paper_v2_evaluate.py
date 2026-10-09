import json

import pytest

from paper_v2.evaluate import (
    _finish_reason,
    _has_choices,
    _read_partial,
    evaluate_model,
    parse_answer,
)


@pytest.mark.parametrize(
    ("text", "answer"),
    [("Yes", True), ("no", False), ("YES.", True), ("Because. Answer: No", False)],
)
def test_parse_answer_accepts_restricted_answers(text, answer):
    assert parse_answer(text) == (answer, "ok")


@pytest.mark.parametrize("text", ["I think Yes", "Yes because...", "", None])
def test_parse_answer_rejects_malformed_or_missing(text):
    answer, status = parse_answer(text)
    assert answer is None
    assert status in {"malformed", "missing"}


def test_finish_reason_supports_provider_dicts():
    assert _finish_reason({"choices": [{"finish_reason": "length"}]}) == "length"


def test_provider_failure_has_no_choices():
    class Failure:
        pass

    assert not _has_choices(Failure())
    assert not _has_choices({"error": "rate limit"})
    assert _has_choices({"choices": [{"message": {"content": "Yes"}}]})


def test_partial_records_are_validated_and_resumed(tmp_path):
    path = tmp_path / "responses.partial"
    path.write_text(
        '{"model":"openai/test","pair_id":"p","side":"a","prompt_hash":"h"}\n'
    )
    prompts = [{"pair_id": "p", "side": "a", "prompt_hash": "h"}]
    assert len(_read_partial(path, prompts, "openrouter/openai/test")) == 1


def test_partial_rejects_duplicate_records(tmp_path):
    path = tmp_path / "responses.partial"
    row = '{"model":"openai/test","pair_id":"p","side":"a","prompt_hash":"h"}\n'
    path.write_text(row + row)
    prompts = [{"pair_id": "p", "side": "a", "prompt_hash": "h"}]
    with pytest.raises(ValueError, match="invalid partial"):
        _read_partial(path, prompts, "openrouter/openai/test")


def test_reasoning_parameter_is_forwarded(tmp_path, monkeypatch):
    requests = []

    def fake_complete(prompts, **kwargs):
        requests.append(kwargs)
        return [
            {"choices": [{"message": {"content": "Yes"}}], "model": "test"}
            for _ in prompts
        ]

    monkeypatch.setattr("paper_v2.evaluate.complete", fake_complete)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {"pair_id": "p", "side": "a", "prompt_hash": "h", "prompt": "Q"}
        )
        + "\n"
    )
    parameters = {
        "temperature": 0,
        "max_tokens": 8,
        "batch_size": 1,
        "max_concurrency": 1,
        "reasoning": {"enabled": False},
        "timeout_seconds": 60,
    }

    evaluate_model(
        {"model": "openrouter/anthropic/test"},
        prompts,
        tmp_path / "raw.jsonl",
        parameters,
        profile="instant",
    )

    assert requests[0]["reasoning"] == {"enabled": False}


def test_limit_selects_deterministic_prompt_prefix(tmp_path, monkeypatch):
    batches = []

    def fake_complete(prompts, **kwargs):
        batches.append(prompts)
        return [
            {"choices": [{"message": {"content": "Yes"}}], "model": "test"}
            for _ in prompts
        ]

    monkeypatch.setattr("paper_v2.evaluate.complete", fake_complete)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps(
                {"pair_id": str(i), "side": "a", "prompt_hash": str(i), "prompt": str(i)}
            )
            + "\n"
            for i in range(3)
        )
    )
    parameters = {
        "temperature": 0,
        "max_tokens": 8,
        "batch_size": 2,
        "max_concurrency": 1,
        "timeout_seconds": 60,
    }

    evaluate_model(
        {"model": "openrouter/anthropic/test"},
        prompts,
        tmp_path / "raw.jsonl",
        parameters,
        limit=2,
    )

    assert batches == [["0", "1"]]


def test_parse_lenient_accepts_formatted_answers_only():
    from paper_v2.evaluate import parse_lenient

    assert parse_lenient("**Yes**") == (True, "ok")
    assert parse_lenient("No\n\nBecause the frame is symmetric.") == (False, "ok")
    assert parse_lenient("Working...\nAnswer: yes") == (True, "ok")
    assert parse_lenient(r"So the implication fails. \boxed{\text{No}}") == (False, "ok")
    assert parse_lenient("The implication fails at w.") == (None, "malformed")
    assert parse_lenient(None) == (None, "missing")
