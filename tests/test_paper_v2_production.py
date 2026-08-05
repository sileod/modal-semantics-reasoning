import hashlib
import json

import pytest

from paper_v2.production import audit, build


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_production_prompts_are_normalized_and_partitioned(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "production"
    _write_jsonl(
        pairs,
        [
            {"pair_id": "frame-1", "axis": "frame"},
            {"pair_id": "domain-1", "axis": "domain"},
        ],
    )
    _write_jsonl(
        prompts,
        [
            {
                "pair_id": pair_id,
                "side": "a",
                "prompt": "Question\nAnswer only YES or NO.",
                "prompt_hash": "old",
            }
            for pair_id in ("frame-1", "domain-1")
        ],
    )

    build(pairs, prompts, output)

    rows = [json.loads(line) for line in (output / "prompts.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["prompt"].endswith("Answer only Yes or No.")
    assert rows[0]["prompt_hash"] == hashlib.sha256(rows[0]["prompt"].encode()).hexdigest()
    assert len((output / "frame_prompts.jsonl").read_text().splitlines()) == 1
    assert len((output / "domain_prompts.jsonl").read_text().splitlines()) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["pairs_by_axis"] == {"domain": 1, "frame": 1}
    with pytest.raises(FileExistsError):
        build(pairs, prompts, output)


def test_production_audit_hashes_frozen_responses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.yaml").write_text("models: []\n")
    raw = tmp_path / "production-1.1"
    raw.mkdir()
    response = {
        "model": "test",
        "evaluation_profile": "direct",
        "request_parameters": {"temperature": 0},
        "parse_status": "ok",
        "response_cost_usd": 0.01,
        "provider_model": "test-v1",
    }
    _write_jsonl(raw / "test.jsonl", [response])

    output = tmp_path / "manifest.json"
    audit(raw, output)
    manifest = json.loads(output.read_text())

    assert manifest["responses"] == 1
    assert manifest["cost_usd"] == 0.01
    assert manifest["files"][0]["sha256"] == hashlib.sha256(
        (raw / "test.jsonl").read_bytes()
    ).hexdigest()
