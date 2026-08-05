import json

from paper_v2.subset import build_stratified_subset


def test_subset_is_balanced_by_axis(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    rows = [
        {"pair_id": f"{axis}-{index}", "axis": axis}
        for axis in ("frame", "domain")
        for index in range(3)
    ]
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prompts.write_text(
        "".join(
            json.dumps({"pair_id": row["pair_id"], "side": side}) + "\n"
            for row in rows
            for side in ("a", "b")
        )
    )
    output = tmp_path / "subset"
    build_stratified_subset(pairs, prompts, output, per_axis=2)
    selected = [
        json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()
    ]
    assert [row["axis"] for row in selected].count("frame") == 2
    assert [row["axis"] for row in selected].count("domain") == 2
