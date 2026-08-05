from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_stratified_subset(
    pairs_path: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    per_axis: int,
    formula_sensitive: bool = False,
) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite subset: {output}")
    pairs = _read_jsonl(pairs_path)
    selected = []
    for axis in ("frame", "domain"):
        axis_pairs = [pair for pair in pairs if pair["axis"] == axis]
        if len(axis_pairs) < per_axis:
            raise ValueError(f"not enough {axis} pairs for subset")
        if formula_sensitive:
            groups = {}
            for pair in axis_pairs:
                key = (
                    pair["metadata"]["formula_valid_under"],
                    pair["metadata"]["mode"],
                )
                groups.setdefault(key, []).append(pair)
            if per_axis % len(groups):
                raise ValueError("per-axis count must divide formula-sensitive strata")
            quota = per_axis // len(groups)
            for key in sorted(groups):
                rows = sorted(groups[key], key=lambda row: _hash(row["pair_id"]))
                selected.extend(rows[:quota])
        else:
            selected.extend(axis_pairs[:per_axis])
    pair_ids = {pair["pair_id"] for pair in selected}
    prompts = [
        prompt for prompt in _read_jsonl(prompts_path)
        if prompt["pair_id"] in pair_ids
    ]
    if len(prompts) != 2 * len(selected):
        raise ValueError("subset prompt count does not match pair count")

    output.mkdir(parents=True)
    _write_jsonl(output / "pairs.jsonl", selected)
    _write_jsonl(output / "prompts.jsonl", prompts)
    manifest = {
        "selection": (
            "lowest SHA-256(pair_id) per axis/orientation/mode stratum"
            if formula_sensitive
            else "first pairs in frozen order within each axis"
        ),
        "per_axis": per_axis,
        "pair_count": len(selected),
        "source_pairs_sha256": _sha256(pairs_path),
        "source_prompts_sha256": _sha256(prompts_path),
        "pairs_sha256": _sha256(output / "pairs.jsonl"),
        "prompts_sha256": _sha256(output / "prompts.jsonl"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def seed_partial(
    prompts_path: str | Path,
    source_responses: str | Path,
    output_partial: str | Path,
) -> None:
    output = Path(output_partial)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite partial responses: {output}")
    prompts = {
        (row["pair_id"], row["side"], row["prompt_hash"])
        for row in _read_jsonl(prompts_path)
    }
    responses = [
        row for row in _read_jsonl(source_responses)
        if (row["pair_id"], row["side"], row["prompt_hash"]) in prompts
    ]
    _write_jsonl(output, responses)


def _read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--pairs", required=True)
    build.add_argument("--prompts", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--per-axis", type=int, required=True)
    build.add_argument("--formula-sensitive", action="store_true")
    seed = subparsers.add_parser("seed-partial")
    seed.add_argument("--prompts", required=True)
    seed.add_argument("--responses", required=True)
    seed.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.command == "build":
        build_stratified_subset(
            args.pairs,
            args.prompts,
            args.out,
            args.per_axis,
            args.formula_sensitive,
        )
    else:
        seed_partial(args.prompts, args.responses, args.out)


if __name__ == "__main__":
    main()
