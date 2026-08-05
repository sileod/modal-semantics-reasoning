from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .statistics import wilson_interval


def build_affinity_prompts(pairs_path: str | Path, output_path: str | Path) -> None:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite affinity prompts: {output}")

    records = []
    for pair in _read_jsonl(pairs_path):
        prompt_a = _omit_axis_clause(
            pair["controlled_english"]["prompt_a"], pair["axis"]
        )
        prompt_b = _omit_axis_clause(
            pair["controlled_english"]["prompt_b"], pair["axis"]
        )
        if prompt_a != prompt_b:
            raise ValueError(f"omitted prompts differ for {pair['pair_id']}")
        records.append(
            {
                "pair_id": pair["pair_id"],
                "side": "omitted",
                "axis": pair["axis"],
                "contrast": pair["contrast"],
                "prompt": prompt_a,
                "prompt_hash": hashlib.sha256(prompt_a.encode()).hexdigest(),
                "label_a": pair["label_a"],
                "label_b": pair["label_b"],
                "semantics_a": pair["semantics_a"],
                "semantics_b": pair["semantics_b"],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "source_pairs": str(pairs_path),
        "source_sha256": hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest(),
        "prompt_count": len(records),
        "prompts_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_name("semantic_affinity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"froze {len(records)} semantic-affinity prompts at {output}")


def score_affinity(prompts_path: str | Path, raw_path: str | Path) -> dict:
    prompts = {row["pair_id"]: row for row in _read_jsonl(prompts_path)}
    responses = _read_jsonl(raw_path)
    rows = []
    for response in responses:
        prompt = prompts[response["pair_id"]]
        if response["prompt_hash"] != prompt["prompt_hash"]:
            raise ValueError(f"prompt mismatch: {response['pair_id']}")
        answer = response["parsed_answer"]
        match = None if answer is None else ("a" if answer == prompt["label_a"] else "b")
        rows.append(
            {
                "pair_id": response["pair_id"],
                "axis": prompt["axis"],
                "contrast": prompt["contrast"],
                "match": match,
                "value_a": _axis_value(prompt["axis"], prompt["semantics_a"]),
                "value_b": _axis_value(prompt["axis"], prompt["semantics_b"]),
            }
        )
    return {
        "model": responses[0]["model"],
        "count": len(rows),
        "parse_rate": sum(row["match"] is not None for row in rows) / len(rows),
        "parse_rate_ci95": wilson_interval(
            sum(row["match"] is not None for row in rows), len(rows)
        ),
        "by_axis": _summaries(rows, lambda row: row["axis"]),
        "by_contrast": _summaries(rows, lambda row: "/".join(row["contrast"])),
        "rows": rows,
    }


def _omit_axis_clause(prompt: str, axis: str) -> str:
    lines = prompt.splitlines()
    if axis == "frame":
        lines = [
            line
            for line in lines
            if not line.startswith("- The accessibility relation")
        ]
    elif axis == "domain":
        lines = [
            line
            for line in lines
            if not (
                line.startswith("- Objects")
                or line.startswith("- New objects")
                or line.startswith("- Exactly the same objects")
            )
        ]
    else:
        raise ValueError(f"unsupported affinity axis: {axis}")

    if "Semantic specification:" in lines:
        index = lines.index("Semantic specification:")
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if not next_line.startswith("- "):
            del lines[index]
            if index < len(lines) and lines[index] == "":
                del lines[index]
    text = "\n".join(lines)
    replacement = (
        "when no frame condition is specified"
        if axis == "frame"
        else "when domain behavior is unspecified"
    )
    return text.replace("under this semantic specification", replacement)


def _summaries(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    output = {}
    for name, group in sorted(groups.items()):
        parsed = [row for row in group if row["match"] is not None]
        counts: dict[str, int] = {}
        for row in parsed:
            value = row[f"value_{row['match']}"]
            counts[value] = counts.get(value, 0) + 1
        output[name] = {
            "count": len(group),
            "parsed": len(parsed),
            "parse_rate": len(parsed) / len(group),
            "parse_rate_ci95": wilson_interval(len(parsed), len(group)),
            "added_property_rate": (
                sum(row["match"] == "b" for row in parsed) / len(parsed)
                if parsed
                else None
            ),
            "added_property_ci95": wilson_interval(
                sum(row["match"] == "b" for row in parsed), len(parsed)
            ),
            "affinity_counts": counts,
        }
    return output


def _axis_value(axis: str, semantics: dict) -> str:
    return semantics["system"] if axis == "frame" else semantics["domain"]


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument(
        "--pairs", default="data/paper_v2/frozen/pilot-0.2/pairs.jsonl"
    )
    build.add_argument(
        "--out",
        default="data/paper_v2/derived/pilot-0.2/semantic_affinity_prompts.jsonl",
    )
    scorer = subparsers.add_parser("score")
    scorer.add_argument("--prompts", required=True)
    scorer.add_argument("--raw", required=True)
    scorer.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.command == "build":
        build_affinity_prompts(args.pairs, args.out)
    else:
        result = score_affinity(args.prompts, args.raw)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"scored semantic affinity for {result['model']}")


if __name__ == "__main__":
    main()
