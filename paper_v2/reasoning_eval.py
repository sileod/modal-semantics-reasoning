from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .evaluate import parse_answer


def build(
    pairs_path: str | Path,
    surface_prompts_path: str | Path,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reasoning mini-set: {output}")

    named = [
        row for row in _read_jsonl(surface_prompts_path) if row["surface"] == "named"
    ]
    pair_ids = {row["pair_id"] for row in named}
    pairs = [row for row in _read_jsonl(pairs_path) if row["pair_id"] in pair_ids]
    direct = [_prompt_variant(row, rationale=False) for row in named]
    rationale = [_prompt_variant(row, rationale=True) for row in named]

    output.mkdir(parents=True)
    paths = {
        "pairs": output / "pairs.jsonl",
        "direct": output / "direct_prompts.jsonl",
        "rationale": output / "rationale_prompts.jsonl",
    }
    _write_jsonl(paths["pairs"], sorted(pairs, key=lambda row: row["pair_id"]))
    _write_jsonl(paths["direct"], direct)
    _write_jsonl(paths["rationale"], rationale)
    manifest = {
        "version": output.name,
        "source_pairs": str(pairs_path),
        "source_prompts": str(surface_prompts_path),
        "pairs": len(pairs),
        "prompts_per_condition": len(direct),
        "counts_by_contrast": {
            contrast: sum("/".join(row["contrast"]) == contrast for row in pairs)
            for contrast in sorted({"/".join(row["contrast"]) for row in pairs})
        },
        "sha256": {name: _sha256(path) for name, path in paths.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def compare(
    direct_score_path: str | Path,
    enhanced_score_path: str | Path,
    direct_raw_path: str | Path,
    enhanced_raw_path: str | Path,
    enhanced_label: str,
    output_path: str | Path,
    samples: int = 10_000,
    seed: int = 20260717,
) -> None:
    direct = json.loads(Path(direct_score_path).read_text())
    enhanced = json.loads(Path(enhanced_score_path).read_text())
    direct_pairs = {row["pair_id"]: row for row in direct["pairs"]}
    enhanced_pairs = {row["pair_id"]: row for row in enhanced["pairs"]}
    if (
        direct_pairs.keys() != enhanced_pairs.keys()
        or direct["model"] != enhanced["model"]
    ):
        raise ValueError("reasoning comparison requires matched pairs and model")

    deltas = []
    ids = sorted(direct_pairs)
    rng = random.Random(seed)
    for _ in range(samples):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        deltas.append(
            sum(
                enhanced_pairs[pair_id]["pair_correct"]
                - direct_pairs[pair_id]["pair_correct"]
                for pair_id in sample
            )
            / len(sample)
        )
    deltas.sort()
    raw = {
        "direct": _raw_summary(direct_raw_path, set(ids)),
        enhanced_label: _raw_summary(enhanced_raw_path, set(ids)),
    }
    result = {
        "model": direct["model"],
        "pair_count": len(ids),
        "axes": sorted({row["axis"] for row in direct_pairs.values()}),
        "conditions": {
            "direct": direct["metrics"],
            enhanced_label: enhanced["metrics"],
        },
        "strict_difference": (
            enhanced["metrics"]["strict_flip_accuracy"]
            - direct["metrics"]["strict_flip_accuracy"]
        ),
        "strict_difference_ci95": [
            deltas[int(0.025 * samples)],
            deltas[int(0.975 * samples)],
        ],
        "raw": raw,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _prompt_variant(row: dict, rationale: bool) -> dict:
    prompt = row["prompt"].replace("Answer only YES or NO.", "Answer only Yes or No.")
    if rationale:
        prompt = prompt.replace(
            "Answer only Yes or No.",
            "Reason step by step about how the stated semantic conditions affect "
            "the inference. In at most five sentences, without headings or "
            "restating the problem, give a concise derivation and end with exactly "
            "`Answer: Yes` or `Answer: No`.",
        )
    result = {
        key: value
        for key, value in row.items()
        if key not in {"surface", "source_side", "prompt_hash", "side"}
    }
    result.update(
        {
            "side": row["source_side"],
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "condition": "rationale" if rationale else "direct",
        }
    )
    return result


def _raw_summary(path: str | Path, pair_ids: set[str] | None = None) -> dict:
    rows = [
        row
        for row in _read_jsonl(path)
        if pair_ids is None or row["pair_id"] in pair_ids
    ]
    reasoning = [row for row in rows if row.get("reasoning_content")]
    return {
        "file": str(path),
        "sha256": _sha256(Path(path)),
        "responses": len(rows),
        "parsed": sum(parse_answer(row.get("raw_response"))[1] == "ok" for row in rows),
        "reasoning_responses": len(reasoning),
        "reasoning_tokens": sum(
            ((row.get("usage") or {}).get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            )
            or 0
            for row in rows
        ),
        "cost_usd": sum(row.get("response_cost_usd") or 0 for row in rows),
        "provider_models": sorted({row.get("provider_model") for row in rows}),
        "request_parameters": rows[0]["request_parameters"],
    }


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--pairs", required=True)
    builder.add_argument("--surface-prompts", required=True)
    builder.add_argument("--out", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--direct-score", required=True)
    comparison.add_argument("--enhanced-score", required=True)
    comparison.add_argument("--direct-raw", required=True)
    comparison.add_argument("--enhanced-raw", required=True)
    comparison.add_argument("--enhanced-label", required=True)
    comparison.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.pairs, args.surface_prompts, args.out)
    else:
        compare(
            args.direct_score,
            args.enhanced_score,
            args.direct_raw,
            args.enhanced_raw,
            args.enhanced_label,
            args.out,
        )


if __name__ == "__main__":
    main()
