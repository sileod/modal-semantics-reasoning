from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .statistics import wilson_interval


DEFINITIONS = {
    "serial": "Every world accesses at least one world.",
    "reflexive": "Every world accesses itself.",
    "symmetric": (
        "Whenever one world accesses a second world, the second world also "
        "accesses the first."
    ),
    "transitive": (
        "Whenever one world accesses a second world and that second world "
        "accesses a third, the first world also accesses the third."
    ),
}
SURFACES = ("named", "defined", "symbolic")


def build_prompts(
    pairs_path: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    per_contrast: int = 10,
) -> None:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite terminology prompts: {output}")
    pairs = _read_jsonl(pairs_path)
    prompts = {
        (row["pair_id"], row["side"]): row for row in _read_jsonl(prompts_path)
    }
    selected = []
    contrasts = sorted({tuple(row["contrast"]) for row in pairs if row["axis"] == "frame"})
    for contrast_index, contrast in enumerate(contrasts):
        cells = [
            [
                row
                for row in pairs
                if row["axis"] == "frame"
                and tuple(row["contrast"]) == contrast
                and row["metadata"]["mode"] == mode
                and row["label_a"] is label_a
            ]
            for mode in ("validity", "nli")
            for label_a in (False, True)
        ]
        for cell in cells:
            cell.sort(key=lambda row: hashlib.sha256(row["pair_id"].encode()).hexdigest())
        quotas = [per_contrast // 4] * 4
        for offset in range(per_contrast % 4):
            quotas[(contrast_index + offset) % 4] += 1
        selected.extend(row for cell, quota in zip(cells, quotas) for row in cell[:quota])

    records = []
    for pair in selected:
        for side in ("a", "b"):
            base = prompts[(pair["pair_id"], side)]
            for surface in SURFACES:
                prompt = _surface_prompt(base["prompt"], pair, side, surface)
                records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "side": f"{surface}_{side}",
                        "source_side": side,
                        "surface": surface,
                        "axis": "frame",
                        "contrast": pair["contrast"],
                        "label": pair[f"label_{side}"],
                        "prompt": prompt,
                        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                )
    records.sort(key=lambda row: (row["pair_id"], row["surface"], row["source_side"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, records)
    manifest = {
        "source_pairs": str(pairs_path),
        "source_prompts": str(prompts_path),
        "pairs": len(selected),
        "prompts": len(records),
        "per_contrast": per_contrast,
        "surfaces": list(SURFACES),
        "prompts_sha256": _sha256(output),
    }
    output.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def score(
    prompts_path: str | Path,
    raw_path: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 20260716,
) -> dict:
    prompts = {
        (row["pair_id"], row["side"]): row for row in _read_jsonl(prompts_path)
    }
    responses = _read_jsonl(raw_path)
    answers = {}
    for response in responses:
        key = (response["pair_id"], response["side"])
        prompt = prompts[key]
        if response["prompt_hash"] != prompt["prompt_hash"]:
            raise ValueError(f"prompt mismatch: {key}")
        answers[key] = response["parsed_answer"]

    pairs = []
    parse_counts = {surface: 0 for surface in SURFACES}
    for pair_id in sorted({key[0] for key in prompts}):
        prompt = prompts[(pair_id, "named_a")]
        row = {"pair_id": pair_id, "contrast": "/".join(prompt["contrast"])}
        for surface in SURFACES:
            values = []
            for side in ("a", "b"):
                item = prompts[(pair_id, f"{surface}_{side}")]
                answer = answers.get((pair_id, f"{surface}_{side}"))
                parse_counts[surface] += answer is not None
                values.append(answer == item["label"])
            row[surface] = all(values)
        pairs.append(row)

    by_surface = {
        surface: _accuracy([row[surface] for row in pairs])
        for surface in SURFACES
    }
    for surface, metrics in by_surface.items():
        total = 2 * len(pairs)
        parsed = parse_counts[surface]
        metrics.update(
            {
                "parsed_sides": parsed,
                "side_count": total,
                "parse_rate": parsed / total,
                "parse_rate_ci95": wilson_interval(parsed, total),
            }
        )
    differences = {surface: [] for surface in ("defined", "symbolic")}
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        for surface in differences:
            differences[surface].append(
                sum(row[surface] - row["named"] for row in sample) / len(sample)
            )
    for values in differences.values():
        values.sort()
    result = {
        "model": responses[0]["model"],
        "pair_count": len(pairs),
        "by_surface": by_surface,
        "by_contrast": {
            contrast: {
                surface: _accuracy(
                    [row[surface] for row in pairs if row["contrast"] == contrast]
                )
                for surface in SURFACES
            }
            for contrast in sorted({row["contrast"] for row in pairs})
        },
        "pairs": pairs,
    }
    for surface, values in differences.items():
        result[f"{surface}_minus_named"] = (
            by_surface[surface]["accuracy"] - by_surface["named"]["accuracy"]
        )
        result[f"{surface}_minus_named_ci95"] = [
            values[int(0.025 * bootstrap_samples)],
            values[int(0.975 * bootstrap_samples)],
        ]
    return result


def _define_frame_properties(prompt: str, properties: list[str]) -> str:
    lines = prompt.splitlines()
    index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("- The accessibility relation")
    )
    replacements = (
        [f"- {DEFINITIONS[name]}" for name in properties]
        if properties
        else ["- No additional conditions are imposed on which worlds access which."]
    )
    lines[index : index + 1] = replacements
    return "\n".join(lines)


def _surface_prompt(prompt: str, pair: dict, side: str, surface: str) -> str:
    if surface == "named":
        return prompt
    if surface == "defined":
        return _define_frame_properties(
            prompt, pair[f"semantics_{side}"]["frame_properties"]
        )
    return _tptp_prompt(pair, side)


def _tptp_prompt(pair: dict, side: str) -> str:
    semantics = pair[f"semantics_{side}"]
    symbols = pair["conjecture"]["ast"]["symbols"]
    lines = [
        "Consider the following non-classical TPTP problem.",
        "```tptp",
        "tff(logic_setup, logic, $modal == [",
        "  $designation == $rigid,",
        "  $domains == $constant,",
        "  $terms == $local,",
        f"  $modalities == $modal_system_{semantics['system']}",
        "]).",
    ]
    lines.extend(f"tff(type_{symbol}, type, {symbol}: $o)." for symbol in symbols)
    lines.extend(
        f"tff(premise_{index}, axiom-local, {premise['tptp']})."
        for index, premise in enumerate(pair["premises"], 1)
    )
    lines.extend(
        [
            f"tff(conjecture, conjecture, {pair['conjecture']['tptp']}).",
            "```",
            (
                "Does the conjecture follow from the local premises?"
                if pair["premises"]
                else "Is the conjecture valid?"
            ),
            "Answer only Yes or No.",
        ]
    )
    return "\n".join(lines)


def _accuracy(values: list[bool]) -> dict:
    correct = sum(values)
    return {
        "correct": correct,
        "count": len(values),
        "accuracy": correct / len(values),
        "ci95": wilson_interval(correct, len(values)),
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
    build = commands.add_parser("build")
    build.add_argument("--pairs", required=True)
    build.add_argument("--prompts", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--per-contrast", type=int, default=10)
    scorer = commands.add_parser("score")
    scorer.add_argument("--prompts", required=True)
    scorer.add_argument("--raw", required=True)
    scorer.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_prompts(args.pairs, args.prompts, args.out, args.per_contrast)
        return
    result = score(args.prompts, args.raw)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
