from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

from .evaluate import parse_answer
from .statistics import wilson_interval


def score(
    pairs_path: str | Path,
    prompts_path: str | Path,
    raw_path: str | Path,
) -> dict[str, Any]:
    all_pairs = {row["pair_id"]: row for row in _read_jsonl(pairs_path)}
    prompts = {(row["pair_id"], row["side"]): row for row in _read_jsonl(prompts_path)}
    prompt_ids = {pair_id for pair_id, _ in prompts}
    if not prompt_ids <= all_pairs.keys():
        raise ValueError("prompts contain an unknown pair")
    pairs = {
        pair_id: pair for pair_id, pair in all_pairs.items() if pair_id in prompt_ids
    }
    raw = [
        row
        for row in _read_jsonl(raw_path)
        if (row["pair_id"], row["side"]) in prompts
    ]
    if len(prompts) != len(pairs) * 2 or len(raw) != len(prompts):
        raise ValueError("prompts or responses do not contain two sides per pair")

    sides: dict[tuple[str, str], dict] = {}
    for response in raw:
        key = (response["pair_id"], response["side"])
        if key not in prompts or response["prompt_hash"] != prompts[key]["prompt_hash"]:
            raise ValueError(f"prompt mismatch: {key}")
        pair = pairs[response["pair_id"]]
        gold = pair[f"label_{response['side']}"]
        if "raw_response" in response:
            parsed_answer, parse_status = parse_answer(response["raw_response"])
        else:
            parsed_answer = response.get("parsed_answer")
            parse_status = response.get("parse_status", "missing")
        sides[key] = {
            **response,
            "parsed_answer": parsed_answer,
            "parse_status": parse_status,
            "gold": gold,
            "correct": parsed_answer == gold,
        }

    pair_rows = []
    for pair_id, pair in pairs.items():
        a, b = sides[(pair_id, "a")], sides[(pair_id, "b")]
        outcome = (
            "correct_correct"
            if a["correct"] and b["correct"]
            else (
                "correct_incorrect"
                if a["correct"]
                else "incorrect_correct" if b["correct"] else "incorrect_incorrect"
            )
        )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "axis": pair["axis"],
                "contrast": pair["contrast"],
                "mode": pair["metadata"]["mode"],
                "premise_status": pair["premise_status"],
                "side_a_correct": a["correct"],
                "side_b_correct": b["correct"],
                "pair_correct": a["correct"] and b["correct"],
                "parse_ok": a["parse_status"] == "ok" and b["parse_status"] == "ok",
                "answer_changed": (
                    a["parsed_answer"] != b["parsed_answer"]
                    if a["parsed_answer"] is not None and b["parsed_answer"] is not None
                    else None
                ),
                "outcome": outcome,
            }
        )

    model = raw[0]["model"] if raw else "unknown"
    return {
        "model": model,
        "raw_file": str(raw_path),
        "pair_count": len(pair_rows),
        "side_count": len(sides),
        "metrics": _metrics(pair_rows),
        "by_axis": _group_metrics(pair_rows, lambda row: row["axis"]),
        "by_mode": _group_metrics(pair_rows, lambda row: row["mode"]),
        "by_contrast": _group_metrics(pair_rows, lambda row: "/".join(row["contrast"])),
        "pairs": pair_rows,
    }


def _metrics(rows: list[dict]) -> dict[str, Any]:
    strict_correct = sum(row["pair_correct"] for row in rows)
    side_correct = sum(row["side_a_correct"] + row["side_b_correct"] for row in rows)
    strict = strict_correct / len(rows)
    side = side_correct / (2 * len(rows))
    parse = sum(row["parse_ok"] for row in rows) / len(rows)
    parsed_rows = [row for row in rows if row["parse_ok"]]
    parsed_correct = sum(row["pair_correct"] for row in parsed_rows)
    parsed_pairs = [row for row in rows if row["answer_changed"] is not None]
    changed = sum(row["answer_changed"] for row in parsed_pairs)
    changed_correct = sum(
        row["pair_correct"] for row in parsed_pairs if row["answer_changed"]
    )
    outcomes = {
        name: sum(row["outcome"] == name for row in rows)
        for name in (
            "correct_correct",
            "correct_incorrect",
            "incorrect_correct",
            "incorrect_incorrect",
        )
    }
    return {
        "strict_flip_accuracy": strict,
        "strict_flip_ci95": wilson_interval(strict_correct, len(rows)),
        "side_accuracy": side,
        "side_accuracy_ci95": wilson_interval(side_correct, 2 * len(rows)),
        "pair_parse_rate": parse,
        "pair_parse_rate_ci95": wilson_interval(
            sum(row["parse_ok"] for row in rows), len(rows)
        ),
        "strict_flip_accuracy_parsed": (
            parsed_correct / len(parsed_rows) if parsed_rows else None
        ),
        "strict_flip_parsed_ci95": wilson_interval(parsed_correct, len(parsed_rows)),
        "answer_change_rate": changed / len(parsed_pairs) if parsed_pairs else None,
        "answer_change_ci95": wilson_interval(changed, len(parsed_pairs)),
        "strict_given_change": changed_correct / changed if changed else None,
        "strict_given_change_ci95": wilson_interval(changed_correct, changed),
        "pair_outcomes": outcomes,
    }


def _group_metrics(
    rows: list[dict], key: Callable[[dict], str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {name: _metrics(group) for name, group in sorted(groups.items())}


def _bootstrap(
    rows: list[dict],
    statistic: Callable[[list[dict]], float],
    samples: int = 10_000,
    seed: int = 20260716,
) -> list[float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        estimates.append(statistic(sample))
    estimates.sort()
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]]


def _strict(rows: list[dict]) -> float:
    return sum(row["pair_correct"] for row in rows) / len(rows)


def _side(rows: list[dict]) -> float:
    return sum(row["side_a_correct"] + row["side_b_correct"] for row in rows) / (
        2 * len(rows)
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--pairs", default="data/paper_v2/frozen/pilot-0.2/pairs.jsonl")
    parser.add_argument(
        "--prompts", default="data/paper_v2/frozen/pilot-0.2/prompts.jsonl"
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    result = score(args.pairs, args.prompts, args.raw)
    output = Path(
        args.out or args.raw.replace("/raw/", "/scored/").replace(".jsonl", ".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{result['model']}: strict={result['metrics']['strict_flip_accuracy']:.3f}, "
        f"side={result['metrics']['side_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
