from __future__ import annotations

import itertools
import json
import random
from collections import Counter, defaultdict


def condition_lookup_baseline(pairs: list[dict]) -> dict:
    by_axis = {axis: _best_lookup(pairs, axis) for axis in ("frame", "domain")}
    return {
        "by_axis": by_axis,
        "mean": sum(row["strict_accuracy"] for row in by_axis.values()) / 2,
        "side_accuracy": sum(row["side_correct"] for row in by_axis.values())
        / (2 * len(pairs)),
    }


def _best_lookup(pairs: list[dict], axis: str) -> dict:
    rows = [pair for pair in pairs if pair["axis"] == axis]
    values = sorted(
        {
            _semantic_value(pair[f"semantics_{side}"], axis)
            for pair in rows
            for side in ("a", "b")
        }
    )
    candidates = []
    for labels in itertools.product((False, True), repeat=len(values)):
        lookup = dict(zip(values, labels))
        pair_correct = side_correct = 0
        for pair in rows:
            correct = []
            for side in ("a", "b"):
                value = _semantic_value(pair[f"semantics_{side}"], axis)
                result = lookup[value] == pair[f"label_{side}"]
                correct.append(result)
                side_correct += result
            pair_correct += all(correct)
        candidates.append((pair_correct, side_correct, lookup))
    pair_correct, side_correct, lookup = max(candidates, key=lambda row: row[:2])
    return {
        "pairs": len(rows),
        "pair_correct": pair_correct,
        "strict_accuracy": pair_correct / len(rows),
        "side_correct": side_correct,
        "side_accuracy": side_correct / (2 * len(rows)),
        "lookup": lookup,
    }


def condition_balanced_accuracy(
    score_rows: list[dict], pairs_by_id: dict[str, dict], condition: str
) -> dict:
    outcomes = {True: [], False: []}
    for row in score_rows:
        pair = pairs_by_id[row["pair_id"]]
        for side in ("a", "b"):
            axis = pair["axis"]
            value = _semantic_value(pair[f"semantics_{side}"], axis)
            if value == condition:
                label = pair[f"label_{side}"]
                outcomes[label].append(row[f"side_{side}_correct"])
    accuracy = {
        label: sum(values) / len(values) if values else None
        for label, values in outcomes.items()
    }
    return {
        "true_accuracy": accuracy[True],
        "false_accuracy": accuracy[False],
        "balanced_accuracy": sum(accuracy.values()) / 2,
        "true_count": len(outcomes[True]),
        "false_count": len(outcomes[False]),
    }


def schema_summary(pairs: list[dict]) -> dict[str, dict]:
    output = {}
    for contrast in sorted({"/".join(pair["contrast"]) for pair in pairs}):
        rows = [pair for pair in pairs if "/".join(pair["contrast"]) == contrast]
        skeletons = Counter(skeleton_key(pair) for pair in rows)
        output[contrast] = {
            "pairs": len(rows),
            "generator_families": len(
                {
                    (pair["metadata"]["mode"], pair["conjecture"]["ast"]["schema"])
                    for pair in rows
                }
            ),
            "skeletons": len(skeletons),
            "premise_skeletons": len(
                {
                    skeleton_key(pair)
                    for pair in rows
                    if pair["metadata"]["mode"] == "nli"
                }
            ),
            "largest_cluster": max(skeletons.values()),
        }
    return output


def cluster_intervals(
    score_rows: list[dict],
    pairs_by_id: dict[str, dict],
    samples: int = 10_000,
    seed: int = 20260716,
) -> dict:
    clusters = defaultdict(list)
    for row in score_rows:
        pair = pairs_by_id[row["pair_id"]]
        clusters[(pair["axis"], skeleton_key(pair))].append(row["pair_correct"])
    by_axis = {
        axis: [
            (sum(values), len(values))
            for (name, _), values in clusters.items()
            if name == axis
        ]
        for axis in ("frame", "domain")
    }
    rng = random.Random(seed)
    estimates = {"frame": [], "domain": [], "mean": []}
    for _ in range(samples):
        axis_scores = {}
        for axis, groups in by_axis.items():
            selected = [groups[rng.randrange(len(groups))] for _ in groups]
            axis_scores[axis] = sum(correct for correct, _ in selected) / sum(
                count for _, count in selected
            )
            estimates[axis].append(axis_scores[axis])
        estimates["mean"].append(sum(axis_scores.values()) / 2)
    return {name: _interval(values) for name, values in estimates.items()}


def skeleton_key(pair: dict) -> str:
    ast = pair["conjecture"]["ast"]
    value = [
        pair["axis"],
        pair["contrast"],
        pair["metadata"]["mode"],
        ast["schema"],
        _placeholder_atoms(ast["content"]),
    ]
    return json.dumps(value, separators=(",", ":"))


def _placeholder_atoms(value):
    if isinstance(value, list):
        if len(value) == 2 and value[0] == "atom":
            return ["atom", "_"]
        return [_placeholder_atoms(child) for child in value]
    return value


def _semantic_value(semantics: dict, axis: str) -> str:
    return semantics["system"] if axis == "frame" else semantics["domain"]


def _interval(values: list[float]) -> list[float]:
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values))]]
