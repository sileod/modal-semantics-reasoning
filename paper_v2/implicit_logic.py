from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .statistics import wilson_interval


SYSTEMS = ("K", "D", "T", "B", "S4", "S5")
AXIOM_BY_CONTRAST = {
    ("K", "D"): "D",
    ("K", "T"): "T",
    ("T", "B"): "B",
    ("T", "S4"): "4",
    ("B", "S5"): "5",
}
SYSTEM_AXIOMS = {
    "K": frozenset(),
    "D": frozenset({"D"}),
    "T": frozenset({"D", "T"}),
    "B": frozenset({"D", "T", "B"}),
    "S4": frozenset({"D", "T", "4"}),
    "S5": frozenset({"D", "T", "B", "4", "5"}),
}


def logic_prediction(system: str, contrast: list[str] | tuple[str, str]) -> bool:
    return AXIOM_BY_CONTRAST[tuple(contrast)] in SYSTEM_AXIOMS[system]


def build_subset(
    pairs_path: str | Path,
    affinity_path: str | Path,
    output_path: str | Path,
    per_contrast: int = 20,
) -> None:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite implicit-logic prompts: {output}")
    if per_contrast % 4:
        raise ValueError("per_contrast must balance two modes and two query polarities")

    pairs = {row["pair_id"]: row for row in _read_jsonl(pairs_path)}
    affinity = {row["pair_id"]: row for row in _read_jsonl(affinity_path)}
    selected = []
    mode_size = per_contrast // 4
    for contrast in AXIOM_BY_CONTRAST:
        for mode in ("validity", "nli"):
            cell = [
                pair
                for pair in pairs.values()
                if pair["axis"] == "frame"
                and tuple(pair["contrast"]) == contrast
                and pair["metadata"]["mode"] == mode
            ]
            cell.sort(
                key=lambda pair: hashlib.sha256(pair["pair_id"].encode()).hexdigest()
            )
            if len(cell) < mode_size:
                raise ValueError(f"insufficient items for {contrast}/{mode}")
            for pair in cell[:mode_size]:
                base = affinity[pair["pair_id"]]
                selected.extend(_polarity_prompts(base))

    selected.sort(key=lambda row: row["pair_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, selected)
    manifest = {
        "source_pairs": str(pairs_path),
        "source_affinity_prompts": str(affinity_path),
        "per_contrast": per_contrast,
        "prompt_count": len(selected),
        "prompt_sha256": _sha256(output),
        "selection": "lowest SHA-256(pair_id) in each contrast/mode cell",
        "query_polarities": ["valid", "fail"],
    }
    output.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def score(
    prompts_path: str | Path,
    raw_path: str | Path,
    primary_pairs_path: str | Path | None = None,
    primary_raw_path: str | Path | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 20260716,
) -> dict:
    prompts = {row["pair_id"]: row for row in _read_jsonl(prompts_path)}
    responses = _read_jsonl(raw_path)
    answers = {}
    for response in responses:
        prompt = prompts[response["pair_id"]]
        if response["prompt_hash"] != prompt["prompt_hash"]:
            raise ValueError(f"prompt mismatch: {response['pair_id']}")
        answers[response["pair_id"]] = response["parsed_answer"]

    grouped: dict[str, dict[str, tuple[dict, bool | None]]] = {}
    for pair_id, prompt in prompts.items():
        grouped.setdefault(prompt["source_pair_id"], {})[prompt["query_polarity"]] = (
            prompt,
            answers.get(pair_id),
        )
    both_parsed = 0
    observations = []
    for group in grouped.values():
        valid_prompt, valid_answer = group["valid"]
        _, fail_answer = group["fail"]
        if valid_answer is None or fail_answer is None:
            continue
        both_parsed += 1
        if valid_answer == (not fail_answer):
            observations.append((valid_prompt, valid_answer))

    fits = _logic_fits(observations)
    support, balanced_intervals = _balanced_bootstrap(
        observations, bootstrap_samples, seed
    )
    for system in SYSTEMS:
        fits[system]["balanced_agreement_ci95"] = balanced_intervals[system]
    best_score = max(fit["balanced_agreement"] for fit in fits.values())
    best = [
        system for system in SYSTEMS if fits[system]["balanced_agreement"] == best_score
    ]
    parsed_answers = sum(answer is not None for answer in answers.values())
    result = {
        "model": responses[0]["model"],
        "count": len(prompts),
        "source_count": len(grouped),
        "parsed": parsed_answers,
        "parse_rate": parsed_answers / len(prompts),
        "parse_rate_ci95": wilson_interval(parsed_answers, len(prompts)),
        "both_polarities_parsed": both_parsed,
        "polarity_consistent": len(observations),
        "polarity_consistency": (
            len(observations) / both_parsed if both_parsed else None
        ),
        "polarity_consistency_ci95": wilson_interval(len(observations), both_parsed),
        "best_logics": best,
        "best_logic_bootstrap_support": sum(support[system] for system in best),
        "bootstrap_support": support,
        "logic_fits": fits,
        "property_affinity": _property_affinity(observations),
    }
    if primary_pairs_path and primary_raw_path:
        source_ids = {prompt["source_pair_id"] for prompt, _ in observations}
        result["explicit_prediction"] = {
            system: _explicit_prediction(
                system,
                source_ids,
                primary_pairs_path,
                primary_raw_path,
                bootstrap_samples,
                seed,
            )
            for system in best
        }
    return result


def _property_affinity(observations: list[tuple[dict, bool]]) -> dict:
    groups: dict[str, list[bool]] = {}
    for prompt, answer in observations:
        property_name = AXIOM_BY_CONTRAST[tuple(prompt["contrast"])]
        groups.setdefault(property_name, []).append(answer)
    return {
        name: {
            "positive": sum(values),
            "count": len(values),
            "rate": sum(values) / len(values),
            "ci95": wilson_interval(sum(values), len(values)),
        }
        for name, values in sorted(groups.items())
    }


def _logic_fits(parsed: list[tuple[dict, bool]]) -> dict:
    log_evidence = {}
    matches = {}
    for system in SYSTEMS:
        count = sum(
            answer == query_prediction(system, prompt) for prompt, answer in parsed
        )
        matches[system] = count
        log_evidence[system] = _log_evidence(count, len(parsed))
    normalizer = _logsumexp(list(log_evidence.values()))
    return {
        system: {
            "matches": matches[system],
            "count": len(parsed),
            "agreement": matches[system] / len(parsed) if parsed else None,
            "agreement_ci95": wilson_interval(matches[system], len(parsed)),
            "balanced_agreement": _balanced_agreement(system, parsed),
            "posterior_probability": math.exp(log_evidence[system] - normalizer),
        }
        for system in SYSTEMS
    }


def _log_evidence(matches: int, total: int, steps: int = 1000) -> float:
    # Uniform prior over an error rate in [0, .5], integrated numerically.
    logs = []
    mismatches = total - matches
    for index in range(steps + 1):
        error = (index + 0.5) * 0.5 / (steps + 1)
        logs.append(matches * math.log1p(-error) + mismatches * math.log(error))
    return _logsumexp(logs) - math.log(len(logs))


def _balanced_agreement(system: str, parsed: list[tuple[dict, bool]]) -> float:
    groups: dict[tuple[str, str], list[tuple[dict, bool]]] = {}
    for prompt, answer in parsed:
        groups.setdefault(tuple(prompt["contrast"]), []).append((prompt, answer))
    if not groups:
        return 0.0
    return sum(
        sum(answer == query_prediction(system, prompt) for prompt, answer in rows)
        / len(rows)
        for rows in groups.values()
    ) / len(groups)


def _balanced_bootstrap(
    parsed: list[tuple[dict, bool]], samples: int, seed: int
) -> tuple[dict[str, float], dict[str, list[float]]]:
    rng = random.Random(seed)
    counts = dict.fromkeys(SYSTEMS, 0)
    estimates = {system: [] for system in SYSTEMS}
    groups: dict[tuple[str, str], list[tuple[dict, bool]]] = {}
    for row in parsed:
        groups.setdefault(tuple(row[0]["contrast"]), []).append(row)
    for _ in range(samples):
        sample = [
            row
            for group in groups.values()
            for row in (group[rng.randrange(len(group))] for _ in group)
        ]
        scores = {system: _balanced_agreement(system, sample) for system in SYSTEMS}
        for system, value in scores.items():
            estimates[system].append(value)
        maximum = max(scores.values())
        tied = [system for system in SYSTEMS if scores[system] == maximum]
        for system in tied:
            counts[system] += 1 / len(tied)
    for values in estimates.values():
        values.sort()
    return (
        {system: counts[system] / samples for system in SYSTEMS},
        {
            system: [
                values[int(0.025 * samples)],
                values[int(0.975 * samples)],
            ]
            for system, values in estimates.items()
        },
    )


def _explicit_prediction(
    system: str,
    pair_ids: set[str],
    pairs_path: str | Path,
    raw_path: str | Path,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    pairs = {
        row["pair_id"]: row
        for row in _read_jsonl(pairs_path)
        if row["pair_id"] in pair_ids
    }
    responses = {
        (row["pair_id"], row["side"]): row["parsed_answer"]
        for row in _read_jsonl(raw_path)
        if row["pair_id"] in pair_ids
    }
    rows = []
    for pair_id, pair in pairs.items():
        if any((pair_id, side) not in responses for side in ("a", "b")):
            continue
        implicit = logic_prediction(system, pair["contrast"])
        row = {"pair_id": pair_id, "profile_prediction": implicit}
        for side in ("a", "b"):
            label = pair[f"label_{side}"]
            answer = responses.get((pair_id, side))
            row[f"answer_{str(label).lower()}"] = answer
        if {"answer_true", "answer_false"} <= set(row):
            rows.append(row)

    prediction = {
        "pairs": len(rows),
        "accuracy_by_gold": {
            str(label).lower(): _accuracy_by_gold(rows, label)
            for label in (True, False)
        },
        "bias_controlled_profile_effect": {
            str(label).lower(): _profile_effect(
                rows, label, bootstrap_samples, seed + int(label)
            )
            for label in (True, False)
        },
    }
    prediction["profile_effect_identifiable"] = all(
        value["identifiable"]
        for value in prediction["bias_controlled_profile_effect"].values()
    )
    return prediction


def _accuracy_by_gold(rows: list[dict], label: bool) -> dict:
    key = f"answer_{str(label).lower()}"
    correct = sum(row[key] is not None and row[key] == label for row in rows)
    return {
        "correct": correct,
        "count": len(rows),
        "accuracy": correct / len(rows),
        "ci95": wilson_interval(correct, len(rows)),
    }


def _profile_effect(
    rows: list[dict], label: bool, bootstrap_samples: int, seed: int
) -> dict:
    key = f"answer_{str(label).lower()}"
    groups = {
        prediction: [
            row[key] is True
            for row in rows
            if row["profile_prediction"] is prediction and row[key] is not None
        ]
        for prediction in (True, False)
    }
    result = {
        f"profile_{str(prediction).lower()}": {
            "yes": sum(values),
            "count": len(values),
            "yes_rate": sum(values) / len(values) if values else None,
        }
        for prediction, values in groups.items()
    }
    if not all(groups.values()):
        return {**result, "identifiable": False, "delta_yes_rate": None, "ci95": None}

    delta = sum(groups[True]) / len(groups[True]) - sum(groups[False]) / len(
        groups[False]
    )
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        rates = {
            prediction: sum(values[rng.randrange(len(values))] for _ in values)
            / len(values)
            for prediction, values in groups.items()
        }
        estimates.append(rates[True] - rates[False])
    estimates.sort()
    return {
        **result,
        "identifiable": True,
        "delta_yes_rate": delta,
        "ci95": [
            estimates[int(0.025 * bootstrap_samples)],
            estimates[int(0.975 * bootstrap_samples)],
        ],
    }


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def query_prediction(system: str, prompt: dict) -> bool:
    validity = logic_prediction(system, prompt["contrast"])
    return validity if prompt["query_polarity"] == "valid" else not validity


def _polarity_prompts(base: dict) -> list[dict]:
    prompt = base["prompt"].replace("Answer only YES or NO.", "Answer only Yes or No.")
    valid = {
        **base,
        "source_pair_id": base["pair_id"],
        "pair_id": f"{base['pair_id']}-valid",
        "query_polarity": "valid",
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    fail_prompt = _failure_question(prompt)
    fail = {
        **base,
        "source_pair_id": base["pair_id"],
        "pair_id": f"{base['pair_id']}-fail",
        "query_polarity": "fail",
        "prompt": fail_prompt,
        "prompt_hash": hashlib.sha256(fail_prompt.encode()).hexdigest(),
    }
    return [valid, fail]


def _failure_question(prompt: str) -> str:
    replacements = {
        "Is the statement valid when no frame condition is specified?": "Can the statement fail when no frame condition is specified?",
        "Does the conjecture follow from the premises when no frame condition is specified?": "Can the conjecture fail to follow from the premises when no frame condition is specified?",
    }
    for question, replacement in replacements.items():
        if question in prompt:
            return prompt.replace(question, replacement)
    raise ValueError("unrecognized omitted-frame question")


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
    build.add_argument("--affinity-prompts", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--per-contrast", type=int, default=20)
    scorer = commands.add_parser("score")
    scorer.add_argument("--prompts", required=True)
    scorer.add_argument("--raw", required=True)
    scorer.add_argument("--out", required=True)
    scorer.add_argument("--primary-pairs")
    scorer.add_argument("--primary-raw")
    args = parser.parse_args()

    if args.command == "build":
        build_subset(args.pairs, args.affinity_prompts, args.out, args.per_contrast)
        return
    result = score(
        args.prompts,
        args.raw,
        args.primary_pairs,
        args.primary_raw,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
