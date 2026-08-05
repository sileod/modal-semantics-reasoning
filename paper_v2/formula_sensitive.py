from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

from .candidates import Candidate
from .freeze import _make_pair, _prompt_records, _write_jsonl
from .oracle import build_tptp, run_side
from .semantics import Semantics
from .schema import content_hash


FAMILIES = (
    ("frame", ("T", "B"), ("B", "S4"), True),
    ("frame", ("T", "S4"), ("B", "S4"), False),
    ("domain", ("varying", "cumulative"), ("cumulative", "decreasing"), True),
    ("domain", ("varying", "decreasing"), ("cumulative", "decreasing"), False),
)


def freeze(
    source_path: str | Path,
    output_dir: str | Path,
    per_family_mode: int = 20,
    timeout: int = 10,
    workers: int = 8,
) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen diagnostic: {output}")
    source_path = Path(source_path)
    source = _read_jsonl(source_path)
    cells = []
    for axis, source_contrast, contrast, first_valid in FAMILIES:
        for mode in ("validity", "nli"):
            available = [
                row
                for row in source
                if row["axis"] == axis
                and tuple(row["contrast"]) == source_contrast
                and row["metadata"]["mode"] == mode
            ]
            available.sort(key=lambda row: hashlib.sha256(row["pair_id"].encode()).hexdigest())
            if len(available) < per_family_mode:
                raise ValueError(f"insufficient source pairs for {source_contrast}/{mode}")
            cells.append((available, contrast, first_valid, mode))

    cache_path = output.parent / f".{output.name}-oracle-cache.jsonl"
    cache = _load_cache(cache_path)
    accepted, rejected = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for available, contrast, first_valid, mode in cells:
            cursor = 0
            accepted_cell = []
            while len(accepted_cell) < per_family_mode:
                batch = [
                    (_candidate(row, contrast), row, first_valid)
                    for row in available[cursor : cursor + workers]
                ]
                cursor += len(batch)
                if not batch:
                    raise RuntimeError(f"cell underfilled for {contrast}/{mode}/{first_valid}")
                futures = {
                    _cache_key(item[0]): pool.submit(
                        _resolve, item[0], item[2], timeout
                    )
                    for item in batch
                    if _cache_key(item[0]) not in cache
                }
                for item in batch:
                    key = _cache_key(item[0])
                    result = cache.get(key) or futures[key].result()
                    if key not in cache:
                        cache[key] = result
                        _append_cache(cache_path, key, result)
                    if result["accepted"]:
                        labels = (result["oracle_a"]["consensus"], result["oracle_b"]["consensus"])
                        expected = (first_valid, not first_valid)
                        if labels == expected:
                            accepted_cell.append((*item, result))
                            if len(accepted_cell) == per_family_mode:
                                break
                        else:
                            rejected.append(
                                {
                                    "source_pair_id": item[1]["pair_id"],
                                    "reason": "wrong_orientation",
                                    "labels": labels,
                                }
                            )
                    else:
                        rejected.append(
                            {
                                "source_pair_id": item[1]["pair_id"],
                                "reason": result["reason"],
                            }
                        )
            accepted.extend(accepted_cell)

    pairs, oracle, prompts = [], [], []
    for index, (candidate, source_pair, first_valid, result) in enumerate(accepted):
        pair = _make_pair(
            candidate,
            result["oracle_a"],
            result["oracle_b"],
            result["premise_status"],
            20260716,
            swap=index % 2 == 1,
        )
        data = pair.to_dict()
        data["metadata"].update(
            source_pair_id=source_pair["pair_id"],
            source_contrast=source_pair["contrast"],
            formula_valid_under=candidate.contrast[0] if first_valid else candidate.contrast[1],
        )
        pairs.append(data)
        oracle.append(
            {
                "pair_id": pair.pair_id,
                "side_a": result["oracle_b"] if index % 2 else result["oracle_a"],
                "side_b": result["oracle_a"] if index % 2 else result["oracle_b"],
                "premise_ablation": result["ablation"],
            }
        )
        prompts.extend(
            _prompt_records(
                pair.pair_id,
                data["controlled_english"]["prompt_a"],
                data["controlled_english"]["prompt_b"],
            )
        )

    _validate(pairs, per_family_mode)
    output.mkdir(parents=True)
    _write_jsonl(output / "pairs.jsonl", pairs)
    _write_jsonl(output / "oracle.jsonl", oracle)
    _write_jsonl(output / "prompts.jsonl", prompts)
    _write_jsonl(output / "rejected.jsonl", rejected)
    manifest = {
        "benchmark": "paper-v2-formula-sensitive",
        "version": "1.0",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "git_commit": _command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(
            _command("git", "status", "--porcelain", "--untracked-files=no")
        ),
        "selection": "lowest SHA-256(pair_id) per source-contrast/mode cell",
        "pairs_per_family_mode": per_family_mode,
        "pairs": len(pairs),
        "rejected": len(rejected),
        "by_axis": {
            axis: sum(row["axis"] == axis for row in pairs)
            for axis in ("frame", "domain")
        },
        "by_orientation": _counts(pairs, lambda row: row["metadata"]["formula_valid_under"]),
        "by_mode": _counts(pairs, lambda row: row["metadata"]["mode"]),
        "oracle_resolutions": _counts(
            [side for row in oracle for side in (row["side_a"], row["side_b"])],
            lambda row: row["resolution"],
        ),
        "python": platform.python_version(),
        "sha256": {
            name: _sha256(output / name)
            for name in ("pairs.jsonl", "oracle.jsonl", "prompts.jsonl", "rejected.jsonl")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _candidate(source: dict, contrast: tuple[str, str]) -> Candidate:
    axis = source["axis"]
    semantics = (
        (Semantics.make("B"), Semantics.make("S4"))
        if axis == "frame"
        else (
            Semantics.make("D", domain="cumulative"),
            Semantics.make("D", domain="decreasing"),
        )
    )
    english_premises, english_conjecture = _english(source)
    ast = {
        **source["conjecture"]["ast"],
        "source_contrast": source["contrast"],
        "diagnostic_contrast": list(contrast),
    }
    candidate = Candidate(
        axis=axis,
        contrast=contrast,
        premises=tuple(row["tptp"] for row in source["premises"]),
        conjecture=source["conjecture"]["tptp"],
        ast=ast,
        semantics_a=semantics[0],
        semantics_b=semantics[1],
        english_premises=english_premises,
        english_conjecture=english_conjecture,
        modal_depth=source["modal_depth"],
        quantifier_depth=source["quantifier_depth"],
        alternation_pattern=source["alternation_pattern"],
    )
    candidate.validate()
    return candidate


def _resolve(candidate: Candidate, first_valid: bool, timeout: int) -> dict:
    valid_side = "a" if first_valid else "b"
    valid_semantics = candidate.semantics_a if first_valid else candidate.semantics_b
    invalid_semantics = candidate.semantics_b if first_valid else candidate.semantics_a
    proof = run_side(candidate, valid_semantics, timeout)
    countermodel = _countermodel_record(candidate, invalid_semantics)
    if proof["consensus"] is not True or proof["resolution"] == "conflict":
        return {
            "accepted": False,
            "reason": "valid_side_unresolved",
            "oracle_a": proof if valid_side == "a" else countermodel,
            "oracle_b": countermodel if valid_side == "a" else proof,
        }
    oracle_a, oracle_b = (
        (proof, countermodel) if first_valid else (countermodel, proof)
    )
    return {
        "accepted": True,
        "oracle_a": oracle_a,
        "oracle_b": oracle_b,
        "premise_status": "premise_dependent" if candidate.premises else "conjecture_only",
        "ablation": {
            "method": "structural",
            "result": (
                "without the premise, the standalone conjecture is not valid "
                "under both diagnostic conditions"
            ),
        } if candidate.premises else None,
    }


def _countermodel_record(candidate: Candidate, semantics: Semantics) -> dict:
    witness = _countermodel(candidate, semantics)
    if _evaluate_witness(candidate, semantics, witness):
        raise ValueError("purported countermodel satisfies the inference")
    source = build_tptp(candidate, semantics)
    return {
        "prover_input": source,
        "prover_input_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "consensus": False,
        "resolution": "explicit_countermodel",
        "provers": {},
        "countermodel": witness,
        "checker": "paper_v2.formula_sensitive._evaluate_witness",
    }


def _countermodel(candidate: Candidate, semantics: Semantics) -> dict:
    source_contrast = tuple(candidate.ast["source_contrast"])
    symbols = candidate.ast["symbols"]
    assignment = lambda target: _assignment(candidate.ast["content"], symbols, target)
    if candidate.axis == "frame" and source_contrast == ("T", "B"):
        return {
            "worlds": [0, 1],
            "accessibility": [[0, 0], [0, 1], [1, 1]],
            "atom_truth": {"0": assignment(True), "1": assignment(False)},
            "property_check": "reflexive and transitive, not necessarily symmetric",
        }
    if candidate.axis == "frame" and source_contrast == ("T", "S4"):
        return {
            "worlds": [0, 1, 2],
            "accessibility": [
                [0, 0], [1, 1], [2, 2], [0, 1], [1, 0], [1, 2], [2, 1]
            ],
            "atom_truth": {
                "0": assignment(True),
                "1": assignment(True),
                "2": assignment(False),
            },
            "property_check": "reflexive and symmetric, not necessarily transitive",
        }
    if candidate.axis == "domain" and source_contrast == ("varying", "cumulative"):
        return {
            "worlds": [0, 1],
            "accessibility": [[0, 1], [1, 1]],
            "domains": {"0": ["a", "b"], "1": ["a"]},
            "atom_truth": {
                "0": {},
                "1": {"a": assignment(True), "b": assignment(False)},
            },
            "property_check": "decreasing and serial",
        }
    if candidate.axis == "domain" and source_contrast == ("varying", "decreasing"):
        return {
            "worlds": [0, 1],
            "accessibility": [[0, 1], [1, 1]],
            "domains": {"0": ["a"], "1": ["a", "b"]},
            "atom_truth": {
                "0": {},
                "1": {"a": assignment(True), "b": assignment(False)},
            },
            "property_check": "cumulative and serial",
        }
    raise ValueError(f"unsupported countermodel family: {source_contrast}")


def _evaluate_witness(
    candidate: Candidate, semantics: Semantics, witness: dict
) -> bool:
    worlds = witness["worlds"]
    relation = {tuple(edge) for edge in witness["accessibility"]}
    successors = lambda world: [target for source, target in relation if source == world]
    if not all(successors(world) for world in worlds):
        raise ValueError("diagnostic countermodel must be serial")
    source_contrast = tuple(candidate.ast["source_contrast"])
    if candidate.axis == "frame":
        reflexive = all((world, world) in relation for world in worlds)
        symmetric = all((target, source) in relation for source, target in relation)
        transitive = all(
            (source, target) in relation
            for source, middle in relation
            for origin, target in relation
            if middle == origin
        )
        required = semantics.frame_properties
        checks = {"reflexive": reflexive, "symmetric": symmetric, "transitive": transitive}
        if not all(checks[property_name] for property_name in required):
            raise ValueError("countermodel violates its declared frame semantics")
        truth = {
            int(world): _evaluate_content(candidate, assignment)
            for world, assignment in witness["atom_truth"].items()
        }
        box = lambda world, fn: all(fn(target) for target in successors(world))
        diamond = lambda world, fn: any(fn(target) for target in successors(world))
        if source_contrast == ("T", "B"):
            premise = truth[0]
            conclusion = box(0, lambda world: diamond(world, truth.__getitem__))
        else:
            premise = box(0, truth.__getitem__)
            conclusion = box(0, lambda world: box(world, truth.__getitem__))
    else:
        domains = {int(world): values for world, values in witness["domains"].items()}
        monotone = all(
            set(domains[source]) <= set(domains[target])
            for source, target in relation
        )
        antitone = all(
            set(domains[target]) <= set(domains[source])
            for source, target in relation
        )
        if (semantics.domain == "cumulative" and not monotone) or (
            semantics.domain == "decreasing" and not antitone
        ):
            raise ValueError("countermodel violates its declared domain semantics")
        truth = {int(world): values for world, values in witness["atom_truth"].items()}
        atom = lambda world, obj: _evaluate_content(candidate, truth[world][obj])
        box_forall = all(
            all(atom(world, obj) for obj in domains[world])
            for world in successors(0)
        )
        forall_box = all(
            all(atom(world, obj) for world in successors(0))
            for obj in domains[0]
        )
        premise, conclusion = (
            (box_forall, forall_box)
            if source_contrast == ("varying", "cumulative")
            else (forall_box, box_forall)
        )
    return (not premise) or conclusion


def _assignment(ast: list, symbols: list[str], target: bool) -> dict[str, bool]:
    for values in product((False, True), repeat=len(symbols)):
        assignment = dict(zip(symbols, values))
        if _evaluate_ast(ast, assignment, symbols) is target:
            return assignment
    raise ValueError(f"content formula cannot realize truth value {target}")


def _evaluate_content(candidate: Candidate, assignment: dict[str, bool]) -> bool:
    return _evaluate_ast(candidate.ast["content"], assignment, candidate.ast["symbols"])


def _evaluate_ast(ast: list, assignment: dict[str, bool], symbols: list[str]) -> bool:
    operator = ast[0]
    if operator == "atom":
        return assignment[symbols[ast[1]]]
    if operator == "not":
        return not _evaluate_ast(ast[1], assignment, symbols)
    left = _evaluate_ast(ast[1], assignment, symbols)
    right = _evaluate_ast(ast[2], assignment, symbols)
    return left and right if operator == "and" else left or right


def _english(pair: dict) -> tuple[tuple[str, ...], str]:
    prompt = pair["controlled_english"]["prompt_a"]
    if pair["metadata"]["mode"] == "validity":
        return (), prompt.split("Statement: ", 1)[1].split("\n\nQuestion:", 1)[0]
    premise_block = prompt.split("Premises:\n", 1)[1].split("\n\nConjecture:", 1)[0]
    premises = tuple(line.split(". ", 1)[1] for line in premise_block.splitlines())
    conjecture = prompt.split("\n\nConjecture: ", 1)[1].split("\n\nQuestion:", 1)[0]
    return premises, conjecture


def _validate(pairs: list[dict], per_family_mode: int) -> None:
    expected = 4 * 2 * per_family_mode
    if len(pairs) != expected or len({row["pair_id"] for row in pairs}) != expected:
        raise ValueError("incorrect or duplicate formula-sensitive pairs")
    for axis in ("frame", "domain"):
        rows = [row for row in pairs if row["axis"] == axis]
        orientations = _counts(rows, lambda row: row["metadata"]["formula_valid_under"])
        if len(set(orientations.values())) != 1:
            raise ValueError(f"unbalanced validity orientation for {axis}")
        for condition in sorted({value for row in rows for value in row["contrast"]}):
            labels = [
                row[f"label_{side}"]
                for row in rows
                for side in ("a", "b")
                if _condition(row[f"semantics_{side}"], axis) == condition
            ]
            if labels.count(True) != labels.count(False):
                raise ValueError(f"condition {condition} is not label-balanced")


def _condition(semantics: dict, axis: str) -> str:
    return semantics["system"] if axis == "frame" else semantics["domain"]


def _counts(rows: list[dict], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(key(row))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _cache_key(candidate: Candidate) -> str:
    return content_hash(
        {
            "axis": candidate.axis,
            "contrast": candidate.contrast,
            "premises": candidate.premises,
            "conjecture": candidate.conjecture,
            "semantics_a": candidate.semantics_a.to_dict(),
            "semantics_b": candidate.semantics_b.to_dict(),
        }
    )


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        row["key"]: row["resolved"]
        for row in _read_jsonl(path)
    }


def _append_cache(path: Path, key: str, resolved: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "resolved": resolved}, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(*command: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/paper_v2/frozen/scaled-0.2/pairs.jsonl")
    parser.add_argument("--out", default="data/paper_v2/frozen/formula-sensitive-1.0")
    parser.add_argument("--per-family-mode", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    freeze(args.source, args.out, args.per_family_mode, args.timeout, args.workers)


if __name__ == "__main__":
    main()
