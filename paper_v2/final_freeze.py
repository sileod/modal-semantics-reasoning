from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .candidates import generate_candidates
from .freeze import (
    _clean_flip,
    _make_pair,
    _premise_ablation,
    _prompt_records,
    _rejection_reason,
    _write_jsonl,
)
from .oracle import run_side
from .schema import content_hash
from .semantics import load_config


def freeze_final(
    output_dir: str | Path = "data/paper_v2/frozen/scaled-0.2",
) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen data: {output}")

    config = load_config()
    final = config["final"]
    source_commit = _command(["git", "rev-parse", "HEAD"])
    source_dirty = bool(
        _command(["git", "status", "--porcelain", "--untracked-files=no"])
    )
    pairs, oracle_records, prompts, rejected = [], [], [], []
    formula_hashes: set[str] = set()
    cache_path = (
        Path("data/paper_v2/cache")
        / final["oracle_cache_version"]
        / "oracle.jsonl"
    )
    cache = _load_cache(cache_path)

    cells = _cells(config)
    with ThreadPoolExecutor(max_workers=final["workers"]) as executor:
        for cell_index, (axis, contrast, mode, quota) in enumerate(cells):
            candidates = generate_candidates(
                axis,
                contrast,
                config["benchmark"]["seed"],
                final["grammar_pool_per_cell"],
                mode=mode,
            )
            rng = random.Random(
                f"{config['benchmark']['seed']}:{axis}:{contrast}:{mode}"
            )
            rng.shuffle(candidates)
            accepted = 0
            cursor = 0
            while accepted < quota and cursor < len(candidates):
                batch = []
                while cursor < len(candidates) and len(batch) < final["workers"]:
                    candidate = candidates[cursor]
                    cursor += 1
                    if _candidate_formula_hash(candidate) in formula_hashes:
                        continue
                    batch.append(candidate)
                futures = {
                    _candidate_cache_key(candidate): executor.submit(
                        _resolve, candidate, final["oracle_timeout_seconds"]
                    )
                    for candidate in batch
                    if _candidate_cache_key(candidate) not in cache
                }
                for candidate in batch:
                    cache_key = _candidate_cache_key(candidate)
                    if cache_key in cache:
                        resolved = cache[cache_key]
                    else:
                        resolved = futures[cache_key].result()
                        cache[cache_key] = resolved
                        _append_cache(cache_path, cache_key, resolved)
                    if not resolved["accepted"]:
                        rejected.append(
                            {
                                "axis": axis,
                                "contrast": list(contrast),
                                "mode": mode,
                                "ast": candidate.ast,
                                "reason": resolved["reason"],
                                "oracle_a": resolved["oracle_a"],
                                "oracle_b": resolved["oracle_b"],
                            }
                        )
                        continue
                    if accepted >= quota:
                        continue
                    swap = (cell_index + accepted) % 2 == 1
                    pair = _make_pair(
                        candidate,
                        resolved["oracle_a"],
                        resolved["oracle_b"],
                        resolved["premise_status"],
                        config["benchmark"]["seed"],
                        swap,
                    )
                    data = pair.to_dict()
                    formula_hashes.add(data["canonical_formula_hash"])
                    pairs.append(data)
                    oracle_records.append(
                        {
                            "pair_id": pair.pair_id,
                            "side_a": (
                                resolved["oracle_b"] if swap else resolved["oracle_a"]
                            ),
                            "side_b": (
                                resolved["oracle_a"] if swap else resolved["oracle_b"]
                            ),
                            "premise_ablation": resolved["ablation"],
                        }
                    )
                    prompts.extend(
                        _prompt_records(
                            pair.pair_id,
                            data["controlled_english"]["prompt_a"],
                            data["controlled_english"]["prompt_b"],
                        )
                    )
                    accepted += 1
            if accepted != quota:
                raise RuntimeError(
                    f"cell underfilled: {axis} {contrast} {mode}: {accepted}/{quota}"
                )
            print(f"accepted {accepted}: {axis} {contrast} {mode}", flush=True)

    _validate_final(pairs, config)
    output.mkdir(parents=True)
    _write_jsonl(output / "pairs.jsonl", pairs)
    _write_jsonl(output / "oracle.jsonl", oracle_records)
    _write_jsonl(output / "prompts.jsonl", prompts)
    _write_jsonl(output / "rejected.jsonl", rejected)
    shutil.copy2("configs/paper_v2.yaml", output / "paper_v2.yaml")
    manifest = _manifest(
        output,
        config,
        pairs,
        oracle_records,
        rejected,
        source_commit,
        source_dirty,
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"froze final dataset: {len(pairs)} pairs at {output}")


def _resolve(candidate, timeout: int) -> dict:
    oracle_a = run_side(candidate, candidate.semantics_a, timeout)
    oracle_b = run_side(candidate, candidate.semantics_b, timeout)
    if not _clean_flip(oracle_a, oracle_b):
        return {
            "accepted": False,
            "reason": _rejection_reason(oracle_a, oracle_b),
            "oracle_a": oracle_a,
            "oracle_b": oracle_b,
        }
    premise_status, ablation = _premise_ablation(
        candidate, oracle_a["consensus"], oracle_b["consensus"], timeout
    )
    return {
        "accepted": True,
        "oracle_a": oracle_a,
        "oracle_b": oracle_b,
        "premise_status": premise_status,
        "ablation": ablation,
    }


def _candidate_formula_hash(candidate) -> str:
    return content_hash(
        {
            "premises": tuple({"tptp": premise} for premise in candidate.premises),
            "conjecture": {"tptp": candidate.conjecture, "ast": candidate.ast},
        }
    )


def _candidate_cache_key(candidate) -> str:
    return content_hash(
        {
            "axis": candidate.axis,
            "contrast": candidate.contrast,
            "premises": candidate.premises,
            "conjecture": candidate.conjecture,
            "ast": candidate.ast,
            "semantics_a": candidate.semantics_a.to_dict(),
            "semantics_b": candidate.semantics_b.to_dict(),
        }
    )


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["key"] in cache:
            raise ValueError(f"duplicate oracle cache key: {record['key']}")
        cache[record["key"]] = record["resolved"]
    return cache


def _append_cache(path: Path, key: str, resolved: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "resolved": resolved}, sort_keys=True) + "\n")


def _cells(config: dict) -> list[tuple[str, tuple[str, str], str, int]]:
    final = config["final"]
    cells = []
    for axis in ("frame", "domain"):
        contrasts = [tuple(value) for value in config["axes"][axis]["contrasts"]]
        combinations = [
            (contrast, mode)
            for contrast in contrasts
            for mode in final["modes"]
        ]
        target = final["target_pairs"][axis]
        base, remainder = divmod(target, len(combinations))
        for index, (contrast, mode) in enumerate(combinations):
            cells.append(
                (axis, contrast, mode, base + (1 if index < remainder else 0))
            )
    return cells


def _validate_final(pairs: list[dict], config: dict) -> None:
    ids = [pair["pair_id"] for pair in pairs]
    hashes = [pair["canonical_formula_hash"] for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair identifiers")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate canonical formulas")
    for axis, target in config["final"]["target_pairs"].items():
        if sum(pair["axis"] == axis for pair in pairs) != target:
            raise ValueError(f"incorrect final count for {axis}")
    positive_a = sum(pair["label_a"] for pair in pairs)
    if abs(positive_a - len(pairs) / 2) > 1:
        raise ValueError("side orientation is not balanced")


def _manifest(
    output: Path,
    config: dict,
    pairs: list[dict],
    oracle_records: list[dict],
    rejected: list[dict],
    source_commit: str,
    source_dirty: bool,
) -> dict:
    counts = {
        "pairs": len(pairs),
        "sides": 2 * len(pairs),
        "by_axis": _count(pairs, lambda row: row["axis"]),
        "by_contrast": _count(pairs, lambda row: "/".join(row["contrast"])),
        "by_mode": _count(pairs, lambda row: row["metadata"]["mode"]),
        "by_premise_status": _count(pairs, lambda row: row["premise_status"]),
        "rejected": len(rejected),
    }
    resolutions = _count(
        [
            side
            for record in oracle_records
            for side in (record["side_a"], record["side_b"])
        ],
        lambda row: row["resolution"],
    )
    return {
        "benchmark": "paper-v2",
        "version": config["final"]["version"],
        "git_commit": source_commit,
        "git_dirty_before_generation": source_dirty,
        "generation_seed": config["benchmark"]["seed"],
        "generator_command": "make paper-v2-freeze",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tools": {
            "let": _command(["java", "-jar", "tools/logic-embedding.jar", "--version"]),
            "vampire": _command(["tools/vampire", "--version"]).splitlines()[0],
            "leo3": _command(["java", "-jar", "tools/leo3.jar", "--version"]),
        },
        "counts": counts,
        "oracle_resolution_counts": resolutions,
        "sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("pairs.jsonl", "oracle.jsonl", "prompts.jsonl", "paper_v2.yaml")
        },
    }


def _count(rows: list[dict], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(key(row))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/paper_v2/frozen/scaled-0.2")
    args = parser.parse_args()
    freeze_final(args.out)


if __name__ == "__main__":
    main()
