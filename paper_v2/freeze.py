from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from .candidates import Candidate, generate_candidates
from .oracle import run_side
from .render import render_prompt
from .schema import Pair
from .semantics import load_config


def freeze_pilot(output_dir: str | Path = "data/paper_v2/frozen/pilot-0.2") -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen data: {output}")

    config = load_config()
    pilot = config["pilot"]
    timeout = pilot["oracle_timeout_seconds"]
    pairs, oracle_records, prompts, rejected = [], [], [], []

    for axis in ("frame", "domain"):
        for raw_contrast in config["axes"][axis]["contrasts"]:
            contrast = tuple(raw_contrast)
            for mode in pilot["modes"]:
                candidate = generate_candidates(
                    axis, contrast, config["benchmark"]["seed"], 1, mode=mode
                )[0]
                oracle_a = run_side(candidate, candidate.semantics_a, timeout)
                oracle_b = run_side(candidate, candidate.semantics_b, timeout)
                if not _clean_flip(oracle_a, oracle_b):
                    rejected.append(
                        {
                            "axis": axis,
                            "contrast": list(contrast),
                            "mode": mode,
                            "reason": _rejection_reason(oracle_a, oracle_b),
                            "oracle_a": oracle_a,
                            "oracle_b": oracle_b,
                        }
                    )
                    continue

                premise_status, ablation = _premise_ablation(
                    candidate, oracle_a["consensus"], oracle_b["consensus"], timeout
                )
                swap = len(pairs) % 2 == 1
                pair = _make_pair(
                    candidate,
                    oracle_a,
                    oracle_b,
                    premise_status,
                    config["benchmark"]["seed"],
                    swap,
                )
                data = pair.to_dict()
                pairs.append(data)
                oracle_records.append(
                    {
                        "pair_id": pair.pair_id,
                        "side_a": oracle_b if swap else oracle_a,
                        "side_b": oracle_a if swap else oracle_b,
                        "premise_ablation": ablation,
                    }
                )
                prompts.extend(
                    _prompt_records(
                        pair.pair_id,
                        data["controlled_english"]["prompt_a"],
                        data["controlled_english"]["prompt_b"],
                    )
                )

    output.mkdir(parents=True)
    _write_jsonl(output / "pairs.jsonl", pairs)
    _write_jsonl(output / "oracle.jsonl", oracle_records)
    _write_jsonl(output / "prompts.jsonl", prompts)
    _write_jsonl(output / "rejected.jsonl", rejected)
    shutil.copy2("configs/paper_v2.yaml", output / "paper_v2.yaml")
    manifest = _manifest(output, config, pairs, oracle_records, rejected)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"froze pilot: {len(pairs)} pairs, {len(rejected)} rejected at {output}"
    )


def _clean_flip(a: dict, b: dict) -> bool:
    return (
        a["consensus"] is not None
        and b["consensus"] is not None
        and a["consensus"] != b["consensus"]
        and "conflict" not in {a["resolution"], b["resolution"]}
    )


def _premise_ablation(
    candidate: Candidate, label_a: bool, label_b: bool, timeout: int
) -> tuple[str, dict | None]:
    if not candidate.premises:
        return "conjecture_only", None
    ablated = replace(candidate, premises=(), english_premises=())
    a = run_side(ablated, ablated.semantics_a, timeout)
    b = run_side(ablated, ablated.semantics_b, timeout)
    record = {"side_a": a, "side_b": b}
    if a["consensus"] is None or b["consensus"] is None:
        return "ablation_unresolved", record
    if (a["consensus"], b["consensus"]) == (label_a, label_b):
        return "conjecture_only", record
    return "premise_dependent", record


def _make_pair(
    candidate: Candidate,
    oracle_a: dict,
    oracle_b: dict,
    premise_status: str,
    seed: int,
    swap: bool,
) -> Pair:
    semantics_a, semantics_b = candidate.semantics_a, candidate.semantics_b
    label_a, label_b = oracle_a["consensus"], oracle_b["consensus"]
    summary_a, summary_b = _oracle_summary(oracle_a), _oracle_summary(oracle_b)
    prompt_a = render_prompt(candidate, semantics_a)
    prompt_b = render_prompt(candidate, semantics_b)
    if swap:
        semantics_a, semantics_b = semantics_b, semantics_a
        label_a, label_b = label_b, label_a
        summary_a, summary_b = summary_b, summary_a
        prompt_a, prompt_b = prompt_b, prompt_a

    return Pair(
        axis=candidate.axis,
        contrast=candidate.contrast,
        premises=tuple({"tptp": premise} for premise in candidate.premises),
        conjecture={"tptp": candidate.conjecture, "ast": candidate.ast},
        semantics_a=semantics_a,
        semantics_b=semantics_b,
        label_a=label_a,
        label_b=label_b,
        oracle_a=summary_a,
        oracle_b=summary_b,
        controlled_english={"prompt_a": prompt_a, "prompt_b": prompt_b},
        generator_seed=seed,
        modal_depth=candidate.modal_depth,
        quantifier_depth=candidate.quantifier_depth,
        premise_status=premise_status,
        alternation_pattern=candidate.alternation_pattern,
        metadata={"mode": "nli" if candidate.premises else "validity"},
    )


def _oracle_summary(record: dict) -> dict:
    return {
        "label": record["consensus"],
        "resolution": record["resolution"],
        "provers": {
            name: {
                "szs_status": result["szs_status"],
                "runtime_seconds": result["runtime_seconds"],
            }
            for name, result in record["provers"].items()
        },
    }


def _prompt_records(pair_id: str, prompt_a: str, prompt_b: str) -> list[dict]:
    return [
        {
            "pair_id": pair_id,
            "side": side,
            "prompt": prompt,
            "prompt_hash": _hash_text(prompt),
        }
        for side, prompt in (("a", prompt_a), ("b", prompt_b))
    ]


def _rejection_reason(a: dict, b: dict) -> str:
    if "conflict" in {a["resolution"], b["resolution"]}:
        return "oracle_conflict"
    if a["consensus"] is None or b["consensus"] is None:
        return "oracle_timeout"
    return "same_oracle_label"


def _manifest(
    output: Path,
    config: dict,
    pairs: list[dict],
    oracle_records: list[dict],
    rejected: list[dict],
) -> dict:
    return {
        "benchmark": "paper-v2-pilot",
        "version": config["pilot"]["version"],
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
        "generation_seed": config["benchmark"]["seed"],
        "generator_command": "make paper-v2-pilot-freeze",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tools": {
            "let": _command(["java", "-jar", "tools/logic-embedding.jar", "--version"]),
            "vampire": _command(["tools/vampire", "--version"]).splitlines()[0],
            "leo3": _command(["java", "-jar", "tools/leo3.jar", "--version"]),
        },
        "counts": {
            "pairs": len(pairs),
            "sides": len(pairs) * 2,
            "by_axis": _counts(pairs, "axis"),
            "by_mode": _counts(pairs, lambda row: row["metadata"]["mode"]),
            "by_premise_status": _counts(pairs, "premise_status"),
            "rejected": len(rejected),
        },
        "oracle_resolution_counts": _oracle_counts(oracle_records),
        "sha256": {
            name: _hash_file(output / name)
            for name in ("pairs.jsonl", "oracle.jsonl", "prompts.jsonl", "paper_v2.yaml")
        },
    }


def _counts(rows: list[dict], key) -> dict[str, int]:
    get = (lambda row: row[key]) if isinstance(key, str) else key
    values: dict[str, int] = {}
    for row in rows:
        value = str(get(row))
        values[value] = values.get(value, 0) + 1
    return values


def _oracle_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for side in ("side_a", "side_b"):
            resolution = record[side]["resolution"]
            counts[resolution] = counts.get(resolution, 0) + 1
    return counts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/paper_v2/frozen/pilot-0.2")
    args = parser.parse_args()
    freeze_pilot(args.out)


if __name__ == "__main__":
    main()
