from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidates import generate_candidates
from .oracle import run_side
from .render import render_prompt
from .schema import content_hash
from .semantics import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local paper-v2 smoke pipeline.")
    parser.add_argument("--per-axis", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=3)
    parser.add_argument("--out", default="data/paper_v2/candidates/smoke.jsonl")
    args = parser.parse_args()

    config = load_config()
    seed = config["benchmark"]["seed"]
    records = []
    for axis, axis_config in _enabled_axes(config):
        contrasts = [tuple(contrast) for contrast in axis_config["contrasts"]]
        for index in range(args.per_axis):
            contrast = contrasts[index % len(contrasts)]
            candidate = generate_candidates(
                axis, contrast, seed + index, index + 1
            )[index]
            oracle_a = run_side(candidate, candidate.semantics_a, args.timeout)
            oracle_b = run_side(candidate, candidate.semantics_b, args.timeout)
            main = (
                oracle_a["resolution"] == "dual_agreement"
                and oracle_b["resolution"] == "dual_agreement"
                and oracle_a["consensus"] != oracle_b["consensus"]
            )
            clean_flip = (
                oracle_a["consensus"] is not None
                and oracle_b["consensus"] is not None
                and oracle_a["consensus"] != oracle_b["consensus"]
                and "conflict" not in {oracle_a["resolution"], oracle_b["resolution"]}
            )
            pool = "main" if main else "reserve" if clean_flip else "rejected"
            identity = {
                "contrast": contrast,
                "ast": candidate.ast,
                "semantics_a": candidate.semantics_a.to_dict(),
                "semantics_b": candidate.semantics_b.to_dict(),
            }
            records.append(
                {
                    "candidate_id": f"{axis}-{content_hash(identity)[:16]}",
                    "axis": axis,
                    "contrast": list(contrast),
                    "ast": candidate.ast,
                    "semantics_a": candidate.semantics_a.to_dict(),
                    "semantics_b": candidate.semantics_b.to_dict(),
                    "prompt_a": render_prompt(candidate, candidate.semantics_a),
                    "prompt_b": render_prompt(candidate, candidate.semantics_b),
                    "oracle_a": oracle_a,
                    "oracle_b": oracle_b,
                    "pool": pool,
                    "accepted": main,
                    "rejection_reason": (
                        None if clean_flip else _rejection_reason(oracle_a, oracle_b)
                    ),
                }
            )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    counts = {
        pool: sum(record["pool"] == pool for record in records)
        for pool in ("main", "reserve", "rejected")
    }
    print(
        f"paper-v2 smoke: {counts['main']} main, {counts['reserve']} reserve, "
        f"{counts['rejected']} rejected; wrote {output}"
    )


def _enabled_axes(config: dict) -> list[tuple[str, dict]]:
    return [
        (axis, axis_config)
        for axis, axis_config in config["axes"].items()
        if axis_config.get("enabled", True)
    ]


def _rejection_reason(a: dict, b: dict) -> str:
    if "conflict" in {a["resolution"], b["resolution"]}:
        return "oracle_conflict"
    if a["consensus"] is None or b["consensus"] is None:
        return "oracle_timeout"
    if a["consensus"] == b["consensus"]:
        return "same_oracle_label"
    return "one_prover_only"


if __name__ == "__main__":
    main()
