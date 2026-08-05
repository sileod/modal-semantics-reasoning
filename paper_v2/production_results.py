from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import load_models
from .implicit_logic import score as score_affinity
from .reasoning_eval import compare
from .score import score


def reproduce(run_name: str, config_path: str | Path = "configs/models.yaml") -> None:
    run = load_models(config_path)["production_runs"][run_name]
    pairs = run["pairs"]
    prompts = Path(run["prompts"])
    affinity_prompts = run["affinity_prompts"]
    raw_dir = Path("results/paper_v2/raw") / run_name
    scored_dir = Path("results/paper_v2/scored") / run_name
    scored_dir.mkdir(parents=True, exist_ok=True)
    direct = {
        cell["model"]: cell
        for cell in run["cells"]
        if cell["scope"] == "primary_and_affinity"
    }

    for cell in direct.values():
        stem = cell["stem"]
        raw_primary = raw_dir / f"{stem}__primary.jsonl"
        primary = score(pairs, prompts / "prompts.jsonl", raw_primary)
        _write(scored_dir / f"{stem}__primary.json", primary)
        affinity = score_affinity(
            affinity_prompts,
            raw_dir / f"{stem}__affinity.jsonl",
            pairs,
            raw_primary,
        )
        _write(scored_dir / f"{stem}__affinity.json", affinity)

    for cell in (row for row in run["cells"] if row["scope"] == "frame"):
        stem = cell["stem"]
        direct_stem = direct[cell["model"]]["stem"]
        direct_raw = raw_dir / f"{direct_stem}__primary.jsonl"
        direct_score = scored_dir / f"{direct_stem}__frame.json"
        enhanced_raw = raw_dir / f"{stem}__frame.jsonl"
        enhanced_score = scored_dir / f"{stem}__frame.json"
        _write(direct_score, score(pairs, prompts / "frame_prompts.jsonl", direct_raw))
        _write(
            enhanced_score,
            score(pairs, prompts / "frame_prompts.jsonl", enhanced_raw),
        )
        label = cell["profile"].removeprefix("reasoning_").removesuffix("_production")
        compare(
            direct_score,
            enhanced_score,
            direct_raw,
            enhanced_raw,
            f"reasoning_{label}",
            scored_dir / f"{stem}__comparison.json",
        )


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="production-1.1")
    parser.add_argument("--config", default="configs/models.yaml")
    args = parser.parse_args()
    reproduce(args.run, args.config)


if __name__ == "__main__":
    main()
