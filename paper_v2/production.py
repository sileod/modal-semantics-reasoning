from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(
    pairs_path: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite production prompts: {output}")

    pairs = _read_jsonl(pairs_path)
    prompts = [_normalize(row) for row in _read_jsonl(prompts_path)]
    ids_by_axis = {
        axis: {row["pair_id"] for row in pairs if row["axis"] == axis}
        for axis in ("frame", "domain")
    }
    files = {
        "prompts": ("prompts.jsonl", prompts),
        **{
            f"{axis}_prompts": (
                f"{axis}_prompts.jsonl",
                [row for row in prompts if row["pair_id"] in pair_ids],
            )
            for axis, pair_ids in ids_by_axis.items()
        },
    }
    output.mkdir(parents=True)
    for filename, rows in files.values():
        _write_jsonl(output / filename, rows)
    manifest = {
        "source_pairs": str(pairs_path),
        "source_prompts": str(prompts_path),
        "pairs": len(pairs),
        "prompts": len(prompts),
        "pairs_by_axis": {axis: len(ids) for axis, ids in ids_by_axis.items()},
        "normalization": "Answer only YES or NO -> Answer only Yes or No",
        "sha256": {
            name: _sha256(output / filename)
            for name, (filename, _) in files.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _normalize(row: dict) -> dict:
    prompt = row["prompt"].replace("Answer only YES or NO.", "Answer only Yes or No.")
    return {
        **row,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
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


def audit(raw_dir: str | Path, output_path: str | Path) -> None:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite production audit: {output}")
    files = []
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        rows = _read_jsonl(path)
        parameters = {json.dumps(row["request_parameters"], sort_keys=True) for row in rows}
        if len(parameters) != 1:
            raise ValueError(f"mixed request parameters: {path}")
        files.append(
            {
                "file": str(path),
                "sha256": _sha256(path),
                "responses": len(rows),
                "model": rows[0]["model"],
                "profile": rows[0]["evaluation_profile"],
                "request_parameters": json.loads(parameters.pop()),
                "parsed": sum(row["parse_status"] == "ok" for row in rows),
                "cost_usd": sum(row.get("response_cost_usd") or 0 for row in rows),
                "provider_models": sorted({row.get("provider_model") for row in rows}),
            }
        )
    manifest = {
        "version": Path(raw_dir).name,
        "responses": sum(row["responses"] for row in files),
        "cost_usd": sum(row["cost_usd"] for row in files),
        "configuration_sha256": _sha256(Path("configs/models.yaml")),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs")
    parser.add_argument("--prompts")
    parser.add_argument("--audit-raw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.audit_raw:
        audit(args.audit_raw, args.out)
    elif args.pairs and args.prompts:
        build(args.pairs, args.prompts, args.out)
    else:
        parser.error("--pairs and --prompts are required unless --audit-raw is used")


if __name__ == "__main__":
    main()
