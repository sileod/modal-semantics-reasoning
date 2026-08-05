from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from litlm import complete


ANSWER = re.compile(r"^\s*(YES|NO)[.!]?\s*$", re.IGNORECASE)
FINAL_ANSWER = re.compile(r"\bAnswer:\s*(YES|NO)[.!]?\s*$", re.IGNORECASE)


def parse_answer(text: str | None) -> tuple[bool | None, str]:
    if text is None:
        return None, "missing"
    match = ANSWER.match(text) or FINAL_ANSWER.search(text)
    if not match:
        return None, "malformed"
    return match.group(1).upper() == "YES", "ok"


def evaluate_model(
    model: dict[str, Any],
    prompts_path: str | Path,
    output_path: str | Path,
    parameters: dict[str, Any],
    max_cost_usd: float | None = None,
    profile: str = "direct",
    limit: int | None = None,
) -> None:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite raw responses: {output}")

    prompts = [json.loads(line) for line in Path(prompts_path).read_text().splitlines()]
    if limit is not None:
        prompts = prompts[:limit]
    partial = output.with_suffix(output.suffix + ".partial")
    records = _read_partial(partial, prompts, model["model"], parameters, profile)
    spent = sum(record.get("response_cost_usd") or 0 for record in records)
    if max_cost_usd is not None and spent >= max_cost_usd:
        raise RuntimeError(f"cost cap already reached: ${spent:.6f}")
    completed = {
        (record["pair_id"], record["side"], record["prompt_hash"]) for record in records
    }
    pending = [
        prompt
        for prompt in prompts
        if (prompt["pair_id"], prompt["side"], prompt["prompt_hash"]) not in completed
    ]
    batch_size = parameters.get("batch_size", 32)
    output.parent.mkdir(parents=True, exist_ok=True)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        requested_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        request = {
            "model": model["model"],
            "show_progress": False,
            "caching": False,
            "num_retries": 2,
            "temperature": parameters["temperature"],
            "max_tokens": parameters["max_tokens"],
            "timeout": parameters["timeout_seconds"],
            "max_concurrency": parameters["max_concurrency"],
        }
        if "reasoning" in parameters:
            request["reasoning"] = parameters["reasoning"]
        responses = complete([prompt["prompt"] for prompt in batch], **request)
        if len(responses) != len(batch):
            raise RuntimeError("provider response count does not match request batch")
        batch_latency = time.perf_counter() - started
        batch_records = [
            _response_record(
                prompt,
                response,
                model,
                parameters,
                profile,
                requested_at,
                batch_latency,
            )
            for prompt, response in zip(batch, responses)
        ]
        _append_jsonl(partial, batch_records)
        records.extend(batch_records)
        spent += sum(record.get("response_cost_usd") or 0 for record in batch_records)
        print(
            f"[{len(records)}/{len(prompts)}] completed batch of {len(batch)} "
            f"(cost ${spent:.6f})",
            flush=True,
        )
        if max_cost_usd is not None and spent > max_cost_usd:
            raise RuntimeError(
                f"cost cap exceeded after completed batch: "
                f"${spent:.6f} > ${max_cost_usd:.6f}"
            )

    by_key = {
        (record["pair_id"], record["side"], record["prompt_hash"]): record
        for record in records
    }
    ordered = [
        by_key[(prompt["pair_id"], prompt["side"], prompt["prompt_hash"])]
        for prompt in prompts
    ]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in ordered),
        encoding="utf-8",
    )
    partial.unlink(missing_ok=True)
    print(f"froze {len(ordered)} raw responses at {output}")


def _response_record(
    prompt: dict,
    response: Any,
    model: dict[str, Any],
    parameters: dict[str, Any],
    profile: str,
    requested_at: str,
    batch_latency: float,
) -> dict:
    raw_text = _content(response)
    parsed, parse_status = parse_answer(raw_text)
    return {
        "pair_id": prompt["pair_id"],
        "side": prompt["side"],
        "prompt_hash": prompt["prompt_hash"],
        "provider": "openrouter",
        "model": model["model"].removeprefix("openrouter/"),
        "request_parameters": parameters,
        "evaluation_profile": profile,
        "request_timestamp": requested_at,
        "raw_response": raw_text,
        "parsed_answer": parsed,
        "parse_status": parse_status,
        "latency_seconds": round(batch_latency, 3),
        "batch_latency_seconds": round(batch_latency, 3),
        "provider_request_id": _field(response, "id"),
        "provider_model": _field(response, "model"),
        "finish_reason": _finish_reason(response),
        "usage": _serializable(_field(response, "usage")),
        "reasoning_content": _field(response, "reasoning"),
        "response_cost_usd": _response_cost(response),
    }


def _read_partial(
    path: Path,
    prompts: list[dict],
    configured_model: str,
    parameters: dict | None = None,
    profile: str | None = None,
) -> list[dict]:
    if not path.exists():
        return []
    expected = {
        (prompt["pair_id"], prompt["side"], prompt["prompt_hash"]) for prompt in prompts
    }
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    keys = [
        (record["pair_id"], record["side"], record["prompt_hash"]) for record in records
    ]
    if len(keys) != len(set(keys)) or any(key not in expected for key in keys):
        raise ValueError(f"invalid partial response file: {path}")
    model = configured_model.removeprefix("openrouter/")
    if any(record["model"] != model for record in records):
        raise ValueError(f"model mismatch in partial response file: {path}")
    if parameters is not None and any(
        record.get("request_parameters") != parameters for record in records
    ):
        raise ValueError(f"parameter mismatch in partial response file: {path}")
    if profile is not None and any(
        record.get("evaluation_profile") != profile for record in records
    ):
        raise ValueError(f"profile mismatch in partial response file: {path}")
    return records


def _append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    return response.choices[0].message.content or ""


def _field(response: Any, name: str) -> Any:
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


def _finish_reason(response: Any) -> str | None:
    choices = response.get("choices", []) if isinstance(response, dict) else response.choices
    if not choices:
        return None
    choice = choices[0]
    return choice.get("finish_reason") if isinstance(choice, dict) else choice.finish_reason


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def _response_cost(response: Any) -> float | None:
    for source in (response, getattr(response, "_hidden_params", None)):
        if source is None:
            continue
        for key in ("response_cost", "_response_cost"):
            value = (
                source.get(key)
                if isinstance(source, dict)
                else getattr(source, key, None)
            )
            if value is not None:
                return float(value)
    return None


def load_models(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Exact configured model identifier."
    )
    parser.add_argument(
        "--prompts", default="data/paper_v2/frozen/pilot-0.2/prompts.jsonl"
    )
    parser.add_argument("--out")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--profile", default="direct")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_models()
    try:
        model = next(item for item in config["models"] if item["model"] == args.model)
    except StopIteration:
        raise SystemExit(f"model is not configured: {args.model}")
    slug = args.model.replace("/", "__")
    output = args.out or (f"results/paper_v2/raw/pilot-0.2-reasoning-off/{slug}.jsonl")
    try:
        parameters = config["evaluation_profiles"][args.profile]
    except KeyError:
        raise SystemExit(f"evaluation profile is not configured: {args.profile}")
    evaluate_model(
        model,
        args.prompts,
        output,
        parameters,
        max_cost_usd=args.max_cost_usd,
        profile=args.profile,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
