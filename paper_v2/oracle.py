from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .candidates import Candidate
from .semantics import Semantics


DECISIVE = {"Theorem": True, "Unsatisfiable": True, "CounterSatisfiable": False, "Satisfiable": False}


def build_tptp(candidate: Candidate, semantics: Semantics) -> str:
    designation = f"${semantics.designation}"
    domain = f"${semantics.domain}"
    system = f"$modal_system_{semantics.system}"
    declarations = _declarations(candidate)
    premises = "\n".join(
        f"tff(premise_{index}, axiom-local, {formula})."
        for index, formula in enumerate(candidate.premises, start=1)
    )
    return (
        "tff(logic_setup, logic, $modal == [\n"
        f"  $designation == {designation},\n"
        f"  $domains == {domain},\n"
        f"  $terms == ${semantics.term_interpretation},\n"
        f"  $modalities == {system}\n"
        "]).\n"
        f"{declarations}{premises}\n"
        f"tff(conjecture, conjecture, {candidate.conjecture}).\n"
    )


def run_side(
    candidate: Candidate,
    semantics: Semantics,
    timeout: int,
    tools_dir: str | Path = "tools",
) -> dict:
    tools = Path(tools_dir)
    source = build_tptp(candidate, semantics)
    embed_command = ["java", "-jar", str(tools / "logic-embedding.jar")]
    if candidate.axis == "designation":
        embed_command.extend(["-p", "FORCE_HIGHERORDER"])

    with tempfile.NamedTemporaryFile("w", suffix=".p") as source_file:
        source_file.write(source)
        source_file.flush()
        command = [*embed_command, source_file.name]
        embedded = subprocess.run(command, capture_output=True, text=True)

    record = {
        "prover_input": source,
        "embed_command": command,
        "embed_exit_code": embedded.returncode,
        "hol_problem": embedded.stdout,
        "embed_stdout_hash": _hash(embedded.stdout),
        "embed_stderr_hash": _hash(embedded.stderr),
    }
    if embedded.returncode:
        diagnostic = (embedded.stderr or embedded.stdout).strip()[:500]
        return {
            **record,
            "embed_diagnostic": diagnostic,
            "consensus": None,
            "resolution": "embedding_failure",
            "provers": {},
        }

    with tempfile.NamedTemporaryFile("w", suffix=".p") as hol_file:
        hol_file.write(embedded.stdout)
        hol_file.flush()
        commands = {
            "vampire": [
                str(tools / "vampire"),
                "--mode",
                "casc",
                "-t",
                str(timeout),
                hol_file.name,
            ],
            "leo3": ["java", "-jar", str(tools / "leo3.jar"), hol_file.name],
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                name: pool.submit(_run_prover, command, timeout)
                for name, command in commands.items()
            }
            provers = {name: future.result() for name, future in futures.items()}

    labels = [result["label"] for result in provers.values() if result["label"] is not None]
    if len(labels) == 2 and labels[0] != labels[1]:
        consensus, resolution = None, "conflict"
    elif len(labels) == 2:
        consensus, resolution = labels[0], "dual_agreement"
    elif len(labels) == 1:
        consensus, resolution = labels[0], "one_prover_only"
    else:
        consensus, resolution = None, "unresolved"
    return {**record, "consensus": consensus, "resolution": resolution, "provers": provers}


def _declarations(candidate: Candidate) -> str:
    if candidate.axis == "frame":
        return "".join(
            f"tff(type_{symbol}, type, {symbol}: $o).\n"
            for symbol in candidate.ast["symbols"]
        )
    if candidate.axis == "domain":
        return "".join(
            f"tff(type_{symbol}, type, {symbol}: $i > $o).\n"
            for symbol in candidate.ast["symbols"]
        )
    return (
        f"tff(type_left, type, {candidate.ast['left']}: $i).\n"
        f"tff(type_right, type, {candidate.ast['right']}: $i).\n"
    )


def _run_prover(command: list[str], timeout: int) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 2)
        status = _szs_status(result.stdout)
        return {
            "command": command,
            "exit_code": result.returncode,
            "szs_status": status,
            "label": DECISIVE.get(status),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "stdout_hash": _hash(result.stdout),
            "stderr_hash": _hash(result.stderr),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "exit_code": None,
            "szs_status": "Timeout",
            "label": None,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "stdout_hash": _hash(error.stdout or b""),
            "stderr_hash": _hash(error.stderr or b""),
        }


def _szs_status(output: str) -> str | None:
    match = re.search(r"SZS status\s+([A-Za-z]+)", output)
    return match.group(1) if match else None


def _hash(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()
