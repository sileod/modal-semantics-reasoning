from __future__ import annotations

from .candidates import Candidate
from .semantics import Semantics


def render_prompt(candidate: Candidate, semantics: Semantics) -> str:
    lines = ["Semantic specification:"]
    lines.extend(f"- {line}" for line in _semantic_lines(candidate.axis, semantics))
    lines.append("")
    if candidate.english_premises:
        lines.append("Premises:")
        lines.extend(
            f"{index}. {premise}"
            for index, premise in enumerate(candidate.english_premises, start=1)
        )
        lines.extend(
            [
                "",
                f"Conjecture: {candidate.english_conjecture}",
                "",
                "Question: Does the conjecture follow from the premises "
                "under this semantic specification?",
            ]
        )
    else:
        lines.extend(
            [
                f"Statement: {candidate.english_conjecture}",
                "",
                "Question: Is the statement valid under this semantic specification?",
            ]
        )
    lines.append("Answer only Yes or No.")
    return "\n".join(lines)


def _semantic_lines(axis: str, semantics: Semantics) -> list[str]:
    properties = semantics.frame_properties
    if properties:
        frame = "The accessibility relation is " + ", ".join(sorted(properties)) + "."
    else:
        frame = "The accessibility relation has no additional constraints."

    domains = {
        "varying": "Objects may appear or disappear at accessible worlds.",
        "cumulative": "Objects cannot disappear when moving to an accessible world.",
        "decreasing": "New objects cannot appear when moving to an accessible world.",
        "constant": "Exactly the same objects exist at every world.",
    }
    designation = {
        "rigid": "Each individual description denotes the same object in every world.",
        "flexible": "An individual description may denote different objects in different worlds.",
    }
    if axis == "frame":
        return [frame]
    if axis == "domain":
        return [frame, domains[semantics.domain]]
    return [frame, domains[semantics.domain], designation[semantics.designation]]
