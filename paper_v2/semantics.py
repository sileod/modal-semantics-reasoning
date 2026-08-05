from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


FRAME_PROPERTIES = {
    "K": frozenset(),
    "D": frozenset({"serial"}),
    "T": frozenset({"reflexive"}),
    "B": frozenset({"reflexive", "symmetric"}),
    "S4": frozenset({"reflexive", "transitive"}),
    "S5": frozenset({"reflexive", "symmetric", "transitive"}),
}

FRAME_CONTRASTS = {
    ("K", "D"): "serial",
    ("K", "T"): "reflexive",
    ("T", "B"): "symmetric",
    ("T", "S4"): "transitive",
    ("B", "S5"): "transitive",
}

# Non-nested contrasts used only by the formula-sensitive diagnostic.  They
# exchange two restrictions, so neither side is uniformly stronger.
FRAME_DIAGNOSTIC_CONTRASTS = {
    ("B", "S4"): frozenset({"symmetric", "transitive"}),
}

DOMAIN_CONTRASTS = {
    ("varying", "cumulative"),
    ("varying", "decreasing"),
    ("cumulative", "constant"),
    ("decreasing", "constant"),
}
DOMAIN_DIAGNOSTIC_CONTRASTS = {("cumulative", "decreasing")}

DESIGNATION_CONTRASTS = {("rigid", "flexible")}


@dataclass(frozen=True)
class Semantics:
    system: str
    frame_properties: frozenset[str]
    domain: str
    designation: str
    term_interpretation: str = "local"

    @classmethod
    def make(
        cls,
        system: str,
        domain: str = "constant",
        designation: str = "rigid",
        term_interpretation: str = "local",
    ) -> "Semantics":
        if system not in FRAME_PROPERTIES:
            raise ValueError(f"unknown modal system: {system}")
        if domain not in {"varying", "cumulative", "decreasing", "constant"}:
            raise ValueError(f"unknown domain condition: {domain}")
        if designation not in {"rigid", "flexible"}:
            raise ValueError(f"unknown designation: {designation}")
        if term_interpretation not in {"local", "global"}:
            raise ValueError(f"unknown term interpretation: {term_interpretation}")
        return cls(
            system=system,
            frame_properties=FRAME_PROPERTIES[system],
            domain=domain,
            designation=designation,
            term_interpretation=term_interpretation,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["frame_properties"] = sorted(self.frame_properties)
        return data


def changed_axes(a: Semantics, b: Semantics) -> set[str]:
    changes = set()
    if a.frame_properties != b.frame_properties:
        changes.add("frame")
    if a.domain != b.domain:
        changes.add("domain")
    if a.designation != b.designation:
        changes.add("designation")
    if a.term_interpretation != b.term_interpretation:
        changes.add("term_interpretation")
    return changes


def validate_contrast(axis: str, contrast: tuple[str, str]) -> None:
    allowed = {
        "frame": set(FRAME_CONTRASTS) | set(FRAME_DIAGNOSTIC_CONTRASTS),
        "domain": DOMAIN_CONTRASTS | DOMAIN_DIAGNOSTIC_CONTRASTS,
        "designation": DESIGNATION_CONTRASTS,
    }
    if axis not in allowed:
        raise ValueError(f"unknown paper-v2 axis: {axis}")
    if contrast not in allowed[axis]:
        raise ValueError(f"unsupported {axis} contrast: {contrast}")


def validate_semantic_difference(
    axis: str, contrast: tuple[str, str], a: Semantics, b: Semantics
) -> None:
    validate_contrast(axis, contrast)
    if changed_axes(a, b) != {axis}:
        raise ValueError(f"pair must change only {axis}: {changed_axes(a, b)}")

    if axis == "frame":
        if {a.system, b.system} != set(contrast):
            raise ValueError("frame systems do not match the declared contrast")
        expected = (
            {FRAME_CONTRASTS[contrast]}
            if contrast in FRAME_CONTRASTS
            else FRAME_DIAGNOSTIC_CONTRASTS[contrast]
        )
        if a.frame_properties ^ b.frame_properties != expected:
            raise ValueError("frame contrast changes unexpected properties")
    elif axis == "domain":
        if {a.domain, b.domain} != set(contrast):
            raise ValueError("domains do not match the declared contrast")
        if a.system != "D" or b.system != "D":
            raise ValueError("domain contrasts require modal system D")
    else:
        if {a.designation, b.designation} != set(contrast):
            raise ValueError("designation values do not match the declared contrast")
        if a.system != "D" or a.domain != "constant":
            raise ValueError("designation contrasts require D and constant domains")


def load_config(path: str | Path = "configs/paper_v2.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    benchmark = config["benchmark"]
    if benchmark["name"] != "paper-v2" or str(benchmark["version"]) != "2.0":
        raise ValueError("expected paper-v2 configuration version 2.0")
    target = benchmark["target_pairs_per_axis"]
    if not isinstance(target, int) or target <= 0:
        raise ValueError("target_pairs_per_axis must be a positive integer")
    final_targets = config["final"]["target_pairs"]
    if any(final_targets[axis] != target for axis in ("frame", "domain")):
        raise ValueError("enabled axis targets must match target_pairs_per_axis")

    axes = config["axes"]
    for axis in ("frame", "domain", "designation"):
        for raw_contrast in axes[axis]["contrasts"]:
            validate_contrast(axis, tuple(raw_contrast))
    if axes["domain"]["fixed_system"] != "D":
        raise ValueError("domain axis must fix modal system D")
    designation = axes["designation"]
    if (
        designation["fixed_system"],
        designation["fixed_domain"],
        designation["fixed_term_interpretation"],
    ) != ("D", "constant", "local"):
        raise ValueError("designation background semantics are not paper-v2 compliant")
    if designation.get("enabled") is not False:
        raise ValueError("designation must remain disabled until its oracle gate passes")
