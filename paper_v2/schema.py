from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .semantics import Semantics, validate_semantic_difference


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Pair:
    axis: str
    contrast: tuple[str, str]
    premises: tuple[dict[str, Any], ...]
    conjecture: dict[str, Any]
    semantics_a: Semantics
    semantics_b: Semantics
    label_a: bool
    label_b: bool
    oracle_a: dict[str, Any]
    oracle_b: dict[str, Any]
    controlled_english: dict[str, Any]
    generator_seed: int
    modal_depth: int
    quantifier_depth: int
    premise_status: str
    alternation_pattern: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_formula_hash(self) -> str:
        return content_hash({"premises": self.premises, "conjecture": self.conjecture})

    @property
    def pair_id(self) -> str:
        identity = {
            "axis": self.axis,
            "contrast": self.contrast,
            "formula_hash": self.canonical_formula_hash,
            "semantics_a": self.semantics_a.to_dict(),
            "semantics_b": self.semantics_b.to_dict(),
        }
        return f"{self.axis}-{content_hash(identity)[:16]}"

    def validate(self) -> None:
        validate_semantic_difference(
            self.axis, self.contrast, self.semantics_a, self.semantics_b
        )
        if self.label_a == self.label_b:
            raise ValueError("oracle labels must be opposite")
        if self.premise_status not in {
            "premise_dependent",
            "conjecture_only",
            "ablation_unresolved",
        }:
            raise ValueError(f"unknown premise status: {self.premise_status}")
        if self.modal_depth < 1 or self.quantifier_depth < 0:
            raise ValueError("invalid formula depth")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "pair_id": self.pair_id,
            "axis": self.axis,
            "contrast": list(self.contrast),
            "premises": list(self.premises),
            "conjecture": self.conjecture,
            "semantics_a": self.semantics_a.to_dict(),
            "semantics_b": self.semantics_b.to_dict(),
            "label_a": self.label_a,
            "label_b": self.label_b,
            "oracle_a": self.oracle_a,
            "oracle_b": self.oracle_b,
            "controlled_english": self.controlled_english,
            "canonical_formula_hash": self.canonical_formula_hash,
            "generator_seed": self.generator_seed,
            "modal_depth": self.modal_depth,
            "quantifier_depth": self.quantifier_depth,
            "premise_status": self.premise_status,
            "premise_dependent": self.premise_status == "premise_dependent",
            "alternation_pattern": self.alternation_pattern,
            "metadata": self.metadata,
        }
