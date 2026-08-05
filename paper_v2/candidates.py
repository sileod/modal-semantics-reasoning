from __future__ import annotations

from dataclasses import dataclass

from .grammar import (
    generate_expressions,
    render_clause,
    render_formula,
    render_property,
)
from .semantics import Semantics, validate_semantic_difference


@dataclass(frozen=True)
class Candidate:
    axis: str
    contrast: tuple[str, str]
    premises: tuple[str, ...]
    conjecture: str
    ast: dict
    semantics_a: Semantics
    semantics_b: Semantics
    english_premises: tuple[str, ...]
    english_conjecture: str
    modal_depth: int
    quantifier_depth: int
    alternation_pattern: str | None = None

    def validate(self) -> None:
        validate_semantic_difference(
            self.axis, self.contrast, self.semantics_a, self.semantics_b
        )
        text = " ".join((*self.premises, self.conjecture))
        if self.axis == "frame" and any(token in text for token in ("![", "?[", "$i")):
            raise ValueError("frame candidates must be propositional")
        if self.axis == "domain" and "the_" in text:
            raise ValueError("domain candidates cannot contain constants")
        if self.axis == "designation" and "the_" not in text:
            raise ValueError("designation candidates require description constants")


FRAME_TEMPLATES = {
    ("K", "D"): lambda p: f"({{$box}} @ ({p}) => {{$dia}} @ ({p}))",
    ("K", "T"): lambda p: f"({{$box}} @ ({p}) => {p})",
    ("T", "B"): lambda p: f"({p} => {{$box}} @ ({{$dia}} @ ({p})))",
    ("T", "S4"): lambda p: (
        f"({{$box}} @ ({p}) => {{$box}} @ ({{$box}} @ ({p})))"
    ),
    ("B", "S5"): lambda p: (
        f"({{$dia}} @ ({p}) => {{$box}} @ ({{$dia}} @ ({p})))"
    ),
}

FRAME_ENGLISH = {
    ("K", "D"): lambda p: (
        f"If it is necessary that {p}, then it is possible that {p}."
    ),
    ("K", "T"): lambda p: f"If it is necessary that {p}, then {p}.",
    ("T", "B"): lambda p: (
        f"If {p}, then at every accessible world it is possible that {p}."
    ),
    ("T", "S4"): lambda p: (
        f"If it is necessary that {p}, then at every accessible world it remains "
        f"necessary that {p}."
    ),
    ("B", "S5"): lambda p: (
        f"If it is possible that {p}, then it is necessarily possible that {p}."
    ),
}

FRAME_NLI = {
    ("K", "D"): (
        lambda p: f"{{$box}} @ ({p})",
        lambda p: f"{{$dia}} @ ({p})",
        lambda clause: f"At every accessible world, the following holds: {clause}.",
        lambda clause: f"At some accessible world, the following holds: {clause}.",
    ),
    ("K", "T"): (
        lambda p: f"{{$box}} @ ({p})",
        lambda p: p,
        lambda clause: f"At every accessible world, the following holds: {clause}.",
        lambda clause: f"At the current world, the following holds: {clause}.",
    ),
    ("T", "B"): (
        lambda p: p,
        lambda p: f"{{$box}} @ ({{$dia}} @ ({p}))",
        lambda clause: f"At the current world, the following holds: {clause}.",
        lambda clause: (
            "At every world accessible from the current world, there is a world "
            f"accessible from that world where the following holds: {clause}."
        ),
    ),
    ("T", "S4"): (
        lambda p: f"{{$box}} @ ({p})",
        lambda p: f"{{$box}} @ ({{$box}} @ ({p}))",
        lambda clause: f"At every accessible world, the following holds: {clause}.",
        lambda clause: (
            "At every world accessible from the current world, the following "
            f"holds at every world accessible from that world: {clause}."
        ),
    ),
    ("B", "S5"): (
        lambda p: f"{{$dia}} @ ({p})",
        lambda p: f"{{$box}} @ ({{$dia}} @ ({p}))",
        lambda clause: f"At some accessible world, the following holds: {clause}.",
        lambda clause: (
            "At every world accessible from the current world, there is a world "
            f"accessible from that world where the following holds: {clause}."
        ),
    ),
}

DOMAIN_TEMPLATES = {
    ("varying", "cumulative"): (
        "box-forall",
        lambda atom: (
            f"({{$box}} @ (![X:$i]: ({atom})) => "
            f"![X:$i]: ({{$box}} @ ({atom})))"
        ),
        "If necessarily every object is marked, then every object is necessarily marked.",
    ),
    ("varying", "decreasing"): (
        "forall-box",
        lambda atom: (
            f"(![X:$i]: ({{$box}} @ ({atom})) => "
            f"{{$box}} @ (![X:$i]: ({atom})))"
        ),
        "If every object is necessarily marked, then necessarily every object is marked.",
    ),
    ("cumulative", "constant"): (
        "forall-box",
        lambda atom: (
            f"(![X:$i]: ({{$box}} @ ({atom})) => "
            f"{{$box}} @ (![X:$i]: ({atom})))"
        ),
        "If every object is necessarily marked, then necessarily every object is marked.",
    ),
    ("decreasing", "constant"): (
        "box-forall",
        lambda atom: (
            f"({{$box}} @ (![X:$i]: ({atom})) => "
            f"![X:$i]: ({{$box}} @ ({atom})))"
        ),
        "If necessarily every object is marked, then every object is necessarily marked.",
    ),
}

DOMAIN_NLI = {
    ("varying", "cumulative"): (
        "box-forall",
        lambda atom: f"{{$box}} @ (![X:$i]: ({atom}))",
        lambda atom: f"![X:$i]: ({{$box}} @ ({atom}))",
        "At every accessible world, every object existing there is registered.",
        "Every object existing at the current world is registered at every accessible world.",
    ),
    ("varying", "decreasing"): (
        "forall-box",
        lambda atom: f"![X:$i]: ({{$box}} @ ({atom}))",
        lambda atom: f"{{$box}} @ (![X:$i]: ({atom}))",
        "Every object existing at the current world is registered at every accessible world.",
        "At every accessible world, every object existing there is registered.",
    ),
    ("cumulative", "constant"): (
        "forall-box",
        lambda atom: f"![X:$i]: ({{$box}} @ ({atom}))",
        lambda atom: f"{{$box}} @ (![X:$i]: ({atom}))",
        "Every object existing at the current world is registered at every accessible world.",
        "At every accessible world, every object existing there is registered.",
    ),
    ("decreasing", "constant"): (
        "box-forall",
        lambda atom: f"{{$box}} @ (![X:$i]: ({atom}))",
        lambda atom: f"![X:$i]: ({{$box}} @ ({atom}))",
        "At every accessible world, every object existing there is registered.",
        "Every object existing at the current world is registered at every accessible world.",
    ),
}


def generate_candidates(
    axis: str,
    contrast: tuple[str, str],
    seed: int,
    n: int,
    mode: str = "validity",
) -> list[Candidate]:
    del seed  # Templates are deterministic; the seed is retained by the caller.
    if mode not in {"validity", "nli"}:
        raise ValueError(f"unknown candidate mode: {mode}")
    if axis == "designation":
        return [_designation_candidate(index) for index in range(n)]
    expressions = generate_expressions(n)
    if len(expressions) != n:
        raise ValueError(f"grammar produced only {len(expressions)} of {n} expressions")
    if axis == "frame":
        return [
            _frame_candidate(contrast, expression, mode)
            for expression in expressions
        ]
    if axis == "domain":
        return [
            _domain_candidate(contrast, expression, mode)
            for expression in expressions
        ]
    raise ValueError(f"unknown paper-v2 axis: {axis}")


def _frame_candidate(
    contrast: tuple[str, str], expression, mode: str
) -> Candidate:
    symbols = (
        "claim_alpha",
        "claim_beta",
        "claim_gamma",
        "claim_delta",
        "claim_epsilon",
    )
    content = render_formula(expression, symbols)
    clause = render_clause(expression)
    if mode == "nli":
        premise, conjecture, english_premise, english_conjecture = FRAME_NLI[contrast]
        premises = (premise(content),)
        english_premises = (english_premise(clause),)
        formula = conjecture(content)
        english_conjecture = english_conjecture(clause)
    else:
        premises = ()
        english_premises = ()
        formula = FRAME_TEMPLATES[contrast](content)
        english_conjecture = FRAME_ENGLISH[contrast](clause)
    candidate = Candidate(
        axis="frame",
        contrast=contrast,
        premises=premises,
        conjecture=formula,
        ast={
            "schema": "frame",
            "contrast": contrast,
            "content": expression.canonical(),
            "symbols": [symbols[index] for index in expression.atoms()],
            "mode": mode,
        },
        semantics_a=Semantics.make(contrast[0]),
        semantics_b=Semantics.make(contrast[1]),
        english_premises=english_premises,
        english_conjecture=english_conjecture,
        modal_depth=2 if contrast in {("T", "B"), ("T", "S4"), ("B", "S5")} else 1,
        quantifier_depth=0,
    )
    candidate.validate()
    return candidate


def _domain_candidate(
    contrast: tuple[str, str], expression, mode: str
) -> Candidate:
    symbols = (
        "marked_alpha",
        "marked_beta",
        "marked_gamma",
        "marked_delta",
        "marked_epsilon",
    )
    atom = render_formula(expression, symbols, variable="X")
    property_phrase = render_property(expression)
    if mode == "nli":
        pattern, premise, conjecture, _, _ = (
            DOMAIN_NLI[contrast]
        )
        premises = (premise(atom),)
        formula = conjecture(atom)
        english_premise, english_conjecture = _domain_english(
            pattern, property_phrase, mode
        )
        english_premises = (english_premise,)
    else:
        pattern, template, _ = DOMAIN_TEMPLATES[contrast]
        premises = ()
        formula = template(atom)
        english_premises = ()
        _, english_conjecture = _domain_english(pattern, property_phrase, mode)
    candidate = Candidate(
        axis="domain",
        contrast=contrast,
        premises=premises,
        conjecture=formula,
        ast={
            "schema": pattern,
            "content": expression.canonical(),
            "symbols": [symbols[index] for index in expression.atoms()],
            "mode": mode,
        },
        semantics_a=Semantics.make("D", domain=contrast[0]),
        semantics_b=Semantics.make("D", domain=contrast[1]),
        english_premises=english_premises,
        english_conjecture=english_conjecture,
        modal_depth=1,
        quantifier_depth=1,
        alternation_pattern=pattern,
    )
    candidate.validate()
    return candidate


def _domain_english(pattern: str, property_phrase: str, mode: str) -> tuple[str, str]:
    accessible = (
        "At every accessible world, every object existing there has status "
        f"{property_phrase}."
    )
    current = (
        "Every object existing at the current world has status "
        f"{property_phrase} at every accessible world."
    )
    premise, conjecture = (
        (accessible, current) if pattern == "box-forall" else (current, accessible)
    )
    if mode == "nli":
        return premise, conjecture
    return "", f"If {premise[:-1].lower()}, then {conjecture[:-1].lower()}."


def _designation_candidate(index: int) -> Candidate:
    left = ("the_director", "the_winner", "the_owner")[index % 3]
    right = f"the_counterpart_{index}"
    candidate = Candidate(
        axis="designation",
        contrast=("rigid", "flexible"),
        premises=(f"({left} = {right})",),
        conjecture=f"{{$box}} @ ({left} = {right})",
        ast={"schema": "identity-preservation", "left": left, "right": right},
        semantics_a=Semantics.make("D", designation="rigid"),
        semantics_b=Semantics.make("D", designation="flexible"),
        english_premises=(f"{left} and {right} denote the same object.",),
        english_conjecture=f"Necessarily, {left} and {right} denote the same object.",
        modal_depth=1,
        quantifier_depth=0,
    )
    candidate.validate()
    return candidate
