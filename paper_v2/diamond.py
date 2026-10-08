"""Diamond set: a harder balanced core with deeper formulas.

Same two diagnostic contrasts and balance as formula-sensitive-1.0 (B<->S4 frames,
cumulative<->decreasing domains; half the formulas valid under each side, so the
condition-only baseline is exactly 50%), but with modal depth 3-4 (frame) or
quantifier depth 2 with modal depth 2-3 (domain), generated rather than templated.

Labels: the valid side is proved by Leo-III/Vampire (paper_v2.oracle.run_side); the
invalid side carries an explicit countermodel found and checked by paper_v2.kripke,
and the provers are also run on it as a cross-check (a proof there aborts the freeze).
Selection is model-independent: candidates are drawn from a seeded generator, ordered by
SHA-256 of their TPTP, and accepted in that order per (axis, orientation, mode) cell.

    python -m paper_v2.diamond --out data/paper_v2/frozen/diamond-1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from . import kripke
from .candidates import Candidate
from .freeze import _hash_text, _oracle_summary, _write_jsonl
from .formula_sensitive import _command, _counts, _validate
from .oracle import build_tptp, run_side
from .render import _semantic_lines
from .schema import Pair
from .semantics import Semantics

SEED = 20261008
CONTRASTS = {"frame": ("B", "S4"), "domain": ("cumulative", "decreasing")}
FRAME_SYMBOLS = ("claim_alpha", "claim_beta", "claim_gamma")
DOMAIN_SYMBOLS = ("marked_alpha", "marked_beta")
CLAUSES = (("the signal is active", "the signal is not active"),
           ("the alarm is sounding", "the alarm is not sounding"),
           ("the gate is open", "the gate is not open"))
PROPERTIES = ("registered", "approved")
SCOPE_LINE = {
    "frame": "Each \"accessible world\" phrase is relative to the world at which that phrase is evaluated.",
    "domain": "Each \"accessible world\" phrase is relative to the world at which that phrase is evaluated, "
              "and \"existing\" means existing at that world.",
}


# ---------------------------------------------------------------- generation

def _frame_formula(rng, k, depth, modal):
    r = rng.random()
    if depth <= 0 or r < 0.2:
        f = ("atom", int(rng.integers(k)))
        return ("not", f) if rng.random() < 0.25 else f
    if modal > 0 and r < 0.6:
        return (("box", "dia")[rng.integers(2)], _frame_formula(rng, k, depth - 1, modal - 1))
    op = ("and", "or", "imp")[rng.integers(3)]
    return (op, _frame_formula(rng, k, depth - 2, modal), _frame_formula(rng, k, depth - 2, modal))


def frame_candidates(rng):
    """Implications A -> C over 2-3 atoms; total modal depth 3-4."""
    while True:
        k = int(rng.integers(2, 4))
        f = ("imp", _frame_formula(rng, k, 4, 2), _frame_formula(rng, k, 5, 3))
        if 3 <= kripke.modal_depth(f) <= 4 and _size(f) <= 18:
            yield f


def _matrix(rng, depth):
    if depth <= 0 or rng.random() < 0.3:
        f = ("pred", int(rng.integers(2)), ("X", "Y")[rng.integers(2)])
        return ("not", f) if rng.random() < 0.25 else f
    return (("and", "or", "imp")[rng.integers(3)], _matrix(rng, depth - 1), _matrix(rng, depth - 1))


def _wrap(prefix, m):
    for op in reversed(prefix):
        m = (op[0], op[1], m) if op[0] in ("all", "ex") else (op[0], m)
    return m


def domain_candidates(rng):
    """Prefix A -> permuted prefix C over the same matrix: two quantifiers (X, Y) and 2-3 modalities.
    Moving a quantifier across a modality is exactly where cumulative and decreasing domains differ."""
    while True:
        prefix = [(("box", "dia")[rng.integers(2)],) for _ in range(int(rng.integers(2, 4)))]
        prefix += [(("all", "ex")[rng.integers(2)], v) for v in ("X", "Y")]
        prefix = [prefix[i] for i in rng.permutation(len(prefix))]
        other = [prefix[i] for i in rng.permutation(len(prefix))]
        m = _matrix(rng, 2)
        if other == prefix or {"X", "Y"} - {g[2] for g in kripke.walk(m) if g[0] == "pred"}:
            continue
        yield ("imp", _wrap(prefix, m), _wrap(other, m))


def _size(f):
    return sum(1 for _ in kripke.walk(f))


# ---------------------------------------------------------------- quality and labels

def _contingent_prop(f, k):
    """Every compound subformula of a modal-free (or matrix) formula is neither a tautology nor a contradiction."""
    def ev(g, v):
        op = g[0]
        if op == "atom":
            return v[g[1]]
        if op == "pred":
            return v[2 * g[1] + (g[2] == "Y")]
        if op == "not":
            return not ev(g[1], v)
        a, b = ev(g[1], v), ev(g[2], v)
        return a and b if op == "and" else a or b if op == "or" else (not a) or b
    for g in kripke.walk(f):
        if g[0] in ("and", "or", "imp", "not") and g[1][0] not in ("atom", "pred"):
            values = {ev(g, v) for v in product((False, True), repeat=k)}
            if len(values) == 1:
                return False
    return True


def _well_formed(f) -> bool:
    for g in kripke.walk(f):
        if g[0] in ("and", "or", "imp") and g[1] == g[2]:
            return False
        if g[0] == "not" and g[1][0] == "not":
            return False
    return True


FRAMES = {}


def _frame_cm(f, system, max_worlds):
    """Exhaustive countermodel search on all frames up to max_worlds (frames cached)."""
    k = max(kripke.n_atoms(f), 1)
    for n in range(1, max_worlds + 1):
        frames = FRAMES.setdefault((n, system), list(kripke.frames(n, system)))
        val = np.array(list(product((False, True), repeat=k * n))).reshape(-1, k, n)
        for R in frames:
            bad = np.argwhere(~kripke._eval_prop(f, val, R))
            if len(bad):
                v, w = bad[0]
                return {"worlds": n, "R": R.astype(int).tolist(), "world": int(w),
                        "valuation": {FRAME_SYMBOLS[i]: val[v][i].astype(int).tolist() for i in range(k)}}
    return None


def _domain_cm(f, domain, seed, rounds):
    cm = kripke.fo_countermodel(f, domain, seed, rounds=rounds)
    if cm:
        cm["P"] = {DOMAIN_SYMBOLS[i]: [row[i] for row in cm["P"]] for i in range(len(cm["P"][0]))}
    return cm


def _subformulas_nontrivial(f, axis, sides) -> bool:
    """No compound subformula is valid or unsatisfiable under either side (T for frames)."""
    for g in kripke.walk(f):
        if g is f or g[0] in ("atom", "pred") or (g[0] == "not" and g[1][0] in ("atom", "pred")):
            continue
        if axis == "domain" and _free_vars(g):
            continue
        for h in (g, ("not", g)):
            if axis == "frame" and _frame_cm(h, "T", 3) is None:
                return False
            if axis == "domain" and any(_domain_cm(h, d, 0, 4) is None for d in sides):
                return False
    return True


def _free_vars(f, bound=()):
    if f[0] == "pred":
        return set() if f[2] in bound else {f[2]}
    if f[0] in ("all", "ex"):
        return _free_vars(f[2], (*bound, f[1]))
    return set().union(*(_free_vars(a, bound) for a in f[1:] if isinstance(a, tuple)))


def classify(f, axis, mode):
    """Return (valid_side, countermodel on the other side) or None if not a clean, non-trivial separation."""
    sides = CONTRASTS[axis]
    if not _well_formed(f):
        return None
    if axis == "frame":
        cms = [_frame_cm(f, s, 3) for s in sides]
    else:
        if not _contingent_prop(_innermost(f[1]), 4):
            return None
        cms = [_domain_cm(f, d, 1, 12) for d in sides]
    if (cms[0] is None) == (cms[1] is None):
        return None
    valid = 0 if cms[0] is None else 1
    # stronger search on the side we will send to the provers
    if axis == "frame" and _frame_cm(f, sides[valid], 4) is not None:
        return None
    if axis == "domain" and _domain_cm(f, sides[valid], 2, 40) is not None:
        return None
    if mode == "nli":  # the premise must matter: the conjecture alone is not valid on the valid side
        alone = _frame_cm(f[2], sides[valid], 3) if axis == "frame" else _domain_cm(f[2], sides[valid], 3, 8)
        if alone is None:
            return None
    if not _subformulas_nontrivial(f, axis, sides):
        return None
    return sides[valid], cms[1 - valid]


def _innermost(f):
    while f[0] in ("box", "dia", "all", "ex"):
        f = f[-1]
    return f


# ---------------------------------------------------------------- rendering

def to_tptp(f, symbols):
    op = f[0]
    if op == "atom":
        return symbols[f[1]]
    if op == "pred":
        return f"{symbols[f[1]]}({f[2]})"
    if op == "not":
        return f"(~ ({to_tptp(f[1], symbols)}))"
    if op in ("and", "or", "imp"):
        c = {"and": "&", "or": "|", "imp": "=>"}[op]
        return f"(({to_tptp(f[1], symbols)}) {c} ({to_tptp(f[2], symbols)}))"
    if op in ("box", "dia"):
        return f"({{${op}}} @ ({to_tptp(f[1], symbols)}))"
    q = "!" if op == "all" else "?"
    return f"({q}[{f[1]}:$i]: ({to_tptp(f[2], symbols)}))"


def _literal(f):
    return f[0] in ("atom", "pred") or (f[0] == "not" and f[1][0] in ("atom", "pred"))


def to_english(f):
    op = f[0]
    if op == "atom":
        return CLAUSES[f[1]][0]
    if op == "pred":
        return f"{f[2].lower()} is {PROPERTIES[f[1]]}"
    if op == "not" and f[1][0] == "atom":
        return CLAUSES[f[1][1]][1]
    if op == "not" and f[1][0] == "pred":
        return f"{f[1][2].lower()} is not {PROPERTIES[f[1][1]]}"
    # A unary operator's scope runs to the end of its bracket, so its operand needs brackets only when it is
    # a binary connective; operands of binary connectives are always bracketed unless they are literals.
    unary = op in ("not", "box", "dia", "all", "ex")
    sub = lambda g: to_english(g) if _literal(g) or (unary and g[0] in ("box", "dia", "all", "ex")) else f"[{to_english(g)}]"
    if op == "not":
        return f"it is not the case that {sub(f[1])}"
    if op == "and":
        return f"{sub(f[1])} and {sub(f[2])}"
    if op == "or":
        return f"{sub(f[1])} or {sub(f[2])}"
    if op == "imp":
        return f"if {sub(f[1])}, then {sub(f[2])}"
    if op == "box":
        return f"in every accessible world, {sub(f[1])}"
    if op == "dia":
        return f"in some accessible world, {sub(f[1])}"
    q = "every" if op == "all" else "some"
    return f"for {q} existing object {f[1].lower()}, {sub(f[2])}"


def _cap(s):
    return s[0].upper() + s[1:]


def render(axis, f, mode, semantics):
    lines = ["Semantic specification:"]
    lines += [f"- {line}" for line in _semantic_lines(axis, semantics)]
    lines += [f"- {SCOPE_LINE[axis]}", ""]
    if mode == "nli":
        lines += ["Premises:", f"1. {_cap(to_english(f[1]))}.", "",
                  f"Conjecture: {_cap(to_english(f[2]))}.", "",
                  "Question: Does the conjecture follow from the premises under this semantic specification?"]
    else:
        lines += [f"Statement: {_cap(to_english(f))}.", "",
                  "Question: Is the statement valid under this semantic specification?"]
    lines.append("Answer only Yes or No.")
    return "\n".join(lines)


def candidate(axis, f, mode) -> Candidate:
    symbols = FRAME_SYMBOLS if axis == "frame" else DOMAIN_SYMBOLS
    used = sorted({g[1] for g in kripke.walk(f) if g[0] in ("atom", "pred")})
    sem = ((Semantics.make("B"), Semantics.make("S4")) if axis == "frame"
           else (Semantics.make("D", domain="cumulative"), Semantics.make("D", domain="decreasing")))
    premises = (to_tptp(f[1], symbols),) if mode == "nli" else ()
    conjecture = to_tptp(f[2] if mode == "nli" else f, symbols)
    c = Candidate(
        axis=axis, contrast=CONTRASTS[axis], premises=premises, conjecture=conjecture,
        ast={"schema": "diamond", "formula": json.loads(json.dumps(f)), "symbols": [symbols[i] for i in used],
             "mode": mode, "diagnostic_contrast": list(CONTRASTS[axis])},
        semantics_a=sem[0], semantics_b=sem[1],
        english_premises=(_cap(to_english(f[1])) + ".",) if mode == "nli" else (),
        english_conjecture=_cap(to_english(f[2] if mode == "nli" else f)) + ".",
        modal_depth=kripke.modal_depth(f), quantifier_depth=kripke.quantifier_depth(f),
        alternation_pattern=None,
    )
    c.validate()
    return c


# ---------------------------------------------------------------- freezing

def _pool(axis, mode, n, seed):
    """n distinct well-formed candidates, ordered by SHA-256 of their TPTP (model-independent)."""
    rng = np.random.default_rng([seed, ("frame", "domain").index(axis), ("validity", "nli").index(mode)])
    gen = frame_candidates(rng) if axis == "frame" else domain_candidates(rng)
    seen = {}
    while len(seen) < n:
        f = next(gen)
        if _well_formed(f):
            key = hashlib.sha256(to_tptp(f, FRAME_SYMBOLS + DOMAIN_SYMBOLS).encode()).hexdigest()
            seen.setdefault(key, f)
    return [seen[k] for k in sorted(seen)]


def _prove(axis, f, mode, valid_side, cm, timeout):
    c = candidate(axis, f, mode)
    sides = CONTRASTS[axis]
    sem = {sides[0]: c.semantics_a, sides[1]: c.semantics_b}
    invalid_side = sides[1 - sides.index(valid_side)]
    proof = run_side(c, sem[valid_side], timeout)
    check = run_side(c, sem[invalid_side], timeout)
    if check["consensus"] is True:
        raise RuntimeError(f"prover proves the countermodel side: {c.conjecture}")
    refutation = {
        "prover_input": build_tptp(c, sem[invalid_side]),
        "consensus": False, "resolution": "explicit_countermodel",
        "provers": check["provers"], "countermodel": cm, "checker": "paper_v2.kripke",
    }
    ok = proof["consensus"] is True and proof["resolution"] != "conflict"
    return c, ok, (proof, refutation) if valid_side == sides[0] else (refutation, proof)


def freeze(out, per_cell=20, timeout=20, workers=8, pool_size=40000, seed=SEED):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    accepted, rejected = [], []
    for axis in ("frame", "domain"):
        for mode in ("validity", "nli"):
            want = {s: per_cell for s in CONTRASTS[axis]}
            pool = _pool(axis, mode, pool_size if axis == "frame" else pool_size // 8, seed)
            screened = ((f, classify(f, axis, mode)) for f in pool)
            with ThreadPoolExecutor(workers) as ex:
                pending = []
                for f, cls in screened:
                    if cls is None or want[cls[0]] <= 0:
                        continue
                    pending.append((f, cls, ex.submit(_prove, axis, f, mode, cls[0], cls[1], timeout)))
                    if sum(want.values()) <= 0:
                        break
                    if len(pending) >= workers:
                        f0, cls0, fut = pending.pop(0)
                        _collect(axis, mode, f0, cls0, fut, want, accepted, rejected)
                    if sum(want.values()) <= 0:
                        break
                for f0, cls0, fut in pending:
                    _collect(axis, mode, f0, cls0, fut, want, accepted, rejected)
            if any(v > 0 for v in want.values()):
                raise RuntimeError(f"cell underfilled {axis}/{mode}: {want}")
            print(axis, mode, "done", flush=True)

    pairs, oracle, prompts = [], [], []
    for i, (axis, mode, f, valid_side, c, (oa, ob)) in enumerate(accepted):
        swap = i % 2 == 1
        sa, sb, la, lb = c.semantics_a, c.semantics_b, oa["consensus"], ob["consensus"]
        pa, pb = render(axis, f, mode, sa), render(axis, f, mode, sb)
        if swap:
            sa, sb, la, lb, oa, ob, pa, pb = sb, sa, lb, la, ob, oa, pb, pa
        pair = Pair(
            axis=axis, contrast=c.contrast, premises=tuple({"tptp": p} for p in c.premises),
            conjecture={"tptp": c.conjecture, "ast": c.ast}, semantics_a=sa, semantics_b=sb,
            label_a=la, label_b=lb, oracle_a=_summary(oa), oracle_b=_summary(ob),
            controlled_english={"prompt_a": pa, "prompt_b": pb}, generator_seed=seed,
            modal_depth=c.modal_depth, quantifier_depth=c.quantifier_depth,
            premise_status="premise_dependent" if mode == "nli" else "conjecture_only",
            metadata={"mode": mode, "formula_valid_under": valid_side},
        )
        pair.validate()
        data = pair.to_dict()
        pairs.append(data)
        oracle.append({"pair_id": pair.pair_id, "side_a": oa, "side_b": ob})
        prompts += [{"pair_id": pair.pair_id, "side": s, "prompt": p, "prompt_hash": _hash_text(p)}
                    for s, p in (("a", pa), ("b", pb))]
    _validate(pairs, per_cell)
    out.mkdir(parents=True)
    for name, rows in (("pairs", pairs), ("oracle", oracle), ("prompts", prompts), ("rejected", rejected)):
        _write_jsonl(out / f"{name}.jsonl", rows)
    manifest = {
        "benchmark": "paper-v2-diamond", "version": out.name.split("-")[-1], "seed": seed,
        "git_commit": _command("git", "rev-parse", "HEAD"),
        "selection": "seeded generator, lowest SHA-256(TPTP) first per axis/mode cell, first per_cell per orientation",
        "pairs_per_cell": per_cell, "pairs": len(pairs), "rejected_after_screening": len(rejected),
        "by_axis": _counts(pairs, lambda r: r["axis"]),
        "by_orientation": _counts(pairs, lambda r: r["metadata"]["formula_valid_under"]),
        "by_mode": _counts(pairs, lambda r: r["metadata"]["mode"]),
        "modal_depth": _counts(pairs, lambda r: str(r["modal_depth"])),
        "quantifier_depth": _counts(pairs, lambda r: str(r["quantifier_depth"])),
        "python": platform.python_version(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


def _summary(record):
    s = _oracle_summary(record)
    if "countermodel" in record:
        s["countermodel"] = record["countermodel"]
    return s


def _collect(axis, mode, f, cls, fut, want, accepted, rejected):
    c, ok, oracles = fut.result()
    if ok and want[cls[0]] > 0:
        want[cls[0]] -= 1
        accepted.append((axis, mode, f, cls[0], c, oracles))
    elif not ok:
        rejected.append({"axis": axis, "mode": mode, "conjecture": c.conjecture, "reason": "valid_side_unproved",
                         "provers": {k: v["szs_status"] for k, v in oracles[0 if cls[0] == CONTRASTS[axis][0] else 1]["provers"].items()}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/paper_v2/frozen/diamond-1.0")
    p.add_argument("--per-cell", type=int, default=20)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    freeze(args.out, args.per_cell, args.timeout, args.workers)


if __name__ == "__main__":
    main()
