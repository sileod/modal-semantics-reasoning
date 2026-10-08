"""Small Kripke-model checker for countermodel search on the diamond set.

Formulas are nested tuples:
    ("atom", i) | ("pred", i, var) | ("not", f) | ("and", f, g) | ("or", f, g) | ("imp", f, g)
    ("box", f) | ("dia", f) | ("all", var, f) | ("ex", var, f)
Propositional formulas are checked on every frame up to a size bound; first-order ones
(actualist quantifiers, unary predicates, variables "X"/"Y") on random models.
"""
from __future__ import annotations

from itertools import product

import numpy as np

FRAME_PROPERTIES = {
    "K": (), "D": ("serial",), "T": ("reflexive",), "B": ("reflexive", "symmetric"),
    "S4": ("reflexive", "transitive"), "S5": ("reflexive", "symmetric", "transitive"),
}


def has_properties(R: np.ndarray, properties) -> bool:
    for p in properties:
        if p == "serial" and not R.any(axis=1).all():
            return False
        if p == "reflexive" and not R.diagonal().all():
            return False
        if p == "symmetric" and not (R == R.T).all():
            return False
        if p == "transitive" and ((R.astype(int) @ R.astype(int) > 0) & ~R).any():
            return False
    return True


def frames(n: int, system: str):
    props = FRAME_PROPERTIES[system]
    for bits in product((False, True), repeat=n * n):
        R = np.array(bits).reshape(n, n)
        if has_properties(R, props):
            yield R


def walk(f):
    yield f
    for a in f[1:]:
        if isinstance(a, tuple):
            yield from walk(a)


def modal_depth(f) -> int:
    kids = [modal_depth(a) for a in f[1:] if isinstance(a, tuple)]
    return (f[0] in ("box", "dia")) + max(kids, default=0)


def quantifier_depth(f) -> int:
    kids = [quantifier_depth(a) for a in f[1:] if isinstance(a, tuple)]
    return (f[0] in ("all", "ex")) + max(kids, default=0)


def n_atoms(f) -> int:
    return 1 + max((g[1] for g in walk(f) if g[0] in ("atom", "pred")), default=-1)


# ---- propositional: values are bool arrays [V, W] (valuations x worlds)

def _eval_prop(f, val, R):
    op = f[0]
    if op == "atom":
        return val[:, f[1], :]
    if op == "not":
        return ~_eval_prop(f[1], val, R)
    if op in ("and", "or", "imp"):
        a, b = _eval_prop(f[1], val, R), _eval_prop(f[2], val, R)
        return a & b if op == "and" else a | b if op == "or" else ~a | b
    if op in ("box", "dia"):
        a = _eval_prop(f[1], val, R).astype(np.int32)
        if op == "box":  # true at w iff no successor falsifies a
            return ((1 - a) @ R.T.astype(np.int32)) == 0
        return (a @ R.T.astype(np.int32)) > 0
    raise ValueError(op)


def prop_countermodel(f, system: str, max_worlds: int = 3):
    """Smallest (frame, valuation, world) falsifying f in `system`, or None."""
    k = max(n_atoms(f), 1)
    for n in range(1, max_worlds + 1):
        val = np.array(list(product((False, True), repeat=k * n))).reshape(-1, k, n)
        for R in frames(n, system):
            truth = _eval_prop(f, val, R)
            bad = np.argwhere(~truth)
            if len(bad):
                v, w = bad[0]
                return {"worlds": n, "R": R.astype(int).tolist(), "world": int(w),
                        "valuation": val[v].astype(int).tolist()}
    return None


# ---- first-order: values are bool arrays [M, W, O, O] (models x worlds x X x Y)

def _eval_fo(f, m):
    op = f[0]
    if op == "pred":
        ext = m["P"][:, :, f[1], :]  # [M, W, O]
        return ext[:, :, :, None] if f[2] == "X" else ext[:, :, None, :]
    if op == "not":
        return ~_eval_fo(f[1], m)
    if op in ("and", "or", "imp"):
        a, b = _eval_fo(f[1], m), _eval_fo(f[2], m)
        return a & b if op == "and" else a | b if op == "or" else ~a | b
    if op in ("box", "dia"):
        a = np.broadcast_to(_eval_fo(f[1], m), m["shape"]).astype(np.int32)
        R = m["R"].astype(np.int32)
        if op == "box":
            return np.einsum("mwv,mvxy->mwxy", R, 1 - a) == 0
        return np.einsum("mwv,mvxy->mwxy", R, a) > 0
    if op in ("all", "ex"):
        a = np.broadcast_to(_eval_fo(f[2], m), m["shape"])
        axis = 2 if f[1] == "X" else 3
        D = m["D"][:, :, :, None] if axis == 2 else m["D"][:, :, None, :]
        r = (a | ~D).all(axis=axis, keepdims=True) if op == "all" else (a & D).any(axis=axis, keepdims=True)
        return r
    raise ValueError(op)


def random_fo_models(rng, n_models, worlds, objects, n_preds, domain: str, system: str = "D"):
    props = FRAME_PROPERTIES[system]
    R = rng.random((n_models * 4, worlds, worlds)) < 0.5
    if props == ("serial",):
        R = R[R.any(axis=2).all(axis=1)][:n_models]
    else:
        R = R[[has_properties(r, props) for r in R]][:n_models]
    D = rng.random((len(R), worlds, objects)) < 0.6
    D[:, :, 0] |= ~D.any(axis=2)  # non-empty domains
    # repair domains along accessibility: cumulative = D(w) <= D(v), decreasing = D(v) <= D(w)
    for _ in range(worlds):
        for w, v in product(range(worlds), repeat=2):
            link = R[:, w, v][:, None]
            if domain == "cumulative":
                D[:, v] |= link & D[:, w]
            elif domain == "decreasing":
                D[:, w] |= link & D[:, v]
    if domain == "constant":
        D[:] = True
    P = rng.random((len(R), worlds, n_preds, objects)) < 0.5
    return {"R": R, "D": D, "P": P, "shape": (len(R), worlds, objects, objects)}


def check_domain(D, R, domain):
    for w, v in zip(*np.nonzero(R)):
        if domain == "cumulative" and (D[w] & ~D[v]).any():
            return False
        if domain == "decreasing" and (D[v] & ~D[w]).any():
            return False
    return True


_BANKS = {}


def _bank(seed, r, n_models, domain, system, k):
    """Model banks are deterministic in (seed, round) and cached, so screening many formulas is cheap."""
    key = (seed, r, n_models, domain, system, k)
    if key not in _BANKS:
        worlds, objects = ((2, 2), (3, 2), (3, 3), (4, 2))[r % 4]
        _BANKS[key] = random_fo_models(np.random.default_rng([seed, r]), n_models, worlds, objects, k, domain, system)
    return _BANKS[key]


def fo_countermodel(f, domain: str, seed: int = 0, system: str = "D", rounds: int = 20, n_models: int = 4000):
    k = 2  # the diamond set uses two unary predicates
    assert n_atoms(f) <= k
    for r in range(rounds):
        m = _bank(seed, r, n_models, domain, system, k)
        worlds, objects = m["D"].shape[1:]
        truth = np.broadcast_to(_eval_fo(f, m), m["shape"])[:, :, 0, 0]  # closed: X, Y unused
        bad = np.argwhere(~truth)
        if len(bad):
            i, w = bad[0]
            assert check_domain(m["D"][i], m["R"][i], domain) and has_properties(m["R"][i], FRAME_PROPERTIES[system])
            return {"worlds": worlds, "objects": objects, "R": m["R"][i].astype(int).tolist(),
                    "D": m["D"][i].astype(int).tolist(), "P": m["P"][i].astype(int).tolist(), "world": int(w)}
    return None
