# Oracle reproduction

The existing repository provides:

- `tools/logic-embedding.jar` (LET/embedproblem);
- `tools/vampire`;
- `tools/leo3.jar`.

Paper-v2 calls these through a single oracle wrapper and retains the exact
input and output provenance for each side. Both provers must agree when both are
decisive. Contradictions are always rejected; timeout or unknown is never
interpreted as invalidity.

For scaled-0.1, a side may be retained when one prover is decisive and the other
is unknown or times out, provided there is no contradiction. The manifest
separates dual-agreement and one-prover-only sides. This pre-final policy is
necessary because Leo-III supplies validity proofs while Vampire supplies the
countermodels in the current environment; it must not be described as
dual-prover agreement.
