# Paper-v2 semantics

Paper-v2 asks whether a model changes its entailment judgment when one declared
semantic parameter changes. A pair has identical premises and conjecture,
opposite oracle labels, and exactly one changed axis.

## Frame

The public systems are K, D, T, B, S4, and S5. The data also stores their
explicit frame properties:

| System | Serial | Reflexive | Symmetric | Transitive |
|---|---:|---:|---:|---:|
| K | | | | |
| D | yes | | | |
| T | | yes | | |
| B | | yes | yes | |
| S4 | | yes | | yes |
| S5 | | yes | yes | yes |

Only K–D, K–T, T–B, T–S4, and B–S5 are benchmark contrasts. Their symmetric
differences are respectively seriality, reflexivity, symmetry, transitivity,
and transitivity.

## Domain

The semantics uses a common object universe and a nonempty local domain at each
world. Quantifiers are actualist: they range over the local domain of the world
of evaluation. Variable assignments persist across modal evaluation. Predicate
extensions are world-relative over the common universe and unconstrained on
objects outside the local domain; existence guards restrict quantification.

- varying: no monotonicity constraint;
- cumulative: objects cannot disappear along accessibility;
- decreasing: new objects cannot appear along accessibility;
- constant: the same objects exist at every world.

All domain pairs use system D, variables and predicates, and no constants or
functions. This guarantees an accessible world while avoiding designation
effects.

## Designation

Rigid descriptions denote the same object at every world. Flexible descriptions
may denote different objects at different worlds. Designation pairs use system D,
constant domains, local term interpretation, and neutral description symbols.

The bundled LET embedding distinguishes rigid and flexible designation only in
its higher-order translation: rigid constants retain type `$i`, while flexible
constants become functions from worlds to individuals. The first diagnostic
showed that Leo-III proves the rigid case, but the flexible case was unresolved;
Vampire timed out on both higher-order problems. Therefore designation candidates
must never be accepted from failure to prove. This axis remains behind the
400-pair go/no-go gate.

## Consequence relation

Evaluation starts at a designated current world. Ordinary premises are local
hypotheses and conjectures are local. Every accepted pair is also rerun without
its premises and receives one of:
`premise_dependent`, `conjecture_only`, or `ablation_unresolved`.

## Formula-sensitive diagnostic

The primary benchmark contrasts nested model classes, which makes a semantic
condition predictive of label direction. The separate `formula-sensitive-1.0`
diagnostic uses two non-nested contrasts:

- B versus S4 (symmetry versus transitivity on a reflexive base);
- cumulative versus decreasing domains (growth-only versus shrink-only).

Each condition occurs equally often with both labels. Valid sides use the same
LET/Leo-III/Vampire pipeline. Every invalid side additionally ships with a
two- or three-world Kripke countermodel checked by
`paper_v2.formula_sensitive._evaluate_witness`; the exact witness and checker
name are stored in `oracle.jsonl`.

Model prompts state the defining rules directly (for example, “reflexive and
symmetric” or “objects cannot disappear”) and do not expose conventional
labels such as B, S4, cumulative, or decreasing. Those labels are metadata and
paper-facing shorthand only.

Multi-agent systems, interaction axioms, and local/global term interpretation
are legacy experiments and are not paper-v2 axes.
