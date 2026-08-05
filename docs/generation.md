# Paper-v2 grammar decision

We compared a small native grammar with an equivalent Gramforge grammar before
scaling the benchmark.

## Protocol

Both implementations used the same three atoms and productions:

```text
expression -> atom
expression -> not expression
expression -> expression and expression
expression -> expression or expression
```

Both generated synchronized formal, propositional-English, and
predicate-English surfaces. We canonicalized atom renaming and commutative
children, then measured 1,000 depth-four generations.

## Result

| Implementation | Unique structures | Duplicate rate | Constraint violations | Time |
|---|---:|---:|---:|---:|
| Native bounded grammar | 78 | 0% during enumeration | 0 | 0.016 s |
| Gramforge recursive sampling | 83 | 91.7% | 190/1,000 | 0.309 s |

Gramforge's default sequential mode generated only its first terminal rule.
Recursive mode restored diversity but did not reliably enforce the
sibling-distinctness constraint. Carrying a typed AST also required serializing
it as a string surface and parsing it back because Gramforge renderers expect
string-like outputs.

## Decision

Paper-v2 uses the native grammar. It is still grammar-based rather than a bank of
complete formula templates: recursive productions construct content expressions,
while context-sensitive constraints enforce canonical atom binding, commutative
deduplication, and non-identical siblings. Axis-specific semantic contexts then
place those expressions under modality and quantification.

The legacy generator continues to use Gramforge; this decision applies only to
the smaller paper-v2 grammar.
