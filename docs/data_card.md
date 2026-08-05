# Paper-v2 data card

The scaled-0.2 release contains 800 contrastive pairs: 400 frame pairs and 400
domain pairs. Each pair keeps its formal problem fixed while changing one
semantic condition. Strict pair accuracy requires both sides to be correct.

The primary surface is deterministic controlled English. Formal inputs, HOL
translations, prover commands, versions, statuses, runtimes, and output hashes
are retained for audit. Model calls were made only after the dataset was frozen.

The benchmark tests semantic-parameter following, not broad natural-language
understanding. It uses no stories, proper names, multi-agent interaction,
designation contrast, or paraphrase augmentation.

Scaled-0.2 uses the documented clean portfolio policy: decisive valid-side proof
evidence, decisive invalid-side countermodel evidence, and no contradictory
prover result. Of 1,600 accepted sides, 727 have dual-prover agreement and 873
have one decisive prover without contradiction. The release contains 400
validity and 400 NLI pairs; every Frame contrast has 80 pairs, and every
contrast is balanced between validity and NLI. Modal and quantifier depths are
fixed within contrast and reported with AST sizes and oracle coverage in the
generated benchmark-audit table.

Because every retained contrast moves from a weaker to a nested stronger
semantic class, the condition name carries label information. The released
scorer therefore includes an optimal name-only baseline (60% Frame, 67%
Domain), within-condition balanced accuracy for T, B, and cumulative settings,
and label-controlled affinity diagnostics. Formula-skeleton clusters replace
predicates and propositions with placeholders; there are 61--93 per contrast,
with largest cluster size five, and primary intervals resample these clusters.

Premise ablation labels 395 pairs premise-dependent, 400 conjecture-only, and
five unresolved. Generated tables report strict accuracy by this status and
both official and parse-conditional accuracy; the latter is diagnostic only.
