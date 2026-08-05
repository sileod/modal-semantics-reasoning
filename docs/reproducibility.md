# Paper-v2 reproducibility boundary

## What reproduces without network access

`make verify-paper` reparses the frozen raw responses, reruns every scorer,
regenerates all paper artifacts, compiles the PDF, and runs ACL PubCheck. It
does not call a model API. Dataset, prompt, and oracle files are content-hashed;
raw model responses retain prompt hashes, request parameters, timestamps,
provider identifiers, usage, costs, and returned reasoning content. The oracle
manifest records the generation commit, seed, Python version, and exact LET,
Leo-III, and Vampire versions.

The Python dependency export pins versions and hashes. The current development
environment should not be treated as the reference environment: it resolves
`litlm` from a local checkout and its installed `litellm` version may differ
from `requirements.txt`. Published artifact reconstruction depends only on the
frozen JSON/JSONL files and scoring dependencies, not on another API call.

## What cannot be bit-for-bit reproduced

New API responses are not deterministic artifacts, even at temperature zero.
Production uses dated identifiers where providers expose them, but OpenRouter
may still change serving infrastructure. Re-running inference therefore tests
replication rather than byte-identical reproduction; the returned provider
model and route are recorded for every request.

## Production diagnostics

`production-1.2-diagnostics` leaves `production-1.1` untouched. It contains a
50-pair, five-contrast representation study for the main-table Terra endpoint
and high-effort reasoning on all 400 Domain pairs for DeepSeek V4 Flash. The
representation prompts and Domain prompts are deterministic partitions of the
same frozen pairs and normalized production prompts. Raw responses, scored
files, costs, and manifests live under version-matched result directories.

`make paper-v2-diagnostics-score` reconstructs these diagnostic scores without
network access. `make verify-paper` includes that command before rebuilding the
paper artifacts.

## Reasoning mini-experiment

The accepted mini-set has 50 Frame pairs, 10 per contrast. GPT-4.1 compares a
fresh direct run with a prompt requesting at most five sentences of reasoning
and a final `Answer: Yes/No` marker. DeepSeek V4 Flash compares fresh direct
answers with the identical prompt in `reasoning.effort: high` mode
and a 2,048-token output allowance. The scorer reparses the immutable raw text;
stored parse fields are metadata rather than the scoring source of truth.

The earlier 256-token GPT rationale run and 512/1,024-token DeepSeek reasoning
runs are retained as rejected configuration diagnostics because their output
budgets truncated final answers.
