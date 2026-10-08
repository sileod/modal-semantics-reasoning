# Modal Semantics Reasoning

Benchmark and evaluation code for testing whether language models follow
explicitly stated modal semantics.

Each benchmark item pairs the same premises and conclusion with two semantic
specifications. One frame or domain condition changes, and the correct answer
changes with it. Labels are checked with automated theorem provers.

Paper: [*Same Formulas, Different Semantics: Do Language Models Follow Modal Logic Specifications?*](https://arxiv.org/abs/2608.05097) (Andrieu and Sileo, 2026).

## Dataset

| Subset | Pairs | Description |
|---|---:|---|
| Balanced core | 160 | Conditions are balanced across labels, preventing a condition-only shortcut. |
| Broad nested set | 800 | Five frame contrasts and three domain contrasts. |
| Diamond | 160 | Harder balanced set: modal depth 3–4 (frame) or two nested quantifiers (domain). |

The complete dataset, prompts, prover records, model responses, and scores are
available on Hugging Face as [ModalFlip](https://huggingface.co/datasets/sileod/ModalFlip).

```python
from datasets import load_dataset

data = load_dataset(
    "sileod/ModalFlip",
    "balanced_core",
    split="train",
)
```

## Install

```bash
git clone https://github.com/sileod/modal-semantics-reasoning
cd modal-semantics-reasoning
pip install -e ".[test]"
pytest -q
```

Python 3.10 or newer is required.

## Usage

Score a response file:

```bash
python -m paper_v2.score \
  --pairs data/paper_v2/frozen/formula-sensitive-1.0/pairs.jsonl \
  --prompts data/paper_v2/frozen/formula-sensitive-1.0/prompts.jsonl \
  --raw responses.jsonl \
  --out scores.json
```

Rebuild the diamond set (needs the provers; deterministic given the seed):

```bash
python -m paper_v2.diamond --out data/paper_v2/frozen/diamond-1.0
```

Generate a small sample:

```bash
python -m paper_v2.smoke --per-axis 10
```

Generation requires the LET embedding, Leo-III, and Vampire. Published labels
can be inspected and scored without installing the provers. See
[`docs/oracle_reproduction.md`](docs/oracle_reproduction.md) for setup details.

## Structure

```text
paper_v2/   generation, rendering, evaluation, and scoring
configs/    benchmark and model settings
data/       released pairs, prompts, and manifests
docs/       semantics and oracle reproduction
tests/      unit and reproducibility tests
```

The benchmark uses controlled English and one modal operator. It varies frame
and domain semantics; multi-agent and name-designation settings are outside its
scope.

## Citation

If you use this benchmark, please cite the paper:

```bibtex
@article{andrieu2026sameformulas,
  title   = {Same Formulas, Different Semantics: Do Language Models Follow Modal Logic Specifications?},
  author  = {Andrieu, R{\'e}mi and Sileo, Damien},
  journal = {arXiv preprint arXiv:2608.05097},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.05097}
}
```

Citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

