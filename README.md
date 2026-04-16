<div align="center">

# ADC: Abstractive Document Comprehension

<p><em>A benchmark for document-level, synthesis-heavy question answering in RAG systems.</em></p>

[![Paper](https://img.shields.io/badge/%F0%9F%93%84%20Paper-arXiv%20coming%20soon-b31b1b?style=flat-square)](#)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face%20coming%20soon-f9ab00?style=flat-square)](#)
[![Webpage](https://img.shields.io/badge/%F0%9F%8C%90%20Webpage-coming%20soon-2563eb?style=flat-square)](#)

</div>

> ADC evaluates whether a RAG system can read long documents, organize evidence, and produce grounded abstractive answers, rather than only retrieve short facts.

<p align="center">
  <img src="figures/alpha.png" alt="ADC overview: topic-set evaluation versus head-to-head evaluation" width="96%">
</p>

<p align="center"><em>Overview of ADC and its topic-set-based evaluation paradigm.</em></p>

## Why ADC?

Many RAG benchmarks are still dominated by extractive or short-answer QA.
ADC is designed for a harder setting: answers that require summarization, comparison, enumeration, and temporal synthesis over full documents.

ADC focuses on two gaps:

- **Benchmark gap**: existing datasets rarely provide stable, human-curated abstractive references for document-level QA.
- **Evaluation gap**: head-to-head judging is expensive, hard to scale, and often weak at explaining *why* one answer is better.

## What ADC Covers

ADC spans academic papers and news documents, and organizes questions into five abstractive task families:

- **Single-Sum**: summarize a single document into a compact, faithful answer
- **Pair-Comp**: compare two documents, methods, entities, or events
- **Multi-Comp**: synthesize structured comparisons across multiple targets
- **Enum**: enumerate key items, themes, contributions, or findings
- **Temp**: reconstruct temporally evolving events over a time window

To study retrieval difficulty, each question is evaluated under three retrieval scopes:

- **Simple**: tightly scoped evidence only
- **Middle**: a broader but still related corpus
- **Hard**: a distractor-heavy task-level corpus

<p align="center">
  <img src="figures/workflow.png" alt="ADC workflow: data collection, QA generation, and QA refinement" width="96%">
</p>

<p align="center"><em>ADC construction workflow: data collection, QA generation, and QA refinement.</em></p>

## Reference-Based Evaluation

ADC pairs each question with a curated reference topic set and a hallucination set.
Instead of relying only on pairwise preference judgments, we evaluate whether a model:

- covers the key reference topics
- avoids unsupported or hallucinated topics

This gives a more interpretable view of abstractive RAG quality through topic-level coverage and hallucination analysis.

We will add a fuller evaluation guide later.
For now, the intended entry point is:

```bash
# TODO: add a complete public evaluation example
python eval_adc/eval_adc.py \
  --response_path <PATH_TO_MODEL_OUTPUT> \
  --save_path <PATH_TO_SAVE_RESULTS> \
  --difficulty <simple_QA|middle_QA|hard_QA> \
  --question_path <PATH_TO_QUESTION_JSON>
```

## Dataset Snapshot

The current paper-side statistics of ADC are summarized below.
This snapshot follows the five question categories used in the benchmark.
`Tok.` denotes tokens, and `#C` denotes middle-level clusters.

| Task Type | #Q | #Docs | Corpus Tok. | Avg Mid Tok. | #C |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-Sum | 422 | 422 | 9,681,570 | 322,719 | 30 |
| Pair-Comp | 99 | 54 | 1,565,393 | 597,178 | 5 |
| Multi-Comp | 42 | 59 | 1,670,693 | 457,473 | 5 |
| Enum | 150 | 63 | 1,728,257 | 427,368 | 7 |
| Temp | 156 | 1,579 | 1,434,193 | 120,514 | 7 |
| **Total** | **869** | **2,096** | **16,080,106** | **347,963** | **54** |

ADC is designed to stress both abstractive answer quality and retrieval scope under increasingly challenging corpus settings.

## Repository Status

This repository is still being organized.
We are gradually cleaning up and releasing:

- benchmark data and metadata
- evaluation scripts for ADC
- reproducible experiment pipelines
- documentation for running baselines

At the moment, the most complete public piece in this repo is the evaluation code under [`eval_adc/`](./eval_adc).

## Repository Layout

```text
ADC/
├── eval_adc/      # ADC evaluation scripts and aggregation utilities
├── notebooks/     # exploratory analysis and helper notebooks
└── README.md
```

## Planned Links

- `arXiv`: [TBD]
- `Hugging Face dataset`: [TBD]
- `Webpage`: [TBD]

## Citation

If you find ADC useful, please cite our paper once it is released.

```bibtex
@article{adc_benchmark_2026,
  title   = {Abstractive Document Comprehension: A Comprehensive Benchmark for RAG-based Question Answering},
  author  = {TBD},
  journal = {arXiv},
  year    = {2026}
}
```
