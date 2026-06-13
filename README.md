# RAG Evaluation Lab

End-to-end retrieval-augmented generation pipeline, built and evaluated on the `rag-mini-wikipedia` corpus. The project compares a naive top-1 pipeline against an enhanced pipeline with recall-oriented query rewriting and cross-encoder reranking, scoring both with deterministic SQuAD metrics over the full test split and LLM-judged RAGAS metrics on a 100-query evaluation slice.

**Headline finding.** Reranking lifts RAGAS context precision from 69.0% to 86.6% and faithfulness from 67.6% to 78.5%, while exact match drops from 41.5 to 33.7. Grounding quality and literal string accuracy are different axes, and a system that looks worse on string overlap can be substantially better grounded. Measuring both is the point of this project.

## Key Results

RAGAS metrics, 100 samples, judged with GPT-4o-mini:

| System | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| Naive, top-1 | 69.0% | 56.0% | 67.6% | 64.1% |
| Enhanced, rewrite plus rerank | 86.6% | 62.0% | 78.5% | 67.2% |

SQuAD metrics over the full 918-query test split:

| System | Exact Match | F1 |
|---|---|---|
| Naive, top-1 | 41.5 | 49.3 |
| Enhanced, rewrite plus rerank | 33.7 | 42.4 |

Wilson 95% confidence intervals and a two-proportion z-test are reported in `docs/technical_report.md`. Over the 918-query split the enhanced pipeline's EM is significantly lower than the naive baseline (95% CIs 38.4–44.7 vs 30.7–36.8; z=3.47, p<0.001), so the enhancement measurably trades literal accuracy for grounding, while the grounding gains are large and consistent.

## Why the Divergence Matters

Deterministic metrics such as exact match reward surface-form overlap with a reference answer. LLM-judged metrics such as faithfulness and context precision reward grounded, semantically correct answers. The enhanced pipeline retrieves and filters better evidence, then produces more verbose answers that match references less literally. Relying on a single metric family would have led to the wrong conclusion in either direction. The full analysis, including knowledge-leakage caveats where the generator answers from pretraining rather than retrieval, is in `docs/technical_report.md`.

Additional findings from the parameter sweeps:

- Retrieval depth hurts a small generator. With FLAN-T5-base, EM falls from 41.5 at top-1 to 21.0 at top-5 as weakly relevant passages dilute the prompt. Reranking down to top-3 from a top-10 candidate pool recovers precision.
- Bigger embeddings buy little here. MPNet at 768 dimensions edges MiniLM-L6 at 384 dimensions by under one F1 point at triple the encoding cost.
- A verification-style prompt beats the baseline prompt by roughly 2 EM points at top-1.

## Architecture

- Corpus: `rag-datasets/rag-mini-wikipedia` from Hugging Face, roughly 3,200 passages, cleaned and profiled before indexing.
- Vector store: Milvus Lite with an IVF_FLAT index over `all-MiniLM-L6-v2` embeddings.
- Generator: FLAN-T5-base with swappable prompt templates.
- Enhancements: recall-oriented query rewriting plus `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking.
- Evaluation: Hugging Face Evaluate for SQuAD EM and F1, RAGAS with GPT-4o-mini as judge, Wilson confidence intervals, and a two-proportion z-test.

All experiment parameters live in `config/default.yaml` and `config/enhanced.yaml`, so runs are repeatable without editing code or notebooks.

## Repository Map

| Path | Purpose |
|---|---|
| `src/` | Modular pipeline: `naive_rag.py`, `enhanced_rag.py`, `evaluation.py`, `pipeline.py`, `utils.py`, `config.py` |
| `notebooks/complete_analysis.ipynb` | Full end-to-end run with all experiments and visible outputs |
| `notebooks/data_exploration.ipynb` | Corpus profiling and evaluation-slice selection |
| `notebooks/system_evaluation.ipynb` | Smoke test of the modular `src/` pipeline |
| `notebooks/final_analysis.ipynb` | Result visualization with confidence intervals |
| `config/` | YAML experiment configuration |
| `results/` | Exported metrics for every experiment, JSON and CSV |
| `docs/technical_report.md` | Full write-up with statistics and error analysis |
| `AI_USAGE_LOG.md` | Complete log of AI assistance used while building the project |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the `OPENAI_API_KEY` environment variable before running RAGAS cells. The first run downloads the corpus and models from Hugging Face, so it needs network access. Subsequent runs use the local cache.

Run the full scripted evaluation, nine generation configurations plus two RAGAS scoring passes:

```bash
python -m src.pipeline
```

Outputs land in `results/` with a `modular_` prefix. The two RAGAS passes additionally require `OPENAI_API_KEY`; the committed RAGAS metrics in `results/ragas_comparison.csv` and `results/ragas_comparison_delta.csv` come from a prior keyed run, so the pipeline's `modular_ragas_*.csv` files are produced only on a keyed re-run and are not committed in this snapshot. The notebook route through `complete_analysis.ipynb` produces the unprefixed results and includes every chart and intermediate inspection step.

## Limitations

- FLAN-T5-base is a small generator. Results measure pipeline design, not frontier-model capability.
- The 100-query slice keeps RAGAS judging affordable, so confidence intervals are wide.
- RAGAS scores depend on the judge model. GPT-4o-mini was used throughout, and scores from a different judge would shift.
- The corpus overlaps the generator's pretraining data, so some top-1 answers succeed without retrieval. The report discusses this leakage and how to design around it.

## Provenance and AI Use

This project began as graduate coursework at Carnegie Mellon and was built with documented AI assistance. `AI_USAGE_LOG.md` records every session, prompt purpose, and verification step. The headline EM/F1 and RAGAS metrics come from the committed notebook artifacts (`results/naive_results.json`, `results/enhanced_results.json`, `results/ragas_comparison.csv`), which score SQuAD metrics over the full 918-query test split. The scripted pipeline route (`results/modular_*.json`) scores EM/F1 on the 100-query RAGAS subset and so reports different point values (EM 46.0 to 37.0, F1 49.4 to 41.6) while showing the same effect: grounding improves as literal exact-match drops.

## License

MIT
