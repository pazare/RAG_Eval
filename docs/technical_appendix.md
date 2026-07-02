# Technical Appendix

## Configuration Files

- `config/default.yaml`: master settings for paths, models, retrieval depths, enhancement switches, RAGAS parameters, and random seeds.
- `config/enhanced.yaml`: lightweight override file for enhanced pipeline experiments.

## Run Checklist

- [ ] Configure evaluation parameters in `config/default.yaml`.
- [ ] Baseline run: prompt `baseline`, top-1.
- [ ] Prompt variant run: prompt `verification`, top-1.
- [ ] MiniLM-L6 runs: top-3, top-5, top-10.
- [ ] MiniLM-L3 runs: top-3, top-5, top-10.
- [ ] Enhanced run: rewrite+rerank (base top-10 / rerank top-3).
- [ ] RAGAS scoring: naive (100-sample) and enhanced (100-sample).
- [ ] Persist JSON/CSV outputs and export AI usage log.

## Repository Modules (`src/`)

- `utils.py`: data loading/cleaning helpers, text profiling, plotting utilities, results persistence.
- `naive_rag.py`: Milvus ingestion schema, embedding/encoding helpers, baseline retrieval-generation loop, FLAN-T5 resource management, in-memory dense retriever utilities.
- `enhanced_rag.py`: query rewriting cache, cross-encoder reranking, enhanced RAG orchestration with confidence scoring.
- `evaluation.py`: SQuAD formatting helpers, RAGAS dataset builder with downsampling, SentenceTransformer wrapper, metric extraction.
- `pipeline.py`: end-to-end runner consuming YAML config, executing nine generation runs + RAGAS scoring, persisting JSON/CSV artefacts.
- `config.py`: YAML loader with helper accessors.

## Key Configuration Constants

- `CONTEXT_SEPARATOR = "\n\n---\n\n"`
- `RAGAS_SAMPLE_SIZE = 100`
- `ENHANCED_BASE_TOP_K = 10`, `ENHANCED_RERANK_TOP_K = 3`, `ENHANCED_REWRITE_STRATEGY = 'recall'`
- Milvus path `data/rag_wikipedia_mini.db`, collection `rag_mini` (supplied at runtime from `config/default.yaml`)

## Environment Snapshot (captured via notebook)

| Component | Version |
|-----------|---------|
| Python | 3.12.x |
| PyTorch | 2.8.0 |
| SentenceTransformers | 5.1.1 |
| Transformers | 4.56.2 |
| pymilvus | 2.6.2 |
| ragas | 0.3.5 |
| langchain-openai | 0.3.34 |
| evaluate | 0.4.6 |
| datasets | 4.1.1 |


## Data Sources

- `hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet`
- `hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet`

## Evaluation Scope

- Baseline and enhanced generation runs cover the full test split (918 queries), scored with SQuAD EM/F1.
- RAGAS analysis scores a 100-query subset (LLM judging is the cost bottleneck).

## Result Artifacts

- `results/modular_naive_results.json`
- `results/modular_enhanced_results.json`
- `results/modular_prompt_comparison.csv`
- `results/modular_embedding_experiments.csv`
- `results/modular_enhanced_summary.csv`
- `results/modular_ragas_comparison.csv` (produced only on a keyed pipeline re-run; remains uncommitted in this snapshot)
- `results/modular_ragas_delta.csv` (produced only on a keyed pipeline re-run; remains uncommitted in this snapshot)

## AI Usage Log

Use notebook helpers:

```python
record_ai_usage(tool, purpose, input_summary, output_usage, verification)
export_ai_usage_log()
```

Store exported JSON in `results/` and reference it in the final report appendices.

## Reproducibility Checklist

- Fixed random seed (42) for the RAGAS evaluation slice; FLAN-T5 generation is deterministic (greedy decoding).
- Cached HF datasets/models after first download.
- Milvus Lite ingestion performed via scripted helpers (`ingest_records`).
- Full 100-sample RAGAS evaluation for comprehensive system assessment.

## Future Work Items

- Batch generation and search operations for throughput.
- Promote notebooks into scripted pipelines once experiments stabilise.
- Automate report generation (Markdown → PDF) once metrics finalise.
