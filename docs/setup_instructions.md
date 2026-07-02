# Setup Instructions

## 1. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1.5 Configure Parameters

- Edit `config/default.yaml` for paths, model names, retrieval depths, and evaluation settings.
- Use `config/enhanced.yaml` for enhanced pipeline overrides.

## 2. Required Services

- Hugging Face account or anonymous access for `rag-datasets/rag-mini-wikipedia` parquet files (automatic download on first run).
- Optional GPU / Apple MPS support for faster FLAN-T5 inference.
- Personal OpenAI API key (set `OPENAI_API_KEY` or store in `~/.openai_api_key`) for RAGAS evaluation using `gpt-4o-mini`.

## Scripted Execution

The simplest scripted run is `python -m src.pipeline` from the repository root. For programmatic control, drive the pipeline directly:

```bash
python - <<'PY'
from src.pipeline import RAGEvaluationPipeline

pipeline = RAGEvaluationPipeline()
pipeline.run_all()
PY
```

This uses `config/default.yaml` and writes all artefacts to `results/`.

## Execution Plan

1. Activate environment and update configuration (`config/default.yaml`).
2. Run evaluations in this order over the full test split (918 queries):
   - Baseline prompt, top-1.
   - Verification prompt, top-1.
   - MiniLM-L6 (384d) top-3/top-5/top-10.
   - MiniLM-L3 (384d) top-3/top-5/top-10.
   - Enhanced pipeline (rewrite+rerank: base top-10, rerank top-3).
3. Run RAGAS twice (naive vs. enhanced) on the persisted results using the 100-sample evaluation set.
4. Confirm `results/` contains refreshed JSON/CSV artefacts and export the AI usage log.

## 3. Local Data Paths

- Milvus Lite databases stored in `data/` directory:
  - `data/rag_wikipedia_mini.db` (main vector database)
  - `data/rag_enhanced.db` (enhanced pipeline database)
- Results persist under `results/` (`naive_results.json`, `enhanced_results.json`, `comparison_analysis.csv`, etc.). Delete them for a clean run.
- All paths configured in `config/default.yaml` for consistency across modules and notebooks.

## 4. Notebook Execution Order

### Recommended Workflow

1. **`notebooks/data_exploration.ipynb`** (Step 1: Dataset exploration)
   - Load and analyze passages and queries using modular helper functions
   - Generate statistics and visualizations
   - Select evaluation subset

2. **`notebooks/complete_analysis.ipynb`** (Steps 1-7: Comprehensive implementation)
   - Complete end-to-end RAG pipeline
   - Naive RAG ingestion, baseline evaluations
   - Prompt experiments and embedding sweeps
   - Enhanced pipeline (rewrite+rerank)
   - RAGAS evaluation
   - Full reporting and AI usage log

3. **`notebooks/system_evaluation.ipynb`** (Module smoke tests)
   - Test modular pipeline with lightweight settings
   - Validate `src/` module integration
   - Quick verification of system functionality

4. **`notebooks/final_analysis.ipynb`** (Results visualization)
   - Load and analyze results from `results/`
   - Statistical analysis with confidence intervals
   - Publication-ready charts and comparisons

Run cells sequentially; re-execute from the top after restarting kernels.

## 5. Optional Optimisations

- Batch Milvus searches and FLAN-T5 prompts to improve throughput.
- Pin model weights locally (`HF_HOME`, `TRANSFORMERS_CACHE`) for offline or repeated runs.
- Milvus databases already stored in `data/` directory for better organization and deployment readiness.
