# Assignment 2 · Ground the Domain RAG

Modular RAG system for CMU's Assignment 2 (RAG Mini Wikipedia) with organized data flow and reusable source modules.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.8.0, SentenceTransformers 5.1.1, Transformers 4.56.2, Milvus Lite (`pymilvus` 2.6.2), OpenAI + LangChain (0.3.34) for RAGAs, Ragas 0.3.5, Evaluate 0.4.6, Datasets 4.1.1, Pandas 2.3.3 / NumPy 2.3.3 / Matplotlib / Seaborn 0.13.2.

## Datasets

Passages and queries load directly from Hugging Face (`hf://datasets/rag-datasets/rag-mini-wikipedia/...`). Runs require network access on first execution; subsequent uses hit local cache.

**Data Organization:**
- Raw datasets: Loaded from HuggingFace (cached locally)
- Processed data: Stored in `data/` directory
- Milvus databases: `data/rag_wikipedia_mini.db`, `data/rag_enhanced.db`
- Results/metrics: Exported to `results/` directory

## Configuration

- Primary settings live in `config/default.yaml` (paths, models, top-k values, enhancement switches, RAGAs params).
- `config/enhanced.yaml` provides overrides for the enhanced pipeline runs.
- Adjust these files instead of editing notebooks for repeatable experiments.

## Execution Workflow

Run the following 9 generation passes on the same 100-query evaluation slice, then execute RAGAs for the two resulting datasets:

1. Baseline prompt (`baseline`), top-1 retrieval.
2. Verification prompt (`verification`), top-1 retrieval.
3-5. MiniLM-L6 (384d) embedding with top-3, top-5, top-10.
6-8. MiniLM-L3 (256d) embedding with top-3, top-5, top-10.
9. Enhanced pipeline (rewrite+rerank) with base top-10, rerank top-3.

After completions:
- Run RAGAs twice (naive vs. enhanced, using the full 100-sample evaluation set) with identical contexts saved from the 100-query runs.
- Export metrics via notebook persistence helpers (JSON/CSV under `results/`).


## Scripted Pipeline

Run the modular evaluation pipeline (minimum requirements: 9 runs + RAGAs):

```bash
python -m src.pipeline
```

This executes Steps 1-6 per `config/default.yaml`:
- 9 generation runs (100 queries each)
- RAGAs evaluation (naive vs enhanced, n=100 samples)
- Results saved with `modular_` prefix to distinguish from complete_analysis outputs

## Notebook Structure

### 1. `data_exploration.ipynb` (Step 1)
- Dataset loading using modular helper functions from `src/`
- Quality analysis and statistics
- Length distributions and sample inspection
- Evaluation subset selection

### 2. `complete_analysis.ipynb` (Steps 1-7, comprehensive)
- Complete end-to-end implementation (original single-notebook approach)
- Naive RAG: embedding, Milvus ingestion, FLAN-T5 generation
- Prompt experiments and parameter sweeps
- Enhanced RAG: query rewriting + cross-encoder reranking
- RAGAs evaluation with confidence intervals
- Full reporting and AI usage log

### 3. `system_evaluation.ipynb`
- Smoke-test the modular pipeline with lightweight settings
- Validates `src/` module integration
- Quick verification of system functionality

### 4. `final_analysis.ipynb`
- Results visualization and comparison
- Loads exported metrics from `results/`
- Statistical analysis with confidence intervals
- Publication-ready charts

## Usage Notes

- Set `OPENAI_API_KEY` env var or store the key in `~/.openai_api_key` before running RAGAs cells.
- Milvus Lite stores embeddings locally in `data/rag_wikipedia_mini.db` and `data/rag_enhanced.db`; remove files to reset ingestion.
- **Milvus Connection**: The enhanced pipeline automatically reconnects to Milvus before running to prevent gRPC timeout issues after long operations.
- Results persist under `results/`. Delete files if you need a clean slate before reruns.
- All paths are configured in `config/default.yaml` for consistency across modules and notebooks.
- Uses deterministic seeds but downloads models on-demand; ensure network access for first run.

## Deliverables

### Notebooks
- `notebooks/data_exploration.ipynb` – Step 1 dataset exploration with modular imports
- `notebooks/complete_analysis.ipynb` – Comprehensive end-to-end implementation (Steps 1-7)
- `notebooks/system_evaluation.ipynb` – Modular pipeline smoke tests
- `notebooks/final_analysis.ipynb` – Results visualization and statistical analysis

### Source Modules
- `src/naive_rag.py` – Core RAG pipeline functions
- `src/enhanced_rag.py` – Query rewriting and reranking enhancements
- `src/evaluation.py` – SQuAD metrics and RAGAs evaluation
- `src/utils.py` – Data loading and utility functions
- `src/config.py` – Configuration management
- `src/pipeline.py` – Automated evaluation runner

### Data & Results
- `data/` – Milvus databases and processed artifacts
- `results/` – JSON/CSV metrics for all experiments (see Results Organization below)
- `config/` – YAML configuration files

## Results Organization

Results are organized to distinguish between two implementation approaches:

### Complete Analysis Results (No Prefix)
Generated from `notebooks/complete_analysis.ipynb` (comprehensive Steps 1-7):
- `naive_results.json` – Baseline metrics (EM=41.50%, F1=49.27%)
- `enhanced_results.json` – Enhanced metrics (EM=33.66%, F1=42.41%)
- `comparison_analysis.csv` – Prompt strategies (baseline vs verification)
- `embedding_experiments.csv` – All embedding models (MiniLM-L3/L6, MPNet)
- `enhanced_summary.csv` – Naive vs Enhanced delta
- `ragas_comparison.csv` – **RAGAs metrics (n=100 samples)** ← Exceeds minimum
- `ragas_comparison_delta.csv` – RAGAs deltas
- `*.txt` – Formatted summary tables

### Modular Pipeline Results (`modular_` Prefix)
Generated from `run_modular_pipeline.py` or `src/pipeline.py` (Steps 1-6):
- `modular_naive_results.json` – Baseline metrics
- `modular_enhanced_results.json` – Enhanced metrics
- `modular_prompt_comparison.csv` – Prompt strategies
- `modular_embedding_experiments.csv` – MiniLM-L3/L6 sweeps only
- `modular_enhanced_summary.csv` – Naive vs Enhanced delta
- `modular_ragas_comparison.csv` – **RAGAs metrics (n=100 samples)** ← Assignment requirement
- `modular_ragas_delta.csv` – RAGAs deltas

**Key Differences:**
- Complete analysis: Includes all experiments with n=100 RAGAs, plus MPNet experiments
- Modular pipeline: Focused implementation with n=100 RAGAs, only required models

### Documentation
- `docs/technical_report.md` – Phase 5 technical report
- `docs/technical_appendix.md` – Implementation details
- `docs/setup_instructions.md` – Environment setup guide

## Next Work (post-notebook)
