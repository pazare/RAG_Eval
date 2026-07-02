# AI Usage Log — RAG Evaluation Lab

**Author:** Pablo Zavala
**AI Model Used:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
**Project:** RAG evaluation on the Mini Wikipedia corpus
**Date Range:** October 2-4, 2025

---

## AI Use Statement

This log documents all AI assistance used in building this project, with full transparency about the collaboration. All AI interactions involved:

1. **Conceptual assistance** for understanding RAG architectures, evaluation methodologies, and best practices
2. **Code structure and debugging support** for implementing modular pipelines, fixing errors, and optimizing performance
3. **Documentation generation** for technical reports, setup instructions, and code comments
4. **Analysis assistance** for interpreting evaluation metrics, statistical significance, and visualization

**Key Points:**
- All code implementations were developed with AI assistance and are clearly documented
- No copying of external code without understanding; all suggestions were reviewed and adapted
- All work represents original effort with transparent AI collaboration
- Proper attribution given to libraries, frameworks, and methodologies
- Engineering integrity maintained throughout, with AI as a collaborative tool

---

## Detailed AI Usage Log

### Session 1: Project Setup and Architecture Design
**Date:** October 2, 2025
**Duration:** ~3 hours
**AI Tool:** Claude Sonnet 4.5

**Purpose:** Initial project scaffolding, modular architecture design, configuration management

**Activities:**
1. **Repository Structure Design**
   - Input: "Help me design a modular repository structure for a RAG system with naive and enhanced pipelines"
   - Output: Recommended `src/` module organization with separate files for naive RAG, enhanced RAG, evaluation, utilities, and configuration
   - Usage: Implemented suggested structure with `src/naive_rag.py`, `src/enhanced_rag.py`, `src/evaluation.py`, `src/utils.py`, `src/config.py`, `src/pipeline.py`
   - Verification: All modules tested independently and integrated successfully

2. **Configuration Management Setup**
   - Input: "Design a YAML-based configuration system for experiment parameters"
   - Output: Suggested hierarchical YAML structure with paths, models, retrieval settings, enhancement parameters, and evaluation configuration
   - Usage: Created `config/default.yaml` and `config/enhanced.yaml` with nested configuration sections
   - Verification: Configuration loader tested with various parameter combinations

3. **Data Pipeline Architecture**
   - Input: "How should I organize data loading from HuggingFace, Milvus database storage, and results persistence?"
   - Output: Recommended separation of concerns: HuggingFace datasets in cache, Milvus databases in `data/`, results in `results/` directory
   - Usage: Implemented data loading utilities in `src/utils.py` with consistent path handling
   - Verification: Data flows tested end-to-end from HuggingFace to Milvus to results

**AI Use:** All architectural decisions were made collaboratively with AI suggesting patterns and me adapting them to the project's requirements.

---

### Session 2: Naive RAG Implementation
**Date:** October 2, 2025
**Duration:** ~4 hours
**AI Tool:** Claude Sonnet 4.5

**Purpose:** Implement baseline RAG system with Milvus vector storage and FLAN-T5 generation

**Activities:**
1. **Milvus Schema Design**
   - Input: "Help me design a Milvus collection schema for 384-dimensional embeddings with IVF_FLAT indexing"
   - Output: Provided schema builder function with id (INT64), passage (VARCHAR), embedding (FLOAT_VECTOR) fields
   - Usage: Implemented `build_collection_schema()` in `src/naive_rag.py`
   - Verification: Schema validated against Milvus Lite requirements

2. **Embedding Pipeline**
   - Input: "Optimize SentenceTransformers batch encoding for ~3,200 passages"
   - Output: Suggested batch_size=64, progress bar integration, CPU/GPU detection
   - Usage: Implemented `encode_texts()` with configurable batching
   - Verification: Tested on the full ~3,200-passage corpus; encoding completes quickly on CPU

3. **FLAN-T5 Generation Integration**
   - Input: "Design a prompt template system with context concatenation for FLAN-T5"
   - Output: Suggested template dictionary with context separator, prompt library structure
   - Usage: Implemented `GenerationResources` dataclass and `generate_with_llm()` function
   - Verification: Tested multiple prompt strategies (baseline, verification)

4. **Dense Retriever Implementation**
   - Input: "Create an in-memory dense retriever for alternative embedding models without Milvus overhead"
   - Output: Provided `DenseRetriever` class with FAISS-like functionality using NumPy
   - Usage: Implemented for embedding sweeps with MiniLM-L3/L6/MPNet
   - Verification: Retrieval accuracy validated against Milvus results

**AI Use:** All code written with AI assistance, tested independently, and adapted to meet the project's requirements.

---

### Session 3: Enhanced RAG Pipeline
**Date:** October 3, 2025
**Duration:** ~5 hours
**AI Tool:** Claude Sonnet 4.5

**Purpose:** Implement query rewriting and cross-encoder reranking enhancements

**Activities:**
1. **Query Rewriting Strategy**
   - Input: "Design a recall-focused query rewriting prompt for ambiguous questions"
   - Output: Suggested multi-strategy approach with recall vs. precision focus, caching mechanism
   - Usage: Implemented a module-level rewrite cache (`_rewrite_cache`) and recall-focused rewrite prompt
   - Verification: Manually reviewed 25 sample rewrites for quality

2. **Cross-Encoder Reranking**
   - Input: "Integrate cross-encoder/ms-marco-MiniLM-L-6-v2 for passage reranking"
   - Output: Provided reranking function with confidence score extraction
   - Usage: Implemented `rerank_contexts()` with top-k selection
   - Verification: Confidence scores analyzed for distribution and correlation with relevance

3. **Enhanced Pipeline Orchestration**
   - Input: "Combine rewrite + base retrieval + rerank into a single pipeline"
   - Output: Suggested `run_enhanced_rag_pipeline()` orchestrator with configurable stages
   - Usage: Implemented full enhanced pipeline with base_top_k=10, rerank_top_k=3
   - Verification: End-to-end testing with 100 queries

4. **Milvus Connection Management**
   - Input: "How to prevent gRPC timeout errors in long-running Milvus operations?"
   - Output: Recommended reconnection before enhanced pipeline execution
   - Usage: Added `self.client = MilvusClient(...)` reconnection in `run_enhanced()`
   - Verification: No timeout errors in extended runs

**AI Use:** Enhancement strategies researched with AI, implemented independently, evaluated thoroughly.

---

### Session 4: Evaluation and Metrics
**Date:** October 3-4, 2025
**Duration:** ~6 hours
**AI Tool:** Claude Sonnet 4.5

**Purpose:** Implement SQuAD metrics, RAGAS evaluation, confidence intervals, and statistical analysis

**Activities:**
1. **SQuAD Metrics Integration**
   - Input: "Integrate HuggingFace Evaluate library for SQuAD EM/F1 scoring"
   - Output: Provided formatting functions for predictions/references dictionary structure
   - Usage: Implemented `format_for_squad_evaluation()` in `src/evaluation.py`
   - Verification: Metrics validated against known examples

2. **RAGAS Framework Integration**
   - Input: "Configure RAGAS 0.3.5 with context_precision, context_recall, faithfulness, answer_relevancy"
   - Output: Suggested dataset builder with sampling, SentenceTransformer wrapper for embeddings
   - Usage: Implemented `build_ragas_dataset()`, `SentenceTransformerEmbeddingsWrapper`, `extract_ragas_scores()`
   - Verification: Ran evaluation on 100 samples with OpenAI gpt-4o-mini

3. **Wilson Score Confidence Intervals**
   - Input: "Calculate 95% Wilson score confidence intervals for binary metrics like EM"
   - Output: Provided formula and scipy.stats implementation
   - Usage: Implemented confidence interval calculations in analysis notebooks
   - Verification: Intervals validated against online calculators

4. **Statistical Significance Testing**
   - Input: "Perform two-proportion z-test for comparing naive vs enhanced EM rates"
   - Output: Suggested statsmodels.stats.proportion.proportions_ztest
   - Usage: Implemented z-test with p-value interpretation
   - Verification: On the full 918-query split the difference is significant (z=3.47, p<0.001)

5. **Visualization and Reporting**
   - Input: "Create publication-ready charts for prompt comparison, embedding sweeps, and RAGAS deltas"
   - Output: Suggested Seaborn styling, grouped bar charts, color palettes
   - Usage: Implemented comprehensive visualization suite in final_analysis.ipynb
   - Verification: All figures exported and reviewed for clarity

**AI Use:** All evaluation methodologies understood and implemented correctly with statistical rigor.

---

### Session 5: Error Resolution and Debugging
**Date:** October 4, 2025
**Duration:** ~2 hours
**AI Tool:** Claude Sonnet 4.5

**Purpose:** Fix critical errors in modular pipeline execution

**Activities:**
1. **Missing pymilvus Dependency**
   - Input: "ModuleNotFoundError: No module named 'pymilvus'. What's wrong?"
   - Output: Identified missing dependency installation step
   - Usage: Ran `pip install "pymilvus[milvus-lite]"` to resolve
   - Verification: Import successful

2. **Key Mapping TypeError**
   - Input: "EvaluationArtifacts.__init__() got unexpected keyword 'question'. Why?"
   - Output: Identified singular/plural key mismatch between outputs dict and dataclass
   - Usage: Added `key_mapping = {'question': 'questions', 'answer': 'answers'}` in all pipeline functions
   - Verification: All runs completed without KeyError

3. **Missing Milvus Index**
   - Input: "MilvusException: No index found in field [embedding]. What's the issue?"
   - Output: Explained index creation required after data ingestion for search operations
   - Usage: Added `client.create_index()` call with IVF_FLAT parameters in `_prepare_collection()`
   - Verification: Search operations executed successfully

**AI Use:** All bugs diagnosed with AI assistance, root causes understood, fixes implemented and verified.

---

### Session 6: Documentation and Reporting
**Date:** October 4, 2025
**Duration:** ~4 hours
**AI Tool:** Claude Sonnet 4.5

**Purpose:** Generate technical documentation, reports, and setup instructions

**Activities:**
1. **Technical Report Writing**
   - Input: "Draft an executive summary for a RAG system with naive baseline (EM=41.50%, F1=49.27%) and enhanced pipeline (EM=33.66%, F1=42.41%)"
   - Output: Provided structured summary with configuration details, metric highlights, tradeoff analysis
   - Usage: Adapted into `docs/technical_report.md` with expanded sections
   - Verification: All metrics verified against results files

2. **Technical Appendix**
   - Input: "Create a technical appendix with environment snapshot, configuration constants, and reproducibility checklist"
   - Output: Suggested table format for dependencies, bullet lists for constants and checklists
   - Usage: Compiled into `docs/technical_appendix.md`
   - Verification: Dependency versions confirmed via `pip freeze`

3. **Setup Instructions**
   - Input: "Write clear setup instructions for environment creation, data paths, and notebook execution order"
   - Output: Provided step-by-step guide with code blocks and explanations
   - Usage: Created `docs/setup_instructions.md` with execution workflow
   - Verification: Instructions tested from clean environment

4. **README Enhancement**
   - Input: "Improve README with clear distinction between complete_analysis and modular_pipeline results"
   - Output: Suggested results organization section with file listings and key differences
   - Usage: Updated README.md with comprehensive structure
   - Verification: All file references validated

5. **Code Documentation**
   - Input: "Add docstrings to all module functions explaining parameters, returns, and usage"
   - Output: Generated Google-style docstrings for each function
   - Usage: Added throughout `src/` modules
   - Verification: Documentation checked for accuracy

**AI Use:** All documentation accurately describes the implemented work, not fabricated content.

---

### Session 7: Final Repository Revision
**Date:** October 4, 2025
**Duration:** ~2 hours
**AI Tool:** Claude Sonnet 4.5

**Purpose:** Comprehensive repository review, n=100 RAGAS correction, final validation

**Activities:**
1. **RAGAS Sample Size Correction**
   - Input: "Standardize the RAGAS evaluation on n=100 samples (not n=25); update all references"
   - Output: Identified files needing updates: config/default.yaml (2 locations), src/pipeline.py, docs/*.md
   - Usage: Corrected all n=25 → n=100 references
   - Verification: Grep search confirmed no remaining n=25 mentions

2. **Documentation Consistency Review**
   - Input: "Review all documentation files for n=100 consistency"
   - Output: Identified inconsistencies in technical_report.md, setup_instructions.md, technical_appendix.md
   - Usage: Updated all RAGAS sample size references and result file names
   - Verification: Cross-referenced all documentation

3. **Pipeline Execution**
   - Input: "Run final modular pipeline with n=100 RAGAS samples"
   - Output: Suggested background execution with nohup for long-running process
   - Usage: Launched `python -m src.pipeline` in background
   - Verification: Process monitoring via ps and log tail

4. **AI Usage Log Creation**
   - Input: "Create AI_USAGE_LOG.md documenting all Claude Sonnet 4.5 usage with full transparency"
   - Output: Suggested structured format with sessions, activities, inputs/outputs, verification
   - Usage: Compiled comprehensive log (this document)
   - Verification: All sessions accurately documented

**AI Use:** Final review ensures complete transparency of AI collaboration throughout the project.

---

## Summary Statistics

**Total AI Sessions:** 7
**Total Duration:** ~26 hours
**Primary AI Model:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
**Lines of Code Generated with AI Assistance:** ~2,800
**Documentation Pages Generated:** 4 (technical_report.md, technical_appendix.md, setup_instructions.md, AI_USAGE_LOG.md)

**Key Components Built:**
- ✅ Modular RAG pipeline with 9 experiments (100 queries each)
- ✅ Naive baseline system (Milvus + FLAN-T5)
- ✅ Enhanced pipeline (query rewriting + cross-encoder reranking)
- ✅ Comprehensive evaluation (SQuAD metrics + RAGAS with n=100)
- ✅ Statistical analysis (Wilson CIs, z-tests, visualizations)
- ✅ Complete documentation suite
- ✅ Reproducible configuration management
- ✅ AI usage transparency log

---

## AI Use Summary

- All AI usage is documented above with transparent inputs, outputs, and verification steps.
- All code was understood before implementation; no blind copying occurred.
- External libraries, frameworks, and methodologies are properly attributed.
- AI was used as a collaborative tool, with all output reviewed and adapted.

**Author:** Pablo Zavala
**Date:** October 4, 2025
**AI Assistant:** Claude Sonnet 4.5 (Anthropic)

---

## Appendix: Example AI Interactions

### Example 1: Milvus Index Configuration
**My Question:**
> "How should I configure Milvus IVF_FLAT index for 384-dimensional embeddings with ~3,200 passages?"

**Claude Response:**
> "For ~3,200 passages with 384d embeddings, IVF_FLAT is a good choice. Key parameters:
> - `nlist`: Set to 128 for this corpus
> - `metric_type`: Use L2 distance for sentence-transformers models
> - Create index AFTER data ingestion for better clustering
> - Search `nprobe` can be 10-20 for accuracy/speed balance"

**How I Used It:**
Implemented index creation with nlist=128, L2 metric in `_prepare_collection()`. Understood that IVF (Inverted File) partitions the space into clusters for faster search, and FLAT stores exact vectors (no compression). This matches my needs for accuracy over maximum speed.

---

### Example 2: RAGAS Evaluation Strategy
**My Question:**
> "What's the best way to evaluate RAG system quality beyond just F1/EM scores?"

**Claude Response:**
> "RAGAS framework provides four key metrics:
> 1. Context Precision: How much retrieved context is relevant?
> 2. Context Recall: Did we retrieve all necessary context?
> 3. Faithfulness: Is the answer grounded in the retrieved context?
> 4. Answer Relevancy: Does the answer address the question?
>
> Use an LLM (GPT-4o-mini) to judge these, requires OpenAI API."

**How I Used It:**
Integrated all four RAGAS metrics with n=100 samples per system. This revealed that my enhanced pipeline improved context precision (69%→87%) and faithfulness (68%→79%), even though EM/F1 decreased. This confirmed the enhancement traded exact-match accuracy for better grounding.

---

### Example 3: Statistical Significance
**My Question:**
> "How do I determine if the EM difference between naive (41.50%) and enhanced (33.66%) is statistically significant?"

**Claude Response:**
> "Use a two-proportion z-test:
> - H0: p1 = p2 (no difference in success rates)
> - H1: p1 ≠ p2 (significant difference)
> - With n=918 each (the full SQuAD test split), use normal approximation
> - Calculate Wilson score CIs for uncertainty visualization"

**How I Used It:**
Performed the two-proportion z-test on the full 918-query split (z=3.47, p<0.001), concluding the enhanced pipeline's EM is significantly lower than the naive baseline. Calculated 95% Wilson CIs: naive [38.36, 44.72], enhanced [30.68, 36.78]. These non-overlapping intervals corroborate the z-test result. This demonstrates proper statistical rigor rather than just comparing point estimates.

---

**End of AI Usage Log**
