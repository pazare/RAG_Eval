from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from evaluate import load as load_metric
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig
from sentence_transformers import SentenceTransformer

from .config import Config, load_config
from .evaluation import SentenceTransformerEmbeddingsWrapper, build_ragas_dataset, extract_ragas_scores, format_for_squad_evaluation
from .enhanced_rag import run_enhanced_rag_pipeline
from .naive_rag import (
    CONTEXT_SEPARATOR,
    DenseRetriever,
    GenerationResources,
    build_collection_schema,
    build_rag_records,
    encode_texts,
    ensure_collection,
    ingest_records,
    load_generation_resources,
    run_rag_generation,
    run_rag_with_dense_retriever,
)
from .utils import ensure_results_dir, load_queries, load_text_table, persist_dataframe, persist_json


@dataclass
class EvaluationArtifacts:
    questions: List[str]
    answers: List[str]
    contexts: List[List[str]]
    ground_truths: List[List[str]]


class RAGEvaluationPipeline:
    """Pipeline that executes the minimum required evaluation schedule for Assignment 2.

    Runs 9 generation experiments + RAGAs evaluation (Steps 1-6):
    1. Baseline prompt, top-1 (100 queries)
    2. Verification prompt, top-1 (100 queries)
    3-5. MiniLM-L6 (384d) at top-3/5/10
    6-8. MiniLM-L3 (256d) at top-3/5/10
    9. Enhanced pipeline (rewrite+rerank: base top-10, rerank top-3)
    10-11. RAGAs evaluation (naive vs enhanced, n=25 samples)

    Results saved with 'modular_' prefix to distinguish from complete_analysis outputs.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.results_dir = ensure_results_dir()
        self.squad_metric = load_metric("squad")

        paths = self.config.get("paths", default={})
        self.passages = load_text_table(paths.get("passages"), "passage")
        queries_df = load_queries(paths.get("queries"))
        evaluation_queries = self.config.get("retrieval", "evaluation_queries", default=100)
        self.queries = queries_df.head(evaluation_queries)

        self.client = MilvusClient(paths.get("milvus_db"))
        self.collection_name = self.config.get("milvus", "collection")

        baseline_model_name = self.config.get("embedding_models", "baseline")
        self.baseline_embedding_model = SentenceTransformer(baseline_model_name)
        self.embedding_alternatives = {
            alt.get("label"): alt.get("name")
            for alt in self.config.get("embedding_models", "alternatives", default=[])
        }
        self.baseline_doc_embeddings = None
        self._prepare_collection()

        self.generation_model = self.config.get("llm", "generation_model")
        self.prompt_library: Dict[str, str] = self.config.get("llm", "prompt_library", default={})
        self.resources = load_generation_resources(self.generation_model)

        self.comparison_records: List[Dict[str, object]] = []
        self.embedding_records: List[Dict[str, object]] = []

    def _prepare_collection(self) -> None:
        schema = build_collection_schema(self.baseline_embedding_model.get_sentence_embedding_dimension())
        ensure_collection(self.client, name=self.collection_name, schema=schema, drop_existing=True)
        self.baseline_doc_embeddings = encode_texts(self.baseline_embedding_model, self.passages["passage"], batch_size=64)
        records = build_rag_records(self.passages, self.baseline_doc_embeddings)
        ingest_records(self.client, self.collection_name, records)

        # Create index after ingestion for search performance
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.config.get("milvus", "index", "type", default="IVF_FLAT"),
            metric_type=self.config.get("milvus", "metric_type", default="L2"),
            params={"nlist": self.config.get("milvus", "index", "nlist", default=128)}
        )
        self.client.create_index(self.collection_name, index_params)

    # -------------------------- run helpers --------------------------
    def _compute_metrics(self, outputs: Dict[str, List]) -> Dict[str, float]:
        predictions, references = format_for_squad_evaluation(outputs["answer"], outputs["ground_truths"])
        return self.squad_metric.compute(predictions=predictions, references=references)

    def _record_prompt_metrics(self, prompt: str, top_k: int, metrics: Dict[str, float]) -> None:
        self.comparison_records.append(
            {
                "Prompt": prompt,
                "TopK": top_k,
                "Exact Match": metrics.get("exact_match", 0.0),
                "F1": metrics.get("f1", 0.0),
            }
        )

    def _record_embedding_metrics(self, embedding: str, top_k: int, metrics: Dict[str, float]) -> None:
        self.embedding_records.append(
            {
                "Embedding": embedding,
                "TopK": top_k,
                "Exact Match": metrics.get("exact_match", 0.0),
                "F1": metrics.get("f1", 0.0),
            }
        )

    # -------------------------- main runs --------------------------
    def run_baseline(self) -> Tuple[EvaluationArtifacts, Dict[str, float], np.ndarray]:
        question_embeddings = encode_texts(
            self.baseline_embedding_model,
            self.queries["question"],
            batch_size=64,
            show_progress=False,
        )
        outputs = run_rag_generation(
            self.queries,
            client=self.client,
            collection_name=self.collection_name,
            question_embeddings=question_embeddings,
            embedding_model=self.baseline_embedding_model,
            resources=self.resources,
            prompt_template=self.prompt_library.get("baseline", ""),
            top_k=self.config.get("evaluation_plan", "baseline", "top_k", default=1),
        )
        metrics = self._compute_metrics(outputs)
        persist_json(
            {
                "prompt_strategy": "baseline",
                "top_k": self.config.get("evaluation_plan", "baseline", "top_k", default=1),
                "queries": len(self.queries),
                "metrics": metrics,
            },
            self.config.get("results", "files", "naive_metrics", default="naive_results.json"),
        )
        self._record_prompt_metrics("baseline", self.config.get("evaluation_plan", "baseline", "top_k", default=1), metrics)
        # Map singular to plural keys for EvaluationArtifacts
        key_mapping = {'question': 'questions', 'answer': 'answers'}
        artifacts_dict = {key_mapping.get(k, k): v for k, v in outputs.items()}
        return EvaluationArtifacts(**artifacts_dict), metrics, question_embeddings

    def run_prompt_variant(self, question_embeddings: np.ndarray) -> EvaluationArtifacts:
        outputs = run_rag_generation(
            self.queries,
            client=self.client,
            collection_name=self.collection_name,
            question_embeddings=question_embeddings,
            embedding_model=self.baseline_embedding_model,
            resources=self.resources,
            prompt_template=self.prompt_library.get("verification", ""),
            top_k=self.config.get("evaluation_plan", "prompt_variant", "top_k", default=1),
        )
        metrics = self._compute_metrics(outputs)
        self._record_prompt_metrics("verification", self.config.get("evaluation_plan", "prompt_variant", "top_k", default=1), metrics)
        # Map singular to plural keys for EvaluationArtifacts
        key_mapping = {'question': 'questions', 'answer': 'answers'}
        artifacts_dict = {key_mapping.get(k, k): v for k, v in outputs.items()}
        return EvaluationArtifacts(**artifacts_dict)

    def run_embedding_sweeps(self) -> Dict[Tuple[str, int], EvaluationArtifacts]:
        plan = self.config.get("evaluation_plan", "embedding_sweeps", default=[])
        artifacts: Dict[Tuple[str, int], EvaluationArtifacts] = {}
        prompt_template = self.prompt_library.get("baseline", "")
        resources = self.resources
        passages = self.passages["passage"].tolist()

        for sweep in plan:
            label = sweep.get('label') or sweep.get('embedding')
            model_name = sweep.get('name') or sweep.get('model') or self.embedding_alternatives.get(label)
            if not model_name:
                raise ValueError("Missing model name for embedding sweep '{label}'.")
            model = SentenceTransformer(model_name)
            retriever = DenseRetriever.build(label=label, model=model, passages=passages)
            for top_k in sweep.get('top_k_values', []):
                outputs = run_rag_with_dense_retriever(
                    self.queries,
                    retriever=retriever,
                    top_k=top_k,
                    prompt_template=prompt_template,
                    resources=resources,
                )
                metrics = self._compute_metrics(outputs)
                self._record_embedding_metrics(label, top_k, metrics)
                # Map singular to plural keys for EvaluationArtifacts
                key_mapping = {'question': 'questions', 'answer': 'answers'}
                artifacts_dict = {key_mapping.get(k, k): v for k, v in outputs.items()}
                artifacts[(label, top_k)] = EvaluationArtifacts(**artifacts_dict)
        return artifacts

    def run_enhanced(self) -> Tuple[EvaluationArtifacts, Dict[str, float]]:
        # Reconnect to Milvus to ensure fresh connection for enhanced pipeline
        # This prevents gRPC timeout issues after long-running operations
        paths = self.config.get("paths", default={})
        self.client = MilvusClient(paths.get("milvus_db"))

        outputs = run_enhanced_rag_pipeline(
            self.queries,
            client=self.client,
            collection_name=self.collection_name,
            embedding_model=self.baseline_embedding_model,
            resources=self.resources,
            base_top_k=self.config.get("enhancements", "base_top_k", default=10),
            rerank_top_k=self.config.get("enhancements", "rerank_top_k", default=3),
            prompt_template=self.prompt_library.get("baseline", ""),
            rewrite_strategy=self.config.get("enhancements", "rewrite_strategy", default="recall"),
        )
        metrics = self._compute_metrics(outputs)
        persist_json(
            {
                "strategy": "rewrite+rerank",
                "prompt_strategy": "baseline",
                "base_top_k": self.config.get("enhancements", "base_top_k", default=10),
                "rerank_top_k": self.config.get("enhancements", "rerank_top_k", default=3),
                "rewrite_strategy": self.config.get("enhancements", "rewrite_strategy", default="recall"),
                "queries": len(self.queries),
                "metrics": metrics,
            },
            self.config.get("results", "files", "enhanced_metrics", default="enhanced_results.json"),
        )
        # Map singular to plural keys and filter to only keys needed for EvaluationArtifacts
        key_mapping = {'question': 'questions', 'answer': 'answers'}
        artifacts_dict = {
            key_mapping.get(k, k): v
            for k, v in outputs.items()
            if k in {'question', 'answer', 'contexts', 'ground_truths'}
        }
        return EvaluationArtifacts(**artifacts_dict), metrics

    def run_ragas(self, naive: EvaluationArtifacts, enhanced: EvaluationArtifacts) -> None:
        sample_size = self.config.get("raga_evaluation", "sample_size", default=25)
        naive_dataset = build_ragas_dataset(
            {
                "question": naive.questions,
                "answer": naive.answers,
                "contexts": naive.contexts,
                "ground_truths": naive.ground_truths,
            },
            sample_size=sample_size,
        )
        enhanced_dataset = build_ragas_dataset(
            {
                "question": enhanced.questions,
                "answer": enhanced.answers,
                "contexts": enhanced.contexts,
                "ground_truths": enhanced.ground_truths,
            },
            sample_size=sample_size,
        )

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            key_file = Path.home() / ".openai_api_key"
            if key_file.exists():
                openai_key = key_file.read_text(encoding="utf-8").strip()
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is required for RAGAs evaluation.")

        ragas_llm = ChatOpenAI(
            model=self.config.get("raga_evaluation", "openai_model", default="gpt-4o-mini"),
            api_key=openai_key,
        )
        wrapper = SentenceTransformerEmbeddingsWrapper(self.baseline_embedding_model)
        run_config = RunConfig(max_workers=self.config.get("raga_evaluation", "max_workers", default=16))

        naive_result = ragas_evaluate(
            dataset=naive_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=wrapper,
            run_config=run_config,
        )
        enhanced_result = ragas_evaluate(
            dataset=enhanced_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=wrapper,
            run_config=run_config,
        )

        ragas_comparison = pd.DataFrame(
            [
                {"System": "Naive", **extract_ragas_scores(naive_result)},
                {"System": "Enhanced", **extract_ragas_scores(enhanced_result)},
            ]
        )
        persist_dataframe(ragas_comparison, self.config.get("results", "files", "ragas_comparison", default="ragas_comparison.csv"))
        ragas_delta = ragas_comparison.set_index("System").T
        ragas_delta["Delta"] = ragas_delta["Enhanced"] - ragas_delta["Naive"]
        persist_dataframe(
            ragas_delta.reset_index().rename(columns={"index": "metric"}),
            self.config.get("results", "files", "ragas_delta", default="ragas_comparison_delta.csv"),
        )

    # -------------------------- orchestration --------------------------
    def run_all(self) -> None:
        naive_artifacts, naive_metrics, question_embeddings = self.run_baseline()
        self.run_prompt_variant(question_embeddings)
        self.run_embedding_sweeps()
        enhanced_artifacts, enhanced_metrics = self.run_enhanced()

        # Enhanced summary CSV
        enhanced_summary = pd.DataFrame(
            [
                {
                    "Metric": key,
                    "Naive (Top-1)": naive_metrics.get(key),
                    "Enhanced (Rewrite+Rerank)": enhanced_metrics.get(key),
                    "Delta": enhanced_metrics.get(key) - naive_metrics.get(key),
                }
                for key in sorted(set(naive_metrics) | set(enhanced_metrics))
            ]
        )
        persist_dataframe(
            enhanced_summary,
            self.config.get("results", "files", "enhanced_summary", default="enhanced_summary.csv"),
        )

        # Prompt comparison and embedding sweeps
        if self.comparison_records:
            persist_dataframe(
                pd.DataFrame(self.comparison_records),
                self.config.get("results", "files", "prompt_comparison", default="comparison_analysis.csv"),
            )
        if self.embedding_records:
            persist_dataframe(
                pd.DataFrame(self.embedding_records),
                self.config.get("results", "files", "embedding_sweeps", default="embedding_experiments.csv"),
            )

        # RAGAs evaluation (naive vs enhanced)
        self.run_ragas(naive_artifacts, enhanced_artifacts)


if __name__ == "__main__":
    print("=" * 80)
    print("ASSIGNMENT 2 - MODULAR RAG PIPELINE")
    print("=" * 80)
    print("\nExecuting Steps 1-6 (9 experiments + RAGAs):")
    print("  • Baseline prompt, top-1")
    print("  • Verification prompt, top-1")
    print("  • MiniLM-L6 at top-3/5/10")
    print("  • MiniLM-L3 at top-3/5/10")
    print("  • Enhanced (rewrite+rerank)")
    print("  • RAGAs (naive vs enhanced, n=100)")
    print("\nResults saved with 'modular_' prefix in results/")
    print("=" * 80)
    print()

    pipeline = RAGEvaluationPipeline()
    pipeline.run_all()

    print()
    print("=" * 80)
    print("PIPELINE COMPLETE - Results in results/ directory")
    print("=" * 80)

