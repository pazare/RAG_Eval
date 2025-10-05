from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

from .naive_rag import CONTEXT_SEPARATOR, GenerationResources, format_prompt, generate_with_llm

QUERY_REWRITE_PROMPTS: Dict[str, str] = {
    "recall": (
        "Rewrite the user question to maximise recall while keeping the intent unchanged. "
        "Use a single concise sentence."
    ),
    "precision": (
        "Rewrite the question focusing on key entities so irrelevant passages are filtered out."
    ),
}

_rewrite_cache: Dict[Tuple[str, str], str] = {}
_cross_encoder_cache: Dict[str, CrossEncoder] = {}


def _rewrite_single(strategy: str, question: str, resources: GenerationResources) -> str:
    cache_key = (strategy, question)
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]
    template = QUERY_REWRITE_PROMPTS[strategy]
    prompt = (
        f"{template}\n\n"
        f"Original question:\n{question}\n\n"
        "Rewritten question:"
    )
    rewritten = generate_with_llm(resources, prompt, max_new_tokens=64).strip()
    _rewrite_cache[cache_key] = rewritten
    return rewritten


def rewrite_questions_bulk(
    questions: List[str],
    *,
    strategy: str,
    resources: GenerationResources,
) -> List[str]:
    return [
        _rewrite_single(strategy, question, resources)
        for question in tqdm(questions, desc=f"Rewriting ({strategy})", leave=False)
    ]


def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    if model_name not in _cross_encoder_cache:
        _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    return _cross_encoder_cache[model_name]


def rerank_contexts(
    query_text: str,
    contexts: List[str],
    *,
    top_n: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Tuple[List[str], List[float]]:
    if not contexts:
        return [], []
    try:
        cross_encoder = load_cross_encoder(model_name)
        scores = cross_encoder.predict([(query_text, ctx) for ctx in contexts])
        order = np.argsort(scores)[::-1][:top_n]
        return [contexts[idx] for idx in order], [float(scores[idx]) for idx in order]
    except Exception:  # noqa: BLE001 - keep behaviour consistent with notebook
        return contexts[:top_n], [1.0] * min(len(contexts), top_n)


def run_enhanced_rag_pipeline(
    queries_df: pd.DataFrame,
    *,
    client: MilvusClient,
    collection_name: str,
    embedding_model: SentenceTransformer,
    resources: GenerationResources,
    base_top_k: int,
    rerank_top_k: int,
    prompt_template: str,
    rewrite_strategy: str = "recall",
    context_separator: str = CONTEXT_SEPARATOR,
) -> Dict[str, List]:
    """Enhanced RAG pipeline with query rewriting and cross-encoder reranking.

    Note: If using this after long-running operations, ensure the MilvusClient
    connection is fresh by reconnecting before calling this function to avoid
    gRPC timeout errors.
    """
    original_questions = queries_df["question"].tolist()
    answers = queries_df["answer"].tolist()
    rewrites = rewrite_questions_bulk(
        original_questions, strategy=rewrite_strategy, resources=resources
    )

    generated_answers: List[str] = []
    retrieved_contexts: List[List[str]] = []
    ground_truths: List[List[str]] = []
    rewritten_questions: List[str] = []
    confidence_scores: List[float] = []

    for question, rewrite, target in tqdm(
        zip(original_questions, rewrites, answers),
        total=len(original_questions),
        desc="Enhanced RAG",
        leave=False,
    ):
        query_embedding = embedding_model.encode(rewrite, convert_to_numpy=True)
        search_output = client.search(
            collection_name=collection_name,
            data=[query_embedding.tolist()],
            limit=base_top_k,
            output_fields=["passage"],
        )
        hits = search_output[0] if search_output else []
        contexts = [hit["entity"]["passage"] for hit in hits] if hits else ["No relevant context found."]
        reranked_contexts, rerank_scores = rerank_contexts(
            rewrite, contexts, top_n=rerank_top_k
        )
        combined_context = context_separator.join(reranked_contexts)
        prompt = format_prompt(prompt_template, combined_context, question)
        answer = generate_with_llm(resources, prompt)

        generated_answers.append(answer)
        retrieved_contexts.append(reranked_contexts)
        ground_truths.append([target] if isinstance(target, str) else target)
        rewritten_questions.append(rewrite)
        confidence_scores.append(float(max(rerank_scores)) if rerank_scores else 0.0)

    return {
        "question": original_questions,
        "rewritten_question": rewritten_questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truths": ground_truths,
        "confidence": confidence_scores,
        "rewrite_strategy": [rewrite_strategy] * len(original_questions),
    }
