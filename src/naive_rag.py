from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CONTEXT_SEPARATOR = "\n\n---\n\n"


@dataclass(slots=True)
class MilvusConfig:
    """Configuration for Milvus Lite persistence."""

    path: str
    collection_name: str


@dataclass(slots=True)
class GenerationResources:
    """Container for generation assets to simplify function signatures."""

    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    device: torch.device


def build_collection_schema(embedding_dim: int) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ],
        description="RAG Mini Wikipedia passages",
        enable_dynamic_field=False,
    )


def ensure_collection(
    client: MilvusClient,
    *,
    name: str,
    schema: CollectionSchema,
    drop_existing: bool = True,
) -> None:
    if client.has_collection(collection_name=name):
        if drop_existing:
            client.drop_collection(collection_name=name)
        else:
            return
    client.create_collection(collection_name=name, schema=schema)


def encode_texts(
    model: SentenceTransformer,
    texts: Iterable[str],
    *,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def build_rag_records(passages: pd.DataFrame, embeddings: np.ndarray) -> List[Dict[str, object]]:
    passages = passages.reset_index(drop=True)
    passages = passages.assign(embedding=list(map(list, embeddings)))
    records = (
        passages.reset_index()
        .rename(columns={"index": "id"})
        [["id", "passage", "embedding"]]
        .to_dict("records")
    )
    return records


def ingest_records(client: MilvusClient, collection_name: str, records: List[Dict[str, object]]) -> None:
    client.insert(collection_name=collection_name, data=records)


def create_index(
    client: MilvusClient,
    collection_name: str,
    *,
    field_name: str = "embedding",
    index_type: str = "IVF_FLAT",
    metric_type: str = "L2",
    nlist: int = 128,
) -> None:
    """Create vector index on the specified field for efficient similarity search."""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=field_name,
        index_type=index_type,
        metric_type=metric_type,
        params={"nlist": nlist},
    )
    client.create_index(collection_name, index_params)


def format_prompt(prompt_template: str, context: str, question: str) -> str:
    return (
        f"{prompt_template.strip()}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )


def load_generation_resources(model_name: str) -> GenerationResources:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return GenerationResources(tokenizer=tokenizer, model=model, device=device)


def generate_with_llm(
    resources: GenerationResources,
    prompt: str,
    *,
    max_new_tokens: int = 128,
) -> str:
    inputs = resources.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {key: value.to(resources.device) for key, value in inputs.items()}
    with torch.no_grad():
        output_tokens = resources.model.generate(**inputs, max_new_tokens=max_new_tokens)
    return resources.tokenizer.decode(output_tokens[0], skip_special_tokens=True)


def run_rag_generation(
    queries_df: pd.DataFrame,
    *,
    client: MilvusClient,
    collection_name: str,
    question_embeddings: Optional[np.ndarray],
    embedding_model: SentenceTransformer,
    resources: GenerationResources,
    prompt_template: str,
    top_k: int,
    context_separator: str = CONTEXT_SEPARATOR,
) -> Dict[str, List]:
    questions = queries_df["question"].tolist()
    answers = queries_df["answer"].tolist()

    if question_embeddings is None:
        question_embeddings = encode_texts(
            embedding_model,
            questions,
            batch_size=64,
            show_progress=True,
        )

    if isinstance(question_embeddings, np.ndarray):
        embedding_vectors = question_embeddings.tolist()
    else:
        embedding_vectors = list(question_embeddings)

    generated_answers: List[str] = []
    retrieved_contexts: List[List[str]] = []
    ground_truths: List[List[str]] = []

    progress_desc = f"Prompt='{prompt_template[:24]}...' | top-{top_k}"
    for idx in tqdm(range(len(questions)), desc=progress_desc, leave=False):
        question = questions[idx]
        truth = answers[idx]
        query_vector = embedding_vectors[idx]

        try:
            search_output = client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["passage"],
            )
            hits = search_output[0] if search_output else []
            contexts = [hit["entity"]["passage"] for hit in hits] if hits else ["No relevant context found."]
            combined_context = context_separator.join(contexts)
            prompt = format_prompt(prompt_template, combined_context, question)
            answer = generate_with_llm(resources, prompt)
        except Exception as exc:  # noqa: BLE001 - keep notebook parity
            contexts = ["Error retrieving context"]
            answer = f"Error: {exc}"

        generated_answers.append(answer)
        retrieved_contexts.append(contexts)
        if isinstance(truth, list):
            ground_truths.append([t for t in truth if t])
        else:
            ground_truths.append([truth] if truth else [""])

    return {
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truths": ground_truths,
    }


@dataclass
class DenseRetriever:
    """In-memory dense retriever mirroring the notebook implementation."""

    label: str
    model: SentenceTransformer
    passages: List[str]
    doc_embeddings: np.ndarray

    @classmethod
    def build(cls, label: str, model: SentenceTransformer, passages: List[str], *, batch_size: int = 64, precomputed: np.ndarray | None = None) -> 'DenseRetriever':
        if precomputed is None:
            embeddings = model.encode(passages, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        else:
            embeddings = precomputed
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm = np.clip(norm, a_min=1e-9, a_max=None)
        embeddings = embeddings / norm
        return cls(label=label, model=model, passages=passages, doc_embeddings=embeddings)

    def retrieve(self, question: str, top_k: int) -> List[Tuple[str, float]]:
        query_vec = self.model.encode([question], show_progress_bar=False, convert_to_numpy=True)[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec /= query_norm
        scores = self.doc_embeddings @ query_vec
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.passages[idx], float(scores[idx])) for idx in idxs]


def run_rag_with_dense_retriever(
    queries_df: pd.DataFrame,
    *,
    retriever: DenseRetriever,
    top_k: int,
    prompt_template: str,
    resources: GenerationResources,
    context_separator: str = CONTEXT_SEPARATOR,
) -> Dict[str, List]:
    questions = queries_df['question'].tolist()
    answers = queries_df['answer'].tolist()

    generated_answers: List[str] = []
    retrieved_contexts: List[List[str]] = []
    ground_truths: List[List[str]] = []

    for question, truth in tqdm(zip(questions, answers), total=len(questions), desc=f"{retriever.label} | top-{top_k}", leave=False):
        contexts_with_scores = retriever.retrieve(question, top_k)
        contexts = [text for text, _ in contexts_with_scores]
        prompt = format_prompt(prompt_template, context_separator.join(contexts), question)
        answer = generate_with_llm(resources, prompt)
        generated_answers.append(answer)
        retrieved_contexts.append(contexts)
        ground_truths.append([truth] if isinstance(truth, str) else truth)

    return {
        'question': questions,
        'answer': generated_answers,
        'contexts': retrieved_contexts,
        'ground_truths': ground_truths,
    }
