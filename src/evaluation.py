from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer


def format_for_squad_evaluation(
    answers: List[str],
    ground_truths: List[List[str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    predictions = [
        {"prediction_text": answer, "id": str(idx)}
        for idx, answer in enumerate(answers)
    ]
    references = []
    for idx, truths in enumerate(ground_truths):
        texts = truths if isinstance(truths, list) else [truths]
        filtered = [text for text in texts if text]
        references.append(
            {
                "answers": {"text": filtered, "answer_start": [0] * len(filtered)},
                "id": str(idx),
            }
        )
    return predictions, references


def build_ragas_dataset(
    records: Dict[str, List],
    *,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    lengths = {len(values) for values in records.values()}
    if len(lengths) != 1:
        raise ValueError("All columns must have identical lengths.")

    references: List[str] = []
    for truth in records["ground_truths"]:
        if isinstance(truth, list) and truth:
            references.append(truth[0])
        elif isinstance(truth, str) and truth:
            references.append(truth)
        else:
            references.append("")

    ragas_records = {
        "question": records["question"],
        "answer": records["answer"],
        "contexts": records["contexts"],
        "ground_truth": references,
    }

    dataset = Dataset.from_dict(ragas_records)
    if sample_size is not None and dataset.num_rows > sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    return dataset


class SentenceTransformerEmbeddingsWrapper:
    """Minimal wrapper so RAGAs can reuse a SentenceTransformer."""

    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode([text], convert_to_numpy=True)[0]
        return np.asarray(vector).tolist()


def extract_ragas_scores(result_obj) -> Dict[str, float]:
    """Normalize RAGAs results to a flat {metric: float} mapping.

    Supports recent RAGAs EvaluationResult objects (with ``.scores`` or
    ``.to_dict()``), and dict-like fallbacks. Each metric value may be a raw
    float or a small object exposing ``.score``.
    """
    summary: Dict[str, float] = {}

    # Preferred: new-style EvaluationResult exposes a mapping of metrics
    if hasattr(result_obj, "scores") and isinstance(getattr(result_obj, "scores"), dict):
        mapping = result_obj.scores
    elif hasattr(result_obj, "to_dict"):
        try:
            mapping = result_obj.to_dict()
        except Exception:  # pragma: no cover - defensive fallback
            mapping = {}
    elif isinstance(result_obj, dict):
        mapping = result_obj
    else:
        # Last resort: attempt casting to dict
        try:
            mapping = dict(result_obj)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - unsupported type
            raise TypeError(f"Unsupported RAGAs result type: {type(result_obj)}") from exc

    for metric_name, metric_value in mapping.items():
        value = getattr(metric_value, "score", metric_value)
        try:
            summary[metric_name] = float(value)
        except Exception:
            # Some objects may expose alternative numeric attributes
            alt = getattr(value, "value", None)
            if alt is not None:
                summary[metric_name] = float(alt)
            else:  # pragma: no cover - skip non-numeric
                continue
    return summary
