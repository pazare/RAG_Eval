from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_text_table(path: str, text_column: str) -> pd.DataFrame:
    """Load a parquet table, drop empty records, and normalise text."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=[text_column])
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]
    return df.reset_index(drop=True)


def load_queries(path: str) -> pd.DataFrame:
    """Load the questions/answers split and guarantee non-empty strings."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")]
    return df.reset_index(drop=True)


def compute_length_summary(df: pd.DataFrame, column: str) -> Tuple[pd.Series, pd.Series]:
    """Return the raw length series and descriptive statistics."""
    lengths = df[column].str.len()
    summary = lengths.describe().round(2)
    return lengths, summary


def plot_length_diagnostics(lengths: pd.Series, bins: int = 80) -> None:
    """Render histogram + boxplot diagnostics for a length distribution."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(lengths, bins=bins, ax=axes[0], kde=True, color="#4C78A8")
    axes[0].set_title("Length Distribution")
    axes[0].set_xlabel("Characters")
    sns.boxplot(x=lengths, ax=axes[1], color="#F58518")
    axes[1].set_title("Length Boxplot")
    axes[1].set_xlabel("Characters")
    plt.tight_layout()
    plt.show()


def display_text_extremes(df: pd.DataFrame, column: str, *, count: int = 1) -> None:
    """Print the shortest and longest samples for quick inspection."""
    ranked = df.assign(length=df[column].str.len()).sort_values("length", ascending=True)
    for label, sample in (("Shortest", ranked.head(count)), ("Longest", ranked.tail(count))):
        for _, row in sample.iterrows():
            print(f"{label} sample (length={row['length']}):")
            print(row[column][:400])
            print("---")


def summarize_text_table(
    df: pd.DataFrame,
    text_column: str,
    *,
    label: str | None = None,
    extra_categorical_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Produce a compact dataframe of text statistics for reporting."""
    extras = list(extra_categorical_columns or [])
    if df.empty:
        return pd.DataFrame([{"metric": "rows", "value": 0}])
    lengths = df[text_column].str.len()
    rows: List[Dict[str, object]] = [
        {"metric": "label", "value": label or text_column},
        {"metric": "rows", "value": f"{len(df):,}"},
        {"metric": "columns", "value": ", ".join(df.columns)},
        {"metric": "avg_length_chars", "value": f"{lengths.mean():.2f}"},
        {"metric": "median_length_chars", "value": f"{lengths.median():.0f}"},
        {"metric": "min_length_chars", "value": int(lengths.min())},
        {"metric": "max_length_chars", "value": int(lengths.max())},
        {"metric": "std_length_chars", "value": f"{lengths.std(ddof=0):.2f}"},
        {"metric": "duplicate_text_entries", "value": f"{df.duplicated(subset=[text_column]).sum():,}"},
    ]
    for column in extras:
        if column in df.columns:
            rows.append({"metric": f"unique_{column}", "value": f"{df[column].nunique():,}"})
    return pd.DataFrame(rows)


def derive_text_quality_notes(df: pd.DataFrame, text_column: str) -> List[str]:
    """Generate concise bullet points about data quality."""
    if df.empty:
        return ["Dataset empty after preprocessing; verify source artefacts."]
    lengths = df[text_column].str.len()
    empties = int((lengths == 0).sum())
    duplicates = int(df.duplicated(subset=[text_column]).sum())
    notes = [
        f"Length distribution spans {int(lengths.min())}-{int(lengths.max())} characters with median {int(lengths.median())}.",
        f"5th to 95th percentile lies between {int(lengths.quantile(0.05))} and {int(lengths.quantile(0.95))} characters.",
        "No empty records detected after cleaning." if not empties else f"Detected {empties:,} empty entries post-cleaning; consider additional filtering.",
        "No duplicate text entries detected." if not duplicates else f"Found {duplicates:,} duplicate entries; deduplication may help.",
    ]
    return notes


def ensure_results_dir() -> Path:
    """Return (and create if necessary) the writable results directory."""
    cwd = Path.cwd().resolve()
    candidates = [cwd / "results", cwd.parent / "results"]
    for candidate in candidates:
        if candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    fallback = candidates[0]
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def persist_json(payload: Dict, filename: str) -> Path:
    """Write a JSON payload under results/ and return the path."""
    output_path = ensure_results_dir() / filename
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return output_path


def persist_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """Persist a dataframe as CSV under results/."""
    output_path = ensure_results_dir() / filename
    df.to_csv(output_path, index=False)
    return output_path
