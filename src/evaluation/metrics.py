"""
src/evaluation/metrics.py
──────────────────────────
Automated evaluation metrics:
  - Intent classification: accuracy, macro-F1, per-class F1, confusion matrix
  - Retrieval relevance: hit rate, mean similarity@k
  - Reply groundedness: ROUGE-L vs. retrieved context (proxy metric)

WHERE IT RUNS: Local CPU. No GPU or Ollama needed.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground-truth intent labels.
        y_pred: Predicted intent labels.
        label_names: Ordered list of intent names.

    Returns:
        Dict with accuracy, macro_f1, per_class_f1, confusion_matrix_df.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class F1
    per_class = f1_score(
        y_true, y_pred,
        labels=label_names,
        average=None,
        zero_division=0,
    )
    per_class_dict = dict(zip(label_names or sorted(set(y_true)), per_class.tolist()))

    # Confusion matrix
    labels = label_names or sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    report = classification_report(y_true, y_pred, zero_division=0)

    logger.info(f"Classification metrics — Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_dict,
        "confusion_matrix": cm_df,
        "classification_report": report,
    }


def retrieval_metrics(
    retrieved_results: list[list[dict]],
    relevance_labels: Optional[list[list[bool]]] = None,
) -> dict:
    """
    Compute retrieval quality metrics.

    Args:
        retrieved_results: List of top-k retrieved result lists (one per query).
        relevance_labels: Optional binary relevance labels per retrieved result.

    Returns:
        Dict with mean_top1_similarity, mean_topk_similarity, hit_rate_at_1.
    """
    top1_sims = [r[0]["similarity"] for r in retrieved_results if r]
    topk_sims = [
        np.mean([ex["similarity"] for ex in r])
        for r in retrieved_results if r
    ]

    metrics = {
        "mean_top1_similarity": float(np.mean(top1_sims)) if top1_sims else 0.0,
        "mean_topk_similarity": float(np.mean(topk_sims)) if topk_sims else 0.0,
        "n_queries": len(retrieved_results),
        "n_empty": sum(1 for r in retrieved_results if not r),
    }

    if relevance_labels:
        # Hit rate: fraction of queries where top-1 is relevant
        hit_rate = sum(
            1 for labels in relevance_labels if labels and labels[0]
        ) / len(relevance_labels)
        metrics["hit_rate_at_1"] = hit_rate

    return metrics


def groundedness_score(
    draft_reply: str,
    retrieved_examples: list[dict],
) -> float:
    """
    Compute a proxy groundedness score using ROUGE-L overlap between
    the draft reply and the retrieved brand replies.

    This is a cheap proxy — the LLM judge is the authoritative groundedness metric.

    Returns:
        Float in [0, 1] — higher = more overlap with retrieved context.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        logger.warning("rouge_score not installed. Run: pip install rouge-score. Returning 0.")
        return 0.0

    if not retrieved_examples:
        return 0.0

    scores = []
    for ex in retrieved_examples:
        reference = ex.get("brand_reply", "")
        if reference:
            score = scorer.score(reference, draft_reply)
            scores.append(score["rougeL"].fmeasure)

    return float(np.mean(scores)) if scores else 0.0


def compute_full_metrics(
    golden_df: pd.DataFrame,
    predictions: list[dict],
    taxonomy_names: list[str],
) -> dict:
    """
    Compute all metrics from a golden set DataFrame and model predictions.

    Args:
        golden_df: DataFrame with columns [text, intent, draft_reply, ...].
        predictions: List of prediction dicts from inference pipeline.
        taxonomy_names: Ordered intent names.

    Returns:
        Nested dict with classification, retrieval, and groundedness metrics.
    """
    y_true = golden_df["intent"].tolist()
    y_pred = [p["intent"] for p in predictions]
    retrieved = [p.get("retrieved_examples", []) for p in predictions]
    drafts = [p.get("draft_reply", "") for p in predictions]

    clf_metrics = classification_metrics(y_true, y_pred, label_names=taxonomy_names)
    ret_metrics = retrieval_metrics(retrieved)
    ground_scores = [groundedness_score(d, r) for d, r in zip(drafts, retrieved)]

    return {
        "classification": clf_metrics,
        "retrieval": ret_metrics,
        "groundedness": {
            "mean_rouge_l": float(np.mean(ground_scores)),
            "scores": ground_scores,
        },
    }
