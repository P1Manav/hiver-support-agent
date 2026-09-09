"""
src/evaluation/calibration.py
──────────────────────────────
Confidence calibration analysis for the intent classifier.

Plots classifier confidence vs. actual accuracy (reliability diagram).
Uses this to tune the escalation confidence threshold.

WHERE IT RUNS: Local CPU. Generates matplotlib plots.
"""

import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (saves to file)
logger = logging.getLogger(__name__)


def calibration_curve(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve data (confidence bins vs. actual accuracy).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        confidences: Predicted confidence scores.
        n_bins: Number of confidence bins.

    Returns:
        Dict with bin_means, bin_accs, bin_counts, ece (Expected Calibration Error).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = [(low <= c < high) for c in confidences]
        if sum(mask) == 0:
            continue
        bin_confs = [c for c, m in zip(confidences, mask) if m]
        bin_correct = [
            (yt == yp)
            for yt, yp, m in zip(y_true, y_pred, mask)
            if m
        ]
        bin_means.append(np.mean(bin_confs))
        bin_accs.append(np.mean(bin_correct))
        bin_counts.append(sum(mask))

    # Expected Calibration Error (ECE)
    n = len(y_true)
    ece = sum(
        count / n * abs(acc - conf)
        for conf, acc, count in zip(bin_means, bin_accs, bin_counts)
    )

    return {
        "bin_means": bin_means,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
        "ece": round(ece, 4),
    }


def plot_calibration(
    calibration_data: dict,
    output_path: str = "calibration_curve.png",
    title: str = "Classifier Calibration Curve",
) -> str:
    """
    Plot a reliability diagram and save to file.

    Args:
        calibration_data: Output from calibration_curve().
        output_path: Where to save the PNG.
        title: Plot title.

    Returns:
        Path to saved image.
    """
    bin_means = calibration_data["bin_means"]
    bin_accs = calibration_data["bin_accs"]
    bin_counts = calibration_data["bin_counts"]
    ece = calibration_data["ece"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax.plot(bin_means, bin_accs, "b-o", label=f"Model (ECE={ece:.3f})", linewidth=2, markersize=8)
    ax.fill_between(bin_means, bin_means, bin_accs, alpha=0.1, color="red", label="Calibration gap")
    ax.set_xlabel("Mean Confidence", fontsize=12)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Confidence histogram
    ax2 = axes[1]
    ax2.bar(bin_means, bin_counts, width=0.08, alpha=0.7, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Distribution", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Calibration plot saved to {output_path} (ECE={ece:.4f})")
    return output_path


def find_optimal_threshold(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    target_precision: float = 0.90,
) -> float:
    """
    Find the confidence threshold at which auto-handled examples achieve
    at least `target_precision` accuracy.

    Args:
        y_true, y_pred, confidences: Classification results.
        target_precision: Target accuracy for auto-handled examples.

    Returns:
        Optimal threshold float (round up to nearest 0.05).
    """
    thresholds = np.arange(0.5, 1.0, 0.025)
    for thresh in sorted(thresholds, reverse=True):
        mask = [c >= thresh for c in confidences]
        if sum(mask) == 0:
            continue
        correct = sum(
            yt == yp
            for yt, yp, m in zip(y_true, y_pred, mask) if m
        )
        precision = correct / sum(mask)
        coverage = sum(mask) / len(y_true)

        if precision >= target_precision:
            logger.info(
                f"Threshold {thresh:.3f}: precision={precision:.3f}, "
                f"coverage={coverage:.3f} ({sum(mask)}/{len(y_true)} auto-handled)"
            )
            return float(thresh)

    logger.warning("Could not find threshold meeting target precision. Returning 0.9.")
    return 0.9
