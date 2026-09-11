"""src/evaluation/__init__.py"""
from .metrics import classification_metrics, retrieval_metrics, groundedness_score, compute_full_metrics
from .calibration import calibration_curve, plot_calibration, find_optimal_threshold

# LLMJudge is imported lazily (requires ollama package + running Ollama server).
# Import it directly: from src.evaluation.judge import LLMJudge

__all__ = [
    "classification_metrics", "retrieval_metrics", "groundedness_score", "compute_full_metrics",
    "calibration_curve", "plot_calibration", "find_optimal_threshold",
]
