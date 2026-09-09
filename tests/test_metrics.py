"""
tests/test_metrics.py
──────────────────────
Unit tests for evaluation metrics.

Run: pytest tests/test_metrics.py -v
"""

import pytest
import numpy as np
from src.evaluation.metrics import classification_metrics, retrieval_metrics, groundedness_score
from src.evaluation.calibration import calibration_curve, find_optimal_threshold


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        labels = ["a", "b", "c", "a", "b"]
        metrics = classification_metrics(labels, labels, label_names=["a", "b", "c"])
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_all_wrong(self):
        y_true = ["a", "a", "a"]
        y_pred = ["b", "b", "b"]
        metrics = classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        labels = ["a", "b", "c"]
        metrics = classification_metrics(labels, labels, label_names=labels)
        assert metrics["confusion_matrix"].shape == (3, 3)

    def test_per_class_f1_keys(self):
        labels = ["a", "b", "a"]
        preds = ["a", "a", "a"]
        metrics = classification_metrics(labels, preds, label_names=["a", "b"])
        assert "a" in metrics["per_class_f1"]
        assert "b" in metrics["per_class_f1"]


class TestRetrievalMetrics:
    def test_mean_similarity(self):
        retrieved = [
            [{"rank": 1, "similarity": 0.8}, {"rank": 2, "similarity": 0.6}],
            [{"rank": 1, "similarity": 0.9}],
        ]
        metrics = retrieval_metrics(retrieved)
        assert abs(metrics["mean_top1_similarity"] - 0.85) < 0.01

    def test_empty_retrieved(self):
        metrics = retrieval_metrics([[]])
        assert metrics["n_empty"] == 1


class TestGroundedness:
    def test_identical_gives_high_score(self):
        reply = "Please DM us with your order details."
        retrieved = [{"brand_reply": "Please DM us with your order details."}]
        score = groundedness_score(reply, retrieved)
        # ROUGE-L of identical text = 1.0 (or close, depending on tokenization)
        assert score > 0.8

    def test_empty_retrieved_returns_zero(self):
        assert groundedness_score("test reply", []) == 0.0


class TestCalibration:
    def test_calibration_curve_bins(self):
        y_true = ["a"] * 50 + ["b"] * 50
        y_pred = ["a"] * 50 + ["b"] * 50
        confs = [0.9] * 50 + [0.8] * 50
        data = calibration_curve(y_true, y_pred, confs, n_bins=5)
        assert "ece" in data
        assert data["ece"] >= 0.0
        assert len(data["bin_means"]) > 0

    def test_find_threshold_returns_float(self):
        y_true = ["a"] * 10
        y_pred = ["a"] * 10
        confs = [0.9] * 10
        thresh = find_optimal_threshold(y_true, y_pred, confs, target_precision=0.90)
        assert isinstance(thresh, float)
        assert 0.0 < thresh <= 1.0
