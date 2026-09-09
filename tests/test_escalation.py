"""
tests/test_escalation.py
─────────────────────────
Unit tests for the escalation gate.

Run: pytest tests/test_escalation.py -v
"""

import pytest
from src.escalation.gate import EscalationGate, EscalationResult

# Shared gate for all tests (using default thresholds)
gate = EscalationGate(
    confidence_threshold=0.70,
    similarity_threshold=0.60,
    urgency_threshold=0.80,
    denylist_intents={"billing_payment", "account_access", "seller_complaint"},
)

# Helper
def make_retrieved(similarity: float) -> list[dict]:
    return [{"rank": 1, "similarity": similarity, "customer_text": "test", "brand_reply": "test reply"}]


class TestDenylist:
    def test_billing_always_escalates(self):
        result = gate.decide("I was charged twice", "billing_payment", 0.95, make_retrieved(0.85))
        assert result.decision == "escalate"
        assert "denylist" in result.triggered_rules

    def test_account_always_escalates(self):
        result = gate.decide("I can't log in", "account_access", 0.90, make_retrieved(0.80))
        assert result.decision == "escalate"
        assert "denylist" in result.triggered_rules

    def test_non_denylist_not_triggered(self):
        result = gate.decide("Where is my order", "order_status_inquiry", 0.95, make_retrieved(0.85))
        assert "denylist" not in result.triggered_rules


class TestConfidenceThreshold:
    def test_low_confidence_escalates(self):
        result = gate.decide("help", "order_status_inquiry", 0.45, make_retrieved(0.85))
        assert result.decision == "escalate"
        assert "low_confidence" in result.triggered_rules

    def test_high_confidence_ok(self):
        result = gate.decide("where is my package", "order_status_inquiry", 0.92, make_retrieved(0.85))
        assert "low_confidence" not in result.triggered_rules


class TestSimilarityThreshold:
    def test_low_similarity_escalates(self):
        result = gate.decide("my order is late", "order_status_inquiry", 0.90, make_retrieved(0.35))
        assert result.decision == "escalate"
        assert "low_similarity" in result.triggered_rules

    def test_empty_retrieved_escalates(self):
        result = gate.decide("my order is late", "order_status_inquiry", 0.90, [])
        assert result.decision == "escalate"
        assert "low_similarity" in result.triggered_rules


class TestUrgency:
    def test_high_urgency_escalates(self):
        result = gate.decide("THIS IS URGENT", "order_status_inquiry", 0.90, make_retrieved(0.85), urgency_score=0.95)
        assert result.decision == "escalate"
        assert "high_urgency" in result.triggered_rules

    def test_low_urgency_ok(self):
        result = gate.decide("where is my order", "order_status_inquiry", 0.90, make_retrieved(0.85), urgency_score=0.2)
        assert "high_urgency" not in result.triggered_rules


class TestAutoHandle:
    def test_all_signals_ok_auto_handles(self):
        result = gate.decide(
            "where is my package",
            "order_status_inquiry",
            0.95,
            make_retrieved(0.85),
            urgency_score=0.1,
        )
        assert result.decision == "auto_handle"
        assert result.triggered_rules == []

    def test_reason_is_specific(self):
        result = gate.decide(
            "where is my package",
            "order_status_inquiry",
            0.95,
            make_retrieved(0.85),
        )
        assert "confidence" in result.reason
        assert "0.95" in result.reason


class TestReasonQuality:
    def test_escalation_reason_mentions_rule(self):
        result = gate.decide("I was charged twice", "billing_payment", 0.95, make_retrieved(0.85))
        assert "billing_payment" in result.reason
        assert "denylist" in result.reason.lower()

    def test_multiple_rules_all_mentioned(self):
        result = gate.decide("", "billing_payment", 0.40, [], urgency_score=0.95)
        assert len(result.triggered_rules) >= 3
