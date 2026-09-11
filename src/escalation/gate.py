"""
src/escalation/gate.py
───────────────────────
Escalation decision gate — transparent rule + signal fusion.

Design decision (D-07): No LLM call here. Escalation is a pure function of
observable signals. Every signal contributes to the human-readable reason string,
so the decision is auditable and tunable.

Signal fusion logic:
  1. Denylist check: some intents ALWAYS escalate (billing, account security, etc.)
  2. Classifier confidence: below threshold → escalate (uncertain prediction)
  3. Retrieval similarity: below threshold → escalate (no good precedent = can't ground reply)
  4. Sentiment/urgency: above threshold → escalate (customer is very distressed)
  5. Message length: very short messages (< 10 tokens) → escalate (insufficient context)

All thresholds are configurable in config.yaml.

WHERE IT RUNS: Local CPU. < 1ms per message.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EscalationResult:
    """Structured output of the escalation gate."""
    decision: str          # "auto_handle" or "escalate"
    reason: str            # Human-readable, specific reason
    signals: dict          # All signal values for debugging/logging
    triggered_rules: list[str]  # Which rules fired


class EscalationGate:
    """
    Transparent escalation decision gate combining classifier confidence,
    retrieval similarity, intent denylist, and urgency signals.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        similarity_threshold: float = 0.60,
        urgency_threshold: float = 0.80,
        denylist_intents: Optional[set] = None,
        min_message_tokens: int = 3,
    ):
        """
        Args:
            confidence_threshold: Below this confidence → escalate.
            similarity_threshold: Below this retrieval similarity → escalate.
            urgency_threshold: Above this urgency score → escalate.
            denylist_intents: Set of intent names that always escalate.
            min_message_tokens: Messages shorter than this → escalate.
        """
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.urgency_threshold = urgency_threshold
        self.denylist_intents = denylist_intents or {
            "billing_payment",
            "account_access",
            "seller_complaint",
        }
        self.min_message_tokens = min_message_tokens

    def decide(
        self,
        message: str,
        intent: str,
        confidence: float,
        retrieved_examples: list[dict],
        urgency_score: float = 0.0,
        sentiment: str = "neutral",
    ) -> EscalationResult:
        """
        Make an escalation decision based on all available signals.

        Args:
            message: Customer message text (PII-redacted).
            intent: Classified intent name.
            confidence: Classifier confidence score (0–1).
            retrieved_examples: Top-k retrieved results (with 'similarity' key).
            urgency_score: Urgency score from sentiment analysis (0–1).
            sentiment: "positive", "neutral", "negative", "urgent".

        Returns:
            EscalationResult with decision, reason, and signal values.
        """
        triggered_rules = []
        reasons = []

        # Signal values
        top_similarity = retrieved_examples[0]["similarity"] if retrieved_examples else 0.0
        word_count = len(message.split())

        signals = {
            "intent": intent,
            "confidence": round(confidence, 3),
            "top_similarity": round(top_similarity, 3),
            "urgency_score": round(urgency_score, 3),
            "sentiment": sentiment,
            "word_count": word_count,
            "n_retrieved": len(retrieved_examples),
        }

        # Rule 0: Non-English check
        try:
            from langdetect import detect
            if detect(message) != "en":
                triggered_rules.append("non_english")
                reasons.append("Message appears to be non-English; requires specialized handling.")
        except Exception:
            pass  # Fail gracefully if message is too short or langdetect errors

        # Rule 1: Intent denylist (always escalate)
        if intent in self.denylist_intents:
            triggered_rules.append("denylist")
            reasons.append(
                f"Intent '{intent}' is in the high-risk escalation denylist "
                f"(billing/account/security issues require human review)."
            )

        # Rule 2: Low classifier confidence
        if confidence < self.confidence_threshold:
            triggered_rules.append("low_confidence")
            reasons.append(
                f"Classifier confidence is low ({confidence:.2f} < threshold {self.confidence_threshold:.2f}); "
                f"the predicted intent '{intent}' may be incorrect."
            )

        # Rule 3: Low retrieval similarity (no good precedent)
        if top_similarity < self.similarity_threshold:
            triggered_rules.append("low_similarity")
            reasons.append(
                f"Best retrieval similarity is {top_similarity:.2f} (< threshold {self.similarity_threshold:.2f}); "
                f"no sufficiently similar historical resolution found to ground a safe reply."
            )

        # Rule 4: High urgency / very negative sentiment
        if urgency_score >= self.urgency_threshold:
            triggered_rules.append("high_urgency")
            reasons.append(
                f"Urgency score {urgency_score:.2f} (>= threshold {self.urgency_threshold:.2f}); "
                f"customer appears very distressed (sentiment: {sentiment})."
            )

        # Rule 5: Message too short for reliable classification
        if word_count < self.min_message_tokens:
            triggered_rules.append("too_short")
            reasons.append(
                f"Message is very short ({word_count} words); "
                f"insufficient context for reliable classification or reply generation."
            )

        # Final decision
        should_escalate = len(triggered_rules) > 0
        decision = "escalate" if should_escalate else "auto_handle"

        if should_escalate:
            reason = "ESCALATE — " + " | ".join(reasons)
        else:
            reason = (
                f"Auto-handled: confidence {confidence:.2f} >= {self.confidence_threshold}; "
                f"similarity {top_similarity:.2f} >= {self.similarity_threshold}; "
                f"intent '{intent}' not in denylist; urgency score {urgency_score:.2f} is acceptable."
            )

        return EscalationResult(
            decision=decision,
            reason=reason,
            signals=signals,
            triggered_rules=triggered_rules,
        )

    @classmethod
    def from_config(cls, config: dict) -> "EscalationGate":
        """Instantiate from config.yaml escalation section."""
        esc = config.get("escalation", {})
        taxonomy_denylist = set(esc.get("denylist", []))
        return cls(
            confidence_threshold=esc.get("confidence_threshold", 0.70),
            similarity_threshold=esc.get("similarity_threshold", 0.60),
            urgency_threshold=esc.get("sentiment_urgency_threshold", 0.80),
            denylist_intents=taxonomy_denylist,
        )
