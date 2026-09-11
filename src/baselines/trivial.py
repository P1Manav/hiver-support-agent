"""
src/baselines/trivial.py
─────────────────────────
Trivial baseline: always predict majority class, return a canned reply.

This is the "do nothing smart" floor. The purpose is to quantify how much
value the ML pipeline adds over the simplest possible system.

WHERE IT RUNS: Local CPU. Instant.
"""

import logging
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

_CANNED_REPLY = (
    "Hi! Thanks for reaching out. We're sorry to hear you're having an issue. "
    "Please DM us with your order details and we'll look into this right away. ^Team"
)


class TrivialBaseline:
    """
    Always predicts the majority class intent and returns a single canned reply.
    """

    def __init__(self, majority_intent: str = "order_status_inquiry"):
        self.majority_intent = majority_intent
        self.canned_reply = _CANNED_REPLY

    def fit(self, texts: list[str], labels: list[str]) -> "TrivialBaseline":
        """Set majority class from training data."""
        counter = Counter(labels)
        self.majority_intent = counter.most_common(1)[0][0]
        logger.info(f"Trivial baseline majority class: '{self.majority_intent}' "
                    f"({counter[self.majority_intent]:,}/{len(labels):,} = "
                    f"{counter[self.majority_intent]/len(labels):.1%})")
        return self

    def predict(self, text: str) -> dict:
        return {
            "intent": self.majority_intent,
            "confidence": 1.0,
            "draft_reply": self.canned_reply,
            "decision": "auto_handle",
            "reason": "Trivial baseline: always auto-handle with canned reply.",
            "guardrail_flags": [],
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def evaluate(self, golden_df: pd.DataFrame) -> dict:
        """
        Evaluate on a golden set DataFrame.

        Args:
            golden_df: DataFrame with columns [customer_text, human_intent].

        Returns:
            Dict with accuracy and macro_f1.
        """
        from src.evaluation.metrics import classification_metrics

        # golden_set_sampler.py writes 'human_intent' (filled during labeling)
        label_col = "human_intent" if "human_intent" in golden_df.columns else "intent"
        y_true = golden_df[label_col].tolist()
        y_pred = [self.majority_intent] * len(y_true)
        metrics = classification_metrics(y_true, y_pred)

        print(f"\n=== Trivial Baseline Results ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
        print(metrics["classification_report"])
        return metrics
