"""
src/baselines/simple.py
────────────────────────
Simple baseline: TF-IDF + Logistic Regression classifier + nearest-neighbor template reply.

No LLM anywhere. Pure sklearn. This is the "classical ML" comparison point.

WHERE IT RUNS: Local CPU. Training ~30s, inference <1ms.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SimpleBaseline:
    """
    TF-IDF + LogReg intent classifier + cosine-NN template reply retrieval.
    """

    def __init__(self, max_features: int = 30000, C: float = 1.0):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._reply_texts: Optional[list[str]] = None
        self._query_texts: Optional[list[str]] = None
        self._vectorizer: Optional[TfidfVectorizer] = None

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        resolved_queries: Optional[list[str]] = None,
        resolved_replies: Optional[list[str]] = None,
    ) -> "SimpleBaseline":
        """
        Fit the classifier and (optionally) the reply retrieval index.

        Args:
            texts: Training customer messages.
            labels: Training intent labels.
            resolved_queries: Customer messages from resolved threads (for NN retrieval).
            resolved_replies: Corresponding brand replies.
        """
        logger.info(f"Fitting TF-IDF + LogReg on {len(texts):,} examples...")
        self.pipeline.fit(texts, labels)
        logger.info("Classifier fitted.")

        # Build NN retrieval index using the same TF-IDF vectorizer
        if resolved_queries and resolved_replies:
            self._vectorizer = self.pipeline.named_steps["tfidf"]
            self._tfidf_matrix = self._vectorizer.transform(resolved_queries)
            self._reply_texts = resolved_replies
            self._query_texts = resolved_queries
            logger.info(f"Retrieval index built: {len(resolved_replies):,} replies.")

        return self

    def predict(self, text: str) -> dict:
        """
        Predict intent and retrieve nearest-neighbor reply.

        Returns:
            Dict matching the full pipeline output format.
        """
        intent = self.pipeline.predict([text])[0]
        proba = self.pipeline.predict_proba([text])[0]
        confidence = float(max(proba))

        draft_reply = self._retrieve_reply(text)

        return {
            "intent": intent,
            "confidence": confidence,
            "draft_reply": draft_reply,
            "decision": "auto_handle",
            "reason": "Simple baseline: TF-IDF + LogReg + NN template.",
            "guardrail_flags": [],
        }

    def _retrieve_reply(self, query: str) -> str:
        """Find the nearest-neighbor reply via TF-IDF cosine similarity."""
        if self._tfidf_matrix is None or self._vectorizer is None:
            return "Please DM us with more details and we'll assist you. ^Team"

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        best_idx = int(np.argmax(sims))
        return self._reply_texts[best_idx] if self._reply_texts else ""

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def evaluate(self, golden_df: pd.DataFrame) -> dict:
        """Evaluate on golden set DataFrame."""
        from src.evaluation.metrics import classification_metrics

        # golden_set_sampler.py writes 'human_intent' and 'customer_text'
        label_col = "human_intent" if "human_intent" in golden_df.columns else "intent"
        text_col = "customer_text" if "customer_text" in golden_df.columns else "text"

        y_true = golden_df[label_col].tolist()
        texts = golden_df[text_col].tolist()
        preds = self.pipeline.predict(texts)
        metrics = classification_metrics(y_true, preds.tolist())

        print(f"\n=== Simple Baseline Results ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
        print(metrics["classification_report"])
        return metrics

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SimpleBaseline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SimpleBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)
