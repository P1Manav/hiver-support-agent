"""
src/classifier/predict.py
──────────────────────────
Load the fine-tuned DistilBERT classifier and run fast CPU inference.

Design decision (D-05): This is the RUNTIME classifier. It's 100× faster
than calling the LLM at inference time (< 50ms on CPU vs. 3–8s for LLM).

WHERE IT RUNS: Local (CPU fine — 66M params, fast inference). No Ollama needed.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Loads the fine-tuned DistilBERT classifier and provides predict() and
    predict_with_confidence() methods.
    """

    def __init__(
        self,
        model_dir: str = "models/classifier",
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        """
        Args:
            model_dir: Directory containing the saved HF model + tokenizer.
            device: "cpu" or "cuda". None = auto-detect.
            max_length: Token truncation length.
        """
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._pipeline = None

        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"IntentClassifier configured: model_dir={model_dir}, device={self.device}")

    def _load(self):
        if self._pipeline is not None:
            return

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {self.model_dir}. "
                "Run the fine-tuning notebook first: notebooks/03_finetune_classifier.ipynb"
            )

        from transformers import pipeline
        logger.info(f"Loading classifier from {self.model_dir}...")
        self._pipeline = pipeline(
            "text-classification",
            model=str(self.model_dir),
            tokenizer=str(self.model_dir),
            device=0 if self.device == "cuda" else -1,
            top_k=None,  # return all class scores
            truncation=True,
            max_length=self.max_length,
        )
        logger.info("Classifier loaded.")

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict the intent label and confidence for a single text.

        Args:
            text: Customer message (should be PII-redacted).

        Returns:
            (intent_name, confidence) tuple.
        """
        self._load()
        results = self._pipeline(text)[0]  # list of {label, score}
        best = max(results, key=lambda x: x["score"])
        return best["label"], float(best["score"])

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[tuple[str, float]]:
        """
        Predict intents for a list of texts.

        Args:
            texts: Customer messages.
            batch_size: HF pipeline batch size.

        Returns:
            List of (intent_name, confidence) tuples.
        """
        self._load()
        results = self._pipeline(texts, batch_size=batch_size)
        return [
            (max(r, key=lambda x: x["score"])["label"],
             float(max(r, key=lambda x: x["score"])["score"]))
            for r in results
        ]

    def predict_all_scores(self, text: str) -> dict[str, float]:
        """
        Return the full probability distribution over all intents.

        Returns:
            Dict mapping intent_name → confidence score.
        """
        self._load()
        results = self._pipeline(text)[0]
        return {r["label"]: float(r["score"]) for r in results}

    def predict_with_thread(self, turns: list[dict]) -> tuple[str, float]:
        """
        Classify using multi-turn thread context (D-13).
        Concatenates the last 3 customer turns as context.

        Args:
            turns: List of turn dicts from thread reconstruction.

        Returns:
            (intent_name, confidence)
        """
        customer_turns = [t["text"] for t in turns if t.get("inbound", True)]
        # Use last 3 turns for context, joined with separator
        context = " [SEP] ".join(customer_turns[-3:])
        return self.predict(context)
