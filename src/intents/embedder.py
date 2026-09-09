"""
src/intents/embedder.py
────────────────────────
SentenceTransformer wrapper for encoding customer messages.

Uses all-MiniLM-L6-v2 (22M params, 384-dim) by default.
  - CPU: ~1000 sentences/sec
  - T4 GPU: ~5000 sentences/sec

Design decision (D-02): Same model used for clustering AND FAISS retrieval,
keeping the embedding space consistent across both stages.

WHERE IT RUNS: Local CPU (fine) or Colab/Kaggle T4 (faster).
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class MessageEmbedder:
    """
    Wraps SentenceTransformer with batching, caching, and numpy output.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 256,
    ):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            device: "cpu", "cuda", or None (auto-detect).
            batch_size: Sentences per encoding batch.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # lazy load

        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"MessageEmbedder configured: model={model_name}, device={self.device}")

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embedding vectors.

        Args:
            texts: List of raw (or redacted) text strings.
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), float32.
        """
        model = self._load()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text. Returns shape (embedding_dim,)."""
        return self.encode([text], show_progress=False)[0]

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        model = self._load()
        return model.get_sentence_embedding_dimension()
