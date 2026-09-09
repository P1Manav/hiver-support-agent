"""
src/retrieval/retriever.py
───────────────────────────
Query the FAISS index to retrieve top-k similar historical resolutions.

WHERE IT RUNS: Local (CPU fine). < 50ms per query on the 50k index.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Retriever:
    """
    Wraps a FAISS index and metadata DataFrame to retrieve top-k
    (customer_text, brand_reply) pairs for a given query message.
    """

    def __init__(
        self,
        index,           # faiss.Index
        metadata: pd.DataFrame,
        embedder,        # MessageEmbedder
        top_k: int = 5,
        text_column: str = "customer_text",
        reply_column: str = "brand_reply",
    ):
        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.top_k = top_k
        self.text_column = text_column
        self.reply_column = reply_column

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve top-k similar past resolutions for a query message.

        Args:
            query: Customer message (PII-redacted).
            top_k: Override instance top_k.

        Returns:
            List of dicts:
            [{
               "rank": int,
               "similarity": float,
               "customer_text": str,
               "brand_reply": str,
               "thread_id": str,
            }]
        """
        k = top_k or self.top_k
        query_emb = self.embedder.encode_single(query).reshape(1, -1)

        # FAISS inner product search (cosine similarity since embeddings are L2-normalized)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for padding
                continue
            row = self.metadata.iloc[idx]
            results.append({
                "rank": rank + 1,
                "similarity": float(score),
                "customer_text": row.get(self.text_column, ""),
                "brand_reply": row.get(self.reply_column, ""),
                "thread_id": row.get("thread_id", ""),
            })
        return results

    def retrieve_for_thread(
        self,
        turns: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve using multi-turn context (D-13).
        Concatenates last 3 customer turns as the query.
        """
        customer_texts = [t["text"] for t in turns if t.get("inbound", True)]
        query = " ".join(customer_texts[-3:])
        return self.retrieve(query, top_k=top_k)

    @property
    def max_similarity(self) -> float:
        """Max possible similarity score (1.0 for L2-normalized embeddings)."""
        return 1.0
