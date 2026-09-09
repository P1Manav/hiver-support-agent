"""
src/retrieval/indexer.py
─────────────────────────
Build and persist the FAISS index over historical (customer_issue, brand_reply) pairs.

Design decision (D-06): Use IndexFlatIP (exact inner-product = cosine similarity
after L2-normalization). Fast enough for 50k examples and gives true similarity
scores for the escalation gate.

WHERE IT RUNS: Local CPU or Colab. ~3 min for 50k embeddings.
Expected output: faiss_index/ directory with index.faiss + metadata.parquet
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_index(
    threads_df: pd.DataFrame,
    embedder,  # MessageEmbedder instance
    index_dir: str = "faiss_index",
    text_column: str = "customer_text",
    reply_column: str = "brand_reply",
) -> tuple:
    """
    Embed all resolved (customer_text, brand_reply) pairs and build FAISS index.

    Args:
        threads_df: DataFrame with at least [thread_id, customer_text, brand_reply].
        embedder: MessageEmbedder instance.
        index_dir: Directory to save index files.
        text_column: Column containing customer messages.
        reply_column: Column containing brand replies.

    Returns:
        (faiss_index, metadata_df) tuple.
    """
    import faiss

    # Filter to threads that have a brand reply (resolved threads only)
    resolved = threads_df.dropna(subset=[reply_column]).copy()
    resolved = resolved[resolved[reply_column].str.strip() != ""]
    logger.info(f"Building index over {len(resolved):,} resolved threads...")

    texts = resolved[text_column].fillna("").tolist()

    logger.info("Embedding customer messages...")
    embeddings = embedder.encode(texts, show_progress=True)

    # Build FAISS flat index (exact search)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (embeddings are L2-normalized)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}.")

    # Save index + metadata
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))

    metadata = resolved[["thread_id", text_column, reply_column, "n_turns"]].reset_index(drop=True)
    metadata.to_parquet(str(out_dir / "metadata.parquet"), index=False)

    logger.info(f"Index saved to {out_dir}/ (index.faiss + metadata.parquet).")
    return index, metadata


def load_index(index_dir: str = "faiss_index"):
    """
    Load a previously saved FAISS index and metadata.

    Returns:
        (faiss_index, metadata_df) tuple.
    """
    import faiss

    out_dir = Path(index_dir)
    index_path = out_dir / "index.faiss"
    meta_path = out_dir / "metadata.parquet"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/05_build_faiss_index.py first."
        )

    index = faiss.read_index(str(index_path))
    metadata = pd.read_parquet(str(meta_path))
    logger.info(f"Loaded FAISS index: {index.ntotal} vectors from {out_dir}/")
    return index, metadata
