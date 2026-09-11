"""
src/intents/clusterer.py
─────────────────────────
Cluster customer message embeddings to discover natural intent groups.

Supports:
  - KMeans (default): predictable k clusters, fast, easy to tune (D-03)
  - HDBSCAN (optional): finds natural clusters, handles noise, variable k

WHERE IT RUNS: Local CPU or Colab. ~1 min for 5k points, 5 min for 50k.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 12,
    random_seed: int = 42,
) -> tuple[np.ndarray, object]:
    """
    Run KMeans clustering on embeddings.

    Args:
        embeddings: (n_samples, dim) float32 array.
        n_clusters: Number of clusters.
        random_seed: For reproducibility.

    Returns:
        (labels, kmeans_model) — labels shape (n_samples,), int32.
    """
    from sklearn.cluster import KMeans

    logger.info(f"Running KMeans with k={n_clusters} on {len(embeddings):,} embeddings...")
    km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto")
    labels = km.fit_predict(embeddings)
    logger.info(f"KMeans done. Cluster sizes: {np.bincount(labels).tolist()}")
    return labels.astype(np.int32), km


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None,
) -> tuple[np.ndarray, object]:
    """
    Run HDBSCAN clustering on embeddings.

    Args:
        embeddings: (n_samples, dim) float32 array.
        min_cluster_size: Minimum points to form a cluster.
        min_samples: Controls noise sensitivity (default = min_cluster_size).

    Returns:
        (labels, hdbscan_model). Labels: -1 = noise point.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")

    logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size}) on {len(embeddings):,} embeddings...")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric="euclidean",  # embeddings are L2-normalized; euclidean ≈ cosine
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"HDBSCAN found {n_clusters} clusters, {n_noise:,} noise points.")
    return labels.astype(np.int32), clusterer


def get_cluster_samples(
    texts: list[str],
    labels: np.ndarray,
    n_samples: int = 10,
    random_seed: int = 42,
) -> dict[int, list[str]]:
    """
    Return a random sample of texts from each cluster for manual inspection.

    Args:
        texts: List of customer messages (same order as embeddings).
        labels: Cluster label per text.
        n_samples: Texts to sample per cluster.
        random_seed: For reproducibility.

    Returns:
        Dict mapping cluster_id → [sampled texts].
    """
    import random
    random.seed(random_seed)

    cluster_ids = sorted(set(labels))
    samples = {}
    for cid in cluster_ids:
        idxs = [i for i, l in enumerate(labels) if l == cid]
        sampled = random.sample(idxs, min(n_samples, len(idxs)))
        samples[int(cid)] = [texts[i] for i in sampled]
    return samples


def compute_cluster_stats(labels: np.ndarray) -> dict:
    """
    Compute basic cluster statistics.

    Returns:
        Dict with n_clusters, cluster_sizes, noise_count.
    """
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    return {
        "n_clusters": len(unique),
        "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
        "noise_count": int((labels == -1).sum()),
        "total_points": len(labels),
    }
