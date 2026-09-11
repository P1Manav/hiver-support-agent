"""
scripts/03_embed_and_cluster.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Embed customer messages and cluster to discover intent groups.

WHERE IT RUNS: Local CPU (preferred) or Colab. No Ollama needed.
EXPECTED RUNTIME: ~8 min for 50k messages on CPU (embedding ~5 min, clustering ~3 min).
For faster iteration, use --n-sample to cluster a subsample first.

COMMAND:
  python scripts/03_embed_and_cluster.py [--n-sample 5000] [--method kmeans] [--n-clusters 12]

OUTPUT:
  data/processed/embeddings.npy     â€” (N, 384) float32 embedding matrix
  data/processed/cluster_labels.npy â€” cluster label per message
  data/processed/cluster_samples.json â€” 10 sample messages per cluster (for manual inspection)
  data/processed/cluster_stats.json  â€” cluster size statistics

NEXT STEP:
  Inspect data/processed/cluster_samples.json directly to name the clusters,
  then edit config/intent_taxonomy.yaml with your intent names.
  (Run scripts/03_embed_and_cluster.py to generate cluster_samples.json first.)
"""

import sys
import logging
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.intents.embedder import MessageEmbedder
from src.intents.clusterer import cluster_kmeans, cluster_hdbscan, get_cluster_samples, compute_cluster_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Embed + cluster customer messages")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n-sample", type=int, help="Override n_sample_for_clustering")
    parser.add_argument("--method", choices=["kmeans", "hdbscan"], help="Override clustering method")
    parser.add_argument("--n-clusters", type=int, help="Override n_clusters (kmeans)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding if embeddings.npy exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["data"]["processed_dir"]
    emb_cfg = cfg["embedding"]
    clust_cfg = cfg["clustering"]

    n_sample = args.n_sample or clust_cfg["n_sample_for_clustering"]
    method = args.method or clust_cfg["method"]
    n_clusters = args.n_clusters or clust_cfg["n_clusters"]

    # Load customer messages
    messages_path = Path(processed_dir) / "customer_messages.csv"
    if not messages_path.exists():
        console.print(f"[red]ERROR: {messages_path} not found. Run 02_data_pipeline.py first.[/red]")
        sys.exit(1)

    df = pd.read_csv(messages_path)
    texts = df["customer_text"].fillna("").tolist()
    console.print(f"\nLoaded {len(texts):,} customer messages.")

    # Subsample for clustering
    if n_sample and len(texts) > n_sample:
        import random
        random.seed(cfg["data"]["random_seed"])
        sample_idx = random.sample(range(len(texts)), n_sample)
        sample_texts = [texts[i] for i in sample_idx]
        console.print(f"Subsampled {n_sample:,} messages for clustering.")
    else:
        sample_texts = texts
        sample_idx = list(range(len(texts)))

    # Embed
    embed_path = Path(processed_dir) / "embeddings.npy"
    if args.skip_embed and embed_path.exists():
        console.print(f"Loading existing embeddings from {embed_path}...")
        embeddings = np.load(embed_path)
    else:
        console.print(f"\n[bold cyan]Embedding {len(sample_texts):,} messages...[/bold cyan]")
        console.print(f"Model: {emb_cfg['model']} | Device: {emb_cfg['device']}")
        embedder = MessageEmbedder(
            model_name=emb_cfg["model"],
            device=emb_cfg["device"],
            batch_size=emb_cfg["batch_size"],
        )
        embeddings = embedder.encode(sample_texts)
        np.save(embed_path, embeddings)
        console.print(f"Embeddings saved: {embed_path} â€” shape {embeddings.shape}")

    # Cluster
    console.print(f"\n[bold cyan]Clustering ({method}, k={n_clusters if method == 'kmeans' else 'auto'})...[/bold cyan]")

    if method == "hdbscan":
        labels, model = cluster_hdbscan(
            embeddings,
            min_cluster_size=clust_cfg["hdbscan_min_cluster_size"],
        )
    else:
        labels, model = cluster_kmeans(
            embeddings,
            n_clusters=n_clusters,
            random_seed=cfg["data"]["random_seed"],
        )

    np.save(Path(processed_dir) / "cluster_labels.npy", labels)

    # Stats
    stats = compute_cluster_stats(labels)
    with open(Path(processed_dir) / "cluster_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Sample inspection
    samples = get_cluster_samples(sample_texts, labels, n_samples=10)
    with open(Path(processed_dir) / "cluster_samples.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Print cluster table
    table = Table(title=f"Clusters ({method.upper()}, n={stats['n_clusters']})", show_lines=True)
    table.add_column("Cluster ID", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Sample message (first of 10)")

    for cid in sorted(stats["cluster_sizes"].keys(), key=int):
        size = stats["cluster_sizes"][cid]
        # samples dict has int keys; stats may have int or str keys after JSON parse
        sample = samples.get(int(cid), samples.get(str(cid), ["(no samples)"]))[0][:80] + "..."
        table.add_row(str(cid), str(size), sample)

    if stats["noise_count"] > 0:
        table.add_row("[dim]noise[/dim]", str(stats["noise_count"]), "[dim]â€”[/dim]")

    console.print(table)
    console.print(f"\n[bold green]âœ“ Clustering complete![/bold green]")
    console.print(f"  Inspect data/processed/cluster_samples.json to see 10 samples per cluster.")
    console.print(f"  Then edit config/intent_taxonomy.yaml with your intent names.\n")


if __name__ == "__main__":
    main()

