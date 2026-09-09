"""
scripts/05_build_faiss_index.py
────────────────────────────────
Build the FAISS retrieval index over historical resolved (customer, reply) pairs.

WHERE IT RUNS: Local CPU. ~3 min for 50k pairs.
PREREQS: data/processed/resolved_pairs.csv must exist (run 02_data_pipeline.py).

COMMAND:
  python scripts/05_build_faiss_index.py

OUTPUT:
  faiss_index/index.faiss     — FAISS flat inner-product index
  faiss_index/metadata.parquet — per-vector metadata (thread_id, customer_text, brand_reply)
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from rich.console import Console

from src.intents.embedder import MessageEmbedder
from src.retrieval.indexer import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["data"]["processed_dir"]
    index_dir = cfg["retrieval"]["index_dir"]
    emb_cfg = cfg["embedding"]

    # Load resolved pairs
    resolved_path = Path(processed_dir) / "resolved_pairs.csv"
    if not resolved_path.exists():
        console.print(f"[red]ERROR: {resolved_path} not found. Run 02_data_pipeline.py first.[/red]")
        sys.exit(1)

    resolved_df = pd.read_csv(resolved_path)
    console.print(f"\nLoaded {len(resolved_df):,} resolved pairs.")

    # Embedder
    embedder = MessageEmbedder(
        model_name=emb_cfg["model"],
        device=emb_cfg["device"],
        batch_size=emb_cfg["batch_size"],
    )

    console.print(f"\n[bold cyan]Building FAISS index...[/bold cyan]")
    console.print(f"  Embedding model: {emb_cfg['model']}")
    console.print(f"  Index dir: {index_dir}")

    index, metadata = build_index(
        threads_df=resolved_df,
        embedder=embedder,
        index_dir=index_dir,
    )

    console.print(f"\n[bold green]✓ FAISS index built![/bold green]")
    console.print(f"  Vectors: {index.ntotal:,}")
    console.print(f"  Dimension: {index.d}")
    console.print(f"  Saved to: {index_dir}/")
    console.print(f"\nNext step: python scripts/06_inference.py --message 'my order is late'\n")


if __name__ == "__main__":
    main()
