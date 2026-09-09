"""
scripts/02_data_pipeline.py
────────────────────────────
Full data pipeline: download → filter → thread reconstruction → PII redaction → subsample → save.

WHERE IT RUNS: Local CPU or Colab. No GPU/Ollama needed.
EXPECTED RUNTIME: ~5 min for 50k sample on a modern CPU.

COMMAND:
  python scripts/02_data_pipeline.py [--brand AmazonHelp] [--sample-size 50000]

OUTPUT:
  data/processed/threads.parquet      — all reconstructed threads
  data/processed/customer_messages.csv — PII-redacted customer messages for clustering/labeling
  data/processed/resolved_pairs.csv    — (customer_text, brand_reply) pairs for FAISS indexing
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from rich.console import Console
from rich.progress import track

from src.data.ingest import download_dataset, load_raw
from src.data.threads import reconstruct_threads, threads_to_dataframe
from src.data.pii import redact_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run the full data pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--brand", type=str, help="Override brand from config")
    parser.add_argument("--sample-size", type=int, help="Override sample_size from config")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download if data already exists")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    brand = args.brand or cfg["data"]["brand"]
    sample_size = args.sample_size or cfg["data"]["sample_size"]
    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    random_seed = cfg["data"]["random_seed"]

    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Data Pipeline[/bold cyan] — brand=[bold]{brand}[/bold], sample_size={sample_size:,}\n")

    # Step 1: Download
    if not args.skip_download:
        console.print("[1/4] Downloading dataset from Kaggle...")
        download_dataset(raw_dir, cfg["data"]["kaggle_dataset"])
    else:
        console.print("[1/4] Skipping download (--skip-download).")

    # Step 2: Load raw data
    console.print(f"[2/4] Loading raw data (nrows=None for full dataset)...")
    df = load_raw(raw_dir)
    console.print(f"      Loaded {len(df):,} tweets.")

    # Step 3: Reconstruct threads
    console.print(f"[3/4] Reconstructing threads for brand '{brand}'...")
    threads = reconstruct_threads(
        df,
        brand=brand,
        min_length=cfg["data"]["min_thread_length"],
        sample_size=sample_size,
        random_seed=random_seed,
    )
    console.print(f"      Found {len(threads):,} threads.")

    threads_df = threads_to_dataframe(threads)

    # Step 4: PII redaction
    console.print("[4/4] Redacting PII from text columns...")
    redact_dataframe(threads_df, text_columns=["customer_text", "brand_reply"])

    # Save full threads
    threads_path = Path(processed_dir) / "threads.parquet"
    threads_df.to_parquet(threads_path, index=False)
    console.print(f"      Saved {len(threads_df):,} threads to {threads_path}")

    # Save customer messages (for clustering/labeling)
    messages_df = threads_df[["thread_id", "customer_text"]].dropna()
    messages_path = Path(processed_dir) / "customer_messages.csv"
    messages_df.to_csv(messages_path, index=False)
    console.print(f"      Saved {len(messages_df):,} customer messages to {messages_path}")

    # Save resolved pairs (for FAISS indexing)
    resolved_df = threads_df[["thread_id", "customer_text", "brand_reply", "n_turns"]].dropna(
        subset=["brand_reply"]
    )
    resolved_df = resolved_df[resolved_df["brand_reply"].str.strip() != ""]
    resolved_path = Path(processed_dir) / "resolved_pairs.csv"
    resolved_df.to_csv(resolved_path, index=False)
    console.print(f"      Saved {len(resolved_df):,} resolved pairs to {resolved_path}")

    # Summary
    console.print(f"\n[bold green]✓ Data pipeline complete![/bold green]")
    console.print(f"  Threads:         {len(threads_df):,}")
    console.print(f"  Resolved pairs:  {len(resolved_df):,}")
    console.print(f"  Customer msgs:   {len(messages_df):,}")
    console.print(f"\nNext step: python scripts/03_embed_and_cluster.py\n")


if __name__ == "__main__":
    main()
