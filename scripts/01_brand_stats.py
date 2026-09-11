"""
scripts/01_brand_stats.py
──────────────────────────
Report message count and thread completeness per brand in the dataset.

Run this FIRST to pick which brand to focus on before running 02_data_pipeline.py.
This script will download data/raw/twcs.csv automatically if it doesn't exist
(requires ~/.kaggle/kaggle.json). Use --skip-download if the file already exists.

WHERE IT RUNS: Local CPU (or Colab). No GPU/Ollama needed.
PREREQS: ~/.kaggle/kaggle.json with valid API credentials.

COMMAND:
  python scripts/01_brand_stats.py [--nrows 500000] [--top-n 30]
  python scripts/01_brand_stats.py --skip-download   # if twcs.csv already exists

EXPECTED OUTPUT:
  Brand stats table sorted by message count, with thread completeness %.
  Typical runtime: ~2 min for 500k rows on CPU.
"""

import sys
import logging
import argparse
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.data.ingest import download_dataset, load_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def compute_brand_stats(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Compute per-brand statistics.

    Returns:
        DataFrame with columns:
        [brand, total_tweets, inbound_received, outbound_sent,
         threads_started, threads_with_reply, thread_completeness_pct]
    """
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(str)

    # Brand outbound tweets
    brand_tweets = df[~df["inbound"]].copy()
    brand_counts = brand_tweets.groupby("author_id").size().reset_index(name="outbound_sent")

    # Inbound tweets directed to brand (inbound tweets that have a brand reply)
    inbound = df[df["inbound"]].copy()
    inbound_ids = set(inbound["tweet_id"].tolist())

    # For each brand, count how many inbound tweets they replied to
    brand_replied = df[~df["inbound"]].copy()
    brand_replied = brand_replied[brand_replied["in_response_to_tweet_id"].isin(inbound_ids)]
    brand_reply_counts = brand_replied.groupby("author_id")["in_response_to_tweet_id"].nunique().reset_index(
        name="threads_with_reply"
    )

    # Inbound count (rough: total @mentions is hard; use threads started as proxy)
    # Threads started = inbound tweets with no parent in dataset (root tweets) that got a reply
    tweet_id_set = set(df["tweet_id"].tolist())
    inbound["has_parent_in_data"] = inbound["in_response_to_tweet_id"].isin(tweet_id_set)
    root_inbound = inbound[~inbound["has_parent_in_data"]].copy()

    # Merge stats
    stats = brand_counts.merge(brand_reply_counts, on="author_id", how="left").fillna(0)
    stats["threads_with_reply"] = stats["threads_with_reply"].astype(int)
    stats["thread_completeness_pct"] = (
        stats["threads_with_reply"] / stats["outbound_sent"].clip(lower=1) * 100
    ).round(1)

    # Sort by outbound volume
    stats = stats.sort_values("outbound_sent", ascending=False).head(top_n)
    stats = stats.rename(columns={"author_id": "brand"})

    return stats.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Brand statistics for Customer Support on Twitter dataset")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing twcs.csv")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows loaded (for fast preview)")
    parser.add_argument("--top-n", type=int, default=30, help="Show top N brands")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (use if data/raw/twcs.csv already exists)")
    parser.add_argument("--kaggle-dataset", default="thoughtvector/customer-support-on-twitter",
                        help="Kaggle dataset slug")
    args = parser.parse_args()

    # Download if needed (this is the key fix — brand stats should be runnable before 02_data_pipeline)
    import os
    csv_path = os.path.join(args.raw_dir, "twcs.csv")
    if not os.path.exists(csv_path):
        if args.skip_download:
            console.print(f"[red]ERROR: {csv_path} not found and --skip-download was set.[/red]")
            console.print("[yellow]Run without --skip-download to fetch the dataset automatically.[/yellow]")
            sys.exit(1)
        console.print("[bold cyan]Downloading dataset from Kaggle (one-time, ~170 MB)...[/bold cyan]")
        download_dataset(args.raw_dir, args.kaggle_dataset)
    else:
        console.print(f"[dim]Found existing {csv_path} — skipping download.[/dim]")

    console.print("\n[bold cyan]Loading dataset...[/bold cyan]")
    df = load_raw(args.raw_dir, nrows=args.nrows)

    console.print(f"\n[bold]Dataset: {len(df):,} tweets, {df['author_id'].nunique():,} unique authors[/bold]\n")

    stats = compute_brand_stats(df, top_n=args.top_n)

    # Rich table
    table = Table(title=f"Top {args.top_n} Brands by Support Volume", show_lines=True)
    table.add_column("#", style="dim")
    table.add_column("Brand", style="bold cyan")
    table.add_column("Outbound Tweets", justify="right")
    table.add_column("Threads w/ Reply", justify="right")
    table.add_column("Thread Completeness", justify="right")

    for i, row in stats.iterrows():
        completeness = row["thread_completeness_pct"]
        color = "green" if completeness >= 60 else ("yellow" if completeness >= 30 else "red")
        table.add_row(
            str(i + 1),
            row["brand"],
            f"{row['outbound_sent']:,}",
            f"{row['threads_with_reply']:,}",
            f"[{color}]{completeness:.1f}%[/{color}]",
        )

    console.print(table)

    # Recommendation
    best = stats.iloc[0]
    console.print(f"\n[bold green][OK] Recommended brand: {best['brand']}[/bold green]")
    console.print(f"  {best['outbound_sent']:,} outbound tweets, "
                  f"{best['thread_completeness_pct']}% thread completeness")
    console.print("\n[dim]Set your choice in config/config.yaml -> data.brand[/dim]\n")

    # Save to CSV
    out_path = "data/processed/brand_stats.csv"
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_path, index=False)
    console.print(f"[dim]Stats saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()
