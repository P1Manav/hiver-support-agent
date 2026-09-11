"""
src/data/ingest.py
──────────────────
Download and load the Customer Support on Twitter dataset from Kaggle.

WHERE IT RUNS: Local (CPU) or Colab/Kaggle — no GPU needed.
PREREQS: kaggle.json at ~/.kaggle/kaggle.json with API credentials.
"""

import os
import zipfile
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def download_dataset(raw_dir: str, dataset: str = "thoughtvector/customer-support-on-twitter") -> Path:
    """
    Download the Kaggle dataset to raw_dir if not already present.

    Args:
        raw_dir: Local directory to store the raw CSV.
        dataset: Kaggle dataset slug (owner/name).

    Returns:
        Path to the extracted CSV file.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    csv_path = raw_path / "twcs.csv"
    if csv_path.exists():
        logger.info(f"Dataset already exists at {csv_path}. Skipping download.")
        return csv_path

    logger.info(f"Downloading dataset '{dataset}' from Kaggle...")
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError("kaggle package not installed. Run: pip install kaggle")

    os.system(f'kaggle datasets download -d "{dataset}" -p "{raw_dir}"')

    # Unzip
    zip_path = raw_path / "customer-support-on-twitter.zip"
    if zip_path.exists():
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_path)
        zip_path.unlink()

    # Find the CSV — Kaggle may extract into a subfolder (e.g. twcs/twcs.csv)
    if not csv_path.exists():
        found = list(raw_path.rglob("twcs.csv"))
        if found:
            # Move to the flat expected location
            import shutil
            shutil.move(str(found[0]), str(csv_path))
            # Clean up empty subfolder if left behind
            try:
                found[0].parent.rmdir()
            except OSError:
                pass
            logger.info(f"Moved extracted CSV to {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected CSV not found at {csv_path} after download. "
            "Check your Kaggle API credentials (~/.kaggle/kaggle.json)."
        )

    logger.info(f"Dataset ready at {csv_path}")
    return csv_path


def load_raw(raw_dir: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load the raw twcs.csv into a DataFrame.

    Args:
        raw_dir: Directory containing twcs.csv.
        nrows: If set, only load this many rows (for fast iteration).

    Returns:
        DataFrame with columns:
            tweet_id, author_id, inbound, created_at, text,
            response_tweet_id, in_response_to_tweet_id
    """
    csv_path = Path(raw_dir) / "twcs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. Run download_dataset() first."
        )

    logger.info(f"Loading raw data from {csv_path} ({'all rows' if nrows is None else f'{nrows:,} rows'})...")
    # Note: created_at is kept as str here for speed — 2.8M rows with dateutil parsing takes ~3 min.
    # Downstream code that needs actual datetimes can call pd.to_datetime(df['created_at']) locally.
    df = pd.read_csv(
        csv_path,
        nrows=nrows,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
            "created_at": str,  # keep as string — avoids slow per-row dateutil parse
        },
    )

    logger.info(f"Loaded {len(df):,} tweets, {df['author_id'].nunique():,} unique authors.")
    return df


def get_brand_tweets(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Filter to rows where the author is `brand` (outbound) or the
    brand is mentioned in a reply chain (inbound directed at brand).

    Args:
        df: Full raw DataFrame.
        brand: Twitter handle without @.

    Returns:
        Subset DataFrame.
    """
    brand_lower = brand.lower()
    mask = df["author_id"].str.lower() == brand_lower
    return df[mask]  # outbound tweets from this brand only
