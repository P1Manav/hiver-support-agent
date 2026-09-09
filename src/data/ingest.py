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
    df = pd.read_csv(
        csv_path,
        nrows=nrows,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        parse_dates=["created_at"],
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
    return df[mask | ~mask]  # keep all; thread reconstruction filters later
