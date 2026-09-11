"""
src/data/threads.py
───────────────────
Reconstruct multi-turn conversation threads from the flat tweet-reply
structure in the Customer Support on Twitter dataset.

The dataset stores tweets as a flat table with:
  - tweet_id: unique ID of this tweet
  - in_response_to_tweet_id: ID of the tweet this is replying to (NaN if root)
  - response_tweet_id: IDs of tweets that replied to this one (comma-separated)
  - inbound: True if this is a customer tweet, False if brand tweet
  - author_id: Twitter handle

Thread reconstruction: follow the reply chain from each inbound root tweet
(no parent, or parent not in dataset) down through the conversation.

WHERE IT RUNS: Local CPU or Colab. O(n) in dataset size; ~2 min for 50k rows.
"""

import logging
from collections import defaultdict
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_reply_graph(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Build a parent → [children] adjacency map from tweet_id / in_response_to_tweet_id.

    Returns:
        Dict mapping tweet_id → list of reply tweet_ids.
    """
    children: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        parent_id = str(row["in_response_to_tweet_id"]) if pd.notna(row["in_response_to_tweet_id"]) else None
        if parent_id and parent_id != "nan":
            children[parent_id].append(str(row["tweet_id"]))
    return dict(children)


def find_thread_roots(df: pd.DataFrame, brand: str) -> list[str]:
    """
    Find root tweets: inbound (customer) tweets that have no parent in the
    dataset AND are directed at the brand (i.e., have a brand reply in their
    response chain).

    Args:
        df: Raw DataFrame.
        brand: Brand handle (no @).

    Returns:
        List of tweet_ids that are thread roots.
    """
    tweet_ids = set(df["tweet_id"].astype(str))
    brand_lower = brand.lower()
    brand_tweet_ids = set(df[df["author_id"].str.lower() == brand_lower]["tweet_id"].astype(str))

    children_map = build_reply_graph(df)

    roots = []
    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        parent = str(row["in_response_to_tweet_id"]) if pd.notna(row["in_response_to_tweet_id"]) else None
        is_inbound = bool(row.get("inbound", True))

        # Root = inbound tweet whose parent is not in our dataset
        if is_inbound and (parent is None or parent == "nan" or parent not in tweet_ids):
            # Check that a brand reply exists in the subtree
            if _has_brand_reply(tid, children_map, brand_tweet_ids, depth=0):
                roots.append(tid)

    logger.info(f"Found {len(roots):,} thread roots for brand '{brand}'.")
    return roots


def _has_brand_reply(
    tweet_id: str,
    children_map: dict,
    brand_tweet_ids: set,
    depth: int,
    max_depth: int = 10,
) -> bool:
    """Recursively check if any descendant tweet is from the brand."""
    if depth > max_depth:
        return False
    for child_id in children_map.get(tweet_id, []):
        if child_id in brand_tweet_ids:
            return True
        if _has_brand_reply(child_id, children_map, brand_tweet_ids, depth + 1):
            return True
    return False


def extract_thread(
    root_id: str,
    tweet_index: dict[str, dict],
    children_map: dict[str, list[str]],
    max_turns: int = 20,
) -> list[dict]:
    """
    DFS from root_id to collect the linear thread (greedy: always follow
    the first reply at each step).

    Returns:
        List of tweet dicts in chronological order:
        [{tweet_id, author_id, text, inbound, created_at}, ...]
    """
    thread = []
    current_id = root_id
    seen = set()

    while current_id and len(thread) < max_turns:
        if current_id in seen or current_id not in tweet_index:
            break
        seen.add(current_id)
        thread.append(tweet_index[current_id])
        replies = children_map.get(current_id, [])
        current_id = replies[0] if replies else None

    return thread


def reconstruct_threads(
    df: pd.DataFrame,
    brand: str,
    min_length: int = 2,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> list[dict]:
    """
    Full thread reconstruction pipeline.

    Args:
        df: Raw DataFrame.
        brand: Brand handle (no @).
        min_length: Minimum number of turns in a thread.
        sample_size: If set, randomly sample this many root threads.
        random_seed: For reproducibility.

    Returns:
        List of thread dicts:
        {
          "thread_id": str,
          "brand": str,
          "turns": [{"tweet_id", "author_id", "text", "inbound", "created_at"}],
          "customer_text": str,  # first customer message
          "brand_reply": str,    # first brand reply
          "n_turns": int,
        }
    """
    logger.info(f"Building tweet index for {len(df):,} tweets...")
    tweet_index = {
        str(row["tweet_id"]): {
            "tweet_id": str(row["tweet_id"]),
            "author_id": str(row["author_id"]),
            "text": str(row["text"]),
            "inbound": bool(row.get("inbound", True)),
            "created_at": row.get("created_at"),
        }
        for _, row in df.iterrows()
    }

    children_map = build_reply_graph(df)
    roots = find_thread_roots(df, brand)

    if sample_size and len(roots) > sample_size:
        import random
        random.seed(random_seed)
        roots = random.sample(roots, sample_size)
        logger.info(f"Sampled {sample_size:,} threads from {len(roots):,} roots.")

    threads = []
    for root_id in roots:
        turns = extract_thread(root_id, tweet_index, children_map)
        if len(turns) < min_length:
            continue

        # Extract first customer text and first brand reply
        customer_text = next(
            (t["text"] for t in turns if t["inbound"]), turns[0]["text"]
        )
        brand_reply = next(
            (t["text"] for t in turns if not t["inbound"]), None
        )

        threads.append({
            "thread_id": root_id,
            "brand": brand,
            "turns": turns,
            "customer_text": customer_text,
            "brand_reply": brand_reply,
            "n_turns": len(turns),
        })

    logger.info(f"Reconstructed {len(threads):,} valid threads (min_length={min_length}).")
    return threads


def threads_to_dataframe(threads: list[dict]) -> pd.DataFrame:
    """
    Flatten thread list into a DataFrame for easy CSV export.

    Columns: thread_id, brand, customer_text, brand_reply, n_turns, turns_json
    """
    import json
    rows = []
    for t in threads:
        rows.append({
            "thread_id": t["thread_id"],
            "brand": t["brand"],
            "customer_text": t["customer_text"],
            "brand_reply": t["brand_reply"],
            "n_turns": t["n_turns"],
            "turns_json": json.dumps(t["turns"], default=str),
        })
    return pd.DataFrame(rows)
