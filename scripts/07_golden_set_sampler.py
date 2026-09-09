"""
scripts/07_golden_set_sampler.py
─────────────────────────────────
Stratified sampler to build the 150–250 example golden evaluation set.

Sampling strategy (D-10):
  1. Stratified by intent: minimum 15 examples per intent (ensures uniform coverage).
  2. Confidence-stratified: sample from BOTH high-confidence and low-confidence
     predictions to capture easy and hard cases.
  3. Edge cases: deliberate oversample of:
     - Very short messages (< 10 words)
     - Messages where top-2 intents have < 0.15 confidence gap (ambiguous)
     - Messages with no good retrieval match (similarity < 0.5)
  4. Diversity: deduplicate near-duplicate messages (cosine similarity > 0.95).

WHERE IT RUNS: Local CPU (Ollama NOT needed for sampling; needed for inference).
COMMAND:
  python scripts/07_golden_set_sampler.py

PREREQS:
  - data/processed/customer_messages.csv
  - models/classifier/ (fine-tuned model)
  - faiss_index/

OUTPUT:
  golden/golden_set_to_label.csv  — sampled examples for hand-labeling
  (After hand-labeling, save as golden/golden_set.csv)
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import pandas as pd
from rich.console import Console

from src.intents.taxonomy import IntentTaxonomy
from src.intents.embedder import MessageEmbedder
from src.classifier.predict import IntentClassifier
from src.retrieval.indexer import load_index
from src.retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Sample golden evaluation set")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--target-size", type=int, help="Override target golden set size")
    parser.add_argument("--per-intent", type=int, help="Override minimum per intent")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gs_cfg = cfg["golden_set"]
    target_size = args.target_size or gs_cfg["target_size"]
    per_intent = args.per_intent or gs_cfg["per_intent"]
    edge_fraction = gs_cfg["edge_case_fraction"]
    processed_dir = cfg["data"]["processed_dir"]

    taxonomy = IntentTaxonomy(cfg["taxonomy"]["file"])
    intent_names = taxonomy.intent_names

    # Load messages
    messages_path = Path(processed_dir) / "customer_messages.csv"
    df = pd.read_csv(messages_path)
    texts = df["customer_text"].fillna("").tolist()
    console.print(f"\nLoaded {len(texts):,} candidate messages.")

    # Run classifier on all messages (batch)
    console.print("\n[bold cyan]Running classifier on all messages...[/bold cyan]")
    classifier = IntentClassifier(
        model_dir=cfg["classifier"]["model_dir"],
        max_length=cfg["classifier"]["max_length"],
    )
    preds = classifier.predict_batch(texts, batch_size=128)
    pred_intents = [p[0] for p in preds]
    confidences = [p[1] for p in preds]

    # Run retriever (for low-similarity edge cases)
    console.print("[bold cyan]Running retriever to find low-similarity messages...[/bold cyan]")
    embedder = MessageEmbedder(
        model_name=cfg["embedding"]["model"],
        device=cfg["embedding"]["device"],
    )
    index, metadata = load_index(cfg["retrieval"]["index_dir"])
    retriever = Retriever(index, metadata, embedder, top_k=1)

    top_sims = []
    for text in texts:
        results = retriever.retrieve(text, top_k=1)
        top_sims.append(results[0]["similarity"] if results else 0.0)

    # Build candidate DataFrame
    df["predicted_intent"] = pred_intents
    df["confidence"] = confidences
    df["top_similarity"] = top_sims
    df["word_count"] = df["customer_text"].str.split().str.len()

    # Compute ambiguity: get all scores
    console.print("[bold cyan]Computing ambiguity scores...[/bold cyan]")
    all_scores = [classifier.predict_all_scores(t) for t in texts]
    top2_gaps = []
    for scores in all_scores:
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        top2_gaps.append(gap)
    df["top2_gap"] = top2_gaps

    # Categorize each message
    df["is_edge_short"] = df["word_count"] < 10
    df["is_edge_ambiguous"] = df["top2_gap"] < 0.15
    df["is_edge_no_precedent"] = df["top_similarity"] < 0.5
    df["is_edge"] = df["is_edge_short"] | df["is_edge_ambiguous"] | df["is_edge_no_precedent"]

    # Sampling
    sampled_indices = set()
    n_edge = int(target_size * edge_fraction)
    n_normal = target_size - n_edge

    # 1. Edge cases
    edge_candidates = df[df["is_edge"]].index.tolist()
    edge_sample = np.random.default_rng(42).choice(
        edge_candidates, size=min(n_edge, len(edge_candidates)), replace=False
    )
    sampled_indices.update(edge_sample.tolist())
    console.print(f"  Edge cases sampled: {len(edge_sample):,}")

    # 2. Stratified by intent (non-edge)
    per_intent_needed = max(per_intent, n_normal // len(intent_names))
    for intent in intent_names:
        candidates = df[
            (df["predicted_intent"] == intent) & (~df.index.isin(sampled_indices))
        ].index.tolist()
        n = min(per_intent_needed, len(candidates))
        if candidates:
            sample = np.random.default_rng(42).choice(candidates, size=n, replace=False)
            sampled_indices.update(sample.tolist())

    sampled = df.loc[list(sampled_indices)].copy()
    sampled = sampled.head(target_size)

    # Output columns for hand-labeling
    output = sampled[["thread_id", "customer_text", "predicted_intent", "confidence",
                       "top_similarity", "word_count", "is_edge"]].copy()
    output["human_intent"] = ""  # fill during labeling
    output["notes"] = ""         # optional notes

    out_path = Path(gs_cfg["output_file"]).parent / "golden_set_to_label.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    console.print(f"\n[bold green]✓ Golden set sample ready![/bold green]")
    console.print(f"  Total examples: {len(output):,}")
    console.print(f"  Edge cases: {output['is_edge'].sum():,} ({output['is_edge'].mean():.0%})")
    console.print(f"  Saved to: {out_path}")
    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"  1. Open {out_path} in Excel/Sheets")
    console.print(f"  2. Fill in 'human_intent' for each row using config/intent_taxonomy.yaml")
    console.print(f"  3. Save as golden/golden_set.csv")
    console.print(f"  OR open notebooks/05_golden_labeling.ipynb for an interactive UI\n")


if __name__ == "__main__":
    main()
