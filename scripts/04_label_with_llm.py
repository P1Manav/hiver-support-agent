"""
scripts/04_label_with_llm.py
──────────────────────────────
Bulk-label customer messages with intent labels using a local Ollama LLM.

WHERE IT RUNS: Local (Ollama must be running on RTX 3060 Ti) OR Colab T4.
EXPECTED RUNTIME: ~45 min for 10k samples on T4, ~2 hrs locally.

NOTE: If running locally and this is too slow, upload customer_messages.csv
to Colab and run from there using the same command.

COMMAND:
  python scripts/04_label_with_llm.py [--n-samples 10000] [--batch-size 20]

OUTPUT:
  data/processed/labeled_training.csv  — columns: [thread_id, text, intent]
  data/processed/labeling_stats.json   — success rate, label distribution
"""

import sys
import logging
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from rich.console import Console

from src.intents.taxonomy import IntentTaxonomy
from src.classifier.weak_labeler import label_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Bulk-label messages with LLM weak supervision")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n-samples", type=int, help="Number of messages to label")
    parser.add_argument("--batch-size", type=int, default=20, help="Messages per LLM call")
    parser.add_argument("--model", type=str, help="Override Ollama model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["data"]["processed_dir"]
    ws_cfg = cfg["weak_supervision"]
    n_samples = args.n_samples or ws_cfg["n_samples"]
    batch_size = args.batch_size or ws_cfg["batch_size"]
    model = args.model or ws_cfg["model"]
    host = cfg["generation"]["ollama_host"]
    output_file = ws_cfg["output_file"]

    # Load taxonomy
    taxonomy = IntentTaxonomy(cfg["taxonomy"]["file"])
    taxonomy_str = taxonomy.to_prompt_string()

    # Load messages
    messages_path = Path(processed_dir) / "customer_messages.csv"
    if not messages_path.exists():
        console.print(f"[red]ERROR: {messages_path} not found. Run 02_data_pipeline.py first.[/red]")
        sys.exit(1)

    df = pd.read_csv(messages_path)

    # Sample
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=cfg["data"]["random_seed"]).reset_index(drop=True)
        console.print(f"Sampled {n_samples:,} messages for labeling.")

    texts = df["customer_text"].fillna("").tolist()

    console.print(f"\n[bold cyan]Labeling {len(texts):,} messages[/bold cyan]")
    console.print(f"  Model: {model}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Taxonomy: {len(taxonomy)} intents")
    console.print(f"  Estimated time: ~{len(texts) // batch_size * 4 // 60} min\n")

    # Label
    labels = label_dataset(
        texts=texts,
        taxonomy_str=taxonomy_str,
        model=model,
        host=host,
        batch_size=batch_size,
        temperature=ws_cfg["temperature"],
    )

    # Save results
    df["intent"] = labels
    valid_df = df.dropna(subset=["intent"])
    valid_df = valid_df[valid_df["intent"].isin(taxonomy.intent_names)]

    # Rename column for consistency
    valid_df = valid_df.rename(columns={"customer_text": "text"})

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(output_file, index=False)

    # Stats
    stats = {
        "total_attempted": len(texts),
        "total_labeled": len(valid_df),
        "success_rate": len(valid_df) / len(texts),
        "label_distribution": valid_df["intent"].value_counts().to_dict(),
    }
    stats_path = Path(processed_dir) / "labeling_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"\n[bold green]✓ Labeling complete![/bold green]")
    console.print(f"  Labeled: {len(valid_df):,} / {len(texts):,} ({stats['success_rate']:.1%} success rate)")
    console.print(f"  Saved to: {output_file}")
    console.print(f"\nLabel distribution:")
    for intent, count in sorted(stats["label_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * (count // (len(valid_df) // 50 + 1))
        console.print(f"  {intent:<30} {count:>5}  {bar}")

    console.print(f"\nNext step: Open notebooks/03_finetune_classifier.ipynb on Colab\n")


if __name__ == "__main__":
    main()
