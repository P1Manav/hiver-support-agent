"""
scripts/08_judge_agreement.py
──────────────────────────────
Compute judge–human agreement (Cohen's kappa) and run full evaluation.

This script:
  1. Runs the LLM judge on the golden set.
  2. If human_judge_score columns are present, computes Cohen's kappa
     between judge scores and human scores.
  3. Runs all automated metrics (classification F1, retrieval, groundedness).
  4. Prints a full evaluation report.

WHERE IT RUNS: Local (Ollama must be running for LLM judge).
COMMAND:
  python scripts/08_judge_agreement.py --golden golden/golden_set.csv

PREREQS:
  - golden/golden_set.csv with human_intent labels
  - models/classifier/, faiss_index/, Ollama running

OUTPUT:
  reports/evaluation_report.json  — full metrics
  reports/calibration.png         — confidence calibration curve
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
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score
from rich.console import Console
from rich.table import Table

from src.intents.taxonomy import IntentTaxonomy
from src.intents.embedder import MessageEmbedder
from src.classifier.predict import IntentClassifier
from src.retrieval.indexer import load_index
from src.retrieval.retriever import Retriever
from src.generation.generator import ReplyGenerator
from src.generation.guardrails import GuardrailChecker
from src.escalation.gate import EscalationGate
from src.evaluation.metrics import classification_metrics, retrieval_metrics, groundedness_score
from src.evaluation.judge import LLMJudge
from src.evaluation.calibration import calibration_curve, plot_calibration, find_optimal_threshold
from src.baselines.trivial import TrivialBaseline
from src.baselines.simple import SimpleBaseline
from src.data.pii import redact

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Full evaluation + judge–human agreement")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--golden", default="golden/golden_set.csv")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge (faster)")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline eval")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path("reports").mkdir(exist_ok=True)

    # Load golden set
    golden_df = pd.read_csv(args.golden)
    assert "human_intent" in golden_df.columns, "Golden set must have 'human_intent' column."
    golden_df = golden_df.dropna(subset=["human_intent"])
    console.print(f"\nLoaded {len(golden_df):,} labeled golden examples.")

    taxonomy = IntentTaxonomy(cfg["taxonomy"]["file"])
    intent_names = taxonomy.intent_names

    # Load pipeline
    embedder = MessageEmbedder(
        model_name=cfg["embedding"]["model"],
        device=cfg["embedding"]["device"],
    )
    classifier = IntentClassifier(model_dir=cfg["classifier"]["model_dir"])
    index, metadata = load_index(cfg["retrieval"]["index_dir"])
    retriever = Retriever(index, metadata, embedder, top_k=cfg["retrieval"]["top_k"])
    generator = ReplyGenerator(
        model=cfg["generation"]["model"],
        host=cfg["generation"]["ollama_host"],
        brand=cfg["data"]["brand"],
    )
    guardrails = GuardrailChecker()
    gate = EscalationGate.from_config(cfg)

    # Run our pipeline on golden set
    console.print("\n[bold cyan]Running pipeline on golden set...[/bold cyan]")
    predictions = []
    for _, row in golden_df.iterrows():
        text = str(row["customer_text"])
        text_clean = redact(text)
        intent, confidence = classifier.predict(text_clean)
        retrieved = retriever.retrieve(text_clean)

        urgency_data = generator.score_urgency(text_clean)
        urgency_score = urgency_data.get("urgency_score", 0.0)
        sentiment = urgency_data.get("sentiment", "neutral")

        escalation = gate.decide(text_clean, intent, confidence, retrieved, urgency_score, sentiment)

        if escalation.decision == "auto_handle":
            gen = generator.generate(
                customer_message=text_clean,
                intent=intent,
                intent_description=taxonomy.get_description(intent),
                retrieved_examples=retrieved,
            )
            draft = gen["draft_reply"]
        else:
            draft = "[ESCALATED]"

        predictions.append({
            "intent": intent,
            "confidence": confidence,
            "decision": escalation.decision,
            "draft_reply": draft,
            "retrieved_examples": retrieved,
            "groundedness": groundedness_score(draft, retrieved),
        })

    # Classification metrics
    y_true = golden_df["human_intent"].tolist()
    y_pred = [p["intent"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]

    console.print("\n[bold cyan]Computing metrics...[/bold cyan]")
    clf = classification_metrics(y_true, y_pred, label_names=intent_names)

    # Calibration
    cal_data = calibration_curve(y_true, y_pred, confidences)
    cal_plot = plot_calibration(cal_data, output_path="reports/calibration.png")
    optimal_thresh = find_optimal_threshold(y_true, y_pred, confidences, target_precision=0.90)

    # Retrieval metrics
    retrieved_all = [p["retrieved_examples"] for p in predictions]
    ret = retrieval_metrics(retrieved_all)

    # Groundedness
    ground_scores = [p["groundedness"] for p in predictions]

    # LLM Judge
    judge_results = []
    if not args.skip_judge:
        console.print("\n[bold cyan]Running LLM judge...[/bold cyan]")
        judge = LLMJudge(model=cfg["evaluation"]["judge_model"], host=cfg["generation"]["ollama_host"])
        judge_examples = [
            {
                "customer_message": str(row["customer_text"]),
                "draft_reply": pred["draft_reply"],
                "retrieved_examples": pred["retrieved_examples"],
                "intent": pred["intent"],
            }
            for row, pred in zip(golden_df.itertuples(), predictions)
        ]
        judge_results = judge.score_batch(judge_examples)

        # Judge–human agreement (if human_judge_score present)
        if "human_judge_score" in golden_df.columns:
            human_scores = golden_df["human_judge_score"].dropna().astype(int).tolist()
            judge_avgs = [round(r["average"]) for r in judge_results[:len(human_scores)]]
            kappa = cohen_kappa_score(human_scores, judge_avgs)
            console.print(f"\n[bold]Judge–Human Cohen's κ: {kappa:.3f}[/bold]")

    # ── Print Results Table ─────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Our System", justify="right", style="green")
    table.add_column("Trivial", justify="right", style="yellow")
    table.add_column("Simple", justify="right", style="cyan")

    # Run baselines if needed
    trivial_f1, simple_f1 = "—", "—"
    if not args.skip_baselines:
        tb = TrivialBaseline()
        tb.fit(golden_df["customer_text"].tolist(), y_true)
        tm = tb.evaluate(golden_df)
        trivial_f1 = f"{tm['macro_f1']:.4f}"

        sb = SimpleBaseline()
        labeled_df = pd.read_csv(cfg["weak_supervision"]["output_file"])
        sb.fit(labeled_df["text"].tolist(), labeled_df["intent"].tolist())
        sm = sb.evaluate(golden_df)
        simple_f1 = f"{sm['macro_f1']:.4f}"

    table.add_row("Intent Accuracy", f"{clf['accuracy']:.4f}", "—", "—")
    table.add_row("Intent Macro-F1", f"{clf['macro_f1']:.4f}", trivial_f1, simple_f1)
    table.add_row("Mean Top-1 Similarity", f"{ret['mean_top1_similarity']:.4f}", "—", "—")
    table.add_row("Mean Groundedness (ROUGE-L)", f"{np.mean(ground_scores):.4f}", "—", "—")

    if judge_results:
        avg_scores = {d: np.mean([r[d] for r in judge_results]) for d in ["correctness", "tone", "groundedness", "conciseness"]}
        for dim, score in avg_scores.items():
            table.add_row(f"Judge: {dim}", f"{score:.2f}/5", "—", "—")

    table.add_row("ECE (calibration)", f"{cal_data['ece']:.4f}", "—", "—")
    table.add_row("Optimal threshold", f"{optimal_thresh:.3f}", "—", "—")

    console.print(table)

    # Save full report
    report = {
        "classification": {k: v for k, v in clf.items() if k != "confusion_matrix"},
        "retrieval": ret,
        "groundedness": {"mean_rouge_l": float(np.mean(ground_scores))},
        "calibration": cal_data,
        "optimal_threshold": optimal_thresh,
        "judge_summary": {
            d: float(np.mean([r[d] for r in judge_results]))
            for d in ["correctness", "tone", "groundedness", "conciseness", "average"]
        } if judge_results else {},
    }
    with open("reports/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")
    console.print(f"  Full report: reports/evaluation_report.json")
    console.print(f"  Calibration plot: {cal_plot}\n")


if __name__ == "__main__":
    main()
