#!/usr/bin/env python
"""
scripts/score_cli.py
───────────────────────
CLI tool for human judge scoring and escalation ground-truth labeling.
"""

import sys
import argparse
import csv
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from rich.console import Console

from src.classifier.predict import IntentClassifier
from src.retrieval.indexer import load_index
from src.retrieval.retriever import Retriever
from src.intents.embedder import MessageEmbedder
from src.intents.taxonomy import IntentTaxonomy
from src.generation.generator import ReplyGenerator
from src.escalation.gate import EscalationGate
from src.evaluation.judge import LLMJudge
from src.data.pii import redact

console = Console()

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default="golden/golden_set.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(args.input)
    if "human_judge_score" not in df.columns:
        df["human_judge_score"] = ""
    if "human_escalate" not in df.columns:
        df["human_escalate"] = ""

    # Load pipeline
    console.print("[cyan]Loading models (this takes a few seconds)...[/cyan]")
    embedder = MessageEmbedder(model_name=cfg["embedding"]["model"], device="cpu")
    classifier = IntentClassifier(model_dir=cfg["classifier"]["model_dir"])
    index, metadata = load_index(cfg["retrieval"]["index_dir"])
    retriever = Retriever(index, metadata, embedder, top_k=cfg["retrieval"]["top_k"])
    generator = ReplyGenerator(model=cfg["generation"]["model"], host=cfg["generation"]["ollama_host"], brand=cfg["data"]["brand"])
    judge = LLMJudge(model=cfg["evaluation"]["judge_model"], host=cfg["generation"]["ollama_host"])
    gate = EscalationGate.from_config(cfg)
    taxonomy = IntentTaxonomy(cfg["taxonomy"]["file"])

    unscored = df[(df["human_judge_score"].isna()) | (df["human_judge_score"] == "")].index.tolist()
    
    console.print(f"\n[bold green]Ready to score! {len(unscored)} items remaining.[/bold green]")
    console.print("You do NOT need to score all 200. A sample of 50 is statistically sufficient for Cohen's Kappa.")
    console.print("Type 'q' anytime to quit and save.")
    input("Press Enter to begin...")

    for i, idx in enumerate(unscored):
        clear()
        row = df.loc[idx]
        text = str(row["customer_text"])
        
        console.print(f"[bold cyan]Example {i+1}/{len(unscored)}[/bold cyan]")
        console.print(f"[bold]Customer:[/bold] {text}\n")
        
        console.print("[dim]Running pipeline...[/dim]")
        
        text_clean = redact(text)
        intent, conf = classifier.predict(text_clean)
        retrieved = retriever.retrieve(text_clean)
        urgency_data = generator.score_urgency(text_clean)
        
        escalation = gate.decide(text_clean, intent, conf, retrieved, urgency_data.get("urgency_score", 0), urgency_data.get("sentiment", "neutral"))
        
        if escalation.decision == "auto_handle":
            gen = generator.generate(text_clean, intent, taxonomy.get_description(intent), retrieved)
            draft = gen["draft_reply"]
        else:
            draft = "[ESCALATED - NO REPLY GENERATED]"
            
        console.print(f"[bold]System Draft Reply:[/bold]\n{draft}\n")
        
        if escalation.decision == "auto_handle":
            judge_res = judge.score_batch([{
                "customer_message": text_clean,
                "draft_reply": draft,
                "retrieved_examples": retrieved,
                "intent": intent
            }])[0]
            llm_score = judge_res["average"]
            console.print(f"[bold magenta]LLM Judge Score (Avg):[/bold magenta] {llm_score:.1f}/5.0\n")
        else:
            console.print("[bold magenta]LLM Judge Score:[/bold magenta] N/A (Escalated)\n")
            
        # 1. Ask for Escalation Ground Truth
        esc_input = ""
        while esc_input.lower() not in ["y", "n", "q"]:
            esc_input = input("Should this message have been escalated to a human? (y/n or 'q' to quit): ").strip()
            
        if esc_input.lower() == "q":
            break
            
        df.at[idx, "human_escalate"] = "escalate" if esc_input.lower() == "y" else "auto_handle"
        
        # 2. Ask for Score if not escalated
        if escalation.decision == "auto_handle":
            score_input = ""
            while score_input not in ["1", "2", "3", "4", "5", "q"]:
                score_input = input("Your Score for the draft reply (1-5 or 'q' to quit): ").strip()
                
            if score_input == "q":
                break
                
            df.at[idx, "human_judge_score"] = int(score_input)
        else:
            df.at[idx, "human_judge_score"] = "N/A"
            
        # Save after every entry
        df.to_csv(args.input, index=False)

    df.to_csv(args.input, index=False)
    scored = len(df[(df["human_judge_score"].notna()) & (df["human_judge_score"] != "")])
    console.print(f"\n[bold green]Saved! You have scored {scored} total examples.[/bold green]")

if __name__ == "__main__":
    main()
