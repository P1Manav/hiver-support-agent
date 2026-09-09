"""
scripts/06_inference.py
────────────────────────
End-to-end inference pipeline: classify → retrieve → escalate → generate → guardrail.

WHERE IT RUNS: Local (RTX 3060 Ti, Ollama must be running).
EXPECTED LATENCY: 3–8 seconds per message (dominated by LLM generation).

PREREQS:
  - models/classifier/ (fine-tuned DistilBERT, from Colab)
  - faiss_index/ (built by 05_build_faiss_index.py)
  - Ollama running with llama3.1:8b-instruct-q4_K_M pulled

COMMAND:
  python scripts/06_inference.py --message "My order hasn't arrived in 3 weeks"
  python scripts/06_inference.py --message "..." --fast-mode  # uses Phi-3.5-mini
  python scripts/06_inference.py --file data/processed/customer_messages.csv --n 20  # batch

OUTPUT (single message):
  JSON to stdout with intent, confidence, decision, reason, draft_reply, guardrail_flags
"""

import sys
import logging
import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

from src.intents.taxonomy import IntentTaxonomy
from src.intents.embedder import MessageEmbedder
from src.classifier.predict import IntentClassifier
from src.retrieval.indexer import load_index
from src.retrieval.retriever import Retriever
from src.generation.generator import ReplyGenerator
from src.generation.guardrails import GuardrailChecker
from src.escalation.gate import EscalationGate
from src.data.pii import redact

logging.basicConfig(level=logging.WARNING)  # quiet for interactive use
console = Console()


def load_components(cfg: dict, fast_mode: bool = False):
    """Load all pipeline components from config."""
    taxonomy = IntentTaxonomy(cfg["taxonomy"]["file"])
    embedder = MessageEmbedder(
        model_name=cfg["embedding"]["model"],
        device=cfg["embedding"]["device"],
        batch_size=cfg["embedding"]["batch_size"],
    )

    index, metadata = load_index(cfg["retrieval"]["index_dir"])
    retriever = Retriever(index, metadata, embedder, top_k=cfg["retrieval"]["top_k"])

    model = cfg["generation"]["fast_model"] if fast_mode else cfg["generation"]["model"]
    generator = ReplyGenerator(
        model=model,
        host=cfg["generation"]["ollama_host"],
        temperature=cfg["generation"]["temperature"],
        max_tokens=cfg["generation"]["max_tokens"],
        brand=cfg["data"]["brand"],
    )

    classifier = IntentClassifier(
        model_dir=cfg["classifier"]["model_dir"],
        max_length=cfg["classifier"]["max_length"],
    )

    guardrails = GuardrailChecker(
        patterns=cfg.get("guardrails", {}).get("unverifiable_patterns")
    )

    gate = EscalationGate.from_config(cfg)

    return taxonomy, embedder, retriever, generator, classifier, guardrails, gate


def run_inference(
    message: str,
    taxonomy,
    retriever,
    generator,
    classifier,
    guardrails,
    gate,
    thread_turns: list | None = None,
) -> dict:
    """Run the full inference pipeline for a single message."""
    t0 = time.time()

    # PII redact
    message_clean = redact(message)

    # Classify
    intent, confidence = classifier.predict(message_clean)
    intent_desc = taxonomy.get_description(intent)

    # Retrieve
    retrieved = retriever.retrieve(message_clean)

    # Urgency scoring (uses the generator LLM)
    urgency_data = generator.score_urgency(message_clean)
    urgency_score = urgency_data.get("urgency_score", 0.0)
    sentiment = urgency_data.get("sentiment", "neutral")

    # Escalation gate
    escalation = gate.decide(
        message=message_clean,
        intent=intent,
        confidence=confidence,
        retrieved_examples=retrieved,
        urgency_score=urgency_score,
        sentiment=sentiment,
    )

    # Generation (only if auto_handle)
    draft_reply = ""
    guardrail_flags = []

    if escalation.decision == "auto_handle":
        gen_result = generator.generate(
            customer_message=message_clean,
            intent=intent,
            intent_description=intent_desc,
            retrieved_examples=retrieved,
            thread_turns=thread_turns,
            sentiment_label=sentiment,
        )
        draft_reply = gen_result["draft_reply"]

        # Guardrails
        _, guardrail_flags = guardrails.is_safe(draft_reply, retrieved)
    else:
        draft_reply = "[ESCALATED — no reply generated]"

    total_ms = int((time.time() - t0) * 1000)

    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "decision": escalation.decision,
        "reason": escalation.reason,
        "triggered_rules": escalation.triggered_rules,
        "draft_reply": draft_reply,
        "guardrail_flags": [f["message"] for f in guardrail_flags],
        "retrieved_examples": [
            {"rank": r["rank"], "similarity": round(r["similarity"], 3), "brand_reply": r["brand_reply"][:100]}
            for r in retrieved[:3]
        ],
        "sentiment": sentiment,
        "urgency_score": round(urgency_score, 3),
        "latency_ms": total_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end inference pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--message", type=str, help="Single customer message")
    parser.add_argument("--file", type=str, help="CSV file with 'text' column for batch inference")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to process from file")
    parser.add_argument("--fast-mode", action="store_true", help="Use Phi-3.5-mini instead of Llama-3.1-8B")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    console.print(f"\n[bold cyan]Loading pipeline components...[/bold cyan]")
    taxonomy, embedder, retriever, generator, classifier, guardrails, gate = load_components(
        cfg, fast_mode=args.fast_mode
    )
    console.print("[green]✓ All components loaded[/green]\n")

    if args.message:
        # Single message inference
        result = run_inference(
            message=args.message,
            taxonomy=taxonomy,
            retriever=retriever,
            generator=generator,
            classifier=classifier,
            guardrails=guardrails,
            gate=gate,
        )

        # Pretty print
        decision_color = "green" if result["decision"] == "auto_handle" else "red"
        console.print(Panel(
            f"[bold]Intent:[/bold] {result['intent']} (confidence: {result['confidence']:.2%})\n"
            f"[bold]Decision:[/bold] [{decision_color}]{result['decision'].upper()}[/{decision_color}]\n"
            f"[bold]Reason:[/bold] {result['reason']}\n\n"
            f"[bold]Draft Reply:[/bold]\n{result['draft_reply']}\n\n"
            f"[bold]Guardrail Flags:[/bold] {result['guardrail_flags'] or 'None'}\n"
            f"[dim]Latency: {result['latency_ms']}ms[/dim]",
            title="Inference Result",
            border_style="cyan",
        ))
        print("\nJSON output:")
        print(json.dumps(result, indent=2))

    elif args.file:
        # Batch inference
        df = pd.read_csv(args.file).head(args.n)
        results = []
        for _, row in df.iterrows():
            msg = row.get("text") or row.get("customer_text", "")
            result = run_inference(
                message=msg,
                taxonomy=taxonomy,
                retriever=retriever,
                generator=generator,
                classifier=classifier,
                guardrails=guardrails,
                gate=gate,
            )
            result["input_text"] = msg
            results.append(result)
            console.print(f"[{result['decision']}] {msg[:60]}... → {result['intent']} ({result['confidence']:.2%})")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\nResults saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
