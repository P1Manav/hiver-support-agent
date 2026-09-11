#!/usr/bin/env python
"""
scripts/label_cli.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Simple CLI tool for hand-labeling the golden set.

Presents each example one at a time with the predicted intent pre-filled.
Press Enter to accept the prediction, or type the correct intent name.
Type 'q' to quit and save progress (can resume later).
Type '?' to see the full taxonomy list.
Type 'note <text>' to add a note to the current example.
Type 'back' to go back to the previous example.

COMMAND:
  python scripts/label_cli.py --input golden/golden_set_to_label.csv
  python scripts/label_cli.py --input golden/golden_set_to_label.csv --resume

OUTPUT:
  golden/golden_set.csv  â€” labeled golden set (intent column filled)

EXPECTED TIME: ~15â€“25 seconds per example = ~45â€“80 min for 200 examples.
"""

import sys
import argparse
import csv
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.intents.taxonomy import IntentTaxonomy

# â”€â”€ Colors for terminal output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def show_taxonomy(taxonomy: IntentTaxonomy):
    print(f"\n{BOLD}Available intents:{RESET}")
    for i, name in enumerate(taxonomy.intent_names, 1):
        desc = taxonomy.get_description(name)
        short_desc = desc[:70] + "..." if len(desc) > 70 else desc
        print(f"  {CYAN}{i:2}. {name:<30}{RESET} {DIM}{short_desc}{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CLI golden set hand-labeler")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default="golden/golden_set_to_label.csv",
                        help="Input CSV from 07_golden_set_sampler.py")
    parser.add_argument("--output", default="golden/golden_set.csv",
                        help="Output CSV (labeled)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output (skip already-labeled rows)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    taxonomy = IntentTaxonomy(cfg["taxonomy"]["file"])
    valid_intents = set(taxonomy.intent_names)

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"{RED}ERROR: Input file not found: {input_path}{RESET}")
        print("Run: python scripts/07_golden_set_sampler.py")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Load existing progress if resuming
    already_labeled = {}
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("human_intent"):
                    already_labeled[row["thread_id"]] = row["human_intent"]
        print(f"{GREEN}Resuming: {len(already_labeled)} examples already labeled.{RESET}")

    # Ensure output has all required columns
    fieldnames = list(rows[0].keys())
    if "human_intent" not in fieldnames:
        fieldnames.append("human_intent")
    if "notes" not in fieldnames:
        fieldnames.append("notes")
    if "human_judge_score" not in fieldnames:
        fieldnames.append("human_judge_score")

    # Track all labeled rows (pre-populate with resumable progress)
    labeled_rows = []
    for row in rows:
        row.setdefault("human_intent", "")
        row.setdefault("notes", "")
        row.setdefault("human_judge_score", "")
        if row["thread_id"] in already_labeled:
            row["human_intent"] = already_labeled[row["thread_id"]]
        labeled_rows.append(row)

    # Filter to unlabeled
    to_label = [r for r in labeled_rows if not r.get("human_intent")]
    n_total = len(rows)
    n_done = n_total - len(to_label)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}Hiver Golden Set Labeler{RESET}")
    print(f"  Total examples:    {n_total}")
    print(f"  Already labeled:   {n_done}")
    print(f"  Remaining:         {len(to_label)}")
    print(f"  Estimated time:    ~{len(to_label) * 20 // 60} min")
    print(f"\n{DIM}Commands:{RESET}")
    print(f"  {CYAN}Enter{RESET}       â†’ Accept predicted intent")
    print(f"  {CYAN}<name>{RESET}      â†’ Override with correct intent name")
    print(f"  {CYAN}?{RESET}           â†’ Show full taxonomy list")
    print(f"  {CYAN}note <text>{RESET} â†’ Add a note to this example")
    print(f"  {CYAN}skip{RESET}        â†’ Leave this example unlabeled (come back later)")
    print(f"  {CYAN}q{RESET}           â†’ Quit and save progress")
    print(f"\n{BOLD}{'='*60}{RESET}\n")
    input("Press Enter to start labeling...")

    i = 0
    current_note = ""
    while i < len(to_label):
        row = to_label[i]
        clear()

        # Progress bar
        total_done = n_done + i
        pct = total_done / n_total * 100
        bar_len = 40
        filled = int(bar_len * total_done / n_total)
        bar = "â–ˆ" * filled + "â–‘" * (bar_len - filled)
        print(f"\n{CYAN}[{bar}] {total_done}/{n_total} ({pct:.0f}%){RESET}\n")

        # Message display
        text = row.get("customer_text", "")
        predicted = row.get("predicted_intent", "unknown")
        confidence = float(row.get("confidence", 0))
        top_sim = float(row.get("top_similarity", 0))
        is_edge = row.get("is_edge", "False")
        word_count = row.get("word_count", "?")

        conf_color = GREEN if confidence >= 0.7 else (YELLOW if confidence >= 0.5 else RED)
        edge_tag = f" {YELLOW}[EDGE CASE]{RESET}" if str(is_edge).lower() == "true" else ""

        print(f"{BOLD}Message:{RESET}{edge_tag}")
        print(f"  {BOLD}{text}{RESET}")
        print()
        print(f"  {DIM}Word count: {word_count} | Top retrieval sim: {top_sim:.3f}{RESET}")
        print()
        print(f"  {BOLD}Predicted intent:{RESET} {conf_color}{predicted}{RESET} "
              f"({conf_color}confidence: {confidence:.2%}{RESET})")
        if current_note:
            print(f"  {DIM}Note so far: {current_note}{RESET}")
        print()
        print(f"{DIM}[Enter=accept '{predicted}' | type intent name | ? = list | skip | q = quit]{RESET}")

        try:
            user_input = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Interrupted. Saving progress...{RESET}")
            break

        if user_input.lower() == "q":
            print(f"\n{YELLOW}Quitting. Saving progress...{RESET}")
            break

        elif user_input.lower() == "?":
            show_taxonomy(taxonomy)
            input("Press Enter to continue...")
            continue  # don't advance i

        elif user_input.lower() == "skip":
            i += 1
            current_note = ""
            continue

        elif user_input.lower().startswith("note "):
            current_note = user_input[5:].strip()
            continue  # don't advance i â€” allow editing intent next

        elif user_input == "" or user_input.lower() == predicted.lower():
            # Accept prediction
            row["human_intent"] = predicted
            row["notes"] = current_note
            i += 1
            current_note = ""
            print(f"  {GREEN}âœ“ Accepted: {predicted}{RESET}")

        elif user_input.lower() in {name.lower() for name in valid_intents}:
            # Exact match (case insensitive)
            matched = next(n for n in valid_intents if n.lower() == user_input.lower())
            row["human_intent"] = matched
            row["notes"] = current_note
            i += 1
            current_note = ""
            print(f"  {GREEN}âœ“ Labeled: {matched}{RESET}")

        else:
            # Try partial match
            matches = [n for n in valid_intents if user_input.lower() in n.lower()]
            if len(matches) == 1:
                row["human_intent"] = matches[0]
                row["notes"] = current_note
                i += 1
                current_note = ""
                print(f"  {GREEN}âœ“ Matched: {matches[0]}{RESET}")
            elif len(matches) > 1:
                print(f"  {YELLOW}Ambiguous: {matches}. Be more specific.{RESET}")
                input("  Press Enter to retry...")
                continue
            else:
                print(f"  {RED}Unknown intent: '{user_input}'. Type ? to see list.{RESET}")
                input("  Press Enter to retry...")
                continue

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled_rows)

    n_labeled = sum(1 for r in labeled_rows if r.get("human_intent"))
    print(f"\n{BOLD}{GREEN}âœ“ Saved {n_labeled}/{n_total} labeled examples to {output_path}{RESET}")
    if n_labeled < n_total:
        print(f"{YELLOW}  {n_total - n_labeled} examples still need labeling.{RESET}")
        print(f"  Resume with: {CYAN}python scripts/label_cli.py --resume{RESET}")
    else:
        print(f"{GREEN}  Golden set complete! All {n_total} examples labeled.{RESET}")
        print(f"  Next step: {CYAN}python scripts/08_judge_agreement.py{RESET}")
    print()


if __name__ == "__main__":
    main()

