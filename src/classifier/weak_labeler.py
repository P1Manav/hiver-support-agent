"""
src/classifier/weak_labeler.py
───────────────────────────────
Bulk-label customer messages with intent labels using a local Ollama LLM.

Design decision (D-04): Use an instruction-tuned LLM for weak supervision
instead of a HF zero-shot classifier. The LLM takes the full taxonomy as
context and produces more consistent, context-aware labels.

Design decision (D-05): These labels are ONLY used for fine-tuning. At
runtime, we use the fast DistilBERT classifier, NOT the LLM.

WHERE IT RUNS: Local (Ollama must be running) OR Colab T4 with Ollama
installed. Expect ~3–5 seconds per batch of 20 messages.
Expected runtime: ~10k messages in ~45 min on T4, ~2 hrs on 3060 Ti.

COMMAND:
  python scripts/04_label_with_llm.py --n-samples 10000
"""

import json
import logging
import time
from typing import Optional

import ollama

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert customer support analyst.
Your job is to classify customer tweets into exactly ONE intent from the provided taxonomy.
Always respond with valid JSON only. No explanations, no markdown, just JSON."""


def build_labeling_prompt(texts: list[str], taxonomy_str: str) -> str:
    """
    Build the user prompt for bulk labeling a batch of messages.

    Args:
        texts: List of customer tweet texts (already PII-redacted).
        taxonomy_str: Formatted taxonomy string from IntentTaxonomy.to_prompt_string().

    Returns:
        Prompt string for the LLM.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return f"""{taxonomy_str}

Classify each of the following customer messages into exactly one intent.
Respond with a JSON array of objects, one per message, in order.
Each object must have exactly two keys: "index" (1-based int) and "intent" (string matching taxonomy name exactly).

Messages:
{numbered}

Respond with JSON only:"""


def label_batch(
    texts: list[str],
    taxonomy_str: str,
    model: str = "llama3.1:8b-instruct-q4_K_M",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> list[Optional[str]]:
    """
    Label a batch of texts using the Ollama LLM.

    Args:
        texts: Customer messages to label (batch of ~20).
        taxonomy_str: Taxonomy string for the prompt.
        model: Ollama model identifier.
        host: Ollama server URL.
        temperature: 0.0 = deterministic.
        max_retries: Retry on parse failure.

    Returns:
        List of intent label strings (same length as texts). None if failed.
    """
    client = ollama.Client(host=host)
    prompt = build_labeling_prompt(texts, taxonomy_str)

    for attempt in range(max_retries):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": temperature},
            )
            raw = response["message"]["content"].strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)
            labels = [None] * len(texts)
            for item in parsed:
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(texts):
                    labels[idx] = item.get("intent")
            return labels

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning(f"Batch labeling attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff

    logger.error(f"All {max_retries} attempts failed for batch of {len(texts)}. Returning None labels.")
    return [None] * len(texts)


def label_dataset(
    texts: list[str],
    taxonomy_str: str,
    model: str = "llama3.1:8b-instruct-q4_K_M",
    host: str = "http://localhost:11434",
    batch_size: int = 20,
    temperature: float = 0.0,
) -> list[Optional[str]]:
    """
    Label a full dataset in batches with progress reporting.

    Args:
        texts: All customer messages to label.
        taxonomy_str: Taxonomy string.
        model: Ollama model.
        host: Ollama URL.
        batch_size: Messages per LLM call.
        temperature: LLM temperature.

    Returns:
        List of intent labels (same length as texts).
    """
    from tqdm import tqdm

    all_labels: list[Optional[str]] = []
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    logger.info(f"Labeling {len(texts):,} messages in {len(batches):,} batches (batch_size={batch_size})...")
    fail_count = 0

    for batch in tqdm(batches, desc="Labeling"):
        batch_labels = label_batch(batch, taxonomy_str, model, host, temperature)
        all_labels.extend(batch_labels)
        fail_count += sum(1 for l in batch_labels if l is None)

    success_rate = (len(texts) - fail_count) / len(texts) * 100
    logger.info(f"Labeling complete. Success rate: {success_rate:.1f}% ({fail_count} failures).")
    return all_labels
