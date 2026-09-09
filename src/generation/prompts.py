"""
src/generation/prompts.py
──────────────────────────
Prompt templates for:
  1. Reply generation (Llama-3.1-8B or Phi-3.5-mini via Ollama)
  2. LLM-as-judge evaluation (Qwen2.5-7B via Ollama)

WHERE IT RUNS: Anywhere. Pure Python string formatting.
"""


def generation_prompt(
    customer_message: str,
    intent: str,
    intent_description: str,
    retrieved_examples: list[dict],
    thread_turns: list[dict] | None = None,
    brand: str = "AmazonHelp",
    sentiment_label: str | None = None,
) -> list[dict]:
    """
    Build the Ollama chat messages list for reply generation.

    Args:
        customer_message: The customer's latest message (PII-redacted).
        intent: Classified intent name.
        intent_description: Human-readable intent description.
        retrieved_examples: Top-k retrieved (customer_text, brand_reply) dicts.
        thread_turns: Optional full thread for context.
        brand: Brand handle for persona.
        sentiment_label: "positive", "neutral", "negative", or "urgent".

    Returns:
        List of {"role": ..., "content": ...} dicts for Ollama.
    """
    # Build retrieved context block
    context_lines = []
    for ex in retrieved_examples:
        context_lines.append(
            f"[Example {ex['rank']} — similarity {ex['similarity']:.2f}]\n"
            f"Customer: {ex['customer_text']}\n"
            f"Brand reply: {ex['brand_reply']}"
        )
    context_block = "\n\n".join(context_lines) if context_lines else "No similar precedents found."

    # Build thread context if available
    thread_block = ""
    if thread_turns and len(thread_turns) > 1:
        thread_lines = []
        for turn in thread_turns[-6:]:  # last 6 turns max
            role = "Customer" if turn.get("inbound") else brand
            thread_lines.append(f"{role}: {turn['text']}")
        thread_block = "\n".join(thread_lines)

    # Sentiment note
    sentiment_note = ""
    if sentiment_label == "urgent":
        sentiment_note = "\n⚠️ This customer appears very distressed. Prioritize empathy and a clear action plan."
    elif sentiment_label == "negative":
        sentiment_note = "\nNote: The customer is frustrated. Acknowledge their frustration before solving."

    system = f"""You are a helpful customer support agent for {brand}.
Your job is to draft a brief, professional, empathetic reply to the customer's message.

Guidelines:
- Be concise (1–3 sentences max for Twitter; 2–4 sentences for email/chat).
- Only reference information that appears in the retrieved precedents below.
- Do NOT invent specific dates, refund amounts, order IDs, or policy details not in the precedents.
- Acknowledge the customer's frustration if they seem upset.
- End with a clear next step or action.
- Sign off as "^Team" to indicate a team account.{sentiment_note}"""

    user = f"""Intent: {intent} — {intent_description}

RETRIEVED PRECEDENTS (ground your reply in these):
{context_block}

{"CONVERSATION HISTORY:" + chr(10) + thread_block + chr(10) if thread_block else ""}Customer's latest message:
{customer_message}

Write the draft reply now (do not include "Brand:" or any prefix — just the reply text):"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def judge_prompt(
    customer_message: str,
    draft_reply: str,
    retrieved_examples: list[dict],
    intent: str,
) -> list[dict]:
    """
    Build the Ollama chat messages list for LLM-as-judge scoring.

    The judge scores the draft reply on 4 dimensions (1–5 each).
    Uses a DIFFERENT model than the generator (D-08).

    Returns:
        List of {"role": ..., "content": ...} dicts for Ollama.
    """
    context_block = "\n\n".join(
        f"[Precedent {ex['rank']}]\nCustomer: {ex['customer_text']}\nReply: {ex['brand_reply']}"
        for ex in retrieved_examples
    ) if retrieved_examples else "None"

    system = """You are an expert evaluator of customer support reply quality.
Score the draft reply on four dimensions, each 1–5 (integer only).
Respond with valid JSON only. No explanations outside the JSON.

Scoring rubrics:
- correctness (1–5): Is the reply factually correct and relevant to the customer's issue?
  5=perfectly addresses the issue, 1=completely irrelevant or wrong
- tone (1–5): Is the reply empathetic, professional, and appropriate in tone?
  5=excellent tone, 1=rude, dismissive, or robotic
- groundedness (1–5): Is the reply grounded in the retrieved precedents? Does it avoid hallucinated specifics?
  5=fully grounded, 1=invents facts not in precedents
- conciseness (1–5): Is the reply appropriately brief and not padded?
  5=perfectly concise, 1=overly long or extremely terse

Also provide a one-sentence "rationale" (string) for your overall assessment."""

    user = f"""Intent: {intent}

Customer message:
{customer_message}

Retrieved precedents (ground truth context):
{context_block}

Draft reply to evaluate:
{draft_reply}

Respond with JSON:
{{
  "correctness": <1-5>,
  "tone": <1-5>,
  "groundedness": <1-5>,
  "conciseness": <1-5>,
  "rationale": "<one sentence>"
}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def urgency_scoring_prompt(text: str) -> list[dict]:
    """
    Prompt for sentiment/urgency scoring (used as escalation signal).
    Returns JSON with sentiment and urgency fields.
    """
    system = """You are a sentiment analysis system. Classify the sentiment and urgency of a customer message.
Respond with valid JSON only."""

    user = f"""Classify this customer message:
"{text}"

Respond with JSON:
{{
  "sentiment": "<positive|neutral|negative>",
  "urgency": "<low|medium|high>",
  "urgency_score": <0.0-1.0>
}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
