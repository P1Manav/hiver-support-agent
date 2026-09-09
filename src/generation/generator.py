"""
src/generation/generator.py
─────────────────────────────
Draft reply generation via Ollama.

Default model: llama3.1:8b-instruct-q4_K_M (RTX 3060 Ti, ~4.7 GB VRAM)
Fast-iter swap: phi3.5:3.8b-mini-instruct-q4_K_M (~2.2 GB, ~2× faster)

Switch by setting generator_model in config.yaml or passing model= directly.

WHERE IT RUNS: Local (Ollama must be running on the RTX 3060 Ti).
Expected latency: 3–8 seconds per reply on RTX 3060 Ti.
"""

import json
import logging
import time

import ollama

from .prompts import generation_prompt, urgency_scoring_prompt

logger = logging.getLogger(__name__)


class ReplyGenerator:
    """
    Generates draft customer support replies using a local Ollama LLM.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b-instruct-q4_K_M",
        host: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 256,
        brand: str = "AmazonHelp",
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brand = brand
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    def generate(
        self,
        customer_message: str,
        intent: str,
        intent_description: str,
        retrieved_examples: list[dict],
        thread_turns: list[dict] | None = None,
        sentiment_label: str | None = None,
    ) -> dict:
        """
        Generate a draft reply for a customer message.

        Args:
            customer_message: PII-redacted customer message.
            intent: Classified intent name.
            intent_description: Human-readable description.
            retrieved_examples: Top-k retrieved (customer, reply) pairs.
            thread_turns: Full thread for multi-turn context.
            sentiment_label: Optional sentiment/urgency label.

        Returns:
            Dict: {
                "draft_reply": str,
                "model": str,
                "latency_ms": int,
                "tokens_used": int,
            }
        """
        messages = generation_prompt(
            customer_message=customer_message,
            intent=intent,
            intent_description=intent_description,
            retrieved_examples=retrieved_examples,
            thread_turns=thread_turns,
            brand=self.brand,
            sentiment_label=sentiment_label,
        )

        client = self._get_client()
        t0 = time.time()

        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            draft = response["message"]["content"].strip()
            latency_ms = int((time.time() - t0) * 1000)

            return {
                "draft_reply": draft,
                "model": self.model,
                "latency_ms": latency_ms,
                "tokens_used": response.get("eval_count", 0),
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "draft_reply": "We apologize for the inconvenience. Please DM us with your order details and we'll look into this right away. ^Team",
                "model": self.model,
                "latency_ms": int((time.time() - t0) * 1000),
                "tokens_used": 0,
                "error": str(e),
            }

    def score_urgency(self, text: str) -> dict:
        """
        Score the urgency/sentiment of a customer message.

        Returns:
            Dict: {"sentiment": str, "urgency": str, "urgency_score": float}
        """
        messages = urgency_scoring_prompt(text)
        client = self._get_client()

        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.0},
            )
            raw = response["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Urgency scoring failed: {e}. Defaulting to neutral/low.")
            return {"sentiment": "neutral", "urgency": "low", "urgency_score": 0.0}
