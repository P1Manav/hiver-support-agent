"""
src/evaluation/judge.py
────────────────────────
LLM-as-judge evaluation using Qwen2.5-7B-Instruct (different from generator).

Design decision (D-08): Use a DIFFERENT model than the generator.
Judge model: qwen2.5:7b-instruct-q4_K_M via Ollama (local)
             OR run on Colab T4 in fp16 for the final scoring pass.

Scores each draft reply on 4 dimensions (1–5 each):
  - correctness: Does the reply address the customer's actual issue?
  - tone: Is it empathetic and professional?
  - groundedness: Is it grounded in retrieved context (no hallucinations)?
  - conciseness: Is it appropriately brief?

WHERE IT RUNS: Local (Ollama, 3060 Ti) or Colab T4 (fp16 via vLLM or Ollama).
Expected latency: 3–6s per example on RTX 3060 Ti.
"""

import json
import logging
import time
from typing import Optional

import ollama

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Evaluates draft replies using a local Ollama LLM judge.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    def score(
        self,
        customer_message: str,
        draft_reply: str,
        retrieved_examples: list[dict],
        intent: str,
        max_retries: int = 3,
    ) -> dict:
        """
        Score a single draft reply.

        Returns:
            Dict: {
                "correctness": int (1-5),
                "tone": int (1-5),
                "groundedness": int (1-5),
                "conciseness": int (1-5),
                "average": float,
                "rationale": str,
                "model": str,
                "latency_ms": int,
            }
        """
        from src.generation.prompts import judge_prompt as build_judge_prompt

        messages = build_judge_prompt(
            customer_message=customer_message,
            draft_reply=draft_reply,
            retrieved_examples=retrieved_examples,
            intent=intent,
        )

        client = self._get_client()
        t0 = time.time()

        for attempt in range(max_retries):
            try:
                response = client.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature},
                )
                raw = response["message"]["content"].strip()

                # Strip markdown fences
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                parsed = json.loads(raw)
                dims = ["correctness", "tone", "groundedness", "conciseness"]
                scores = {d: int(parsed.get(d, 3)) for d in dims}
                avg = sum(scores.values()) / len(dims)

                return {
                    **scores,
                    "average": round(avg, 2),
                    "rationale": str(parsed.get("rationale", "")),
                    "model": self.model,
                    "latency_ms": int((time.time() - t0) * 1000),
                }

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Judge parse attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        # Fallback if all retries fail
        logger.error(f"Judge failed after {max_retries} attempts. Returning default scores.")
        return {
            "correctness": 3, "tone": 3, "groundedness": 3, "conciseness": 3,
            "average": 3.0,
            "rationale": "Scoring failed — default scores assigned.",
            "model": self.model,
            "latency_ms": int((time.time() - t0) * 1000),
            "error": "parse_failure",
        }

    def score_batch(
        self,
        examples: list[dict],
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Score a batch of examples.

        Args:
            examples: List of dicts with keys:
                [customer_message, draft_reply, retrieved_examples, intent]

        Returns:
            List of score dicts (same order as input).
        """
        from tqdm import tqdm
        results = []
        it = tqdm(examples, desc="LLM Judge scoring") if show_progress else examples

        for ex in it:
            result = self.score(
                customer_message=ex["customer_message"],
                draft_reply=ex["draft_reply"],
                retrieved_examples=ex.get("retrieved_examples", []),
                intent=ex.get("intent", "unknown"),
            )
            results.append(result)

        return results
