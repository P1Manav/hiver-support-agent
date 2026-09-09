"""
src/generation/guardrails.py
──────────────────────────────
Flag draft replies that assert unverifiable claims — dates, refund amounts,
policy specifics — not present in the retrieved precedents.

Design decision (D-11): Regex-based, not LLM-based. Fast, interpretable,
catches the most common hallucination failure mode without adding model latency.

WHERE IT RUNS: Local CPU. < 1ms per reply.
"""

import re
import logging
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default patterns (can be overridden via config.yaml guardrails.unverifiable_patterns)
_DEFAULT_PATTERNS = [
    (r"\b\d+\s*(?:business\s*)?days?\b", "specific_timeframe"),
    (r"\$\d+(?:\.\d{2})?\b", "specific_dollar_amount"),
    (r"\b\d+\s*%\s*(?:refund|discount|off|back)\b", "specific_percentage"),
    (r"\b(?:policy|guarantee|warranty|terms)\s+(?:says?|states?|allows?|requires?)\b", "policy_assertion"),
    (r"\b(?:we|our)\s+(?:promise|guarantee|ensure|assure)\b", "promise_language"),
    (r"\bwithin\s+\d+\s*(?:hours?|days?|weeks?)\b", "specific_timeframe"),
]


class GuardrailChecker:
    """
    Checks draft replies against retrieved precedents for unverifiable claims.
    """

    def __init__(self, patterns: Optional[list[str]] = None):
        """
        Args:
            patterns: List of regex strings. None = use defaults from code.
                      Pass config['guardrails']['unverifiable_patterns'] to override.
        """
        if patterns:
            self._compiled = [(re.compile(p, re.IGNORECASE), "config_pattern") for p in patterns]
        else:
            self._compiled = [(re.compile(p, re.IGNORECASE), label) for p, label in _DEFAULT_PATTERNS]

    def check(
        self,
        draft_reply: str,
        retrieved_examples: list[dict],
    ) -> list[dict]:
        """
        Check if the draft reply contains claims not grounded in retrieved context.

        Args:
            draft_reply: Generated reply text.
            retrieved_examples: Top-k retrieved (customer, reply) pairs.

        Returns:
            List of flag dicts:
            [{
                "pattern_type": str,
                "matched_text": str,
                "grounded": bool,  # True if the matched text also appears in a precedent
            }]
        """
        flags = []

        # Build combined precedent text for groundedness check
        precedent_text = " ".join(
            ex.get("brand_reply", "") for ex in retrieved_examples
        ).lower()

        for pattern, label in self._compiled:
            for match in pattern.finditer(draft_reply):
                matched = match.group(0)
                # Check if this exact claim (or close variant) appears in precedents
                grounded = matched.lower() in precedent_text

                if not grounded:
                    flags.append({
                        "pattern_type": label,
                        "matched_text": matched,
                        "grounded": False,
                        "message": (
                            f"Unverifiable claim detected: '{matched}' "
                            f"(type: {label}) not found in retrieved precedents."
                        ),
                    })
                    logger.debug(f"Guardrail flagged: '{matched}' ({label})")

        return flags

    def is_safe(
        self,
        draft_reply: str,
        retrieved_examples: list[dict],
    ) -> tuple[bool, list[dict]]:
        """
        Convenience method: returns (is_safe, flags).

        Returns:
            (True, []) if no unverifiable claims, (False, [flags]) otherwise.
        """
        flags = self.check(draft_reply, retrieved_examples)
        return len(flags) == 0, flags
