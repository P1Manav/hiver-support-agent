"""
src/intents/taxonomy.py
────────────────────────
Load, validate, and query the intent taxonomy from config/intent_taxonomy.yaml.

The taxonomy YAML is the single source of truth for intent names, descriptions,
keywords, and escalation denylist flags. Edit it after cluster inspection.

WHERE IT RUNS: Anywhere. No GPU/network needed.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class IntentTaxonomy:
    """
    Loads and exposes the intent taxonomy from a YAML file.
    """

    def __init__(self, taxonomy_path: str = "config/intent_taxonomy.yaml"):
        self.taxonomy_path = Path(taxonomy_path)
        self._intents: list[dict] = []
        self._name_to_intent: dict[str, dict] = {}
        self._load()

    def _load(self):
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(
                f"Taxonomy file not found: {self.taxonomy_path}. "
                "Edit config/intent_taxonomy.yaml to define your intents."
            )
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._intents = data.get("intents", [])
        self._name_to_intent = {i["name"]: i for i in self._intents}
        logger.info(f"Loaded {len(self._intents)} intents from {self.taxonomy_path}.")

    @property
    def intent_names(self) -> list[str]:
        """Ordered list of intent name strings."""
        return [i["name"] for i in self._intents]

    @property
    def denylist_intents(self) -> set[str]:
        """Intent names that are always escalated (high-risk)."""
        return {
            i["name"]
            for i in self._intents
            if i.get("escalation_denylist", False)
        }

    def get_intent(self, name: str) -> Optional[dict]:
        """Return the intent dict for a given name, or None."""
        return self._name_to_intent.get(name)

    def get_description(self, name: str) -> str:
        """Return the intent description string."""
        intent = self.get_intent(name)
        return intent["description"].strip() if intent else ""

    def get_examples(self, name: str) -> list[str]:
        """Return example messages for an intent."""
        intent = self.get_intent(name)
        return intent.get("examples", []) if intent else []

    def to_prompt_string(self) -> str:
        """
        Format the full taxonomy as a numbered list suitable for inclusion
        in an LLM prompt (used by weak_labeler.py).
        """
        lines = ["Intent taxonomy (use EXACTLY these names):"]
        for i, intent in enumerate(self._intents, 1):
            desc = intent["description"].strip().replace("\n", " ")
            lines.append(f"{i}. {intent['name']}: {desc}")
        return "\n".join(lines)

    def id_to_name(self, idx: int) -> str:
        """Convert 0-indexed integer to intent name."""
        return self._intents[idx]["name"]

    def name_to_id(self, name: str) -> int:
        """Convert intent name to 0-indexed integer."""
        return self.intent_names.index(name)

    def __len__(self) -> int:
        return len(self._intents)

    def __repr__(self) -> str:
        return f"IntentTaxonomy({len(self._intents)} intents: {self.intent_names})"
