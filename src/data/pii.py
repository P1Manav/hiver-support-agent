"""
src/data/pii.py
───────────────
PII (Personally Identifiable Information) redaction for tweet text.

Redacts:
  - Twitter handles (@username)
  - Order/reference numbers (e.g., Amazon order IDs: 3-digit-7digit-7digit)
  - Phone-like strings (10+ digit sequences, formatted phone numbers)
  - Email addresses
  - URLs (may contain tracking params with PII)
  - Credit card patterns (just-in-case)

Design decision (D-09): Redaction happens BEFORE any text is embedded or
stored in FAISS. This prevents PII from appearing in retrieved context and
potentially being echoed in generated replies.

WHERE IT RUNS: Local CPU. No GPU or network needed. ~1ms per tweet.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex Patterns ────────────────────────────────────────────────────────────

_PATTERNS = {
    # Email addresses — MUST come before handle to avoid eating the @ in emails
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),

    # URLs (http/https) — before handles to catch embedded params
    "url": re.compile(r"https?://\S+", re.IGNORECASE),

    # Twitter handles — @username (after email so emails are caught first)
    "handle": re.compile(r"@[A-Za-z0-9_]{1,50}", re.IGNORECASE),

    # Amazon-style order IDs: 3-digit-7digit-7digit (e.g., 113-1234567-1234567)
    "order_id_amazon": re.compile(r"\b\d{3}-\d{7}-\d{7}\b"),

    # Generic order/reference numbers: 8–20 alphanumeric chars preceded by
    # order/ref/case/ticket keywords
    "order_id_generic": re.compile(
        r"\b(?:order|ref|reference|case|ticket|confirmation|tracking)[\s#:]*[A-Z0-9\-]{6,20}\b",
        re.IGNORECASE,
    ),

    # Phone numbers: various formats (US + international)
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),

    # Long digit strings that look like account/order numbers (10+ digits)
    "digit_string": re.compile(r"\b\d{10,}\b"),

    # Credit card patterns (16 digits, possibly spaced/dashed)
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}

_REPLACEMENTS = {
    "handle": "[USER]",
    "order_id_amazon": "[ORDER_ID]",
    "order_id_generic": "[ORDER_ID]",
    "phone": "[PHONE]",
    "digit_string": "[NUMBER]",
    "email": "[EMAIL]",
    "url": "[URL]",
    "credit_card": "[CARD]",
}


def redact(text: str, patterns: Optional[list[str]] = None) -> str:
    """
    Redact PII from a single text string.

    Args:
        text: Raw tweet text.
        patterns: List of pattern names to apply. None = apply all.

    Returns:
        Redacted text with PII replaced by placeholder tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    active = patterns or list(_PATTERNS.keys())

    for name in active:
        if name in _PATTERNS:
            text = _PATTERNS[name].sub(_REPLACEMENTS[name], text)

    return text.strip()


def redact_batch(texts: list[str], patterns: Optional[list[str]] = None) -> list[str]:
    """
    Redact PII from a list of texts.

    Args:
        texts: List of raw tweet texts.
        patterns: Pattern names to apply. None = all.

    Returns:
        List of redacted texts.
    """
    return [redact(t, patterns) for t in texts]


def redact_dataframe(df, text_columns: list[str], patterns: Optional[list[str]] = None):
    """
    In-place redaction on a DataFrame's text columns.

    Args:
        df: pandas DataFrame.
        text_columns: Column names to redact.
        patterns: Pattern names. None = all.

    Returns:
        DataFrame with redacted columns (modifies in place AND returns).
    """
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(lambda t: redact(t, patterns))
            logger.debug(f"Redacted column '{col}' ({len(df)} rows).")
        else:
            logger.warning(f"Column '{col}' not found in DataFrame.")
    return df


def audit_redaction(original: str, redacted: str) -> dict:
    """
    Compare original and redacted text to count replacements per category.
    Useful for quality-checking the redaction pipeline.

    Returns:
        Dict of {pattern_name: count_replaced}
    """
    stats = {}
    for name, pattern in _PATTERNS.items():
        original_matches = len(pattern.findall(original))
        redacted_matches = len(pattern.findall(redacted))
        stats[name] = original_matches - redacted_matches
    return stats


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "@AmazonHelp my order 114-1234567-9876543 hasn't arrived, call me at 555-867-5309",
        "I emailed support@amazon.com about case REF-ABC123 but no reply",
        "Check https://amazon.com/orders/tracking?id=12345678901234",
    ]
    for s in samples:
        r = redact(s)
        print(f"IN:  {s}")
        print(f"OUT: {r}")
        print()
