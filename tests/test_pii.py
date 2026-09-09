"""
tests/test_pii.py
──────────────────
Unit tests for PII redaction.

Run: pytest tests/test_pii.py -v
"""

import pytest
from src.data.pii import redact, audit_redaction


class TestHandleRedaction:
    def test_basic_handle(self):
        assert redact("@AmazonHelp my order is late") == "[USER] my order is late"

    def test_multiple_handles(self):
        result = redact("@user1 and @user2 both complained to @AmazonHelp")
        assert "[USER]" in result
        assert "@user1" not in result
        assert "@user2" not in result

    def test_no_handle(self):
        text = "my package is missing"
        assert redact(text) == text


class TestOrderIdRedaction:
    def test_amazon_order_id(self):
        result = redact("my order 114-1234567-9876543 hasn't arrived")
        assert "[ORDER_ID]" in result
        assert "114-1234567-9876543" not in result

    def test_generic_order_ref(self):
        result = redact("case REF-ABC12345 was opened")
        assert "[ORDER_ID]" in result


class TestPhoneRedaction:
    def test_us_phone(self):
        result = redact("call me at 555-867-5309")
        assert "[PHONE]" in result
        assert "555-867-5309" not in result

    def test_formatted_phone(self):
        result = redact("(800) 123-4567")
        assert "[PHONE]" in result


class TestEmailRedaction:
    def test_email(self):
        result = redact("email me at john@example.com")
        assert "[EMAIL]" in result
        assert "john@example.com" not in result


class TestUrlRedaction:
    def test_url(self):
        result = redact("check https://amazon.com/orders/123")
        assert "[URL]" in result


class TestAudit:
    def test_audit_counts_matches(self):
        original = "@AmazonHelp order 114-1234567-9876543"
        redacted = redact(original)
        audit = audit_redaction(original, redacted)
        assert audit["handle"] == 1
        assert audit["order_id_amazon"] == 1


class TestEdgeCases:
    def test_empty_string(self):
        assert redact("") == ""

    def test_none_safe(self):
        # Should not raise, return as-is (or empty)
        result = redact(None)
        assert result is None or result == ""

    def test_clean_message_unchanged(self):
        text = "my package is missing and I am very upset"
        assert redact(text) == text
