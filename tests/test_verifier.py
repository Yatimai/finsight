"""Tests for the Verifier module."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.verifier import VERIFICATION_SYSTEM_PROMPT, Verifier


@dataclass
class FakePage:
    """Minimal stand-in for RetrievedPage (avoids PIL/torch deps)."""

    point_id: int = 1
    document_id: str = "doc1"
    source_filename: str = "walmart_10k.pdf"
    page_number: int = 12
    total_pages: int = 50
    image_path: str = "/tmp/fake_page.png"
    score: float = 0.95


@pytest.fixture
def verifier(config):
    with patch("app.models.verifier.anthropic"):
        v = Verifier(config, client=AsyncMock())
        return v


# ── _parse_verification ─────────────────────────────────────────


class TestParseVerification:
    def test_basic_confirmed(self, verifier):
        """Valid JSON with all claims CONFIRMÉ → status 'verified'."""
        text = json.dumps(
            {
                "claims": [
                    {
                        "id": 1,
                        "claim": "Revenue is $648B",
                        "verdict": "CONFIRMÉ",
                        "evidence": "Page 12",
                        "correction": None,
                    }
                ],
                "confidence": 0.95,
                "summary": "All claims confirmed.",
            }
        )
        result = verifier._parse_verification(text)
        assert result["status"] == "verified"
        assert result["confidence"] == 0.95
        assert result["claims_verified"] == 1
        assert result["claims_contradicted"] == 0
        assert result["claims_not_found"] == 0

    def test_contradicted_flags(self, verifier):
        """A single CONTREDIT claim → status 'flagged'."""
        text = json.dumps(
            {
                "claims": [
                    {
                        "id": 1,
                        "claim": "Revenue is $648B",
                        "verdict": "CONFIRMÉ",
                        "evidence": "Page 12",
                        "correction": None,
                    },
                    {
                        "id": 2,
                        "claim": "Net income is $10B",
                        "verdict": "CONTREDIT",
                        "evidence": "Page 15",
                        "correction": "$8B",
                    },
                ],
                "confidence": 0.6,
                "summary": "One contradiction found.",
            }
        )
        result = verifier._parse_verification(text)
        assert result["status"] == "flagged"
        assert result["claims_verified"] == 1
        assert result["claims_contradicted"] == 1

    def test_code_fence_json(self, verifier):
        """JSON wrapped in ```json ``` code fences → parse OK."""
        inner = json.dumps(
            {
                "claims": [
                    {"id": 1, "claim": "Q4 revenue", "verdict": "CONFIRMÉ", "evidence": "p3", "correction": None},
                ],
                "confidence": 0.9,
                "summary": "OK",
            }
        )
        text = f"```json\n{inner}\n```"
        result = verifier._parse_verification(text)
        assert result["status"] == "verified"
        assert result["confidence"] == 0.9
        assert result["claims_verified"] == 1

    def test_invalid_json_parse_error(self, verifier):
        """Non-JSON text → status 'parse_error'."""
        result = verifier._parse_verification("This is not JSON at all")
        assert result["status"] == "parse_error"
        assert result["confidence"] == 0.0
        assert result["claims"] == []

    def test_majority_not_found_flags(self, verifier):
        """More than 50% NON TROUVÉ → status 'flagged'."""
        text = json.dumps(
            {
                "claims": [
                    {"id": 1, "claim": "A", "verdict": "NON TROUVÉ", "evidence": "", "correction": None},
                    {"id": 2, "claim": "B", "verdict": "NON TROUVÉ", "evidence": "", "correction": None},
                    {"id": 3, "claim": "C", "verdict": "CONFIRMÉ", "evidence": "p1", "correction": None},
                ],
                "confidence": 0.5,
                "summary": "Mostly not found.",
            }
        )
        result = verifier._parse_verification(text)
        assert result["status"] == "flagged"
        assert result["claims_not_found"] == 2

    def test_low_confidence_status(self, verifier):
        """All confirmed but confidence below threshold → 'low_confidence'."""
        text = json.dumps(
            {
                "claims": [
                    {"id": 1, "claim": "X", "verdict": "CONFIRMÉ", "evidence": "p1", "correction": None},
                ],
                "confidence": 0.5,
                "summary": "Low confidence.",
            }
        )
        result = verifier._parse_verification(text)
        assert result["status"] == "low_confidence"
        assert result["confidence"] == 0.5


# ── _disabled_result ─────────────────────────────────────────────


class TestDisabledResult:
    def test_disabled_result(self, verifier):
        result = verifier._disabled_result()
        assert result["status"] == "disabled"
        assert result["confidence"] is None
        assert result["claims"] == []
        assert result["summary"] == "Verification disabled"


# ── should_abstain ───────────────────────────────────────────────


class TestShouldAbstain:
    def test_below_threshold_abstains(self, verifier):
        """confidence < 0.7 → should abstain."""
        assert verifier.should_abstain({"confidence": 0.5}) is True
        assert verifier.should_abstain({"confidence": 0.69}) is True

    def test_at_threshold_does_not_abstain(self, verifier):
        """confidence >= 0.7 → should NOT abstain."""
        assert verifier.should_abstain({"confidence": 0.7}) is False
        assert verifier.should_abstain({"confidence": 0.95}) is False

    def test_missing_confidence_defaults_high(self, verifier):
        """No confidence key → defaults to 1.0, no abstention."""
        assert verifier.should_abstain({}) is False


# ── _build_verification_content ──────────────────────────────────


class TestBuildVerificationContent:
    def test_structure_no_pages(self, verifier):
        """With no encodable pages, content has header + question block."""
        with patch.object(verifier, "_encode_image", return_value=None):
            content = verifier._build_verification_content("Q?", "A.", [FakePage()])
        # First block: DOCUMENTS SOURCES header
        assert content[0]["type"] == "text"
        assert "DOCUMENTS SOURCES" in content[0]["text"]
        # Last block: QUESTION + RÉPONSE
        last = content[-1]
        assert last["type"] == "text"
        assert "Q?" in last["text"]
        assert "A." in last["text"]

    def test_structure_with_image(self, verifier):
        """With an encodable page, content includes image block."""
        with patch.object(verifier, "_encode_image", return_value="BASE64DATA"):
            content = verifier._build_verification_content("Q?", "A.", [FakePage()])
        types = [c["type"] for c in content]
        assert "image" in types
        img_block = next(c for c in content if c["type"] == "image")
        assert img_block["source"]["data"] == "BASE64DATA"
        assert img_block["source"]["media_type"] == "image/png"


# ── verify (disabled / sync) ────────────────────────────────────


class TestVerify:
    async def test_verify_disabled_returns_disabled(self, config):
        """When mode='disabled', verify() returns disabled result immediately."""
        config.verification.mode = "disabled"
        with patch("app.models.verifier.anthropic"):
            v = Verifier(config, client=AsyncMock())
            result = await v.verify("Q?", "A.", [])
        assert result["status"] == "disabled"

    async def test_verify_not_enabled_returns_disabled(self, config):
        """When enabled=False, verify() returns disabled result."""
        config.verification.enabled = False
        with patch("app.models.verifier.anthropic"):
            v = Verifier(config, client=AsyncMock())
            result = await v.verify("Q?", "A.", [])
        assert result["status"] == "disabled"

    async def test_verify_empty_content_returns_error(self, verifier):
        """Bug 3 regression: empty response.content must not raise IndexError."""
        # Mock response with empty content list
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage = MagicMock()

        with patch("app.models.verifier.call_anthropic_with_retry", return_value=mock_response):
            result = await verifier.verify("Q?", "A.", [FakePage()])

        assert result["status"] == "error"
        assert result["confidence"] is None


# ── submit_batch (batch_async) ───────────────────────────────────


class TestBatchAsync:
    async def test_submit_batch_constructs_request(self, config):
        """submit_batch() calls messages.batches.create with correct structure."""
        with patch("app.models.verifier.anthropic") as mock_anthropic:
            mock_sync = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_sync
            mock_sync.messages.batches.create.return_value = MagicMock(id="batch_abc123")

            v = Verifier(config, client=AsyncMock())
            with patch.object(v, "_encode_image", return_value=None):
                batch_id = await v.submit_batch("q-1", "What is revenue?", "Revenue is $10B.", [FakePage()])

        assert batch_id == "batch_abc123"
        call_args = mock_sync.messages.batches.create.call_args
        requests = call_args.kwargs["requests"]
        assert len(requests) == 1
        assert requests[0]["custom_id"] == "q-1"
        params = requests[0]["params"]
        assert params["model"] == config.verification.model
        assert params["temperature"] == 0.0
        assert params["max_tokens"] == 2048

    async def test_submit_batch_failure_returns_none(self, config):
        """submit_batch() returns None on API error."""
        with patch("app.models.verifier.anthropic") as mock_anthropic:
            mock_sync = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_sync
            mock_sync.messages.batches.create.side_effect = Exception("API down")

            v = Verifier(config, client=AsyncMock())
            with patch.object(v, "_encode_image", return_value=None):
                batch_id = await v.submit_batch("q-1", "Q?", "A.", [FakePage()])

        assert batch_id is None

    async def test_disabled_mode_no_submit(self, config):
        """When mode='disabled', verify() returns immediately, no batch submit."""
        config.verification.mode = "disabled"
        with patch("app.models.verifier.anthropic") as mock_anthropic:
            mock_sync = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_sync

            v = Verifier(config, client=AsyncMock())
            result = await v.verify("Q?", "A.", [])

        assert result["status"] == "disabled"
        mock_sync.messages.batches.create.assert_not_called()


# ── System prompt content ────────────────────────────────────────


class TestSystemPrompt:
    def test_multi_locale_instruction_present(self):
        """Regression: system prompt must include numeric normalization rules."""
        assert "FORMATS NUMÉRIQUES" in VERIFICATION_SYSTEM_PROMPT
        assert "normalise" in VERIFICATION_SYSTEM_PROMPT
        assert "CONTREDIT" in VERIFICATION_SYSTEM_PROMPT
