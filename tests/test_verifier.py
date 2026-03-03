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
        """No confidence key → defaults to None, no abstention."""
        assert verifier.should_abstain({}) is False

    def test_confidence_none_does_not_abstain(self, verifier):
        """Regression: confidence=None (from _error_result) must not raise TypeError."""
        assert verifier.should_abstain({"confidence": None}) is False

    def test_status_error_does_not_abstain(self, verifier):
        """Regression: status=error → don't discard the answer."""
        assert verifier.should_abstain({"status": "error", "confidence": None}) is False

    def test_disabled_result_does_not_abstain(self, verifier):
        """_disabled_result has confidence=None → should not abstain."""
        assert verifier.should_abstain(verifier._disabled_result()) is False


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
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage = MagicMock()

        with patch("app.models.verifier.call_anthropic_with_retry", return_value=mock_response):
            result = await verifier.verify("Q?", "A.", [FakePage()])

        assert result["status"] == "error"
        assert result["confidence"] is None

    async def test_verify_success_returns_parsed_result(self, verifier):
        """Successful verify() returns parsed claims with token usage."""
        verification_json = json.dumps(
            {
                "claims": [
                    {"id": 1, "claim": "Revenue is $10B", "verdict": "CONFIRMÉ", "evidence": "p12", "correction": None}
                ],
                "confidence": 0.95,
                "summary": "All confirmed.",
            }
        )
        mock_response = MagicMock()
        block = MagicMock()
        block.text = verification_json
        mock_response.content = [block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 200
        mock_response.usage.cache_read_input_tokens = 100

        with (
            patch("app.models.verifier.call_anthropic_with_retry", return_value=mock_response),
            patch.object(verifier, "_encode_image", return_value=None),
        ):
            result = await verifier.verify("Q?", "A.", [FakePage()])

        assert result["status"] == "verified"
        assert result["confidence"] == 0.95
        assert result["claims_verified"] == 1
        assert result["input_tokens"] == 500
        assert result["output_tokens"] == 200
        assert result["cache_read_tokens"] == 100

    async def test_verify_api_error_returns_error_result(self, verifier):
        """API failure in verify() returns error result after exhausting all models."""
        from app.errors import ServiceUnavailableError

        with (
            patch(
                "app.models.verifier.call_anthropic_with_retry",
                side_effect=ServiceUnavailableError("API down"),
            ),
            patch.object(verifier, "_encode_image", return_value=None),
        ):
            result = await verifier.verify("Q?", "A.", [FakePage()])

        assert result["status"] == "error"
        assert "indisponibles" in result["summary"]


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


# ── poll_batch ───────────────────────────────────────────────────


class TestPollBatch:
    async def test_poll_batch_success(self, verifier):
        """poll_batch returns parsed verification on successful batch."""
        verification_json = json.dumps(
            {
                "claims": [{"id": 1, "claim": "X", "verdict": "CONFIRMÉ", "evidence": "p1", "correction": None}],
                "confidence": 0.9,
                "summary": "OK",
            }
        )

        # Mock batch retrieve → ended
        mock_batch = MagicMock()
        mock_batch.processing_status = "ended"
        verifier.sync_client.messages.batches.retrieve.return_value = mock_batch

        # Mock batch results stream
        mock_entry = MagicMock()
        mock_entry.custom_id = "q-1"
        mock_entry.result.type = "succeeded"
        content_block = MagicMock()
        content_block.text = verification_json
        mock_entry.result.message.content = [content_block]
        mock_entry.result.message.usage.input_tokens = 300
        mock_entry.result.message.usage.output_tokens = 150
        verifier.sync_client.messages.batches.results.return_value = [mock_entry]

        result = await verifier.poll_batch("batch_123", "q-1")

        assert result["status"] == "verified"
        assert result["confidence"] == 0.9
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 150
        assert result["batch_id"] == "batch_123"

    async def test_poll_batch_empty_content(self, verifier):
        """poll_batch with empty content returns error (Bug 3)."""
        mock_batch = MagicMock()
        mock_batch.processing_status = "ended"
        verifier.sync_client.messages.batches.retrieve.return_value = mock_batch

        mock_entry = MagicMock()
        mock_entry.custom_id = "q-1"
        mock_entry.result.type = "succeeded"
        mock_entry.result.message.content = []
        verifier.sync_client.messages.batches.results.return_value = [mock_entry]

        result = await verifier.poll_batch("batch_123", "q-1")
        assert result["status"] == "error"
        assert "Empty content" in result["summary"]

    async def test_poll_batch_request_failed(self, verifier):
        """poll_batch with failed request returns error."""
        mock_batch = MagicMock()
        mock_batch.processing_status = "ended"
        verifier.sync_client.messages.batches.retrieve.return_value = mock_batch

        mock_entry = MagicMock()
        mock_entry.custom_id = "q-1"
        mock_entry.result.type = "errored"
        verifier.sync_client.messages.batches.results.return_value = [mock_entry]

        result = await verifier.poll_batch("batch_123", "q-1")
        assert result["status"] == "error"
        assert "failed" in result["summary"]

    async def test_poll_batch_query_id_not_found(self, verifier):
        """poll_batch with missing query_id returns error."""
        mock_batch = MagicMock()
        mock_batch.processing_status = "ended"
        verifier.sync_client.messages.batches.retrieve.return_value = mock_batch

        mock_entry = MagicMock()
        mock_entry.custom_id = "other-query"
        mock_entry.result.type = "succeeded"
        verifier.sync_client.messages.batches.results.return_value = [mock_entry]

        result = await verifier.poll_batch("batch_123", "q-1")
        assert result["status"] == "error"
        assert "not found" in result["summary"]

    async def test_poll_batch_waits_for_processing(self, verifier):
        """poll_batch polls until batch is ended."""
        batch_processing = MagicMock()
        batch_processing.processing_status = "in_progress"
        batch_ended = MagicMock()
        batch_ended.processing_status = "ended"
        verifier.sync_client.messages.batches.retrieve.side_effect = [batch_processing, batch_ended]

        verification_json = json.dumps(
            {
                "claims": [],
                "confidence": 0.8,
                "summary": "OK",
            }
        )
        mock_entry = MagicMock()
        mock_entry.custom_id = "q-1"
        mock_entry.result.type = "succeeded"
        block = MagicMock()
        block.text = verification_json
        mock_entry.result.message.content = [block]
        mock_entry.result.message.usage.input_tokens = 100
        mock_entry.result.message.usage.output_tokens = 50
        verifier.sync_client.messages.batches.results.return_value = [mock_entry]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await verifier.poll_batch("batch_123", "q-1")

        assert result["status"] == "verified"
        assert verifier.sync_client.messages.batches.retrieve.call_count == 2

    async def test_poll_batch_poll_error_recovers(self, verifier):
        """poll_batch recovers from transient errors during polling."""
        batch_error_then_ended = [
            Exception("connection reset"),
            MagicMock(processing_status="ended"),
        ]
        verifier.sync_client.messages.batches.retrieve.side_effect = batch_error_then_ended

        verification_json = json.dumps(
            {
                "claims": [],
                "confidence": 0.85,
                "summary": "OK",
            }
        )
        mock_entry = MagicMock()
        mock_entry.custom_id = "q-1"
        mock_entry.result.type = "succeeded"
        block = MagicMock()
        block.text = verification_json
        mock_entry.result.message.content = [block]
        mock_entry.result.message.usage.input_tokens = 100
        mock_entry.result.message.usage.output_tokens = 50
        verifier.sync_client.messages.batches.results.return_value = [mock_entry]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await verifier.poll_batch("batch_123", "q-1")

        assert result["status"] == "verified"


# ── System prompt content ────────────────────────────────────────


# ── verify() fallback models ─────────────────────────────────────


class TestVerifyFallback:
    async def test_fallback_model_used_on_primary_failure(self, verifier):
        """When primary model fails, fallback model is tried and succeeds."""
        verifier.config.verification.fallback_models = ["claude-fallback"]

        verification_json = json.dumps(
            {
                "claims": [{"id": 1, "claim": "X", "verdict": "CONFIRMÉ", "evidence": "p1", "correction": None}],
                "confidence": 0.9,
                "summary": "OK via fallback.",
            }
        )
        mock_response = MagicMock()
        block = MagicMock()
        block.text = verification_json
        mock_response.content = [block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 400
        mock_response.usage.output_tokens = 150
        mock_response.usage.cache_read_input_tokens = 0

        call_count = 0

        async def side_effect(fn, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("529 overloaded")
            return await fn()

        with (
            patch("app.models.verifier.call_anthropic_with_retry", side_effect=side_effect),
            patch.object(verifier, "_encode_image", return_value=None),
        ):
            # Make the async client return the mock response for fallback call
            verifier.client.messages.create.return_value = mock_response
            result = await verifier.verify("Q?", "A.", [FakePage()])

        assert result["status"] == "verified"
        assert result["confidence"] == 0.9
        assert result["model_used"] == "claude-fallback"

    async def test_all_models_fail_returns_error(self, verifier):
        """When all models fail, returns error result."""
        verifier.config.verification.fallback_models = ["claude-fallback"]

        with (
            patch(
                "app.models.verifier.call_anthropic_with_retry",
                side_effect=Exception("all down"),
            ),
            patch.object(verifier, "_encode_image", return_value=None),
        ):
            result = await verifier.verify("Q?", "A.", [FakePage()])

        assert result["status"] == "error"
        assert result["confidence"] is None
        assert "indisponibles" in result["summary"]

    async def test_primary_model_success_no_fallback(self, verifier):
        """When primary model succeeds, model_used is the primary model."""
        verification_json = json.dumps(
            {
                "claims": [],
                "confidence": 0.85,
                "summary": "OK",
            }
        )
        mock_response = MagicMock()
        block = MagicMock()
        block.text = verification_json
        mock_response.content = [block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 300
        mock_response.usage.output_tokens = 100
        mock_response.usage.cache_read_input_tokens = 0

        with (
            patch("app.models.verifier.call_anthropic_with_retry", return_value=mock_response),
            patch.object(verifier, "_encode_image", return_value=None),
        ):
            result = await verifier.verify("Q?", "A.", [FakePage()])

        assert result["model_used"] == verifier.config.verification.model
        assert result["status"] == "verified"


# ── System prompt content ────────────────────────────────────────


class TestSystemPrompt:
    def test_multi_locale_instruction_present(self):
        """Regression: system prompt must include numeric normalization rules."""
        assert "FORMATS NUMÉRIQUES" in VERIFICATION_SYSTEM_PROMPT
        assert "normalise" in VERIFICATION_SYSTEM_PROMPT
        assert "CONTREDIT" in VERIFICATION_SYSTEM_PROMPT
