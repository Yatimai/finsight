"""Tests for the Generator module (answer generation with Claude Sonnet)."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import AppConfig
from app.models.generator import GENERATION_SYSTEM_PROMPT, Generator


@dataclass
class FakePage:
    """Minimal stand-in for RetrievedPage."""

    point_id: int = 1
    document_id: str = "doc1"
    source_filename: str = "rapport_annuel.pdf"
    page_number: int = 12
    total_pages: int = 50
    image_path: str = "/tmp/fake_page.png"
    score: float = 0.95


@pytest.fixture
def config():
    return AppConfig()


@pytest.fixture
def generator(config):
    return Generator(config, client=AsyncMock())


def _make_response(text="Réponse test [Page 12].", input_tokens=100, output_tokens=50):
    """Create a mock Anthropic response."""
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    response.content = [content_block]
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    return response


# ── _extract_citations ──────────────────────────────────────────────


class TestExtractCitations:
    def test_single_citation(self, generator):
        answer = "Le CA est de 86M [Page 12]."
        result = generator._extract_citations(answer)
        assert result == [{"page": 12}]

    def test_multiple_citations_deduped(self, generator):
        answer = "CA [Page 12] en hausse [Page 14]. Confirmé [Page 12]."
        result = generator._extract_citations(answer)
        assert result == [{"page": 12}, {"page": 14}]

    def test_no_citations_returns_empty(self, generator):
        answer = "Cette information n'apparaît pas dans les documents fournis."
        result = generator._extract_citations(answer)
        assert result == []

    def test_malformed_citation_ignored(self, generator):
        """[Page abc] is not a valid citation and should be ignored."""
        answer = "Voir [Page abc] pour les détails [Page 5]."
        result = generator._extract_citations(answer)
        assert result == [{"page": 5}]


# ── _build_content ──────────────────────────────────────────────────


class TestBuildContent:
    def test_content_has_page_images(self, generator):
        pages = [FakePage(page_number=5), FakePage(page_number=12)]
        with patch.object(generator, "_encode_image", return_value="base64data"):
            content = generator._build_content("Quel est le CA ?", pages)
        types = [c["type"] for c in content]
        assert types.count("image") == 2
        assert "QUESTION" in content[-1]["text"]

    def test_content_with_conversation_history(self, generator):
        history = [{"question": "CA 2023 ?", "answer": "86M euros."}]
        with patch.object(generator, "_encode_image", return_value="base64data"):
            content = generator._build_content("Et en 2022 ?", [FakePage()], history)
        texts = [c["text"] for c in content if c["type"] == "text"]
        joined = "\n".join(texts)
        assert "CONTEXTE DE CONVERSATION" in joined
        assert "CA 2023" in joined

    def test_conversation_truncated_to_3_exchanges(self, generator):
        history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        with patch.object(generator, "_encode_image", return_value="base64data"):
            content = generator._build_content("Next?", [FakePage()], history)
        texts = [c["text"] for c in content if c["type"] == "text"]
        joined = "\n".join(texts)
        # Only last 3 exchanges should be included
        assert "Q7" in joined
        assert "Q8" in joined
        assert "Q9" in joined
        assert "Q0" not in joined

    def test_image_encoding_failure_skips_page(self, generator):
        """When _encode_image returns None, page is skipped (no image block)."""
        pages = [FakePage(page_number=5)]
        with patch.object(generator, "_encode_image", return_value=None):
            content = generator._build_content("Q?", pages)
        types = [c["type"] for c in content]
        assert "image" not in types


# ── generate ────────────────────────────────────────────────────────


class TestGenerate:
    async def test_generate_returns_answer_and_citations(self, generator):
        mock_response = _make_response("Le CA est de 86M [Page 12].")
        with (
            patch("app.models.generator.call_anthropic_with_retry", return_value=mock_response),
            patch.object(generator, "_encode_image", return_value="base64"),
        ):
            result = await generator.generate("CA ?", [FakePage()])

        assert "answer" in result
        assert result["answer"] == "Le CA est de 86M [Page 12]."
        assert result["citations"] == [{"page": 12}]

    async def test_generate_records_token_usage(self, generator):
        mock_response = _make_response(input_tokens=200, output_tokens=75)
        with (
            patch("app.models.generator.call_anthropic_with_retry", return_value=mock_response),
            patch.object(generator, "_encode_image", return_value="base64"),
        ):
            result = await generator.generate("Q?", [FakePage()])

        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 75

    async def test_generate_with_cache_read_tokens(self, generator):
        mock_response = _make_response()
        mock_response.usage.cache_read_input_tokens = 1500
        mock_response.usage.cache_creation_input_tokens = 500
        with (
            patch("app.models.generator.call_anthropic_with_retry", return_value=mock_response),
            patch.object(generator, "_encode_image", return_value="base64"),
        ):
            result = await generator.generate("Q?", [FakePage()])

        assert result["cache_read_tokens"] == 1500
        assert result["cache_creation_tokens"] == 500

    async def test_generate_calls_correct_model(self, generator, config):
        mock_response = _make_response()
        with (
            patch("app.models.generator.call_anthropic_with_retry", return_value=mock_response) as mock_retry,
            patch.object(generator, "_encode_image", return_value="base64"),
        ):
            await generator.generate("Q?", [FakePage()])

        # The _api_call function passed to call_anthropic_with_retry
        mock_retry.assert_called_once()
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["component"] == "generator"

    async def test_system_prompt_has_cache_control(self, generator):
        """The system prompt must use cache_control for prompt caching."""
        mock_response = _make_response()
        captured_kwargs = {}

        async def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        generator.client.messages.create = capture_create
        with patch("app.models.generator.call_anthropic_with_retry") as mock_retry:
            # Make retry call the actual function
            async def call_fn(fn, *args, **kwargs):
                return await fn()

            mock_retry.side_effect = call_fn
            with patch.object(generator, "_encode_image", return_value="base64"):
                await generator.generate("Q?", [FakePage()])

        assert "system" in captured_kwargs
        system = captured_kwargs["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}


# ── System prompt content ───────────────────────────────────────────


class TestSystemPrompt:
    def test_prompt_is_french(self):
        assert "analyste financier" in GENERATION_SYSTEM_PROMPT

    def test_prompt_forbids_external_knowledge(self):
        assert "jamais inventer" in GENERATION_SYSTEM_PROMPT

    def test_prompt_requires_citations(self):
        assert "[Page X]" in GENERATION_SYSTEM_PROMPT
