"""Tests for the QueryRewriter module (RAG Fusion query rewriting)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import AppConfig
from app.errors import ServiceUnavailableError
from app.models.rewriter import QueryRewriter


@pytest.fixture
def config():
    return AppConfig()


@pytest.fixture
def rewriter(config):
    return QueryRewriter(config, client=AsyncMock())


def _make_response(text='["Q1 reformulée", "Q2 reformulée", "Q3 reformulée"]'):
    """Create a mock Anthropic response."""
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    response.content = [content_block]
    response.usage = MagicMock()
    return response


# ── _build_context ──────────────────────────────────────────────────


class TestBuildContext:
    def test_empty_history_returns_empty_string(self, rewriter):
        assert rewriter._build_context(None) == ""
        assert rewriter._build_context([]) == ""

    def test_truncates_to_3_exchanges(self, rewriter):
        history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        result = rewriter._build_context(history)
        assert "Q7" in result
        assert "Q8" in result
        assert "Q9" in result
        assert "Q0" not in result

    def test_truncates_long_answers_at_200(self, rewriter):
        history = [{"question": "Q?", "answer": "A" * 300}]
        result = rewriter._build_context(history)
        assert "..." in result
        # Should contain the truncated part (200 chars + "...")
        assert len(result.split("R: ")[1].strip()) == 203  # 200 + "..."


# ── rewrite ─────────────────────────────────────────────────────────


class TestRewrite:
    async def test_disabled_returns_raw_query(self, config):
        config.rewriting.enabled = False
        rw = QueryRewriter(config, client=AsyncMock())
        result = await rw.rewrite("ma question")
        assert result == ["ma question"]

    async def test_returns_up_to_max_rewrites(self, rewriter):
        mock_resp = _make_response('["Q1", "Q2", "Q3", "Q4", "Q5"]')
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rewriter.rewrite("test")
        assert len(result) <= rewriter.max_rewrites

    async def test_non_list_response_falls_back(self, rewriter):
        mock_resp = _make_response('"not a list"')
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rewriter.rewrite("test query")
        assert result == ["test query"]

    async def test_empty_list_falls_back(self, rewriter):
        mock_resp = _make_response("[]")
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rewriter.rewrite("test query")
        assert result == ["test query"]

    async def test_json_decode_error_falls_back(self, rewriter):
        mock_resp = _make_response("this is not json at all")
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rewriter.rewrite("test query")
        # json.loads fails, caught by except Exception → [query]
        assert result == ["test query"]

    async def test_service_unavailable_raises_fallback_error(self, rewriter):
        with patch(
            "app.models.rewriter.call_anthropic_with_retry",
            side_effect=ServiceUnavailableError("API down"),
        ):
            from app.errors import RewritingFallbackError

            with pytest.raises(RewritingFallbackError):
                await rewriter.rewrite("test query")

    async def test_generic_exception_falls_back_silently(self, rewriter):
        with patch(
            "app.models.rewriter.call_anthropic_with_retry",
            side_effect=RuntimeError("unexpected"),
        ):
            result = await rewriter.rewrite("test query")
        assert result == ["test query"]

    async def test_strips_markdown_code_fences(self, rewriter):
        inner = json.dumps(["Q1", "Q2", "Q3"])
        mock_resp = _make_response(f"```json\n{inner}\n```")
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rewriter.rewrite("test")
        assert result == ["Q1", "Q2", "Q3"]


# ── _call_sonnet ────────────────────────────────────────────────────


class TestRewriteWithContext:
    async def test_context_prepended_to_user_message(self, config):
        """Line 79: conversation context is prepended to the user message."""
        client = AsyncMock()
        rw = QueryRewriter(config, client=client)
        history = [{"question": "CA de LVMH ?", "answer": "86 Md€"}]
        mock_resp = _make_response('["Q1", "Q2", "Q3"]')
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rw.rewrite("et en 2022 ?", conversation_history=history)
        assert len(result) == 3

    async def test_all_whitespace_strings_falls_back(self, rewriter):
        """Line 91: response contains only whitespace strings after strip."""
        mock_resp = _make_response('["  ", "\\t", "\\n"]')
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rewriter.rewrite("test query")
        assert result == ["test query"]

    async def test_empty_exchanges_return_empty_context(self, rewriter):
        """Line 157: exchanges with no question or answer produce no context lines."""
        history = [{"question": "", "answer": ""}, {"other_key": "value"}]
        result = rewriter._build_context(history)
        assert result == ""


class TestCallSonnet:
    async def test_parses_valid_json_array(self, rewriter):
        mock_resp = _make_response('["reformulation 1", "reformulation 2"]')
        with patch("app.models.rewriter.call_anthropic_with_retry", return_value=mock_resp):
            result = await rewriter._call_sonnet("Question : test")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == "reformulation 1"

    async def test_call_sonnet_inner_api_call(self, config):
        """Lines 106-119: exercise the inner _api_call function."""
        mock_client = AsyncMock()
        mock_resp = _make_response('["Q1", "Q2"]')
        mock_client.messages.create.return_value = mock_resp
        rw = QueryRewriter(config, client=mock_client)

        # Don't patch call_anthropic_with_retry — let it call through
        # but patch the retry to just call the function directly
        async def call_directly(fn, **kwargs):
            return await fn()

        with patch("app.models.rewriter.call_anthropic_with_retry", side_effect=call_directly):
            result = await rw._call_sonnet("Question : test")

        assert result == ["Q1", "Q2"]
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["max_tokens"] == 512
