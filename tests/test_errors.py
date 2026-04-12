"""Tests for error handling and retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIError, APIStatusError, RateLimitError

from app.errors import (
    EmptyResponseError,
    ServiceUnavailableError,
    call_anthropic_sync_with_retry,
    call_anthropic_with_retry,
    extract_text_from_response,
)

# ── extract_text_from_response ─────────────────────────────────────


class TestExtractTextFromResponse:
    def test_returns_text(self):
        response = MagicMock()
        block = MagicMock()
        block.text = "hello"
        response.content = [block]
        assert extract_text_from_response(response) == "hello"

    def test_empty_content_raises(self):
        response = MagicMock()
        response.content = []
        with pytest.raises(EmptyResponseError):
            extract_text_from_response(response)


# ── Async retry ────────────────────────────────────────────────────


def _make_rate_limit_error(retry_after=None):
    """Helper to create a RateLimitError with optional retry-after header."""
    response = MagicMock()
    response.status_code = 429
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    response.headers = headers
    error = RateLimitError.__new__(RateLimitError)
    error.response = response
    error.message = "rate limited"
    return error


def _make_api_status_error(status_code=529):
    """Helper to create an APIStatusError."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    error = APIStatusError.__new__(APIStatusError)
    error.response = response
    error.status_code = status_code
    error.message = "server error"
    return error


def _make_api_error():
    """Helper to create a generic APIError."""
    error = APIError.__new__(APIError)
    error.message = "generic api error"
    return error


class TestCallAnthropicWithRetry:
    async def test_successful_call(self):
        func = AsyncMock(return_value="success")
        result = await call_anthropic_with_retry(func, max_retries=3, component="test")
        assert result == "success"
        func.assert_called_once()

    async def test_exhausts_retries_rate_limit(self):
        error = _make_rate_limit_error()
        func = AsyncMock(side_effect=error)

        with pytest.raises(ServiceUnavailableError, match="test_component"):
            await call_anthropic_with_retry(func, max_retries=1, backoff_base=0, component="test_component")

        assert func.call_count == 2

    async def test_succeeds_after_rate_limit_retry(self):
        error = _make_rate_limit_error()
        func = AsyncMock(side_effect=[error, "success"])

        result = await call_anthropic_with_retry(func, max_retries=2, backoff_base=0, component="test")
        assert result == "success"
        assert func.call_count == 2

    async def test_rate_limit_uses_retry_after_header(self):
        error = _make_rate_limit_error(retry_after=0)
        func = AsyncMock(side_effect=[error, "ok"])

        with patch("app.errors.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await call_anthropic_with_retry(
                func, max_retries=2, backoff_base=10, use_retry_after=True, component="test"
            )
        assert result == "ok"
        # Should use retry-after value (0) instead of backoff (10**0=1)
        mock_sleep.assert_called_once_with(0.0)

    async def test_api_status_error_retries(self):
        """APIStatusError (529 overloaded) triggers retry."""
        error = _make_api_status_error(529)
        func = AsyncMock(side_effect=[error, "recovered"])

        result = await call_anthropic_with_retry(func, max_retries=2, backoff_base=0, component="test")
        assert result == "recovered"
        assert func.call_count == 2

    async def test_api_status_error_500_retries(self):
        """APIStatusError (500 server error) triggers retry."""
        error = _make_api_status_error(500)
        func = AsyncMock(side_effect=[error, "ok"])

        result = await call_anthropic_with_retry(func, max_retries=2, backoff_base=0, component="test")
        assert result == "ok"

    async def test_api_status_error_exhausted(self):
        """APIStatusError exhausts retries → ServiceUnavailableError."""
        error = _make_api_status_error(529)
        func = AsyncMock(side_effect=error)

        with pytest.raises(ServiceUnavailableError):
            await call_anthropic_with_retry(func, max_retries=1, backoff_base=0, component="test")

    async def test_generic_api_error_retries(self):
        """Generic APIError triggers retry."""
        error = _make_api_error()
        func = AsyncMock(side_effect=[error, "ok"])

        result = await call_anthropic_with_retry(func, max_retries=2, backoff_base=0, component="test")
        assert result == "ok"

    async def test_generic_api_error_exhausted(self):
        """Generic APIError exhausts retries → ServiceUnavailableError."""
        error = _make_api_error()
        func = AsyncMock(side_effect=error)

        with pytest.raises(ServiceUnavailableError):
            await call_anthropic_with_retry(func, max_retries=1, backoff_base=0, component="test")


# ── Sync retry ─────────────────────────────────────────────────────


class TestCallAnthropicSyncWithRetry:
    def test_successful_call(self):
        func = MagicMock(return_value="ok")
        result = call_anthropic_sync_with_retry(func, max_retries=1, component="test")
        assert result == "ok"
        func.assert_called_once()

    def test_rate_limit_retries_and_recovers(self):
        error = _make_rate_limit_error()
        func = MagicMock(side_effect=[error, "recovered"])

        with patch("app.errors.time.sleep"):
            result = call_anthropic_sync_with_retry(func, max_retries=2, backoff_base=0, component="test")
        assert result == "recovered"
        assert func.call_count == 2

    def test_rate_limit_uses_retry_after_header(self):
        error = _make_rate_limit_error(retry_after=0)
        func = MagicMock(side_effect=[error, "ok"])

        with patch("app.errors.time.sleep") as mock_sleep:
            result = call_anthropic_sync_with_retry(
                func, max_retries=2, backoff_base=10, use_retry_after=True, component="test"
            )
        assert result == "ok"
        mock_sleep.assert_called_once_with(0.0)

    def test_rate_limit_exhausted(self):
        error = _make_rate_limit_error()
        func = MagicMock(side_effect=error)

        with (
            patch("app.errors.time.sleep"),
            pytest.raises(ServiceUnavailableError, match="test_sync"),
        ):
            call_anthropic_sync_with_retry(func, max_retries=1, backoff_base=0, component="test_sync")

        assert func.call_count == 2

    def test_api_status_error_retries(self):
        error = _make_api_status_error(529)
        func = MagicMock(side_effect=[error, "ok"])

        with patch("app.errors.time.sleep"):
            result = call_anthropic_sync_with_retry(func, max_retries=2, backoff_base=0, component="test")
        assert result == "ok"

    def test_api_error_exhausted(self):
        error = _make_api_status_error(500)
        func = MagicMock(side_effect=error)

        with (
            patch("app.errors.time.sleep"),
            pytest.raises(ServiceUnavailableError),
        ):
            call_anthropic_sync_with_retry(func, max_retries=1, backoff_base=0, component="test")
