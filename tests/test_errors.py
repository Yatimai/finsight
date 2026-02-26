"""Tests for error handling and retry logic."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import RateLimitError

from app.errors import (
    ServiceUnavailableError,
    call_anthropic_with_retry,
)


class TestCallAnthropicWithRetry:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        func = AsyncMock(return_value="success")
        result = await call_anthropic_with_retry(func, max_retries=3, component="test")
        assert result == "success"
        func.assert_called_once()

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        response = MagicMock()
        response.headers = {}
        response.status_code = 429
        error = RateLimitError.__new__(RateLimitError)
        error.response = response
        error.message = "rate limited"

        func = AsyncMock(side_effect=error)

        with pytest.raises(ServiceUnavailableError, match="test_component"):
            await call_anthropic_with_retry(func, max_retries=1, backoff_base=0, component="test_component")

        # Initial attempt + 1 retry = 2 calls
        assert func.call_count == 2

    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self):
        response = MagicMock()
        response.headers = {}
        response.status_code = 429
        error = RateLimitError.__new__(RateLimitError)
        error.response = response
        error.message = "rate limited"

        func = AsyncMock(side_effect=[error, "success"])

        result = await call_anthropic_with_retry(func, max_retries=2, backoff_base=0, component="test")
        assert result == "success"
        assert func.call_count == 2
