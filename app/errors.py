"""
Error handling and retry logic for Anthropic API calls.
Implements exponential backoff with retry-after header support.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from anthropic import (
    APIError,
    APIStatusError,
    RateLimitError,
)

from app.logging import get_logger

logger = get_logger("errors")


class ServiceUnavailableError(Exception):
    """Raised when all retries are exhausted."""

    pass


class RewritingFallbackError(Exception):
    """Raised when rewriting fails and we should fall back to raw query."""

    pass


async def call_anthropic_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    backoff_base: int = 2,
    use_retry_after: bool = True,
    component: str = "unknown",
    **kwargs,
) -> Any:
    """
    Call an Anthropic API function with retry logic.

    Handles:
    - 429 (rate limit): uses retry-after header if available
    - 529 (overloaded): exponential backoff
    - 500 (api error): exponential backoff

    Args:
        func: The async function to call
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (seconds)
        use_retry_after: Whether to use the retry-after header
        component: Name of the calling component (for logging)

    Returns:
        The result of the function call

    Raises:
        ServiceUnavailableError: If all retries are exhausted
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):  # +1 for the initial attempt
        try:
            result = await func(*args, **kwargs)
            return result

        except RateLimitError as e:
            last_error = e
            if attempt >= max_retries:
                break

            # Use retry-after header if available
            wait_time = backoff_base**attempt
            if use_retry_after and hasattr(e, "response") and e.response is not None:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait_time = float(retry_after)

            _log_retry(component, attempt, max_retries, "rate_limit", wait_time)
            await asyncio.sleep(wait_time)

        except APIStatusError as e:
            last_error = e
            if attempt >= max_retries:
                break

            # 529 overloaded or 500 server error
            wait_time = backoff_base**attempt
            error_type = "overloaded" if e.status_code == 529 else "api_error"

            _log_retry(component, attempt, max_retries, error_type, wait_time)
            await asyncio.sleep(wait_time)

        except APIError as e:
            last_error = e
            if attempt >= max_retries:
                break

            wait_time = backoff_base**attempt
            _log_retry(component, attempt, max_retries, "api_error", wait_time)
            await asyncio.sleep(wait_time)

    raise ServiceUnavailableError(
        f"{component}: Anthropic API indisponible après {max_retries} tentatives. Dernière erreur: {last_error}"
    )


def call_anthropic_sync_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    backoff_base: int = 2,
    use_retry_after: bool = True,
    component: str = "unknown",
    **kwargs,
) -> Any:
    """Synchronous version of call_anthropic_with_retry."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return result

        except RateLimitError as e:
            last_error = e
            if attempt >= max_retries:
                break

            wait_time = backoff_base**attempt
            if use_retry_after and hasattr(e, "response") and e.response is not None:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait_time = float(retry_after)

            _log_retry(component, attempt, max_retries, "rate_limit", wait_time)
            time.sleep(wait_time)

        except (APIStatusError, APIError) as e:
            last_error = e
            if attempt >= max_retries:
                break

            wait_time = backoff_base**attempt
            _log_retry(component, attempt, max_retries, "api_error", wait_time)
            time.sleep(wait_time)

    raise ServiceUnavailableError(
        f"{component}: Anthropic API indisponible après {max_retries} tentatives. Dernière erreur: {last_error}"
    )


def _log_retry(component: str, attempt: int, max_retries: int, error_type: str, wait_time: float):
    """Log retry attempt."""
    logger.warning(
        "api_retry",
        component=component,
        attempt=attempt + 1,
        max_retries=max_retries,
        error_type=error_type,
        wait_seconds=round(wait_time, 1),
    )
