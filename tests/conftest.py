"""Shared fixtures for all test modules."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import AppConfig


@dataclass
class FakePage:
    """Minimal stand-in for RetrievedPage (avoids PIL/torch deps)."""

    point_id: int = 1
    document_id: str = "doc1"
    source_filename: str = "rapport_annuel.pdf"
    page_number: int = 12
    total_pages: int = 50
    image_path: str = "/tmp/fake_page.png"
    score: float = 0.95


@pytest.fixture
def config():
    """Default AppConfig for tests."""
    return AppConfig()


@pytest.fixture
def fake_pages():
    """Three FakePages with distinct page numbers."""
    return [
        FakePage(point_id=1, page_number=5, score=0.98),
        FakePage(point_id=2, page_number=12, score=0.92),
        FakePage(point_id=3, page_number=23, score=0.87),
    ]


@pytest.fixture
def mock_anthropic_response():
    """Factory that creates a mock Anthropic response with .content[0].text and .usage."""

    def _factory(text="Réponse test.", input_tokens=100, output_tokens=50):
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

    return _factory


@pytest.fixture
def mock_empty_anthropic_response():
    """Mock Anthropic response with empty content list (Bug 3 scenario)."""
    response = MagicMock()
    response.content = []
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 0
    return response


@pytest.fixture
def mock_anthropic_client():
    """AsyncMock Anthropic client."""
    return AsyncMock()
