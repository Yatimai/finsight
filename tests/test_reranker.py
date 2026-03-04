"""Tests for the VisualReranker and its pipeline integration."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from app.config import AppConfig, RerankingConfig
from app.models.reranker import VisualReranker
from app.pipeline import Pipeline


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
    image: object = None

    def load_image(self):
        self.image = "fake_image"
        return self.image


# ── VisualReranker unit tests ──────────────────────────────────────


class TestVisualReranker:
    def test_rerank_reorders_by_score(self):
        """Pages should be reordered by reranking score (desc)."""
        config = RerankingConfig(enabled=True)
        reranker = VisualReranker(config)

        pages = [
            FakePage(page_number=1, score=0.9),
            FakePage(page_number=2, score=0.5),
            FakePage(page_number=3, score=0.7),
        ]

        # Mock _score_single to return known scores
        scores = {1: 0.3, 2: 0.9, 3: 0.6}
        reranker._score_single = MagicMock(
            side_effect=lambda q, img: scores[pages[[p.image for p in pages].index(img)].page_number] if img else 0
        )

        # Simpler: mock by call order
        reranker._score_single = MagicMock(side_effect=[0.3, 0.9, 0.6])

        result = reranker.rerank("test query", pages)

        assert [p.page_number for p in result] == [2, 3, 1]

    def test_rerank_empty_pages(self):
        """Empty input → empty output."""
        config = RerankingConfig(enabled=True)
        reranker = VisualReranker(config)

        result = reranker.rerank("test query", [])

        assert result == []

    def test_rerank_loads_images(self):
        """Each page should have load_image() called."""
        config = RerankingConfig(enabled=True)
        reranker = VisualReranker(config)
        reranker._score_single = MagicMock(return_value=0.5)

        pages = [FakePage(page_number=1), FakePage(page_number=2)]
        for page in pages:
            page.load_image = MagicMock(return_value="fake_image")
            page.image = "fake_image"

        reranker.rerank("test query", pages)

        for page in pages:
            page.load_image.assert_called_once()


# ── Pipeline integration tests ─────────────────────────────────────


@pytest.fixture
def config_reranking_enabled():
    config = AppConfig()
    config.reranking.enabled = True
    config.reranking.top_k = 3
    return config


@pytest.fixture
def config_reranking_disabled():
    return AppConfig()  # enabled=False by default


@pytest.fixture
def mock_pipeline_with_reranker(config_reranking_enabled):
    with (
        patch("app.pipeline.anthropic"),
        patch("app.pipeline.Retriever"),
        patch("app.pipeline.VisualReranker"),
        patch("app.pipeline.SemanticCache"),
        patch("app.pipeline.VerificationStore"),
    ):
        p = Pipeline(config_reranking_enabled)
        p.rewriter = AsyncMock()
        p.retriever = MagicMock()
        p.generator = AsyncMock()
        p.verifier = AsyncMock()
        p.cache = MagicMock()
        p.verification_store = MagicMock()

        # encode_query
        mock_qe = MagicMock()
        mock_qe.filtered = torch.randn(10, 128)
        p.retriever.encode_query.return_value = mock_qe

        # Cache miss
        p.cache.lookup.return_value = None

        # Rewriter
        p.rewriter.rewrite.return_value = ["question test"]

        # Retriever returns 5 pages
        fake_pages = [FakePage(page_number=i, score=0.9 - i * 0.1) for i in range(5)]
        p.retriever.retrieve.return_value = (fake_pages, {})

        # Reranker returns pages reordered (reverse order)
        reranked_pages = list(reversed(fake_pages))
        p.reranker.rerank.return_value = reranked_pages

        # Generator
        p.generator.generate.return_value = {
            "answer": "Réponse test",
            "citations": [{"page": 1}],
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 0,
        }

        # Verifier
        p.verifier.verify.return_value = {
            "status": "verified",
            "confidence": 0.95,
            "claims": [],
            "summary": "OK",
        }
        p.verifier.should_abstain = MagicMock(return_value=False)

        yield p


class TestPipelineRerankerIntegration:
    async def test_pipeline_calls_reranker_when_enabled(self, mock_pipeline_with_reranker):
        """When reranking is enabled, reranker.rerank() should be called."""
        await mock_pipeline_with_reranker.query("test question", skip_verification=True)

        mock_pipeline_with_reranker.reranker.rerank.assert_called_once()
        # First arg should be the original question
        call_args = mock_pipeline_with_reranker.reranker.rerank.call_args
        assert call_args[0][0] == "test question"

    async def test_pipeline_skips_reranker_when_disabled(self, config_reranking_disabled):
        """When reranking is disabled, pipeline.reranker should be None."""
        with (
            patch("app.pipeline.anthropic"),
            patch("app.pipeline.Retriever"),
            patch("app.pipeline.SemanticCache"),
            patch("app.pipeline.VerificationStore"),
        ):
            p = Pipeline(config_reranking_disabled)
            assert p.reranker is None

    async def test_pipeline_truncates_to_reranking_top_k(self, mock_pipeline_with_reranker):
        """After reranking, pages should be truncated to reranking.top_k."""
        p = mock_pipeline_with_reranker
        # Reranker returns 5 pages, top_k=3 → result should have 3
        result = await p.query("test question", skip_verification=True)

        assert len(result.pages) == 3
