"""Tests for the Pipeline orchestrator."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from app.config import AppConfig
from app.errors import RewritingFallbackError, ServiceUnavailableError
from app.pipeline import Pipeline, QueryResult


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

    def load_image(self):
        pass


@pytest.fixture
def config():
    return AppConfig()


@pytest.fixture
def mock_pipeline(config):
    with (
        patch("app.pipeline.anthropic"),
        patch("app.pipeline.Retriever"),
        patch("app.pipeline.SemanticCache"),
        patch("app.pipeline.VerificationStore"),
    ):
        p = Pipeline(config)
        p.rewriter = AsyncMock()
        p.retriever = MagicMock()
        p.generator = AsyncMock()
        p.verifier = AsyncMock()
        p.cache = MagicMock()
        p.verification_store = MagicMock()

        # encode_query must return object with .filtered
        mock_qe = MagicMock()
        mock_qe.filtered = torch.randn(10, 128)
        p.retriever.encode_query.return_value = mock_qe

        # Default: cache miss
        p.cache.lookup.return_value = None
        p.cache.enabled = True

        # Default: rewriter returns single query
        p.rewriter.rewrite.return_value = ["question test"]

        # Default: retriever returns pages
        fake_pages = [FakePage(page_number=5), FakePage(page_number=12)]
        p.retriever.retrieve.return_value = (fake_pages, {})

        # Default: generator returns answer
        p.generator.generate.return_value = {
            "answer": "Le CA est de 86M [Page 5].",
            "citations": [{"page": 5}],
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 0,
        }

        # Default: verifier returns verified
        p.verifier.verify.return_value = {
            "status": "verified",
            "confidence": 0.95,
            "claims": [],
            "summary": "OK",
        }
        # should_abstain is sync, not async — use MagicMock
        p.verifier.should_abstain = MagicMock(return_value=False)

        yield p


# ── QueryResult ─────────────────────────────────────────────────────


class TestQueryResult:
    def test_to_api_response_format(self):
        result = QueryResult()
        result.answer = "Réponse test"
        result.pages = [FakePage(page_number=5, score=0.95)]
        result.citations = [{"page": 5}]
        result.verification = {"status": "verified", "confidence": 0.9}
        result.total_latency_ms = 1234.5

        api = result.to_api_response()
        assert api["answer"] == "Réponse test"
        assert api["query_id"] == result.query_id
        assert len(api["sources"]) == 1
        assert api["sources"][0]["page"] == 5
        assert api["confidence"] == 0.9
        assert api["verification_status"] == "verified"
        assert api["latency_ms"] == round(1234.5)

    def test_to_log_entry_format(self):
        result = QueryResult()
        result.question = "CA 2023 ?"
        result.rewritten_queries = ["CA 2023 ?"]
        result.pages = [FakePage(page_number=5)]
        result.citations = [{"page": 5}]
        result.verification = {"status": "verified", "confidence": 0.95}

        log = result.to_log_entry()
        assert log["question"] == "CA 2023 ?"
        assert log["verification"]["status"] == "verified"
        assert "rewriting" in log
        assert "retrieval" in log
        assert "generation" in log


# ── Pipeline: Cache hit ─────────────────────────────────────────────


class TestPipelineCacheHit:
    async def test_cache_hit_skips_pipeline(self, mock_pipeline):
        mock_pipeline.cache.lookup.return_value = {
            "answer": "cached answer",
            "citations": [{"page": 1}],
            "confidence": 0.9,
        }

        result = await mock_pipeline.query("question test")

        assert result.cache_hit is True
        assert result.answer == "cached answer"
        # Rewriter and generator should NOT be called
        mock_pipeline.rewriter.rewrite.assert_not_called()
        mock_pipeline.generator.generate.assert_not_called()

    async def test_cache_hit_returns_cached_answer(self, mock_pipeline):
        mock_pipeline.cache.lookup.return_value = {
            "answer": "cached!",
            "citations": [],
            "confidence": 0.8,
        }

        result = await mock_pipeline.query("test")
        assert result.answer == "cached!"
        assert result.verification["status"] == "cached"


# ── Pipeline: Cache miss ────────────────────────────────────────────


class TestPipelineCacheMiss:
    async def test_full_pipeline_flow(self, mock_pipeline):
        """Full pipeline: rewrite → retrieve → generate → verify."""
        mock_pipeline.config.verification.mode = "sync"

        result = await mock_pipeline.query("CA 2023 ?")

        assert result.cache_hit is False
        mock_pipeline.rewriter.rewrite.assert_called_once()
        mock_pipeline.retriever.retrieve.assert_called_once()
        mock_pipeline.generator.generate.assert_called_once()
        assert result.answer == "Le CA est de 86M [Page 5]."

    async def test_stores_result_in_cache(self, mock_pipeline):
        mock_pipeline.config.verification.mode = "sync"

        await mock_pipeline.query("test")

        mock_pipeline.cache.store.assert_called_once()

    async def test_no_pages_returns_abstention(self, mock_pipeline):
        """When retriever returns no pages, return abstention message."""
        mock_pipeline.retriever.retrieve.return_value = ([], {})

        result = await mock_pipeline.query("question sans résultat")

        assert result.answer == mock_pipeline.config.verification.abstention_message
        mock_pipeline.generator.generate.assert_not_called()


# ── Pipeline: Verification ──────────────────────────────────────────


class TestPipelineVerification:
    async def test_sync_verification(self, mock_pipeline):
        mock_pipeline.config.verification.mode = "sync"

        result = await mock_pipeline.query("test")

        mock_pipeline.verifier.verify.assert_called_once()
        assert result.verification["status"] == "verified"

    async def test_low_confidence_appends_warning(self, mock_pipeline):
        mock_pipeline.config.verification.mode = "sync"
        mock_pipeline.verifier.verify.return_value = {
            "status": "low_confidence",
            "confidence": 0.4,
            "claims": [],
            "summary": "Low",
        }
        mock_pipeline.verifier.should_abstain = MagicMock(return_value=True)

        result = await mock_pipeline.query("test")

        assert "Confiance faible" in result.answer

    async def test_verification_error_appends_warning(self, mock_pipeline):
        """Regression: verification error → answer preserved with warning."""
        mock_pipeline.config.verification.mode = "sync"
        mock_pipeline.verifier.verify.return_value = {
            "status": "error",
            "confidence": None,
            "claims": [],
            "summary": "Verification failed: 529 overloaded",
        }
        mock_pipeline.verifier.should_abstain = MagicMock(return_value=False)

        result = await mock_pipeline.query("test")

        assert "Le CA est de 86M" in result.answer
        assert "non vérifiée" in result.answer
        assert "indisponible" in result.answer

    async def test_skip_verification_flag(self, mock_pipeline):
        await mock_pipeline.query("test", skip_verification=True)

        mock_pipeline.verifier.verify.assert_not_called()


# ── Pipeline: Error handling ────────────────────────────────────────


class TestPipelineErrorHandling:
    async def test_service_unavailable_returns_error_message(self, mock_pipeline):
        mock_pipeline.rewriter.rewrite.side_effect = ServiceUnavailableError("API down")

        result = await mock_pipeline.query("test")

        assert "indisponible" in result.answer
        assert result.error is not None

    async def test_unexpected_error_returns_generic_message(self, mock_pipeline):
        mock_pipeline.rewriter.rewrite.side_effect = RuntimeError("boom")

        result = await mock_pipeline.query("test")

        assert "erreur inattendue" in result.answer
        assert result.error is not None

    async def test_rewriting_fallback_uses_raw_query(self, mock_pipeline):
        mock_pipeline.rewriter.rewrite.side_effect = RewritingFallbackError("fallback")

        result = await mock_pipeline.query("ma question")

        assert result.rewriting_fallback is True
        assert result.rewritten_queries == ["ma question"]


# ── Pipeline: Init ──────────────────────────────────────────────────


class TestPipelineInit:
    def test_async_client_has_timeout(self, config):
        """Bug 2 regression: AsyncAnthropic must be created with timeout."""
        with (
            patch("app.pipeline.anthropic") as mock_anthropic,
            patch("app.pipeline.Retriever"),
            patch("app.pipeline.SemanticCache"),
            patch("app.pipeline.VerificationStore"),
        ):
            Pipeline(config)

        mock_anthropic.AsyncAnthropic.assert_called_once()
        call_kwargs = mock_anthropic.AsyncAnthropic.call_args
        assert "timeout" in call_kwargs.kwargs
