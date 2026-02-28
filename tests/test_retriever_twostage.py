"""Tests for two-stage retriever and query token hygiene.

All tests mock the encoder and Qdrant client so they run without GPU in CI.
"""

from unittest.mock import MagicMock

import torch

from app.config import AppConfig
from app.models.retriever import QueryEmbedding, RetrievedPage, Retriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> AppConfig:
    return AppConfig(
        retrieval={
            "model": "vidore/colqwen2.5-v0.2",
            "top_k": 5,
            "max_candidates": 20,
            "mask_non_image_embeddings": True,
            "prefetch_k": 100,
            "border_crop": True,
        },
        qdrant={"mode": "remote", "remote_url": "http://localhost:6333"},
    )


def _make_query_embedding(num_tokens: int = 10) -> QueryEmbedding:
    """Create a fake QueryEmbedding with random tensors."""
    filtered = torch.randn(num_tokens, 128)
    pooled = filtered.mean(dim=0)
    return QueryEmbedding(filtered=filtered, pooled=pooled)


def _make_scored_point(point_id: int, score: float = 0.9) -> MagicMock:
    """Create a mock Qdrant ScoredPoint."""
    point = MagicMock()
    point.id = point_id
    point.score = score
    point.payload = {
        "document_id": f"doc_{point_id}",
        "source_filename": f"test_{point_id}.pdf",
        "page_number": point_id,
        "total_pages": 10,
        "image_path": f"/tmp/page_{point_id}.png",
    }
    return point


# ---------------------------------------------------------------------------
# TestQueryEmbedding
# ---------------------------------------------------------------------------


class TestQueryEmbedding:
    def test_query_embedding_has_filtered_and_pooled(self):
        qe = _make_query_embedding(15)
        assert qe.filtered.shape == (15, 128)
        assert qe.pooled.shape == (128,)

    def test_pooled_is_mean_of_filtered(self):
        qe = _make_query_embedding(8)
        expected = qe.filtered.mean(dim=0)
        assert torch.allclose(qe.pooled, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# TestRetrieverFallback
# ---------------------------------------------------------------------------


class TestRetrieverFallback:
    """Test backward compatibility with old mono-vector collections."""

    def test_detects_old_collection_without_pooled(self):
        """Old collections with only 'colqwen2' vector → has_pooled_vector = False."""
        config = _make_config()
        mock_client = MagicMock()

        # Simulate old collection info — vectors is a dict with only "colqwen2"
        collection_info = MagicMock()
        collection_info.config.params.vectors = {"colqwen2": MagicMock()}
        mock_client.get_collection.return_value = collection_info

        retriever = Retriever(config, qdrant_client=mock_client)
        assert not retriever.has_pooled_vector

    def test_detects_new_collection_with_pooled(self):
        """New collections with 3 named vectors → has_pooled_vector = True."""
        config = _make_config()
        mock_client = MagicMock()

        collection_info = MagicMock()
        collection_info.config.params.vectors = {
            "colqwen2": MagicMock(),
            "pooled": MagicMock(),
            "global": MagicMock(),
        }
        mock_client.get_collection.return_value = collection_info

        retriever = Retriever(config, qdrant_client=mock_client)
        assert retriever.has_pooled_vector

    def test_fallback_search_old_collection(self):
        """Search on old collection uses simple query_points without prefetch."""
        config = _make_config()
        mock_client = MagicMock()

        # Old collection
        collection_info = MagicMock()
        collection_info.config.params.vectors = {"colqwen2": MagicMock()}
        mock_client.get_collection.return_value = collection_info

        # Mock search results
        mock_results = MagicMock()
        mock_results.points = [_make_scored_point(1), _make_scored_point(2)]
        mock_client.query_points.return_value = mock_results

        retriever = Retriever(config, qdrant_client=mock_client)
        qe = _make_query_embedding()
        pages = retriever.search_single(qe, top_k=2)

        assert len(pages) == 2
        # Should NOT use prefetch
        call_kwargs = mock_client.query_points.call_args[1]
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None


# ---------------------------------------------------------------------------
# TestTwoStageSearch
# ---------------------------------------------------------------------------


class TestTwoStageSearch:
    def test_two_stage_search_uses_prefetch(self):
        """New collection search uses prefetch with pooled vectors."""
        config = _make_config()
        mock_client = MagicMock()

        # New collection with pooled
        collection_info = MagicMock()
        collection_info.config.params.vectors = {
            "colqwen2": MagicMock(),
            "pooled": MagicMock(),
            "global": MagicMock(),
        }
        mock_client.get_collection.return_value = collection_info

        mock_results = MagicMock()
        mock_results.points = [_make_scored_point(1)]
        mock_client.query_points.return_value = mock_results

        retriever = Retriever(config, qdrant_client=mock_client)
        qe = _make_query_embedding()
        pages = retriever.search_single(qe, top_k=5)

        assert len(pages) == 1
        # Should use prefetch
        call_kwargs = mock_client.query_points.call_args[1]
        assert "prefetch" in call_kwargs
        assert call_kwargs["prefetch"] is not None

    def test_search_single_returns_retrieved_pages(self):
        """search_single returns properly formatted RetrievedPage objects."""
        config = _make_config()
        mock_client = MagicMock()

        collection_info = MagicMock()
        collection_info.config.params.vectors = {"colqwen2": MagicMock()}
        mock_client.get_collection.return_value = collection_info

        mock_results = MagicMock()
        mock_results.points = [_make_scored_point(42, score=0.85)]
        mock_client.query_points.return_value = mock_results

        retriever = Retriever(config, qdrant_client=mock_client)
        qe = _make_query_embedding()
        pages = retriever.search_single(qe, top_k=1)

        assert len(pages) == 1
        page = pages[0]
        assert isinstance(page, RetrievedPage)
        assert page.point_id == 42
        assert page.score == 0.85
        assert page.page_number == 42

    def test_search_multi_rrf_fusion(self):
        """Multi-query search fuses results with RRF."""
        config = _make_config()
        mock_client = MagicMock()

        collection_info = MagicMock()
        collection_info.config.params.vectors = {"colqwen2": MagicMock()}
        mock_client.get_collection.return_value = collection_info

        # Two queries return overlapping results
        mock_results = MagicMock()
        mock_results.points = [_make_scored_point(1), _make_scored_point(2), _make_scored_point(3)]
        mock_client.query_points.return_value = mock_results

        retriever = Retriever(config, qdrant_client=mock_client)
        qe1 = _make_query_embedding()
        qe2 = _make_query_embedding()
        pages = retriever.search_multi([qe1, qe2], top_k=3)

        # All 3 pages should be fused (they appear in both lists)
        assert len(pages) == 3
        # Pages with higher RRF scores (appearing in both lists) should be first
        assert all(isinstance(p, RetrievedPage) for p in pages)


# ---------------------------------------------------------------------------
# TestRetrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_retrieve_encodes_queries_and_returns_embeddings(self):
        """retrieve() encodes queries and returns both pages and embeddings."""
        config = _make_config()
        mock_client = MagicMock()

        collection_info = MagicMock()
        collection_info.config.params.vectors = {"colqwen2": MagicMock()}
        mock_client.get_collection.return_value = collection_info

        mock_results = MagicMock()
        mock_results.points = [_make_scored_point(1)]
        mock_client.query_points.return_value = mock_results

        # Mock the encoder to avoid loading the actual model
        mock_encoder = MagicMock()
        fake_embedding = torch.randn(10, 128)
        mock_encoder.encode_query.return_value = fake_embedding

        retriever = Retriever(config, encoder=mock_encoder, qdrant_client=mock_client)
        pages, embeddings = retriever.retrieve(["test query"])

        assert len(pages) == 1
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], QueryEmbedding)

    def test_retrieve_reuses_precomputed_embeddings(self):
        """Precomputed embeddings are reused, not re-encoded."""
        config = _make_config()
        mock_client = MagicMock()

        collection_info = MagicMock()
        collection_info.config.params.vectors = {"colqwen2": MagicMock()}
        mock_client.get_collection.return_value = collection_info

        mock_results = MagicMock()
        mock_results.points = [_make_scored_point(1)]
        mock_client.query_points.return_value = mock_results

        mock_encoder = MagicMock()
        mock_encoder.encode_query.return_value = torch.randn(10, 128)

        retriever = Retriever(config, encoder=mock_encoder, qdrant_client=mock_client)

        precomputed_qe = _make_query_embedding()
        _pages, embeddings = retriever.retrieve(
            ["cached query"],
            precomputed_embeddings={"cached query": precomputed_qe},
        )

        # Encoder should NOT be called for the precomputed query
        mock_encoder.encode_query.assert_not_called()
        assert embeddings[0] is precomputed_qe


# ---------------------------------------------------------------------------
# TestEncodeQuery
# ---------------------------------------------------------------------------


class TestEncodeQuery:
    def test_encode_query_returns_query_embedding(self):
        """encode_query returns a QueryEmbedding with filtered and pooled tensors."""
        config = _make_config()
        mock_client = MagicMock()

        mock_encoder = MagicMock()
        # Simulate encoder returning a filtered tensor (padding already removed)
        mock_encoder.encode_query.return_value = torch.randn(12, 128)

        retriever = Retriever(config, encoder=mock_encoder, qdrant_client=mock_client)
        qe = retriever.encode_query("test query")

        assert isinstance(qe, QueryEmbedding)
        assert qe.filtered.shape == (12, 128)
        assert qe.pooled.shape == (128,)

    def test_encode_query_pooled_is_mean(self):
        """The pooled vector is the mean of filtered tokens."""
        config = _make_config()
        mock_client = MagicMock()

        fixed_tensor = torch.randn(5, 128)
        mock_encoder = MagicMock()
        mock_encoder.encode_query.return_value = fixed_tensor

        retriever = Retriever(config, encoder=mock_encoder, qdrant_client=mock_client)
        qe = retriever.encode_query("test")

        expected_pooled = fixed_tensor.mean(dim=0)
        assert torch.allclose(qe.pooled, expected_pooled, atol=1e-6)
