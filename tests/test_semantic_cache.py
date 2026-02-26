"""Tests for the semantic cache."""

import torch

from app.cache.semantic_cache import SemanticCache
from app.config import CachingConfig


def _make_embedding(seed: int = 0, dim: int = 128, tokens: int = 10) -> torch.Tensor:
    """Create a deterministic embedding for testing."""
    torch.manual_seed(seed)
    return torch.randn(tokens, dim)


class TestSemanticCache:
    def test_cache_miss_on_empty(self):
        config = CachingConfig(semantic_cache_enabled=True, similarity_threshold=0.98)
        cache = SemanticCache(config)
        emb = _make_embedding(seed=1)
        result = cache.lookup("test query", emb)
        assert result is None
        assert cache.misses == 1

    def test_store_and_hit(self):
        config = CachingConfig(semantic_cache_enabled=True, similarity_threshold=0.98)
        cache = SemanticCache(config)
        emb = _make_embedding(seed=1)
        response = {"answer": "test answer", "citations": []}
        cache.store("query A", emb, response)

        # Same embedding should hit
        result = cache.lookup("query A", emb)
        assert result is not None
        assert result["answer"] == "test answer"
        assert cache.hits == 1

    def test_different_embeddings_miss(self):
        config = CachingConfig(semantic_cache_enabled=True, similarity_threshold=0.98)
        cache = SemanticCache(config)
        emb_a = _make_embedding(seed=1)
        emb_b = _make_embedding(seed=999)
        cache.store("query A", emb_a, {"answer": "A"})

        result = cache.lookup("query B", emb_b)
        assert result is None

    def test_disabled_cache(self):
        config = CachingConfig(semantic_cache_enabled=False)
        cache = SemanticCache(config)
        emb = _make_embedding(seed=1)
        cache.store("query", emb, {"answer": "test"})
        result = cache.lookup("query", emb)
        assert result is None

    def test_lru_eviction(self):
        config = CachingConfig(semantic_cache_enabled=True, similarity_threshold=0.98, max_cache_entries=2)
        cache = SemanticCache(config)
        cache.store("q1", _make_embedding(1), {"answer": "1"})
        cache.store("q2", _make_embedding(2), {"answer": "2"})
        cache.store("q3", _make_embedding(3), {"answer": "3"})  # evicts q1
        assert len(cache._cache) == 2
        assert "q1" not in cache._cache

    def test_hit_rate(self):
        config = CachingConfig(semantic_cache_enabled=True, similarity_threshold=0.98)
        cache = SemanticCache(config)
        emb = _make_embedding(seed=1)
        cache.store("q", emb, {"answer": "a"})
        cache.lookup("q", emb)  # hit
        cache.lookup("other", _make_embedding(seed=999))  # miss
        assert cache.hit_rate == 0.5

    def test_stats(self):
        config = CachingConfig(semantic_cache_enabled=True, similarity_threshold=0.98)
        cache = SemanticCache(config)
        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_clear(self):
        config = CachingConfig(semantic_cache_enabled=True, similarity_threshold=0.98)
        cache = SemanticCache(config)
        cache.store("q", _make_embedding(1), {"answer": "a"})
        cache.clear()
        assert len(cache._cache) == 0
        assert cache.hits == 0
