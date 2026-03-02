"""
Semantic cache for query responses.
Uses ColQwen2 query embeddings and cosine similarity to detect
near-duplicate queries and return cached responses.

Addresses the financial domain ambiguity issue:
- Threshold set to 0.98 (not 0.95) to avoid false hits
  between similar but semantically different queries
  (e.g., "CA 2023" vs "CA 2022" should NOT be cache hits)
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from app.config import CachingConfig
from app.logging import get_logger

logger = get_logger("semantic_cache")


@dataclass
class CacheEntry:
    """A cached query-response pair."""

    query: str
    embedding: torch.Tensor  # (num_tokens, dim)
    response: dict  # Full QueryResult.to_api_response() dict
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """
    Query-level semantic cache using ColQwen2 embeddings.

    Before calling the LLM, the pipeline encodes the query with ColQwen2
    (already needed for retrieval) and checks if a similar query exists
    in the cache. If similarity > threshold, returns the cached response.

    Uses MaxSim between multi-vector representations for similarity,
    matching the retrieval scoring mechanism.
    """

    def __init__(self, config: CachingConfig):
        self.enabled = config.semantic_cache_enabled
        self.threshold = config.similarity_threshold
        self.max_entries = config.max_cache_entries

        # LRU cache: OrderedDict with max size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self.hits = 0
        self.misses = 0

    def lookup(self, query: str, query_embedding: torch.Tensor) -> dict | None:
        """
        Check if a similar query exists in the cache.

        Args:
            query: The raw query string
            query_embedding: ColQwen2 multi-vector embedding (num_tokens, dim)

        Returns:
            Cached API response dict if hit, None if miss
        """
        with self._lock:
            if not self.enabled or len(self._cache) == 0:
                self.misses += 1
                return None

            best_score = 0.0
            best_key = None

            for key, entry in self._cache.items():
                score = self._maxsim_similarity(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_score >= self.threshold and best_key is not None:
                entry = self._cache[best_key]
                entry.hit_count += 1
                self.hits += 1

                # Move to end (most recently used)
                self._cache.move_to_end(best_key)

                logger.info(
                    "cache_hit",
                    query=query,
                    cached_query=entry.query,
                    similarity=round(best_score, 4),
                    hit_count=entry.hit_count,
                )

                return entry.response

            self.misses += 1
            logger.debug(
                "cache_miss",
                query=query,
                best_similarity=round(best_score, 4),
                threshold=self.threshold,
            )
            return None

    def store(self, query: str, query_embedding: torch.Tensor, response: dict) -> None:
        """
        Store a query-response pair in the cache.

        Args:
            query: The raw query string (used as key)
            query_embedding: ColQwen2 multi-vector embedding
            response: The full API response dict to cache
        """
        if not self.enabled:
            return

        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("cache_evict", evicted_query=evicted_key)

            self._cache[query] = CacheEntry(
                query=query,
                embedding=query_embedding.cpu(),
                response=response,
            )

    def _maxsim_similarity(self, query_a: torch.Tensor, query_b: torch.Tensor) -> float:
        """
        Compute MaxSim similarity between two multi-vector representations.

        MaxSim(A, B) = (1/|A|) * sum_i max_j cos_sim(a_i, b_j)

        This matches the scoring mechanism used by ColQwen2 + Qdrant.
        """
        # Normalize vectors
        a_norm = torch.nn.functional.normalize(query_a.float(), dim=-1)
        b_norm = torch.nn.functional.normalize(query_b.float(), dim=-1)

        # Compute pairwise cosine similarities: (num_tokens_a, num_tokens_b)
        sim_matrix = torch.mm(a_norm, b_norm.t())

        # MaxSim: for each token in A, take max similarity with any token in B
        max_sims = sim_matrix.max(dim=1).values

        # Average over all tokens in A
        return max_sims.mean().item()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
