"""
Retriever: ColQwen2.5 text encoding → Qdrant two-stage MaxSim search → RRF fusion.

Handles single-query retrieval and multi-query RAG Fusion with
Reciprocal Rank Fusion for merging ranked results.

Two-stage search:
  Stage 1 (prefetch): Fast search using tile-level mean-pooled vectors
  Stage 2 (rerank): Exact MaxSim on full visual patch vectors
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SearchParams

from app.config import AppConfig
from app.logging import get_logger

logger = get_logger("retriever")


@dataclass
class QueryEmbedding:
    """Encoded query with filtered and pooled variants for two-stage search."""

    filtered: torch.Tensor  # Padding-filtered multi-vector [N_real_tokens, 128]
    pooled: torch.Tensor  # Mean of real tokens [128] — for prefetch stage 1


@dataclass
class RetrievedPage:
    """A single retrieved page with metadata."""

    point_id: int
    document_id: str
    source_filename: str
    page_number: int
    total_pages: int
    image_path: str
    score: float
    image: Image.Image | None = field(default=None, repr=False)

    def load_image(self) -> Image.Image:
        """Load the page image from disk."""
        if self.image is None:
            self.image = Image.open(self.image_path).convert("RGB")
        return self.image


class Retriever:
    """
    Retrieves relevant pages using ColQwen2.5 + Qdrant two-stage MaxSim.
    Supports single-query and multi-query (RAG Fusion) modes.

    If the collection has the new 3-vector format (colqwen2 + pooled + global),
    uses two-stage prefetch for better performance. Falls back to single-stage
    search for old mono-vector collections (backward compatible).
    """

    def __init__(self, config: AppConfig, encoder=None, qdrant_client: QdrantClient | None = None):
        """
        Args:
            config: Application configuration
            encoder: ColQwen2Encoder instance (from indexing).
                     If None, creates one in CPU mode for query encoding.
            qdrant_client: Pre-initialized Qdrant client. If None, creates one.
        """
        self.config = config
        self.collection_name = config.qdrant.collection_name

        # Initialize encoder (lazy — only loads when needed)
        self._encoder = encoder

        # Initialize Qdrant client
        if qdrant_client is not None:
            self.client = qdrant_client
        elif config.qdrant.mode == "embedded":
            self.client = QdrantClient(path=config.qdrant.path)
        else:
            self.client = QdrantClient(url=config.qdrant.remote_url)

        # Detect collection capabilities (lazy)
        self._has_pooled_vector: bool | None = None

    @property
    def encoder(self):
        """Lazy-load the encoder on first use."""
        if self._encoder is None:
            from indexing.index_documents import ColQwen2Encoder

            self._encoder = ColQwen2Encoder(
                model_name=self.config.retrieval.model,
                mask_non_image_embeddings=self.config.retrieval.mask_non_image_embeddings,
                border_crop=self.config.retrieval.border_crop,
            )
            self._encoder.load()
        return self._encoder

    @property
    def has_pooled_vector(self) -> bool:
        """Check if collection supports two-stage search (has 'pooled' named vector)."""
        if self._has_pooled_vector is None:
            try:
                collection_info = self.client.get_collection(self.collection_name)
                vectors_config = collection_info.config.params.vectors
                self._has_pooled_vector = isinstance(vectors_config, dict) and "pooled" in vectors_config
            except Exception:
                self._has_pooled_vector = False
            logger.info("collection_capability", has_pooled=self._has_pooled_vector)
        return self._has_pooled_vector

    def encode_query(self, query: str) -> QueryEmbedding:
        """Encode a text query with padding hygiene.

        Returns a QueryEmbedding with:
        - filtered: padding tokens removed (for exact MaxSim search)
        - pooled: mean of real tokens (for prefetch stage 1)
        """
        filtered = self.encoder.encode_query(query)
        pooled = filtered.mean(dim=0)
        return QueryEmbedding(filtered=filtered, pooled=pooled)

    def search_single(
        self,
        query_embedding: QueryEmbedding,
        top_k: int | None = None,
    ) -> list[RetrievedPage]:
        """
        Search Qdrant with a single query embedding.

        Uses two-stage search if the collection supports it:
        - Stage 1: Prefetch top prefetch_k candidates using pooled vectors
        - Stage 2: Exact MaxSim rerank on full patch vectors

        Falls back to single-stage search for old collections.
        """
        top_k = top_k or self.config.retrieval.top_k

        if self.has_pooled_vector:
            prefetch_k = self.config.retrieval.prefetch_k
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.filtered.tolist(),
                using="colqwen2",
                limit=top_k,
                prefetch=[
                    Prefetch(
                        query=query_embedding.filtered.tolist(),
                        using="pooled",
                        limit=prefetch_k,
                    )
                ],
                search_params=SearchParams(exact=True),
            )
        else:
            # Fallback: single-stage search for old collections
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.filtered.tolist(),
                using="colqwen2",
                limit=top_k,
            )

        return [self._to_retrieved_page(r) for r in results.points]

    def search_multi(
        self,
        query_embeddings: list[QueryEmbedding],
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievedPage]:
        """
        Search Qdrant with multiple query embeddings (RAG Fusion).
        Performs N searches and fuses results with RRF.
        """
        top_k = top_k or self.config.retrieval.top_k
        rrf_k = rrf_k or self.config.rewriting.rrf_k

        if len(query_embeddings) == 1:
            return self.search_single(query_embeddings[0], top_k=top_k)

        # Perform N searches in parallel
        max_candidates = self.config.retrieval.max_candidates

        def _search_one(qe: QueryEmbedding):
            if self.has_pooled_vector:
                prefetch_k = self.config.retrieval.prefetch_k
                return self.client.query_points(
                    collection_name=self.collection_name,
                    query=qe.filtered.tolist(),
                    using="colqwen2",
                    limit=max_candidates,
                    prefetch=[
                        Prefetch(
                            query=qe.filtered.tolist(),
                            using="pooled",
                            limit=prefetch_k,
                        )
                    ],
                    search_params=SearchParams(exact=True),
                ).points
            else:
                return self.client.query_points(
                    collection_name=self.collection_name,
                    query=qe.filtered.tolist(),
                    using="colqwen2",
                    limit=max_candidates,
                ).points

        with ThreadPoolExecutor(max_workers=len(query_embeddings)) as pool:
            ranked_lists = list(pool.map(_search_one, query_embeddings))

        # RRF fusion
        fused = self._rrf_fusion(ranked_lists, rrf_k=rrf_k)

        # Take top-k and convert to RetrievedPage
        top_results = sorted(fused.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:top_k]

        pages = []
        for _point_id, data in top_results:
            page = self._to_retrieved_page(data["point"])
            page.score = data["rrf_score"]
            pages.append(page)

        return pages

    def retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
        precomputed_embeddings: dict[str, QueryEmbedding] | None = None,
    ) -> tuple[list[RetrievedPage], list[QueryEmbedding]]:
        """
        Full retrieval pipeline: encode queries → search → (optional RRF) → return.

        Args:
            queries: List of query strings (1 for simple, 3 for RAG Fusion)
            top_k: Number of pages to return
            precomputed_embeddings: Optional dict of {query_string: QueryEmbedding}
                                   to avoid re-encoding already-encoded queries

        Returns:
            Tuple of (retrieved pages, query embeddings)
        """
        precomputed = precomputed_embeddings or {}

        # Encode queries, reusing precomputed embeddings when available
        query_embeddings = []
        for q in queries:
            if q in precomputed:
                query_embeddings.append(precomputed[q])
            else:
                query_embeddings.append(self.encode_query(q))

        # Search
        if len(query_embeddings) == 1:
            pages = self.search_single(query_embeddings[0], top_k=top_k)
        else:
            pages = self.search_multi(query_embeddings, top_k=top_k)

        return pages, query_embeddings

    def _rrf_fusion(
        self,
        ranked_lists: list[list],
        rrf_k: int = 60,
    ) -> dict:
        """
        Reciprocal Rank Fusion.

        RRF_score(doc) = Σ  1 / (k + rank_in_list_i)
        """
        scores = {}

        for ranked_list in ranked_lists:
            for rank, point in enumerate(ranked_list):
                pid = point.id
                rrf_score = 1.0 / (rrf_k + rank + 1)

                if pid not in scores:
                    scores[pid] = {"rrf_score": 0.0, "point": point}

                scores[pid]["rrf_score"] += rrf_score

        return scores

    def _to_retrieved_page(self, point) -> RetrievedPage:
        """Convert a Qdrant ScoredPoint to a RetrievedPage."""
        payload = point.payload or {}
        return RetrievedPage(
            point_id=point.id,
            document_id=payload.get("document_id", ""),
            source_filename=payload.get("source_filename", ""),
            page_number=payload.get("page_number", 0),
            total_pages=payload.get("total_pages", 0),
            image_path=payload.get("image_path", ""),
            score=point.score if hasattr(point, "score") and point.score is not None else 0.0,
        )
