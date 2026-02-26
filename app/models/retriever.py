"""
Retriever: ColQwen2 text encoding → Qdrant MaxSim search → RRF fusion.

Handles single-query retrieval and multi-query RAG Fusion with
Reciprocal Rank Fusion for merging ranked results.
"""

from dataclasses import dataclass, field

import torch
from PIL import Image
from qdrant_client import QdrantClient

from app.config import AppConfig


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
    Retrieves relevant pages using ColQwen2 + Qdrant MaxSim.
    Supports single-query and multi-query (RAG Fusion) modes.
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

    @property
    def encoder(self):
        """Lazy-load the encoder on first use."""
        if self._encoder is None:
            from indexing.index_documents import ColQwen2Encoder

            self._encoder = ColQwen2Encoder(model_name=self.config.retrieval.model)
            self._encoder.load()
        return self._encoder

    def encode_query(self, query: str) -> torch.Tensor:
        """Encode a text query into multi-vector embedding."""
        return self.encoder.encode_query(query)

    def search_single(
        self,
        query_embedding: torch.Tensor,
        top_k: int | None = None,
    ) -> list[RetrievedPage]:
        """
        Search Qdrant with a single query embedding.

        Args:
            query_embedding: Multi-vector query tensor (num_tokens, dim)
            top_k: Number of results to return (default from config)

        Returns:
            List of RetrievedPage sorted by score descending
        """
        top_k = top_k or self.config.retrieval.top_k

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            using="colqwen2",
            limit=top_k,
        )

        return [self._to_retrieved_page(r) for r in results.points]

    def search_multi(
        self,
        query_embeddings: list[torch.Tensor],
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievedPage]:
        """
        Search Qdrant with multiple query embeddings (RAG Fusion).
        Performs N searches and fuses results with RRF.

        Args:
            query_embeddings: List of multi-vector query tensors
            top_k: Final number of results after fusion
            rrf_k: RRF constant (default 60)

        Returns:
            List of RetrievedPage sorted by RRF score descending
        """
        top_k = top_k or self.config.retrieval.top_k
        rrf_k = rrf_k or self.config.rewriting.rrf_k

        if len(query_embeddings) == 1:
            return self.search_single(query_embeddings[0], top_k=top_k)

        # Perform N searches
        max_candidates = self.config.retrieval.max_candidates
        ranked_lists = []

        for qe in query_embeddings:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=qe.tolist(),
                using="colqwen2",
                limit=max_candidates,
            )
            ranked_lists.append(results.points)

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
        precomputed_embeddings: dict[str, "torch.Tensor"] | None = None,
    ) -> tuple[list[RetrievedPage], list["torch.Tensor"]]:
        """
        Full retrieval pipeline: encode queries → search → (optional RRF) → return.

        Args:
            queries: List of query strings (1 for simple, 3 for RAG Fusion)
            top_k: Number of pages to return
            precomputed_embeddings: Optional dict of {query_string: embedding}
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

        Args:
            ranked_lists: List of Qdrant search results (each is a ranked list)
            rrf_k: RRF constant (standard = 60)

        Returns:
            Dict mapping point_id to {"rrf_score": float, "point": ScoredPoint}
        """
        scores = {}  # point_id -> {"rrf_score": float, "point": ScoredPoint}

        for ranked_list in ranked_lists:
            for rank, point in enumerate(ranked_list):
                pid = point.id
                rrf_score = 1.0 / (rrf_k + rank + 1)  # rank is 0-indexed

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
