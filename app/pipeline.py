"""
End-to-end query pipeline orchestrator.
Coordinates: cache check → rewrite → retrieve → generate → verify → respond.
"""

import asyncio
import time
import uuid
from datetime import UTC, datetime

import anthropic

from app.cache.semantic_cache import SemanticCache
from app.cache.verification_store import VerificationStore
from app.config import AppConfig
from app.errors import RewritingFallbackError, ServiceUnavailableError
from app.logging import get_logger
from app.models.generator import Generator
from app.models.retriever import RetrievedPage, Retriever
from app.models.rewriter import QueryRewriter
from app.models.verifier import Verifier
from app.security.output_validator import validate_response

logger = get_logger("pipeline")


class QueryResult:
    """Structured result from the pipeline."""

    def __init__(self):
        self.query_id: str = str(uuid.uuid4())
        self.timestamp: str = datetime.now(UTC).isoformat()
        self.question: str = ""

        # Cache
        self.cache_hit: bool = False

        # Rewriting
        self.rewritten_queries: list[str] = []
        self.rewriting_latency_ms: float = 0
        self.rewriting_fallback: bool = False

        # Retrieval
        self.pages: list[RetrievedPage] = []
        self.retrieval_latency_ms: float = 0

        # Generation
        self.answer: str = ""
        self.citations: list[dict] = []
        self.generation_latency_ms: float = 0
        self.generation_tokens: dict = {}

        # Verification
        self.verification: dict = {}
        self.verification_latency_ms: float = 0

        # Security
        self.validation: dict = {}

        # Totals
        self.total_latency_ms: float = 0
        self.error: str | None = None

    def to_api_response(self) -> dict:
        """Convert to API response format."""
        sources = [
            {
                "document": p.source_filename,
                "page": p.page_number,
                "score": round(p.score, 4),
                "image_path": p.image_path,
            }
            for p in self.pages
        ]

        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "sources": sources,
            "citations": self.citations,
            "confidence": self.verification.get("confidence"),
            "verification_status": self.verification.get("status", "pending"),
            "latency_ms": round(self.total_latency_ms),
        }

    def to_log_entry(self) -> dict:
        """Convert to structured log entry."""
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "semantic_cache_hit": self.cache_hit,
            "rewriting": {
                "queries": self.rewritten_queries,
                "latency_ms": round(self.rewriting_latency_ms),
                "fallback_used": self.rewriting_fallback,
            },
            "retrieval": {
                "latency_ms": round(self.retrieval_latency_ms),
                "top_pages": [p.page_number for p in self.pages],
                "scores": [round(p.score, 4) for p in self.pages],
                "queries_searched": len(self.rewritten_queries),
            },
            "generation": {
                "latency_ms": round(self.generation_latency_ms),
                "citations_found": [c.get("page") for c in self.citations],
                **self.generation_tokens,
            },
            "verification": {
                "status": self.verification.get("status"),
                "confidence": self.verification.get("confidence"),
                "latency_ms": round(self.verification_latency_ms),
                "claims_verified": self.verification.get("claims_verified", 0),
                "claims_contradicted": self.verification.get("claims_contradicted", 0),
                "claims_not_found": self.verification.get("claims_not_found", 0),
            },
            "security": self.validation,
            "total_latency_ms": round(self.total_latency_ms),
            "error": self.error,
        }


class Pipeline:
    """
    End-to-end query pipeline.

    Flow:
    0. Check semantic cache
    1. Rewrite query (RAG Fusion)
    2. Retrieve pages (ColQwen2 + Qdrant + RRF)
    3. Generate answer (Sonnet)
    4. Validate output (security)
    5. Verify answer (Opus, batch async or sync)
    6. Store in cache
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic.api_key)

        # Initialize components
        self.rewriter = QueryRewriter(config, self.client)
        self.retriever = Retriever(config)
        self.generator = Generator(config, self.client)
        self.verifier = Verifier(config, self.client)
        self.cache = SemanticCache(config.caching)
        self.verification_store = VerificationStore()

    async def query(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        skip_verification: bool = False,
    ) -> QueryResult:
        """
        Execute the full query pipeline.

        Args:
            question: User's question
            conversation_history: Previous Q&A pairs for multi-turn
            skip_verification: Skip Opus verification (for testing)

        Returns:
            QueryResult with answer, sources, confidence
        """
        result = QueryResult()
        result.question = question
        t_start = time.time()

        try:
            # Step 0: Semantic cache check
            # Encode query once — reused for cache lookup AND retrieval
            query_embedding = self.retriever.encode_query(question)

            cached = self.cache.lookup(question, query_embedding)
            if cached is not None:
                result.cache_hit = True
                result.answer = cached["answer"]
                result.citations = cached.get("citations", [])
                result.verification = {"status": "cached", "confidence": cached.get("confidence")}
                result.total_latency_ms = (time.time() - t_start) * 1000
                logger.info("cache_hit", query=question, latency_ms=round(result.total_latency_ms))
                return result

            # Step 1: Rewrite
            result.rewritten_queries = await self._rewrite(question, conversation_history, result)

            # Step 2: Retrieve — pass original embedding to avoid re-encoding
            result.pages = self._retrieve(
                result.rewritten_queries,
                result,
                precomputed={question: query_embedding},
            )

            if not result.pages:
                result.answer = self.config.verification.abstention_message
                result.total_latency_ms = (time.time() - t_start) * 1000
                return result

            # Step 3: Generate
            gen_result = await self._generate(question, result.pages, conversation_history, result)
            result.answer = gen_result["answer"]
            result.citations = gen_result["citations"]
            result.generation_tokens = {
                "input_tokens": gen_result["input_tokens"],
                "output_tokens": gen_result["output_tokens"],
                "cache_read_tokens": gen_result.get("cache_read_tokens", 0),
            }

            # Step 4: Validate output
            result.validation = validate_response(result.answer)

            # Step 5: Verify
            if not skip_verification and self.config.verification.enabled:
                if self.config.verification.mode == "sync":
                    result.verification = await self._verify(question, result.answer, result.pages, result)

                    if self.verifier.should_abstain(result.verification):
                        result.answer = (
                            f"{result.answer}\n\n"
                            f"Confiance faible ({result.verification.get('confidence', 0):.0%}). "
                            f"Verifiez les sources."
                        )

                elif self.config.verification.mode == "batch_async":
                    await self._verify_batch_async(result.query_id, question, result.answer, result.pages)

            # Step 6: Cache the response
            self.cache.store(question, query_embedding, result.to_api_response())

        except ServiceUnavailableError as e:
            result.error = str(e)
            result.answer = "Service temporairement indisponible. Veuillez reessayer dans quelques instants."
            logger.error("service_unavailable", error=str(e))

        except Exception as e:
            result.error = str(e)
            result.answer = "Une erreur inattendue s'est produite."
            logger.error("unexpected_error", error=str(e), exc_info=True)

        result.total_latency_ms = (time.time() - t_start) * 1000
        return result

    async def _rewrite(
        self,
        question: str,
        conversation_history: list[dict] | None,
        result: QueryResult,
    ) -> list[str]:
        """Step 1: Rewrite query with RAG Fusion."""
        t0 = time.time()

        try:
            queries = await self.rewriter.rewrite(question, conversation_history)
            result.rewriting_fallback = False
        except RewritingFallbackError:
            queries = [question]
            result.rewriting_fallback = True

        result.rewriting_latency_ms = (time.time() - t0) * 1000
        return queries

    def _retrieve(
        self,
        queries: list[str],
        result: QueryResult,
        precomputed: dict | None = None,
    ) -> list[RetrievedPage]:
        """Step 2: Retrieve pages with ColQwen2 + RRF."""
        t0 = time.time()

        pages, _ = self.retriever.retrieve(
            queries,
            precomputed_embeddings=precomputed,
        )

        for page in pages:
            page.load_image()

        result.retrieval_latency_ms = (time.time() - t0) * 1000
        return pages

    async def _generate(
        self,
        question: str,
        pages: list[RetrievedPage],
        conversation_history: list[dict] | None,
        result: QueryResult,
    ) -> dict:
        """Step 3: Generate answer with Sonnet."""
        t0 = time.time()

        gen_result = await self.generator.generate(question, pages, conversation_history)

        result.generation_latency_ms = (time.time() - t0) * 1000
        return gen_result

    async def _verify(
        self,
        question: str,
        answer: str,
        pages: list[RetrievedPage],
        result: QueryResult,
    ) -> dict:
        """Step 5a: Verify with Opus (sync mode)."""
        t0 = time.time()

        verification = await self.verifier.verify(question, answer, pages)

        result.verification_latency_ms = (time.time() - t0) * 1000
        return verification

    async def _verify_batch_async(
        self,
        query_id: str,
        question: str,
        answer: str,
        pages: list[RetrievedPage],
    ):
        """
        Step 5b: Verify with Opus Batch API (async mode, 50% discount).

        Submits to Anthropic Batch API, stores pending status in persistent
        store, then polls in background task.
        """
        batch_id = await self.verifier.submit_batch(query_id, question, answer, pages)

        if batch_id is None:
            self.verification_store.set(query_id, self.verifier._error_result("Batch submission failed"))
            return

        self.verification_store.set_pending(query_id, batch_id)

        # Poll in background — does not block the response
        self._background_task = asyncio.create_task(self._poll_batch_verification(query_id, batch_id))

    async def _poll_batch_verification(self, query_id: str, batch_id: str):
        """Background task: poll batch API and update verification store."""
        try:
            result = await self.verifier.poll_batch(batch_id, query_id)
            self.verification_store.set(query_id, result)
        except Exception as e:
            logger.error(
                "batch_poll_failed",
                query_id=query_id,
                batch_id=batch_id,
                error=str(e),
            )
            self.verification_store.set(query_id, self.verifier._error_result(str(e)))
