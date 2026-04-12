"""
End-to-end query pipeline orchestrator.
Coordinates: retrieve → generate → verify → respond.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import anthropic
import httpx

from app.config import AppConfig
from app.errors import ServiceUnavailableError
from app.logging import get_logger
from app.models.generator import Generator
from app.models.retriever import RetrievedPage, Retriever
from app.models.verifier import Verifier
from app.security.output_validator import validate_response

logger = get_logger("pipeline")


class QueryResult:
    """Structured result from the pipeline."""

    def __init__(self):
        self.query_id: str = str(uuid.uuid4())
        self.timestamp: str = datetime.now(UTC).isoformat()
        self.question: str = ""

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
            "retrieval": {
                "latency_ms": round(self.retrieval_latency_ms),
                "top_pages": [p.page_number for p in self.pages],
                "scores": [round(p.score, 4) for p in self.pages],
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
    1. Retrieve pages (ColQwen2.5 + Qdrant)
    2. Generate answer (Sonnet)
    3. Validate output (security)
    4. Verify answer (Opus)
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(
            api_key=config.anthropic.api_key,
            timeout=httpx.Timeout(config.anthropic.timeout_seconds),
        )

        # Initialize components
        self.retriever = Retriever(config)
        self.generator = Generator(config, self.client)
        self.verifier = Verifier(config, self.client)

        # Async verification state (keyed by query_id to avoid overwrite races)
        self._verification_store: dict[str, dict] = {}
        self._verification_tasks: dict[str, asyncio.Task] = {}
        self._verification_store_max_size = 100

    async def query(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        skip_verification: bool = False,
        async_verification: bool = False,
    ) -> QueryResult:
        """
        Execute the full query pipeline.

        Args:
            question: User's question
            conversation_history: Previous Q&A pairs for multi-turn
            skip_verification: Skip Opus verification (for testing)
            async_verification: If True, Opus verification runs in a
                background task and the result is returned with
                verification status "pending". The frontend polls
                GET /api/v1/query/{query_id}/verification until the
                terminal status is available. If False (default), the
                pipeline awaits verification synchronously and returns
                the full result in one response.

        Returns:
            QueryResult with answer, sources, confidence
        """
        result = QueryResult()
        result.question = question
        t_start = time.time()

        try:
            # Step 1: Encode query + retrieve (run sync ColQwen2 + Qdrant in a thread
            # to avoid blocking the FastAPI event loop)
            query_embedding = await asyncio.to_thread(self.retriever.encode_query, question)
            result.pages = await self._retrieve(question, query_embedding, result)

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
                if async_verification:
                    # Fire-and-forget: verification runs in background.
                    # The frontend polls GET /api/v1/query/{query_id}/verification
                    # to retrieve the final status. The answer text is returned
                    # as-is; any "low confidence" warning is the frontend's
                    # responsibility via the verification badge.
                    result.verification = {
                        "status": "pending",
                        "confidence": None,
                        "claims": [],
                        "summary": "Verification in progress",
                        "claims_verified": 0,
                        "claims_contradicted": 0,
                        "claims_not_found": 0,
                    }
                    self._start_background_verification(result.query_id, question, result.answer, result.pages)
                else:
                    result.verification = await self._verify(question, result.answer, result.pages, result)

                    if self.verifier.should_abstain(result.verification):
                        result.answer = (
                            f"{result.answer}\n\n"
                            f"Confiance faible ({result.verification.get('confidence', 0):.0%}). "
                            f"Verifiez les sources."
                        )
                    elif result.verification.get("status") == "error":
                        result.answer = (
                            f"{result.answer}\n\nRéponse non vérifiée (service de vérification indisponible)."
                        )

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

    async def query_stream(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[tuple[str, dict]]:
        """
        Stream the query pipeline.

        Yields event tuples of (event_type, payload):
        - ("meta", {"query_id": ..., "sources": [...]}): emitted after
          retrieval, before the first token. Lets the client render
          sources immediately and learn the query_id for verification
          polling.
        - ("token", {"text": "..."}): each incremental chunk from Sonnet.
        - ("done", {"citations": [...], "verification_status": "pending",
          "confidence": null, "latency_ms": int}): final event with
          accumulated metadata. Opus verification is always started as
          a background task in this mode — the client polls
          GET /api/v1/query/{query_id}/verification for the terminal
          status.
        - ("error", {"message": str}): only emitted on failure.

        Streaming mode always uses async verification. For sync
        verification in a single response, call query() instead.
        """
        result = QueryResult()
        result.question = question
        t_start = time.time()

        try:
            # Step 1: Encode query + retrieve
            query_embedding = await asyncio.to_thread(self.retriever.encode_query, question)
            result.pages = await self._retrieve(question, query_embedding, result)

            sources_payload = [
                {
                    "document": p.source_filename,
                    "page": p.page_number,
                    "score": round(p.score, 4),
                    "image_path": p.image_path,
                }
                for p in result.pages
            ]
            yield (
                "meta",
                {"query_id": result.query_id, "sources": sources_payload},
            )

            if not result.pages:
                # No pages retrieved → emit abstention message as a single token
                abstention = self.config.verification.abstention_message
                result.answer = abstention
                yield ("token", {"text": abstention})
                result.total_latency_ms = (time.time() - t_start) * 1000
                yield (
                    "done",
                    {
                        "citations": [],
                        "verification_status": "disabled",
                        "confidence": None,
                        "latency_ms": round(result.total_latency_ms),
                    },
                )
                return

            # Step 3: Stream generation
            t_gen = time.time()
            async for event_type, payload in self.generator.generate_stream(
                question, result.pages, conversation_history
            ):
                if event_type == "token":
                    yield ("token", {"text": payload})
                elif event_type == "final":
                    result.answer = payload["answer"]
                    result.citations = payload["citations"]
                    result.generation_tokens = {
                        "input_tokens": payload["input_tokens"],
                        "output_tokens": payload["output_tokens"],
                        "cache_read_tokens": payload.get("cache_read_tokens", 0),
                    }
            result.generation_latency_ms = (time.time() - t_gen) * 1000

            # Step 4: Validate output
            result.validation = validate_response(result.answer)

            # Step 5: Start background verification (async, never blocks the stream)
            if self.config.verification.enabled:
                self._start_background_verification(result.query_id, question, result.answer, result.pages)
                verification_status = "pending"
            else:
                verification_status = "disabled"

            result.total_latency_ms = (time.time() - t_start) * 1000
            yield (
                "done",
                {
                    "citations": result.citations,
                    "verification_status": verification_status,
                    "confidence": None,
                    "latency_ms": round(result.total_latency_ms),
                },
            )

            # Log the query after the stream is fully emitted
            log_entry = result.to_log_entry()
            logger.info("stream_query_completed", **log_entry)

        except ServiceUnavailableError as e:
            logger.error("stream_service_unavailable", error=str(e))
            yield ("error", {"message": "Service temporairement indisponible."})

        except Exception as e:
            logger.error("stream_unexpected_error", error=str(e), exc_info=True)
            yield ("error", {"message": "Une erreur inattendue s'est produite."})

    async def _retrieve(
        self,
        question: str,
        query_embedding,
        result: QueryResult,
    ) -> list[RetrievedPage]:
        """Step 1: Retrieve pages with ColQwen2 two-stage MaxSim."""
        t0 = time.time()

        pages = await asyncio.to_thread(self.retriever.search_single, query_embedding)

        result.retrieval_latency_ms = (time.time() - t0) * 1000

        for page in pages:
            page.load_image()

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
        """Step 5: Verify with Opus."""
        t0 = time.time()

        verification = await self.verifier.verify(question, answer, pages)

        result.verification_latency_ms = (time.time() - t0) * 1000
        return verification

    def _start_background_verification(
        self,
        query_id: str,
        question: str,
        answer: str,
        pages: list[RetrievedPage],
    ) -> None:
        """
        Schedule verification as a background asyncio task.

        The task is tracked in self._verification_tasks keyed by query_id
        (to avoid the "single variable overwrite" race that affected the
        previous batch_async implementation). A done_callback removes the
        task entry from the dict once it completes. The result is written
        to self._verification_store via _verify_background.
        """
        task = asyncio.create_task(self._verify_background(query_id, question, answer, pages))
        self._verification_tasks[query_id] = task
        # Keyed by query_id via default arg to avoid late-binding closure issues.
        task.add_done_callback(lambda t, qid=query_id: self._verification_tasks.pop(qid, None))  # type: ignore[misc]

    async def _verify_background(
        self,
        query_id: str,
        question: str,
        answer: str,
        pages: list[RetrievedPage],
    ) -> None:
        """
        Run verification in a background task and persist the result.

        Any exception raised by the verifier is caught and stored as an
        error result so the store never holds an orphaned "pending".
        Called via asyncio.create_task from query() when async_verification
        is True.
        """
        try:
            verification = await self.verifier.verify(question, answer, pages)
        except Exception as e:
            logger.error("background_verification_failed", query_id=query_id, error=str(e), exc_info=True)
            verification = {
                "status": "error",
                "confidence": None,
                "claims": [],
                "summary": f"Verification failed: {e}",
                "claims_verified": 0,
                "claims_contradicted": 0,
                "claims_not_found": 0,
            }

        self._store_verification(query_id, verification)

    def _store_verification(self, query_id: str, verification: dict) -> None:
        """
        Write a verification result to the store with FIFO eviction.

        Enforces self._verification_store_max_size by evicting the oldest
        entry (dict preserves insertion order in Python 3.7+).
        """
        if (
            len(self._verification_store) >= self._verification_store_max_size
            and query_id not in self._verification_store
        ):
            oldest_id = next(iter(self._verification_store))
            del self._verification_store[oldest_id]
            logger.debug("verification_store_evicted", evicted_query_id=oldest_id)
        self._verification_store[query_id] = verification

    def get_verification(self, query_id: str) -> dict:
        """
        Return the verification result for a query_id.

        Returns the final verification dict if complete, a dict with
        status "pending" if the background task is still running, or
        status "not_found" if the query_id is unknown.
        """
        if query_id in self._verification_store:
            return self._verification_store[query_id]
        if query_id in self._verification_tasks:
            return {
                "status": "pending",
                "confidence": None,
                "claims": [],
                "summary": "Verification in progress",
                "claims_verified": 0,
                "claims_contradicted": 0,
                "claims_not_found": 0,
            }
        return {
            "status": "not_found",
            "confidence": None,
            "claims": [],
            "summary": "Query not found",
            "claims_verified": 0,
            "claims_contradicted": 0,
            "claims_not_found": 0,
        }
