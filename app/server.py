"""
FastAPI server for the Multimodal RAG system.

Endpoints:
- POST /api/v1/query — Ask a question
- GET /api/v1/query/{query_id}/verification — Get verification status
- GET /api/v1/health — Health check
- GET /api/v1/metrics — System metrics
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_config
from app.logging import get_logger, setup_logging
from app.pipeline import Pipeline, QueryResult

logger = get_logger("server")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[dict] | None = None
    skip_verification: bool = False
    async_verification: bool = False


class SourceResponse(BaseModel):
    document: str
    page: int
    score: float
    image_path: str


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[SourceResponse]
    citations: list[dict]
    confidence: float | None
    verification_status: str
    latency_ms: int


class VerificationResponse(BaseModel):
    status: str
    confidence: float | None
    claims: list[dict]
    summary: str
    claims_verified: int = 0
    claims_contradicted: int = 0
    claims_not_found: int = 0


class HealthResponse(BaseModel):
    status: str
    components: dict


class MetricsResponse(BaseModel):
    queries_total: int
    avg_latency_ms: float
    avg_cost_per_query: float
    abstention_rate: float
    error_rate: float


# ---------------------------------------------------------------------------
# In-memory metrics
# ---------------------------------------------------------------------------

metrics_store = {
    "queries_total": 0,
    "total_latency_ms": 0.0,
    "total_cost": 0.0,
    "abstentions": 0,
    "errors": 0,
}


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup, cleanup on shutdown."""
    config = get_config()
    setup_logging(config.observability.log_level)
    app.state.pipeline = Pipeline(config)
    app.state.config = config
    logger.info("pipeline_initialized", qdrant_mode=config.qdrant.mode)
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Multimodal RAG — Financial Documents",
    description="Visual RAG system for financial document analysis with verified responses.",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting
_config = get_config()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# CORS for frontend — restricted to configured origins
_cors_origins = _config.security.allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/query", response_model=QueryResponse)
# slowapi accepts both `lambda: str` and `lambda key: str` for limit_value;
# the no-arg form (used here) is invoked as `()` by LimitGroup.__iter__.
@limiter.limit(lambda: _config.security.rate_limit)
async def query(request: Request, body: QueryRequest):
    """
    Ask a question about the indexed financial documents.
    Returns an answer with citations, sources, and confidence score.
    """
    pipeline: Pipeline = app.state.pipeline

    result: QueryResult = await pipeline.query(
        question=body.question,
        conversation_history=body.conversation_history,
        skip_verification=body.skip_verification,
        async_verification=body.async_verification,
    )

    # Update metrics
    metrics_store["queries_total"] += 1
    metrics_store["total_latency_ms"] += result.total_latency_ms
    if result.error:
        metrics_store["errors"] += 1

    # Log the request
    log_entry = result.to_log_entry()
    _log_query(log_entry)

    return QueryResponse(**result.to_api_response())


@app.post("/api/v1/query/stream")
@limiter.limit(lambda: _config.security.rate_limit)
async def query_stream(request: Request, body: QueryRequest):
    """
    Stream a query response as Server-Sent Events.

    Emits three kinds of events:
    - meta: {type, query_id, sources}
    - token: {type, text}
    - done: {type, citations, verification_status, confidence, latency_ms}
    - error: {type, message} (only on failure)

    Always uses async verification (Opus runs in the background after the
    stream completes). Clients should poll
    GET /api/v1/query/{query_id}/verification for the terminal status.
    """
    pipeline: Pipeline = app.state.pipeline

    async def event_generator():
        try:
            async for event_type, payload in pipeline.query_stream(
                question=body.question,
                conversation_history=body.conversation_history,
            ):
                data = {"type": event_type, **payload}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:  # pragma: no cover — last-resort guard
            logger.error("query_stream_generator_failed", error=str(e), exc_info=True)
            error_payload = json.dumps({"type": "error", "message": "Stream interrupted"}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/query/{query_id}/verification", response_model=VerificationResponse)
async def get_verification(query_id: str):
    """
    Get the verification result for a query submitted with async_verification=True.

    Returns status "pending" while the background task is still running,
    the final verification dict once complete, or status "not_found" if
    the query_id is unknown (either never submitted or evicted from the
    store after the 100-entry cap).
    """
    pipeline: Pipeline = app.state.pipeline
    verification = pipeline.get_verification(query_id)
    return VerificationResponse(
        status=verification.get("status", "not_found"),
        confidence=verification.get("confidence"),
        claims=verification.get("claims", []),
        summary=verification.get("summary", ""),
        claims_verified=verification.get("claims_verified", 0),
        claims_contradicted=verification.get("claims_contradicted", 0),
        claims_not_found=verification.get("claims_not_found", 0),
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Health check for all components."""
    config = app.state.config
    components = {}

    # Check Qdrant
    try:
        pipeline: Pipeline = app.state.pipeline
        count = pipeline.retriever.client.get_collection(config.qdrant.collection_name).points_count
        components["qdrant"] = f"ok ({count} pages)"
    except Exception as e:
        components["qdrant"] = f"error: {str(e)[:100]}"

    # Check Anthropic API
    try:
        # Lightweight check — just verify the client is configured
        if config.anthropic.api_key:
            components["anthropic_api"] = "ok (key configured)"
        else:
            components["anthropic_api"] = "warning: no API key"
    except Exception as e:
        components["anthropic_api"] = f"error: {str(e)[:100]}"

    # ColQwen2
    components["colqwen2"] = "ok (lazy load)"

    status = "healthy" if all("error" not in v for v in components.values()) else "degraded"

    return HealthResponse(status=status, components=components)


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def metrics():
    """System metrics."""
    total = int(metrics_store["queries_total"])

    return MetricsResponse(
        queries_total=total,
        avg_latency_ms=metrics_store["total_latency_ms"] / max(total, 1),
        avg_cost_per_query=metrics_store["total_cost"] / max(total, 1),
        abstention_rate=metrics_store["abstentions"] / max(total, 1),
        error_rate=metrics_store["errors"] / max(total, 1),
    )


@app.get("/api/v1/pages/{document_id}/{page_number}")
async def get_page_image(document_id: str, page_number: int):
    """Serve a page image for frontend preview."""
    import re
    from pathlib import Path

    from fastapi.responses import FileResponse

    # Validate document_id to prevent path traversal
    if not re.match(r"^[a-zA-Z0-9_-]+$", document_id):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    if page_number < 1 or page_number > 10000:
        raise HTTPException(status_code=400, detail="Invalid page number")

    config = app.state.config
    image_path = Path(config.data.pages_dir) / document_id / f"page_{page_number:04d}.png"

    # Double-check the resolved path is within pages_dir
    pages_dir = Path(config.data.pages_dir).resolve()
    if not image_path.resolve().is_relative_to(pages_dir):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")

    return FileResponse(image_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_query(entry: dict):
    """Log a query result with structlog."""
    logger.info("query_completed", **entry)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
