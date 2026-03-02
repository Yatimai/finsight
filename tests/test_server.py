"""Tests for the FastAPI server endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import reset_config


@pytest.fixture
def client():
    """Create a TestClient with mocked Pipeline and config."""
    mock_pipeline = AsyncMock()
    mock_pipeline.verification_store = MagicMock()
    mock_pipeline.retriever = MagicMock()

    # Default query result
    mock_result = MagicMock()
    mock_result.query_id = "test-query-id"
    mock_result.answer = "Réponse test [Page 5]."
    mock_result.pages = []
    mock_result.citations = [{"page": 5}]
    mock_result.verification = {"status": "verified", "confidence": 0.95}
    mock_result.total_latency_ms = 123.0
    mock_result.cache_hit = False
    mock_result.error = None
    mock_result.to_api_response.return_value = {
        "query_id": "test-query-id",
        "answer": "Réponse test [Page 5].",
        "sources": [],
        "citations": [{"page": 5}],
        "confidence": 0.95,
        "verification_status": "verified",
        "latency_ms": 123,
    }
    mock_result.to_log_entry.return_value = {}
    mock_pipeline.query.return_value = mock_result

    # Patch everything that would trigger real initialization
    with (
        patch("app.pipeline.anthropic"),
        patch("app.pipeline.Retriever"),
        patch("app.pipeline.SemanticCache"),
        patch("app.pipeline.VerificationStore"),
        patch("app.server.Pipeline", return_value=mock_pipeline),
        patch("app.server.setup_logging"),
    ):
        reset_config()
        from fastapi.testclient import TestClient

        from app.server import app

        with TestClient(app) as tc:
            yield tc, mock_pipeline


# ── Query endpoint ──────────────────────────────────────────────────


class TestQueryEndpoint:
    def test_valid_query_returns_200(self, client):
        tc, mock_pipeline = client
        response = tc.post("/api/v1/query", json={"question": "Quel est le CA 2023 ?"})
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "test-query-id"
        assert data["answer"] == "Réponse test [Page 5]."

    def test_empty_question_returns_422(self, client):
        tc, _ = client
        response = tc.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422

    def test_question_too_long_returns_422(self, client):
        tc, _ = client
        response = tc.post("/api/v1/query", json={"question": "x" * 2001})
        assert response.status_code == 422


# ── Verification endpoint ───────────────────────────────────────────


class TestVerificationEndpoint:
    def test_existing_verification_returns_200(self, client):
        tc, mock_pipeline = client
        mock_pipeline.verification_store.get.return_value = {
            "status": "verified",
            "confidence": 0.9,
            "claims": [{"id": 1, "verdict": "CONFIRMÉ"}],
            "summary": "All OK",
        }
        response = tc.get("/api/v1/query/test-id/verification")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "verified"
        assert data["confidence"] == 0.9

    def test_nonexistent_returns_404(self, client):
        tc, mock_pipeline = client
        mock_pipeline.verification_store.get.return_value = None
        response = tc.get("/api/v1/query/nonexistent/verification")
        assert response.status_code == 404


# ── Health endpoint ─────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        tc, mock_pipeline = client
        mock_pipeline.retriever.client.get_collection.side_effect = Exception("no qdrant")
        response = tc.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_shows_components(self, client):
        tc, mock_pipeline = client
        mock_pipeline.retriever.client.get_collection.side_effect = Exception("no qdrant")
        response = tc.get("/api/v1/health")
        data = response.json()
        assert "components" in data
        assert "colqwen2" in data["components"]


# ── Metrics endpoint ────────────────────────────────────────────────


class TestMetricsEndpoint:
    def test_metrics_returns_200_with_zeros(self, client):
        tc, _ = client
        response = tc.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["queries_total"] >= 0
        assert "avg_latency_ms" in data
        assert "cache_hit_rate" in data


# ── Page image endpoint ─────────────────────────────────────────────


class TestPageImageEndpoint:
    def test_path_traversal_blocked(self, client):
        """document_id with special chars returns 400."""
        tc, _ = client
        # The regex ^[a-zA-Z0-9_-]+$ blocks dots and slashes
        response = tc.get("/api/v1/pages/doc..etc/1")
        assert response.status_code == 400

    def test_invalid_document_id_returns_400(self, client):
        tc, _ = client
        response = tc.get("/api/v1/pages/doc@evil!/1")
        assert response.status_code == 400

    def test_missing_image_returns_404(self, client):
        tc, _ = client
        response = tc.get("/api/v1/pages/valid-doc/999")
        assert response.status_code == 404


# ── CORS ────────────────────────────────────────────────────────────


class TestCORS:
    def test_cors_not_wildcard(self, client):
        """Bug 4 regression: CORS must not use wildcard origins."""
        tc, _ = client
        response = tc.options(
            "/api/v1/query",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # evil.com should NOT get access-control-allow-origin
        allow_origin = response.headers.get("access-control-allow-origin")
        assert allow_origin != "*"
        assert allow_origin != "http://evil.com"
