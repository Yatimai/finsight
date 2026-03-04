"""Tests for configuration loading."""

import os
from unittest.mock import patch

from app.config import AppConfig, CachingConfig, load_config, reset_config


class TestAppConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.retrieval.model == "vidore/colqwen2.5-v0.2"
        assert config.retrieval.top_k == 10
        assert config.generation.model == "claude-sonnet-4-5-20250929"
        assert config.verification.model == "claude-opus-4-6"
        assert config.verification.mode == "batch_async"
        assert config.caching.similarity_threshold == 0.98
        assert config.qdrant.mode == "embedded"

    def test_cache_threshold_is_098(self):
        """Regression: threshold must be 0.98, not 0.95.
        'CA 2023' vs 'CA 2022' must NOT be a cache hit."""
        config = CachingConfig()
        assert config.similarity_threshold == 0.98

    def test_load_config_missing_file(self):
        reset_config()
        config = load_config("/nonexistent/path/config.yaml")
        assert isinstance(config, AppConfig)
        assert config.retrieval.model == "vidore/colqwen2.5-v0.2"

    def test_env_var_override(self):
        reset_config()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            config = load_config("/nonexistent/path/config.yaml")
            assert config.anthropic.api_key == "test-key-123"

    def test_reset_config(self):
        reset_config()
        from app.config import _config

        assert _config is None

    def test_anthropic_timeout_default(self):
        """Bug 2 regression: AnthropicConfig must have a timeout_seconds field."""
        config = AppConfig()
        assert hasattr(config.anthropic, "timeout_seconds")
        assert config.anthropic.timeout_seconds == 30.0

    def test_cors_default_not_wildcard(self):
        """Bug 4 regression: CORS must not default to wildcard."""
        config = AppConfig()
        assert "*" not in config.security.allowed_origins

    def test_cors_default_is_localhost(self):
        """Bug 4 regression: default origins should include localhost."""
        config = AppConfig()
        origins = config.security.allowed_origins
        assert any("localhost" in o for o in origins)

    def test_reranking_disabled_by_default(self):
        """Reranking must be disabled by default (no GPU in CI)."""
        config = AppConfig()
        assert config.reranking.enabled is False
        assert config.reranking.top_k == 5

    def test_rate_limit_config_exists(self):
        """Bug 5 regression: SecurityConfig must have a rate_limit field."""
        config = AppConfig()
        assert hasattr(config.security, "rate_limit")
        assert config.security.rate_limit == "10/minute"
