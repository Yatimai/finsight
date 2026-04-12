"""
Configuration management for the Multimodal RAG system.
Loads from config.yaml with environment variable overrides.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    model: str = "vidore/colqwen2.5-v0.2"
    top_k: int = 10
    mask_non_image_embeddings: bool = True
    prefetch_k: int = 500
    border_crop: bool = True


class GenerationConfig(BaseModel):
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024
    temperature: float = 0.0


class VerificationConfig(BaseModel):
    model: str = "claude-opus-4-6"
    fallback_models: list[str] = ["claude-opus-4-5-20250929"]
    enabled: bool = True
    confidence_threshold: float = 0.7
    abstention_message: str = "Cette information n'apparaît pas dans les documents fournis."


class QdrantConfig(BaseModel):
    mode: str = "remote"  # "remote" required in production; "embedded" only for tests
    path: str = "./data/qdrant"
    collection_name: str = "financial_pages"
    remote_url: str | None = "http://localhost:6333"
    timeout_seconds: float = 120.0


class ErrorHandlingConfig(BaseModel):
    generation_max_retries: int = 3
    verification_max_retries: int = 5
    backoff_base: int = 2
    use_retry_after_header: bool = True


class SecurityConfig(BaseModel):
    prompt_separation: bool = True
    output_citation_check: bool = True
    log_anomalies: bool = True
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8501"]
    rate_limit: str = "10/minute"


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    trace_all_requests: bool = True
    export_metrics: bool = True


class AnthropicConfig(BaseModel):
    api_key: str = ""
    timeout_seconds: float = 30.0


class DataConfig(BaseModel):
    documents_dir: str = "./data/documents"
    pages_dir: str = "./data/pages"
    dpi: int = 300
    batch_size: int = 8
    chunk_size: int = 50


class AppConfig(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    data: DataConfig = Field(default_factory=DataConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML file with env var overrides."""

    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Environment variable overrides
    env_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_api_key:
        raw.setdefault("anthropic", {})["api_key"] = env_api_key

    return AppConfig(**raw)


# Singleton instance
_config: AppConfig | None = None


def get_config(config_path: str = "config.yaml") -> AppConfig:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config():
    """Reset the singleton (for testing)."""
    global _config
    _config = None
