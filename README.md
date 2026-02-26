# FinSight

Visual RAG system for financial document analysis — ColQwen2, Qdrant, Claude Sonnet/Opus.

A question-answering system for financial reports (PDFs) that operates on a **full visual pipeline**: documents are processed as page images, never as extracted text. This preserves tables, charts, and layout that traditional OCR destroys.

## Architecture

```
                        INDEXING (offline, GPU)
          PDF --> Page Images --> ColQwen2 Vision Encoder --> Qdrant (multi-vector)

                        QUERY (runtime, CPU + API)
  Question --> Semantic Cache --[HIT]--> Cached Response
                   |
                [MISS]
                   |
            RAG Fusion (Sonnet) --> 3 query interpretations
                   |
            ColQwen2 Text Encoder --> 3x Qdrant MaxSim --> RRF Fusion --> Top-5 Pages
                                                                            |
                                                                    Sonnet Generation
                                                                            |
                                                                    Opus Verification
                                                                      (adversarial)
                                                                            |
                                                                  Confidence >= threshold?
                                                                    /              \
                                                               Response         Abstention
```

## Stack

| Component | Technology |
|---|---|
| Retrieval | ColQwen2 v1.0 (multi-vector, MaxSim) |
| Storage | Qdrant embedded (native multi-vector) |
| Generation | Claude Sonnet 4.5/4.6 with prompt caching |
| Verification | Claude Opus 4.6 via Batch API (50% cost) |
| Query Rewriting | RAG Fusion (3 interpretations + RRF k=60) |
| Semantic Cache | ColQwen2 embeddings, MaxSim threshold 0.98 |
| API | FastAPI async |
| Logging | structlog (JSON) |

## Quick Start

```bash
git clone https://github.com/gillesturpin/finsight.git
cd finsight
pip install -r requirements.txt
cp config.example.yaml config.yaml  # add ANTHROPIC_API_KEY
python -m app.server
# --> http://localhost:8000/docs
```

## Indexing (requires GPU)

```bash
python -m indexing.index_documents --dir data/documents/
```

Designed for Google Colab. PDFs are converted to page images, encoded via ColQwen2 vision encoder, and stored as multi-vector embeddings in Qdrant.

## Project Structure

```
app/
  config.py              Pydantic config from config.yaml
  errors.py              Retry logic, exponential backoff
  logging.py             Structured logging (structlog JSON)
  pipeline.py            End-to-end orchestration
  server.py              FastAPI endpoints
  models/
    rewriter.py           RAG Fusion query rewriting (Sonnet)
    retriever.py          ColQwen2 encoding + Qdrant search + RRF
    generator.py          Sonnet generation with system prompt
    verifier.py           Opus adversarial verification (sync + batch)
  cache/
    semantic_cache.py     ColQwen2-based semantic cache (MaxSim, 0.98)
    verification_store.py Persistent JSON store for async results
  security/
    output_validator.py   Citation check, anomaly detection
indexing/
  index_documents.py      PDF -> images -> ColQwen2 -> Qdrant
  utils.py                PDF processing, image encoding
tests/                    38 tests (config, cache, validation, utils)
```

## Dev

```bash
ruff check .
mypy app/ indexing/
pytest tests/ -v
```

## Design Decisions

See [DESIGN.md](./DESIGN.md) for the full architecture document (980 lines) covering all component choices, trade-offs, and rationale.

## License

MIT
