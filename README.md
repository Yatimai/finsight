# FinSight

Visual RAG system for financial document analysis, ColQwen2.5, Qdrant, Claude Sonnet/Opus.

A question-answering system for financial reports (PDFs) that operates on a **full visual pipeline**: documents are processed as page images, never as extracted text. This preserves tables, charts, and layout that traditional OCR destroys.

## Demo

https://github.com/user-attachments/assets/aac2eef7-e43b-4467-8da1-99ae0fdf1c63

Three queries against the indexed corpus (5982 pages from 10 French annual reports, DEU 2024):

1. **Detailed** — *"Quel est le produit net bancaire (PNB) de BNP Paribas en 2024 ?"* → answer with breakdown by operating pole (Commercial Banking, IPS, CIB).
2. **Abstention** — *"Quelle est la prévision de croissance du PIB mondial selon le FMI pour 2025 ?"* → out of corpus, the system explicitly acknowledges what is and isn't in the documents.
3. **Factual** — *"Quel est le chiffre d'affaires consolidé de LVMH en 2024 ?"* → exact figure with a clickable page citation.

**Streaming + background verification**: tokens appear progressively (Sonnet SSE stream), and Opus verification runs as a detached `asyncio` task. The user perceives only Sonnet's streaming latency (~5-10 s per query) while the verification badge updates from `pending` to `verified` in the background — Opus latency is hidden behind subsequent user interaction.

## Architecture

```
                        INDEXING (offline, GPU)
          PDF --> Page Images --> ColQwen2.5 Vision Encoder --> Qdrant (multi-vector)

                        QUERY (runtime, streaming)
  Question --> ColQwen2.5 Text Encoder --> Qdrant 2-stage MaxSim --> Top-10 Pages
                                                                          |
                                                                   Sonnet Stream (SSE)
                                                                          |
                                                          Response text + citations
                                                                          |
                                                          ┌───────────────┴───────────────┐
                                                          |                               |
                                                   User sees answer           Opus runs in background
                                                   immediately                asyncio.create_task
                                                                                          |
                                                                              Verification badge updates
                                                                              (pending → verified)
```

## Stack

| Component | Technology |
|---|---|
| Retrieval | ColQwen2.5-v0.2 (multi-vector, two-stage MaxSim: prefetch on pooled vectors + exact rerank) |
| Storage | Qdrant (remote, native multi-vector with 3 named vectors: `colqwen2`, `pooled`, `global`) |
| Generation | Claude Sonnet 4.5/4.6 with SSE streaming and system-prompt caching |
| Verification | Claude Opus 4.6 as a detached `asyncio.create_task` — invisible to the user |
| API | FastAPI async with streaming endpoint (`POST /api/v1/query/stream`) and polling endpoint (`GET /api/v1/query/{id}/verification`) |
| Frontend | React + Vite + base-ui, consumes the SSE stream and polls the verification endpoint |
| Logging | structlog (JSON) |

## Quick Start

```bash
git clone https://github.com/Yatimai/finsight.git
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

Requires a GPU (RTX 4090 or better). PDFs are converted to page images, encoded via ColQwen2.5 vision encoder, and stored as multi-vector embeddings in Qdrant.

## Project Structure

```
app/
  config.py              Pydantic config from config.yaml
  errors.py              Retry logic, exponential backoff
  logging.py             Structured logging (structlog JSON)
  pipeline.py            End-to-end orchestration (query + query_stream)
  server.py              FastAPI endpoints (query, stream, verification polling)
  models/
    retriever.py          ColQwen2.5 encoding + Qdrant two-stage MaxSim
    generator.py          Sonnet generation (batch + streaming via SSE)
    verifier.py           Opus adversarial verification
  security/
    output_validator.py   Citation check, anomaly detection
indexing/
  index_documents.py      PDF -> images -> ColQwen2.5 -> Qdrant
  utils.py                PDF processing, image encoding
evaluation/
  evaluate.py             CLI runner (retrieval-only, skip-verification, full)
  metrics.py              Recall@k, citation accuracy, abstention, cost
  ground_truth.json       50 non-circular questions (source pages from direct PDF reading)
frontend/                  React + Vite chat interface (SSE consumer + verification polling)
tests/                    183 tests, 0 failures
```

## Evaluation Results

Corpus: 10 DEU 2024 (French annual reports), 5982 pages indexed. Ground truth: 50 questions, non-circular (source pages extracted by direct PDF reading, not from the retriever).

| Metric | Score |
|---|---|
| Recall@1 | 36% |
| Recall@3 | 68% |
| Recall@5 | 78% |
| **Recall@10** | **90%** |
| **Citation accuracy** | **100%** |
| User-perceived latency (warm) | ~5-10 s per query (streaming Sonnet, Opus hidden) |
| Opus verification latency (background) | ~15-25 s per query |
| Avg cost/query | ~$0.15 (Sonnet ~$0.05 + Opus ~$0.10, full price, no cache, no batch) |

## Dev

```bash
ruff check .
mypy app/ indexing/
pytest tests/ -v
```

## License

MIT
