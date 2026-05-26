# FinSight

Visual RAG system for financial document analysis, ColQwen2.5, Qdrant, Claude Opus/Sonnet.

A question-answering system for financial reports (PDFs) that operates on a **full visual pipeline**: documents are processed as page images, never as extracted text. This preserves tables, charts, and layout that traditional OCR destroys.

## Demo

https://github.com/user-attachments/assets/aac2eef7-e43b-4467-8da1-99ae0fdf1c63

Three queries against the indexed corpus (5982 pages from 10 French annual reports, DEU 2024):

1. **Detailed**: *"Quel est le produit net bancaire (PNB) de BNP Paribas en 2024 ?"* → answer with breakdown by operating pole (Commercial Banking, IPS, CIB).
2. **Abstention**: *"Quelle est la prévision de croissance du PIB mondial selon le FMI pour 2025 ?"* → out of corpus, the system explicitly acknowledges what is and isn't in the documents.
3. **Factual**: *"Quel est le chiffre d'affaires consolidé de LVMH en 2024 ?"* → exact figure with a clickable page citation.

**Streaming + background verification**: tokens appear progressively (Opus SSE stream), and Sonnet verification runs as a detached `asyncio` task. The user perceives only the generation stream while the verification badge updates from `pending` to `verified` in the background. Verification latency is hidden behind subsequent user interaction.

## Architecture

```
                        INDEXING (offline, GPU)
          PDF --> Page Images --> ColQwen2.5 Vision Encoder --> Qdrant (multi-vector)

                        QUERY (runtime)
  Question --> ColQwen2.5 Text Encoder --> Qdrant 2-stage MaxSim --> Top-10 Pages
                                                                          |
                                              Opus 4.7 generation (page images, SSE stream)
                                                                          |
                                                          Response text + citations
                                                                          |
                                                          ┌───────────────┴───────────────┐
                                                          |                               |
                                                   User sees answer           Sonnet verification
                                                   immediately (streaming)    in background, async task
                                                                                          |
                                                                              Verification badge
                                                                              (pending -> verified)
```

## Stack

| Component | Technology |
|---|---|
| Retrieval | ColQwen2.5-v0.2 (`vidore/colqwen2.5-v0.2`): multi-vector, two-stage MaxSim with prefetch on pooled vectors + exact rerank |
| Storage | Qdrant (remote, native multi-vector with 3 named vectors: `colqwen2`, `pooled`, `global`) |
| Generation | Claude **Opus 4.7** (`claude-opus-4-7`) on page images, SSE streaming + system-prompt caching |
| Verification | Claude **Sonnet 4.6** (`claude-sonnet-4-6`) as a detached `asyncio.create_task`, an independent second opinion, invisible to the user |
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
    generator.py          Opus generation (batch + streaming via SSE)
    verifier.py           Sonnet adversarial verification
  security/
    output_validator.py   Citation check, anomaly detection
indexing/
  index_documents.py      PDF -> images -> ColQwen2.5 -> Qdrant
  utils.py                PDF processing, image encoding
evaluation/
  evaluate.py             CLI runner (retrieval-only, skip-verification, full)
  metrics.py              Recall@k, citation, abstention, cost
  llm_judge.py            LLM-as-judge answer-accuracy grading (Claude Haiku)
  ground_truth.json       50 non-circular questions (source pages from direct PDF reading)
frontend/                  React + Vite chat interface (SSE consumer + verification polling)
tests/                    184 tests, 0 failures
```

## Evaluation Results

Corpus: 10 DEU 2024 (French annual reports), 5982 pages indexed. Ground truth: 50 questions, non-circular (source pages extracted by direct PDF reading, not from the retriever).

| Metric | Score |
|---|---|
| Recall@1 | 36% |
| Recall@3 | 66% |
| Recall@5 | 82% |
| **Recall@10** | **92%** |
| **Answer accuracy** (LLM-judge vs verified gold, Opus 4.7) | **84% strict** |
| **Citation accuracy** (every answer cites a valid `[Page X]` from the retrieved context) | **100%** |
| Citation faithfulness (per question: at least one cited page supports the claim, judged by Sonnet 4.6) | 92% |
| Avg cost/query | ~$0.30 (Opus generation + Sonnet verification, full price, no cache/batch) |

*Answer accuracy: the generated answer is compared to a human-verified gold answer by an LLM judge (Claude Haiku 4.5, `claude-haiku-4-5-20251001`). **Strict** = key figure exactly correct. Reproducible via `evaluation/llm_judge.py`. Retrieval measured with the `pooled` two-stage prefetch config.*

## Dev

```bash
ruff check .
mypy app/ indexing/
pytest tests/ -v
```

## License

MIT
