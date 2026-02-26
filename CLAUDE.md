# CLAUDE.md

## Projet
Multimodal RAG system pour l'analyse de documents financiers. Pipeline 100% visuel : les PDFs sont traités comme des images, jamais comme du texte extrait.

## Architecture
Lire **DESIGN.md** pour toutes les décisions architecturales détaillées (980 lignes).

## Stack
- **Retrieval**: ColQwen2 v1.0 (multi-vector, MaxSim)
- **Storage**: Qdrant embedded (multi-vector natif)
- **Generation**: Claude Sonnet 4.5/4.6 via Anthropic API
- **Verification**: Claude Opus 4.6 batch async, prompt adversarial
- **Query Rewriting**: RAG Fusion (3 interpretations + RRF)
- **Frontend**: React + shadcn/ui + Tailwind CSS
- **API**: FastAPI async

## Structure du code
```
app/config.py              → Configuration Pydantic depuis config.yaml
app/errors.py              → Retry logic, exponential backoff (structlog)
app/logging.py             → Structured logging (structlog JSON)
app/models/rewriter.py     → RAG Fusion query rewriting (Sonnet)
app/models/retriever.py    → ColQwen2 encoding + Qdrant search + RRF
app/models/generator.py    → Sonnet generation avec system prompt
app/models/verifier.py     → Opus adversarial verification (sync + batch API)
app/cache/semantic_cache.py→ Cache sémantique ColQwen2 (MaxSim, seuil 0.98)
app/cache/verification_store.py → Persistence JSON des vérifications async
app/security/              → Output validation, anomaly detection
app/pipeline.py            → Orchestration end-to-end (cache → rewrite → retrieve → generate → verify)
app/server.py              → FastAPI endpoints
indexing/                  → PDF → images → ColQwen2 → Qdrant (GPU, Colab)
```

## Commandes
```bash
# Setup
cp config.example.yaml config.yaml  # Ajouter ANTHROPIC_API_KEY
pip install -r requirements.txt

# Indexing (nécessite GPU)
python -m indexing.index_documents --dir data/documents/

# Server
python -m app.server
# → http://localhost:8000/docs

# Docker
docker-compose up
```

## Conventions
- Python 3.11+, type hints partout
- Config externalisée dans config.yaml (pas de hardcoded values)
- Structured logging JSON par requête
- Chaque composant est testable indépendamment
- Français pour les prompts et réponses, anglais pour le code

## Points d'attention
- **Full visual** : JAMAIS d'extraction de texte des PDFs. Tout passe par les images.
- **Citations** : le format `[Page X]` est parsé par le frontend. Ne pas changer sans mettre à jour le parsing.
- **Erreurs API** : chaque composant a sa propre stratégie de retry (voir DESIGN.md section 7.3)
- **Prompt caching** : le system prompt utilise `cache_control: ephemeral` pour le caching Anthropic
