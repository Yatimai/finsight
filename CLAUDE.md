# FinSight — CLAUDE.md

Visual RAG pour documents financiers. ColQwen2 + Qdrant + Claude Sonnet/Opus.

## Architecture

```
Query → Rewriter (Sonnet, RAG Fusion ×3) → Retriever (ColQwen2 + Qdrant + RRF)
      → Generator (Sonnet, avec pages images) → Verifier (Opus, adversarial)
      → SemanticCache (LRU thread-safe)
```

### Modules

| Module | Rôle |
|---|---|
| `app/models/rewriter.py` | Génère 3 variantes de la question (RAG Fusion) |
| `app/models/retriever.py` | Encode query ColQwen2, recherche Qdrant, fusionne par RRF |
| `app/models/generator.py` | Construit le prompt multimodal (images pages), appelle Sonnet |
| `app/models/verifier.py` | Vérifie la réponse contre les sources (Opus sync ou batch async) |
| `app/pipeline.py` | Orchestre le flux complet : cache → rewrite → retrieve → generate → verify |
| `app/server.py` | FastAPI : endpoints `/query`, `/verification`, `/health`, `/metrics`, `/pages` |
| `app/cache/semantic_cache.py` | Cache LRU avec similarité cosinus, thread-safe (Lock) |
| `app/cache/verification_store.py` | Persistance JSON des résultats de vérification |
| `app/config.py` | Configuration YAML + Pydantic (AppConfig) |
| `app/errors.py` | Retry exponential backoff, `extract_text_from_response()`, exceptions custom |
| `app/security/output_validator.py` | Validation Pydantic des sorties |
| `indexing/index_documents.py` | Indexation PDF → images → ColQwen2 → Qdrant (GPU requis) |
| `indexing/preprocessing.py` | Conversion PDF → images par page |

## Coeur métier (coverage cible : 85%)

- `app/models/generator.py` — 98%
- `app/models/verifier.py` — 67% (à monter)
- `app/pipeline.py` — 96%
- `app/errors.py` — 48% (à monter, les fonctions retry sync/async ne sont pas testées)

## Commandes

**IMPORTANT** : toujours utiliser le venv (`.venv/bin/python`, `.venv/bin/pytest`). Le Python système n'a pas `slowapi`, `colpali_engine`, etc. → import errors.

```bash
# Tests (pas de GPU, pas d'API réelle)
.venv/bin/pytest -v -m "not slow and not gpu and not integration"

# Coverage
.venv/bin/pytest --cov=app --cov=indexing --cov-report=term-missing -m "not slow and not gpu and not integration"

# Lint + format
.venv/bin/ruff check . && .venv/bin/ruff format --check .

# Types
.venv/bin/mypy app/ --ignore-missing-imports

# Serveur local (charger les clés API avant)
set -a && source ~/GILLES\ AI/API_KEYS.env && set +a
.venv/bin/uvicorn app.server:app --reload --port 8000
```

## Pièges connus

1. **Qdrant en test** : toujours mocker `app.pipeline.Retriever` et `qdrant_client`. Ne jamais instancier un vrai client Qdrant en test (lock file, crash).

2. **Anthropic en test** : utiliser `AsyncMock` pour le client async (pipeline, generator, rewriter) et `MagicMock` pour le client sync (verifier batch). Ne jamais appeler l'API réelle.

3. **`should_abstain` est sync** : dans `pipeline.py`, `should_abstain()` est une méthode synchrone. Utiliser `MagicMock(return_value=False)`, pas `AsyncMock` (un coroutine est toujours truthy → le test passe pour la mauvaise raison).

4. **`extract_text_from_response()`** : toujours utiliser cette fonction (dans `app/errors.py`) au lieu d'accéder directement à `response.content[0].text`. Protège contre les réponses vides.

5. **`round()` Python** : utilise le banker's rounding. `round(1234.5)` = 1234, pas 1235. Ne pas hardcoder des valeurs arrondies dans les assertions.

6. **FastAPI TestClient** : le fixture doit patcher `app.server.Pipeline` (pas importer Pipeline directement). Utiliser `reset_config()` avant l'import de `app.server` pour éviter les configs stale.

7. **`config.yaml` est gitignored** : le fichier versionné est `config.example.yaml`. Les tests utilisent `AppConfig()` par défaut (sans fichier).

8. **Indexation GPU-only** : `indexing/index_documents.py` nécessite un GPU (ColQwen2). Marqué `@pytest.mark.gpu`. Le snapshot Qdrant est dans `data/qdrant/` (non versionné, 3.2 Go).

9. **Qdrant nécessite Docker** : les données ont été migrées vers le serveur Docker Qdrant. Toujours lancer le container avant l'API : `docker start qdrant`. Le mode `embedded` ne passe pas à l'échelle (5 GB en RAM → freeze).

10. **ColQwen2 en bfloat16** : le modèle est toujours chargé en bfloat16 (~3.7 Go). Sur WSL avec 12 Go, ça passe mais c'est serré avec Docker Qdrant en parallèle.

11. **`evaluate_single` est async** : les tests doivent utiliser `@pytest.mark.asyncio` et `await`. Le pipeline mock utilise `AsyncMock` pour `pipeline.query`.

## Fixtures partagées (`tests/conftest.py`)

- `config()` → `AppConfig()` par défaut
- `fake_pages()` → 3 FakePage avec page_numbers distincts
- `mock_anthropic_response(text, input_tokens, output_tokens)` → factory
- `mock_empty_anthropic_response()` → `.content = []`
- `mock_anthropic_client()` → `AsyncMock()`

## Évaluation (P2)

Infrastructure dans `evaluation/` :

```bash
# Évaluation complète
python -m evaluation.evaluate --ground-truth evaluation/ground_truth.json

# Retrieval seul (gratuit)
python -m evaluation.evaluate --retrieval-only

# Sans vérification Opus (~$2 au lieu de ~$13)
python -m evaluation.evaluate --skip-verification
```

Résultats dans `evaluation/results/` (gitignored).

### Ground truth (`evaluation/ground_truth.json`)

- 50 questions, 48 avec réponse, 2 abstentions (q47, q50)
- **Non-circulaire** : les `source_pages` ont été extraites en lisant directement les PDFs (pas le retriever)
- `source_pages` limitées à 1-3 pages par question (recall mesuré rigoureusement)
- q23 (cours action SocGen) et q39 (prix Brent) reclassées de `abstention` vers `chiffre_exact` (données trouvées dans les PDFs)
- `source_document` correspond aux noms de fichiers dans Qdrant (ex : `LVMH_DEU_2024.pdf`)
- Page numbers = physical PDF pages (1-indexed), matching `page_number` in Qdrant

## État actuel

- 225 tests, 83% coverage
- P0 (indexation) et P1 (tests coeur métier) terminés
- P2 (ground truth) reconstruit non-circulairement — baseline à re-mesurer avec le nouveau GT
- P3 (frontend) pas commencé
- P4 (durcissement) partiel : rate limiting et CORS OK, manque auth/circuit breaker/audit trail

### Baseline P2 (ancien GT circulaire, obsolète)

| Métrique | Valeur |
|---|---|
| Recall@1 | 34% |
| Recall@3 | 52% |
| Recall@5 | 54% |
| Citation accuracy | 52% |
| Abstention recall | 75% |
| Coût/query | $0.029 |

> Ces chiffres sont basés sur l'ancien GT (circulaire, 5 pages/question). Avec le nouveau GT (1-3 pages, non-circulaire), les scores seront différents. Re-lancer l'évaluation pour obtenir la nouvelle baseline.
