# ROADMAP

Priorites classees par impact sur la fiabilite du systeme.

## P0 — Indexation RunPod + premier run end-to-end (FAIT)

### Indexation sur RunPod H200 (fait)

10 documents, 3 836 pages indexees en ~24 min sur H200 (ColQwen2 bfloat16 + flash attention).
Snapshot Qdrant recupere en local (753 Mo compresse, 3.2 Go sur disque dans `data/qdrant/`).
Le pipeline peut tourner en local en mode embedded sans GPU.

### Premier test end-to-end (fait)

5 queries testees sur RunPod (LVMH, Tesla, BNP, Walmart, TotalEnergies) :
- Retrieval 100% top-1
- RAG Fusion 100%
- Batch async ~9s

### Reste a faire

- Gestion des secrets : `.env.example`, validation `ANTHROPIC_API_KEY` au demarrage, `python-dotenv`

---

## P1 — Tests coeur metier (en cours)

82 tests collectes. Les modules support sont couverts. Les modules critiques du pipeline restent a tester.

### Etat des tests

| Module | Fichier test | Statut | Ce qu'il faut tester |
|---|---|---|---|
| `config.py` | `test_config.py` | **FAIT** | |
| `errors.py` | `test_errors.py` | **FAIT** | |
| `utils.py` | `test_utils.py` | **FAIT** | |
| `semantic_cache.py` | `test_semantic_cache.py` | **FAIT** | |
| `verification_store.py` | `test_verification_store.py` | **FAIT** | |
| `retriever.py` | `test_retriever_twostage.py` | **FAIT** | |
| `preprocessing.py` | `test_preprocessing.py` | **FAIT** | |
| `index_documents.py` | `test_index_documents.py` | **FAIT** | |
| `output_validator.py` | `test_output_validator.py` | **FAIT** | |
| `verifier.py` | `test_verifier.py` | **FAIT** (18 tests) | Parse, disabled, abstain, content, batch_async |
| `generator.py` | `test_generator.py` | A faire | Construction du prompt, prompt caching (`cache_control`), parsing de la reponse, gestion des erreurs API |
| `rewriter.py` | `test_rewriter.py` | A faire | Generation des 3 variantes, fallback sur query originale si erreur |
| `pipeline.py` | `test_pipeline.py` | A faire | Orchestration complete avec mocks : cache hit, cache miss, verification async, gestion erreurs |
| `server.py` | `test_server.py` | A faire | Endpoints FastAPI avec `TestClient` : `/query`, `/verification/{id}`, `/health`, `/metrics` |

### Fix Walmart (fait)

Normalisation numerique multi-locale dans le verifier prompt (commit `6c1ecb0`).
Le verifier comparait `611,289` (format US) a `611 289` (format FR) et concluait a tort REFUTED.
Test de regression : `test_verifier.py::TestSystemPrompt::test_multi_locale_instruction_present`.

### Tests unitaires — technique

`unittest.mock.AsyncMock` pour les appels Anthropic, `MagicMock` pour Qdrant et ColQwen2. Aucune API reelle appelee dans les tests.

### Test d'integration (mock)

Un test end-to-end qui traverse tout le pipeline avec des mocks :
```
query → rewriter (mock) → retriever (mock) → generator (mock) → verifier (mock) → response
```
Verifie que les donnees circulent correctement entre les composants.

**Objectif** : coverage > 70% sur chaque module.

---

## P2 — Evaluation et ground truth (EN COURS)

Infrastructure d'evaluation pour mesurer la qualite du RAG.

### Fichiers

| Fichier | Role |
|---|---|
| `evaluation/models.py` | Modeles Pydantic (GroundTruthItem, EvaluationResult, EvaluationReport) |
| `evaluation/metrics.py` | Fonctions de metriques pures (recall, citations, abstention, cout) |
| `evaluation/evaluate.py` | Runner CLI : charge ground truth, execute pipeline, genere rapport |
| `evaluation/bootstrap_ground_truth.py` | Bootstrap : execute retrieval pour decouvrir les pages sources |
| `evaluation/questions_draft.json` | 50 questions candidates (18 chiffre_exact, 10 tendance, 8 comparaison, 6 tableau, 4 graphique, 4 abstention) |
| `evaluation/ground_truth.json` | Ground truth final (sortie du bootstrap, a generer) |
| `tests/test_evaluation.py` | 34 tests (modeles + metriques + runner) |

### Metriques

- **Recall@K** : la bonne page est-elle dans les top-K retrieves ?
- **Citation accuracy** : les `[Page X]` pointent-elles vers les bonnes pages ?
- **Abstention precision/recall** : le systeme refuse-t-il quand il devrait ?
- **Cout moyen** : tokens input/output, prix par query (Sonnet / Opus)

### Reste a faire

- Executer `bootstrap_ground_truth.py` avec Qdrant + ColQwen2 pour remplir les pages sources
- Revue manuelle du ground truth (reponses attendues)
- Premier run d'evaluation complet

---

## P3 — Frontend

### Option A : Streamlit (rapide, 1 fichier)

Fichier : `frontend/app.py`

- Champ de saisie pour la question
- Affichage de la reponse avec citations cliquables
- Pages sources affichees en miniatures
- Badge de verification (VERIFIED / REFUTED / PENDING)

### Option B : React + shadcn/ui (comme dans DESIGN.md)

Plus ambitieux, correspond au design initial. A faire uniquement si le back-end est stable et teste.

---

## P4 — Durcissement production

### Deja en place

- Verification adversariale Opus (sync + batch async -50% cout)
- Abstention si confiance < 0.7
- Citations page par page
- Retry + backoff exponentiel (retry-after header)
- Logs structures JSON (structlog)
- Verification store persistant (JSON file-backed)
- Input validation (Pydantic, min/max length)
- Path traversal protection
- Output validation (`output_validator.py`)
- Endpoints `/health` et `/metrics`
- CORS middleware (a restreindre)

### A implementer

1. **Auth** — middleware API key (~30 lignes)
2. **Rate limiting** — slowapi (~15 lignes)
3. **Circuit breaker** — fast-fail apres N echecs consecutifs (~40 lignes)
4. **Timeout global** — `asyncio.wait_for()` sur le pipeline (~5 lignes)
5. **Versioning modele** — model ID dans les resultats de verification (~10 lignes)
6. **Audit trail complet** — persistance query→reponse→verification (~SQLite)
7. **CORS** — restreindre `allow_origins=["*"]` a la liste des origines autorisees
8. **Graceful shutdown** — finir les taches batch en cours avant arret
9. **Secrets** — `.env.example` + `python-dotenv` (deja liste en P0)
10. **Verification claim-by-claim** — le verifier decoupe la reponse en affirmations et verifie chacune contre la source
11. **Invalidation cache** — TTL sur le cache semantique + invalidation quand un document est re-indexe
12. **Human-in-the-loop** — les reponses basse confiance mises en attente au lieu d'etre envoyees

### Quick win latence (optionnel)

Paralleliser les 3 rewrites avec `asyncio.gather()` : 2.5s → ~0.9s, -1.6s sur le total.

---

## CI/CD

Fichier : `.github/workflows/ci.yml` (existe).

Pipeline : lint (ruff) → type-check (mypy) → tests (pytest).

A verifier : le workflow tourne-t-il sur chaque push/PR ? Ajouter le badge de statut dans le README.

---

## Ordre d'execution

```
P0 Indexation + premier run   ← FAIT
P1 Tests coeur metier         ← EN COURS (verifier fait, reste generator/rewriter/pipeline/server)
P2 Ground truth + eval        ← EN COURS (infra OK, bootstrap a executer)
P3 Frontend                   ← pas commence
P4 Durcissement production    ← partiel (verification OK, infra manque)
```
