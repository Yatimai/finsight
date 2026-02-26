# ROADMAP

Priorites classees par impact sur la fiabilite du systeme.

## P0 — Tests coeur metier

Le cache, la config et les utils sont testes. Les modules critiques ne le sont pas.

### Tests unitaires avec mocks Anthropic

| Module | Fichier test | Ce qu'il faut tester |
|---|---|---|
| `generator.py` | `test_generator.py` | Construction du prompt, prompt caching (`cache_control`), parsing de la reponse, gestion des erreurs API |
| `verifier.py` | `test_verifier.py` | Verification sync, soumission batch, polling batch, parsing verdict (VERIFIED/REFUTED/UNCERTAIN) |
| `rewriter.py` | `test_rewriter.py` | Generation des 3 variantes, fallback sur query originale si erreur |
| `retriever.py` | `test_retriever.py` | Encoding ColQwen2 (mock), recherche Qdrant (mock), RRF scoring, deduplication |
| `pipeline.py` | `test_pipeline.py` | Orchestration complete avec mocks : cache hit, cache miss, verification async, gestion erreurs |
| `server.py` | `test_server.py` | Endpoints FastAPI avec `TestClient` : `/query`, `/verification/{id}`, `/health`, `/metrics` |

**Objectif** : coverage > 70% sur chaque module.

**Technique** : `unittest.mock.AsyncMock` pour les appels Anthropic, `MagicMock` pour Qdrant et ColQwen2. Aucune API reelle appelee dans les tests.

### Test d'integration (mock)

Un test end-to-end qui traverse tout le pipeline avec des mocks :
```
query → rewriter (mock) → retriever (mock) → generator (mock) → verifier (mock) → response
```
Verifie que les donnees circulent correctement entre les composants.

---

## P1 — Evaluation et ground truth

Sans donnees de reference, impossible de mesurer la qualite du RAG.

### Jeu de 50 questions

Fichier : `evaluation/ground_truth.json`

```json
[
  {
    "question": "Quel est le chiffre d'affaires consolide 2023 ?",
    "expected_answer": "86,2 milliards d'euros",
    "source_pages": [12],
    "document": "rapport_annuel_2023.pdf",
    "category": "chiffre_exact"
  }
]
```

**Categories** : `chiffre_exact`, `tendance`, `comparaison`, `tableau`, `graphique`, `abstention` (question hors scope).

### Script d'evaluation

Fichier : `evaluation/evaluate.py`

Metriques :
- **Recall** : la bonne page est-elle dans les top-K retrieves ?
- **Faithfulness** : la reponse est-elle fidele aux pages sources ?
- **Citation accuracy** : les `[Page X]` pointent-elles vers les bonnes pages ?
- **Abstention rate** : le systeme refuse-t-il de repondre quand il devrait ?

---

## P2 — Indexing cle en main

### Notebook Colab

Fichier : `notebooks/index_documents.ipynb`

Cellules :
1. Install des dependances (colpali-engine, qdrant-client, pdf2image)
2. Upload des PDFs ou lien Google Drive
3. Lancement de l'indexation avec barre de progression
4. Verification : nombre de documents, pages, vecteurs indexes
5. Export de la collection Qdrant (snapshot)

### Gestion des secrets

- Ajouter un fichier `.env.example` avec les variables requises
- Valider la presence de `ANTHROPIC_API_KEY` au demarrage dans `config.py`
- Ajouter `python-dotenv` dans les dependances

---

## P3 — Frontend minimal

Deux options selon l'ambition :

### Option A : Streamlit (rapide, 1 fichier)

Fichier : `frontend/app.py`

- Champ de saisie pour la question
- Affichage de la reponse avec citations cliquables
- Pages sources affichees en miniatures
- Badge de verification (VERIFIED / REFUTED / PENDING)

### Option B : React + shadcn/ui (comme dans DESIGN.md)

Plus ambitieux, correspond au design initial. A faire uniquement si le back-end est stable et teste.

---

## Ordre d'execution

```
P0 Tests coeur metier     ← prerequis a tout le reste
P1 Ground truth + eval    ← mesurer avant d'optimiser
P2 Notebook indexing      ← rendre le projet utilisable
P3 Frontend               ← rendre le projet presentable
```
