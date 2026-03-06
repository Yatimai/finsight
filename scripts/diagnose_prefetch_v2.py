"""Diagnose prefetch loss: find GT pages found by exact MaxSim but lost in prefetch.

For each non-abstention GT question, compares:
  - Exact search: MaxSim exact=True, limit=10 (no prefetch)
  - Prefetch search: global vector prefetch limit=500, then MaxSim exact rerank limit=10

Reports questions where exact finds GT pages but prefetch does not.

Usage:
    cd /workspace/finsight
    python scripts/diagnose_prefetch_v2.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from qdrant_client.models import Prefetch, SearchParams

from app.config import load_config
from app.models.retriever import Retriever


def main() -> None:
    gt_path = Path("evaluation/ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)

    config = load_config()
    retriever = Retriever(config)
    prefetch_k = config.retrieval.prefetch_k

    print(f"Loaded {len(gt)} GT questions")
    print(f"prefetch_k = {prefetch_k}")
    print(f"Collection has global vector: {retriever.has_global_vector}")
    print()

    rows: list[dict] = []
    lost_questions: list[dict] = []

    for item in gt:
        if item["category"] == "abstention":
            continue

        qid = item["id"]
        question = item["question"]
        source_doc = item["source_document"]
        source_pages = set(item["source_pages"])

        qe = retriever.encode_query(question)

        # --- Exact search (no prefetch, brute-force MaxSim) ---
        exact_response = retriever.client.query_points(
            retriever.collection_name,
            query=qe.filtered.tolist(),
            using="colqwen2",
            limit=10,
            search_params=SearchParams(exact=True),
        )
        exact_hits = {
            pt.payload["page_number"]
            for pt in exact_response.points
            if pt.payload["source_filename"] == source_doc and pt.payload["page_number"] in source_pages
        }

        # --- Prefetch search (global vector stage 1, MaxSim rerank stage 2) ---
        prefetch_response = retriever.client.query_points(
            retriever.collection_name,
            query=qe.filtered.tolist(),
            using="colqwen2",
            limit=10,
            prefetch=[
                Prefetch(
                    query=qe.pooled.tolist(),
                    using="global",
                    limit=prefetch_k,
                )
            ],
            search_params=SearchParams(exact=True),
        )
        prefetch_hits = {
            pt.payload["page_number"]
            for pt in prefetch_response.points
            if pt.payload["source_filename"] == source_doc and pt.payload["page_number"] in source_pages
        }

        exact_hit = len(exact_hits) > 0
        prefetch_hit = len(prefetch_hits) > 0
        lost = exact_hit and not prefetch_hit

        row = {
            "id": qid,
            "exact_hit": exact_hit,
            "prefetch_hit": prefetch_hit,
            "lost_in_prefetch": lost,
            "exact_pages": sorted(exact_hits),
            "prefetch_pages": sorted(prefetch_hits),
            "gt_pages": sorted(source_pages),
        }
        rows.append(row)

        if lost:
            lost_questions.append(row)

        status = "LOST" if lost else ("ok" if prefetch_hit else ("exact_only" if exact_hit else "miss"))
        print(f"  {qid}: {status}  exact={sorted(exact_hits)}  prefetch={sorted(prefetch_hits)}")

    # Summary table
    total = len(rows)
    n_exact = sum(1 for r in rows if r["exact_hit"])
    n_prefetch = sum(1 for r in rows if r["prefetch_hit"])
    n_lost = sum(1 for r in rows if r["lost_in_prefetch"])
    n_both_miss = sum(1 for r in rows if not r["exact_hit"] and not r["prefetch_hit"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total questions (non-abstention): {total}")
    print(f"  Exact hit (top-10):               {n_exact}/{total} ({n_exact / total * 100:.0f}%)")
    print(f"  Prefetch hit (top-10):            {n_prefetch}/{total} ({n_prefetch / total * 100:.0f}%)")
    print(f"  Lost in prefetch:                 {n_lost}")
    print(f"  Both miss:                        {n_both_miss}")

    if lost_questions:
        print(f"\n{'=' * 60}")
        print("LOST IN PREFETCH (exact finds GT, prefetch does not)")
        print("=" * 60)
        for r in lost_questions:
            print(f"  {r['id']}: GT pages {r['gt_pages']}, exact found {r['exact_pages']}")


if __name__ == "__main__":
    main()
