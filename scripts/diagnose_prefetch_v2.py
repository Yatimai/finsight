"""Diagnose the 16 missed questions: classify into 3 failure groups.

Group A — Both miss: exact=False AND prefetch=False (GT not findable at all)
Group B — Lost in prefetch: exact=True AND prefetch=False (global vector misses GT)
Group C — Prefetch OK but pipeline miss: prefetch=True but eval recall=False

For each question shows: id, question (30 chars), document, GT pages,
best GT rank in prefetch pool (or "absent").

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


def find_gt_rank_in_prefetch(
    retriever: Retriever,
    qe,
    prefetch_k: int,
    source_doc: str,
    source_pages: set[int],
) -> str:
    """Find the best rank of any GT page in the global vector prefetch pool."""
    response = retriever.client.query_points(
        retriever.collection_name,
        query=qe.pooled.tolist(),
        using="global",
        limit=prefetch_k,
    )
    for rank, pt in enumerate(response.points, 1):
        if pt.payload["source_filename"] == source_doc and int(pt.payload["page_number"]) in source_pages:
            return str(rank)
    return "absent"


def load_eval_hits(results_dir: Path) -> dict[str, bool]:
    """Load the latest eval results and return {qid: recall_hit}."""
    result_files = sorted(results_dir.glob("eval_*.json"))
    if not result_files:
        print("WARNING: No eval results found, assuming all miss for Group C detection")
        return {}
    latest = result_files[-1]
    print(f"Using eval results: {latest.name}")
    with open(latest) as f:
        data = json.load(f)
    hits = {}
    for r in data.get("results", []):
        qid = r.get("question_id", "")
        recall_at_k = r.get("recall_at_k", {})
        hits[qid] = recall_at_k.get("5", False)
    return hits


def main() -> None:
    gt_path = Path("evaluation/ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)

    config = load_config()
    retriever = Retriever(config)
    prefetch_k = config.retrieval.prefetch_k

    # Load pipeline eval results for Group C detection
    eval_hits = load_eval_hits(Path("evaluation/results"))

    print(f"Loaded {len(gt)} GT questions, prefetch_k={prefetch_k}")
    print(f"Eval results loaded: {len(eval_hits)} questions")
    print()

    group_a: list[dict] = []  # Both miss
    group_b: list[dict] = []  # Lost in prefetch
    group_c: list[dict] = []  # Prefetch OK but pipeline miss

    for item in gt:
        if item["category"] == "abstention":
            continue

        qid = item["id"]
        question = item["question"]
        source_doc = item["source_document"]
        source_pages = {int(p) for p in item["source_pages"]}

        qe = retriever.encode_query(question)

        # Exact search (brute-force MaxSim, no prefetch)
        exact_response = retriever.client.query_points(
            retriever.collection_name,
            query=qe.filtered.tolist(),
            using="colqwen2",
            limit=10,
            search_params=SearchParams(exact=True),
        )
        exact_hit = any(
            pt.payload["source_filename"] == source_doc and int(pt.payload["page_number"]) in source_pages
            for pt in exact_response.points
        )

        # Prefetch search (global prefetch + MaxSim rerank)
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
        prefetch_hit = any(
            pt.payload["source_filename"] == source_doc and int(pt.payload["page_number"]) in source_pages
            for pt in prefetch_response.points
        )

        # Best GT rank in global prefetch pool
        gt_rank = find_gt_rank_in_prefetch(retriever, qe, prefetch_k, source_doc, source_pages)

        # Pipeline eval hit
        pipeline_hit = eval_hits.get(qid, False)

        row = {
            "id": qid,
            "question": question[:30],
            "document": source_doc.replace("_DEU_2024.pdf", "").replace("_Annual_Report_2024.pdf", ""),
            "gt_pages": sorted(source_pages),
            "gt_rank_in_prefetch": gt_rank,
        }

        if not exact_hit and not prefetch_hit:
            group_a.append(row)
            print(f"  {qid}: GROUP_A (both miss)")
        elif exact_hit and not prefetch_hit:
            group_b.append(row)
            print(f"  {qid}: GROUP_B (lost in prefetch)")
        elif prefetch_hit and not pipeline_hit:
            group_c.append(row)
            print(f"  {qid}: GROUP_C (prefetch ok, pipeline miss)")
        else:
            print(f"  {qid}: ok")

    # Print tables
    header = f"  {'ID':<5} {'Question':<32} {'Document':<20} {'GT pages':<20} {'Prefetch rank'}"
    sep = "  " + "-" * 95

    print(f"\n{'=' * 100}")
    print(f"GROUP A — Both miss ({len(group_a)} questions)")
    print("=" * 100)
    print(header)
    print(sep)
    for r in group_a:
        print(
            f"  {r['id']:<5} {r['question']:<32} {r['document']:<20} {r['gt_pages']!s:<20} {r['gt_rank_in_prefetch']}"
        )

    print(f"\n{'=' * 100}")
    print(f"GROUP B — Lost in prefetch ({len(group_b)} questions)")
    print("=" * 100)
    print(header)
    print(sep)
    for r in group_b:
        print(
            f"  {r['id']:<5} {r['question']:<32} {r['document']:<20} {r['gt_pages']!s:<20} {r['gt_rank_in_prefetch']}"
        )

    print(f"\n{'=' * 100}")
    print(f"GROUP C — Prefetch OK but pipeline miss ({len(group_c)} questions)")
    print("=" * 100)
    print(header)
    print(sep)
    for r in group_c:
        print(
            f"  {r['id']:<5} {r['question']:<32} {r['document']:<20} {r['gt_pages']!s:<20} {r['gt_rank_in_prefetch']}"
        )


if __name__ == "__main__":
    main()
