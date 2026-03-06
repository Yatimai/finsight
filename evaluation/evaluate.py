"""Evaluation runner CLI.

Loads ground truth, executes pipeline on each question,
computes metrics, and generates a report.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from evaluation.metrics import build_report
from evaluation.models import EvaluationResult, GroundTruthItem, RetrievedSource

DEFAULT_GT_PATH = Path("evaluation/ground_truth.json")
DEFAULT_OUTPUT_DIR = Path("evaluation/results")


def load_ground_truth(path: Path) -> list[GroundTruthItem]:
    """Load and validate ground truth from JSON file."""
    with open(path) as f:
        data = json.load(f)
    items = [GroundTruthItem.model_validate(item) for item in data]
    # Validate uniqueness of IDs
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in ground truth")
    return items


async def evaluate_single(
    pipeline,
    item: GroundTruthItem,
    skip_verification: bool = False,
    retrieval_only: bool = False,
) -> EvaluationResult:
    """Evaluate a single ground truth question against the pipeline.

    Args:
        pipeline: Pipeline instance (or mock).
        item: Ground truth question.
        skip_verification: Skip Opus verification.
        retrieval_only: Only test retrieval, skip generation.

    Returns:
        EvaluationResult with recall, citations, abstention, tokens.
    """
    t0 = time.time()

    # Run the pipeline
    query_result = await pipeline.query(
        item.question,
        skip_verification=skip_verification or retrieval_only,
    )

    retrieved_pages = [p.page_number for p in query_result.pages]
    retrieved_sources = [RetrievedSource(document=p.source_filename, page=p.page_number) for p in query_result.pages]

    # Expected (document, page) pairs from ground truth
    expected_pairs = {(item.source_document, p) for p in item.source_pages}

    # Compute recall@k — match on (document, page) pairs, not just page number
    recall_at_k = {}
    for k in (1, 3, 5, 10):
        top_k_pairs = {(s.document, s.page) for s in retrieved_sources[:k]}
        recall_at_k[k] = bool(top_k_pairs & expected_pairs)

    # Citation analysis
    cited_pages: list[int] = []
    citation_correct = False
    if not retrieval_only and query_result.answer:
        cited_pages = [c.get("page", 0) for c in query_result.citations]
        if item.source_pages and cited_pages:
            # Citations only have page numbers — check pages AND that the right document is retrieved
            cited_set = set(cited_pages)
            retrieved_doc_pages = {(s.document, s.page) for s in retrieved_sources}
            citation_correct = bool({(item.source_document, p) for p in cited_set} & retrieved_doc_pages)

    # Abstention detection
    abstention_keywords = [
        "n'apparaît pas",
        "pas dans les documents",
        "ne figure pas",
        "hors du scope",
        "pas disponible dans",
    ]
    did_abstain = any(kw in query_result.answer.lower() for kw in abstention_keywords)

    # Faithfulness score from verification
    faithfulness_score = None
    if query_result.verification and query_result.verification.get("confidence") is not None:
        faithfulness_score = query_result.verification["confidence"]

    latency_ms = (time.time() - t0) * 1000

    return EvaluationResult(
        question_id=item.id,
        retrieved_pages=retrieved_pages,
        retrieved_sources=retrieved_sources,
        recall_at_k=recall_at_k,
        generated_answer=query_result.answer if not retrieval_only else "",
        faithfulness_score=faithfulness_score,
        cited_pages=cited_pages,
        citation_correct=citation_correct,
        should_abstain=item.category == "abstention",
        did_abstain=did_abstain,
        input_tokens=query_result.generation_tokens.get("input_tokens", 0),
        output_tokens=query_result.generation_tokens.get("output_tokens", 0),
        latency_ms=latency_ms,
    )


def format_report_markdown(report) -> str:
    """Format an EvaluationReport as Markdown summary for stdout."""
    lines = [
        "# Evaluation Report",
        "",
        f"**Questions**: {report.total_questions}",
        "",
        "## Recall",
        f"- Recall@1: {report.recall_at_1:.1%}",
        f"- Recall@3: {report.recall_at_3:.1%}",
        f"- Recall@5: {report.recall_at_5:.1%}",
        f"- Recall@10: {report.recall_at_10:.1%}",
        "",
        "## Quality",
        f"- Citation accuracy: {report.citation_accuracy:.1%}",
        f"- Abstention precision: {report.abstention_precision:.1%}",
        f"- Abstention recall: {report.abstention_recall:.1%}",
        "",
        "## Cost",
        f"- Average per query: ${report.cost_per_query_usd:.4f}",
        "",
        "## By Category",
    ]
    for cat, metrics in sorted(report.by_category.items()):
        count = int(metrics.get("count", 0))
        r1 = metrics.get("recall_at_1", 0)
        lines.append(f"- **{cat}** ({count}): recall@1={r1:.1%}")
    return "\n".join(lines)


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate FinSight RAG pipeline against ground truth.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GT_PATH,
        help="Path to ground truth JSON (default: evaluation/ground_truth.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for report JSON (default: evaluation/results/eval_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip Opus verification (~$2 instead of ~$13)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only test retrieval, skip generation (free)",
    )
    parser.add_argument(
        "--no-rewriter",
        action="store_true",
        help="Disable query rewriting (raw query only, no RAG Fusion)",
    )
    args = parser.parse_args(argv)

    # Load ground truth
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        return 1

    items = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(items)} ground truth questions.")

    # Initialize pipeline
    from app.config import get_config
    from app.pipeline import Pipeline

    config = get_config()
    pipeline = Pipeline(config)

    if args.no_rewriter:
        pipeline.rewriter.enabled = False

    # Evaluate
    results: list[EvaluationResult] = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.question[:60]}...")
        result = await evaluate_single(
            pipeline,
            item,
            skip_verification=args.skip_verification,
            retrieval_only=args.retrieval_only,
        )
        results.append(result)

    # Build report
    gt_map = {item.id: item.category.value for item in items}
    model = "sonnet"
    report = build_report(results, gt_map, model=model)

    # Save report
    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"eval_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"\nReport saved to: {output_path}")

    # Print summary
    print()
    print(format_report_markdown(report))

    return 0


def main(argv: list[str] | None = None) -> int:
    import asyncio

    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())
