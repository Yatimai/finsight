"""Pure metric computation functions for evaluation results.

All functions are pure — no API calls, no side effects.
"""

from evaluation.models import EvaluationReport, EvaluationResult

# Pricing per million tokens (USD)
PRICING = {
    "sonnet": {"input": 3.0, "output": 15.0},
    "opus": {"input": 15.0, "output": 75.0},
}


def compute_recall_at_k(results: list[EvaluationResult], k: int) -> float:
    """Fraction of questions with at least 1 correct page in top-k retrieved."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.recall_at_k.get(k, False))
    return hits / len(results)


def compute_citation_accuracy(results: list[EvaluationResult]) -> float:
    """Fraction of questions with correct citations."""
    answerable = [r for r in results if not r.should_abstain]
    if not answerable:
        return 0.0
    correct = sum(1 for r in answerable if r.citation_correct)
    return correct / len(answerable)


def compute_abstention_metrics(results: list[EvaluationResult]) -> dict[str, float]:
    """Compute precision and recall for abstention detection.

    Precision: of those that abstained, how many should have?
    Recall: of those that should abstain, how many did?
    """
    true_positives = sum(1 for r in results if r.should_abstain and r.did_abstain)
    false_positives = sum(1 for r in results if not r.should_abstain and r.did_abstain)
    false_negatives = sum(1 for r in results if r.should_abstain and not r.did_abstain)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return {"precision": precision, "recall": recall}


def compute_cost_per_query(
    results: list[EvaluationResult],
    model: str = "sonnet",
) -> float:
    """Average cost per query in USD based on token counts."""
    if not results:
        return 0.0
    pricing = PRICING[model]
    total_cost = 0.0
    for r in results:
        total_cost += (r.input_tokens * pricing["input"] + r.output_tokens * pricing["output"]) / 1_000_000
    return total_cost / len(results)


def compute_category_breakdown(
    results: list[EvaluationResult],
    ground_truth_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Per-category recall@1, recall@5, citation accuracy, and count.

    Args:
        results: Evaluation results.
        ground_truth_map: Mapping question_id → category name.
    """
    by_cat: dict[str, list[EvaluationResult]] = {}
    for r in results:
        cat = ground_truth_map.get(r.question_id, "unknown")
        by_cat.setdefault(cat, []).append(r)

    breakdown = {}
    for cat, cat_results in by_cat.items():
        breakdown[cat] = {
            "count": float(len(cat_results)),
            "recall_at_1": compute_recall_at_k(cat_results, 1),
            "recall_at_5": compute_recall_at_k(cat_results, 5),
            "recall_at_10": compute_recall_at_k(cat_results, 10),
            "citation_accuracy": compute_citation_accuracy(cat_results),
        }
    return breakdown


def build_report(
    results: list[EvaluationResult],
    ground_truth_map: dict[str, str],
    model: str = "sonnet",
) -> EvaluationReport:
    """Build a full evaluation report from results."""
    abstention = compute_abstention_metrics(results)

    return EvaluationReport(
        recall_at_1=compute_recall_at_k(results, 1),
        recall_at_3=compute_recall_at_k(results, 3),
        recall_at_5=compute_recall_at_k(results, 5),
        recall_at_10=compute_recall_at_k(results, 10),
        citation_accuracy=compute_citation_accuracy(results),
        abstention_precision=abstention["precision"],
        abstention_recall=abstention["recall"],
        cost_per_query_usd=compute_cost_per_query(results, model),
        total_questions=len(results),
        by_category=compute_category_breakdown(results, ground_truth_map),
        results=results,
    )
