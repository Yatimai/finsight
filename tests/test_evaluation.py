"""Tests for evaluation models, metrics, and runner."""

import pytest

from evaluation.models import (
    EvaluationReport,
    EvaluationResult,
    GroundTruthItem,
    QuestionCategory,
)

# ── Models ──────────────────────────────────────────────────────────


class TestGroundTruthItem:
    def test_load_valid_ground_truth(self):
        item = GroundTruthItem(
            id="q01",
            question="Quel est le chiffre d'affaires consolidé de LVMH en 2024 ?",
            expected_answer="86,2 milliards d'euros",
            source_document="LVMH_DEU_2024.pdf",
            source_pages=[12, 15],
            category="chiffre_exact",
        )
        assert item.id == "q01"
        assert item.category == QuestionCategory.chiffre_exact
        assert item.source_pages == [12, 15]

    def test_load_draft_item_empty_fields(self):
        """Draft items have empty source_document, source_pages, expected_answer."""
        item = GroundTruthItem(
            id="q01",
            question="Question test ?",
            category="tendance",
        )
        assert item.expected_answer == ""
        assert item.source_document == ""
        assert item.source_pages == []

    def test_question_ids_unique(self):
        items = [
            GroundTruthItem(id="q01", question="Q1?", category="chiffre_exact"),
            GroundTruthItem(id="q02", question="Q2?", category="tendance"),
            GroundTruthItem(id="q03", question="Q3?", category="comparaison"),
        ]
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids))

    def test_page_numbers_positive(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            GroundTruthItem(
                id="q01",
                question="Test?",
                category="chiffre_exact",
                source_pages=[0],
            )

    def test_page_numbers_negative_rejected(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            GroundTruthItem(
                id="q01",
                question="Test?",
                category="chiffre_exact",
                source_pages=[-1],
            )

    def test_invalid_category_rejected(self):
        with pytest.raises(ValueError):
            GroundTruthItem(
                id="q01",
                question="Test?",
                category="invalid_category",
            )


class TestEvaluationResult:
    def test_create_with_defaults(self):
        result = EvaluationResult(question_id="q01")
        assert result.question_id == "q01"
        assert result.retrieved_pages == []
        assert result.generated_answer == ""
        assert result.faithfulness_score is None
        assert result.input_tokens == 0

    def test_create_full(self):
        result = EvaluationResult(
            question_id="q01",
            retrieved_pages=[5, 12, 23],
            recall_at_k={1: True, 3: True, 5: True},
            generated_answer="Le CA est de 86,2 Md€ [Page 12]",
            faithfulness_score=0.95,
            cited_pages=[12],
            citation_correct=True,
            should_abstain=False,
            did_abstain=False,
            input_tokens=1500,
            output_tokens=200,
            latency_ms=3500.0,
        )
        assert result.recall_at_k[1] is True
        assert result.faithfulness_score == 0.95


class TestEvaluationReport:
    def test_report_serialization_roundtrip(self):
        result = EvaluationResult(
            question_id="q01",
            retrieved_pages=[5, 12],
            recall_at_k={1: True, 3: True, 5: True},
            generated_answer="Test answer",
            citation_correct=True,
            input_tokens=100,
            output_tokens=50,
        )
        report = EvaluationReport(
            recall_at_1=0.8,
            recall_at_3=0.9,
            recall_at_5=0.95,
            avg_faithfulness=0.85,
            citation_accuracy=0.7,
            abstention_precision=1.0,
            abstention_recall=0.5,
            cost_per_query_usd=0.05,
            total_questions=50,
            by_category={"chiffre_exact": {"recall_at_1": 0.9, "count": 18}},
            results=[result],
        )

        json_str = report.model_dump_json()
        restored = EvaluationReport.model_validate_json(json_str)

        assert restored.recall_at_1 == 0.8
        assert restored.recall_at_5 == 0.95
        assert restored.total_questions == 50
        assert len(restored.results) == 1
        assert restored.results[0].question_id == "q01"
        assert restored.by_category["chiffre_exact"]["recall_at_1"] == 0.9

    def test_report_to_dict(self):
        report = EvaluationReport(
            recall_at_1=0.5,
            total_questions=10,
        )
        d = report.model_dump()
        assert isinstance(d, dict)
        assert d["recall_at_1"] == 0.5
        assert d["total_questions"] == 10

    def test_empty_report(self):
        report = EvaluationReport()
        assert report.total_questions == 0
        assert report.results == []
        assert report.by_category == {}
