"""Tests for evaluation models, metrics, and runner."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.evaluate import evaluate_single, load_ground_truth
from evaluation.metrics import (
    build_report,
    compute_abstention_metrics,
    compute_category_breakdown,
    compute_citation_accuracy,
    compute_cost_per_query,
    compute_recall_at_k,
)
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


# ── Metrics ─────────────────────────────────────────────────────────


def _make_result(
    qid: str,
    recall_1: bool = False,
    recall_3: bool = False,
    recall_5: bool = False,
    citation_correct: bool = False,
    should_abstain: bool = False,
    did_abstain: bool = False,
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> EvaluationResult:
    """Helper to create EvaluationResult with common defaults."""
    return EvaluationResult(
        question_id=qid,
        retrieved_pages=[5, 12, 23],
        recall_at_k={1: recall_1, 3: recall_3, 5: recall_5},
        generated_answer="Answer",
        citation_correct=citation_correct,
        should_abstain=should_abstain,
        did_abstain=did_abstain,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class TestRecallAtK:
    def test_recall_at_1_perfect(self):
        results = [
            _make_result("q01", recall_1=True),
            _make_result("q02", recall_1=True),
        ]
        assert compute_recall_at_k(results, 1) == 1.0

    def test_recall_at_1_miss(self):
        results = [
            _make_result("q01", recall_1=False),
            _make_result("q02", recall_1=False),
        ]
        assert compute_recall_at_k(results, 1) == 0.0

    def test_recall_at_5_partial(self):
        results = [
            _make_result("q01", recall_5=True),
            _make_result("q02", recall_5=False),
            _make_result("q03", recall_5=True),
            _make_result("q04", recall_5=False),
        ]
        assert compute_recall_at_k(results, 5) == 0.5

    def test_recall_empty_results(self):
        assert compute_recall_at_k([], 1) == 0.0


class TestCitationAccuracy:
    def test_all_correct(self):
        results = [
            _make_result("q01", citation_correct=True),
            _make_result("q02", citation_correct=True),
        ]
        assert compute_citation_accuracy(results) == 1.0

    def test_none_correct(self):
        results = [
            _make_result("q01", citation_correct=False),
            _make_result("q02", citation_correct=False),
        ]
        assert compute_citation_accuracy(results) == 0.0

    def test_excludes_abstention_questions(self):
        """Abstention questions should not count toward citation accuracy."""
        results = [
            _make_result("q01", citation_correct=True),
            _make_result("q02", citation_correct=False, should_abstain=True),
        ]
        # Only q01 counts (answerable), and it's correct → 1.0
        assert compute_citation_accuracy(results) == 1.0

    def test_empty_results(self):
        assert compute_citation_accuracy([]) == 0.0


class TestAbstentionMetrics:
    def test_precision_recall(self):
        results = [
            _make_result("q01", should_abstain=True, did_abstain=True),  # TP
            _make_result("q02", should_abstain=True, did_abstain=False),  # FN
            _make_result("q03", should_abstain=False, did_abstain=False),  # TN
            _make_result("q04", should_abstain=False, did_abstain=True),  # FP
        ]
        metrics = compute_abstention_metrics(results)
        # TP=1, FP=1, FN=1
        assert metrics["precision"] == 0.5  # 1/(1+1)
        assert metrics["recall"] == 0.5  # 1/(1+1)

    def test_perfect_abstention(self):
        results = [
            _make_result("q01", should_abstain=True, did_abstain=True),
            _make_result("q02", should_abstain=False, did_abstain=False),
        ]
        metrics = compute_abstention_metrics(results)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_no_abstention_cases(self):
        results = [
            _make_result("q01", should_abstain=False, did_abstain=False),
        ]
        metrics = compute_abstention_metrics(results)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestCostPerQuery:
    def test_sonnet_cost(self):
        results = [
            _make_result("q01", input_tokens=1000, output_tokens=200),
        ]
        # Sonnet: $3/M input, $15/M output
        # Cost = (1000 * 3 + 200 * 15) / 1_000_000 = (3000 + 3000) / 1_000_000 = 0.006
        cost = compute_cost_per_query(results, model="sonnet")
        assert abs(cost - 0.006) < 1e-9

    def test_opus_cost(self):
        results = [
            _make_result("q01", input_tokens=1000, output_tokens=200),
        ]
        # Opus: $15/M input, $75/M output
        # Cost = (1000 * 15 + 200 * 75) / 1_000_000 = (15000 + 15000) / 1_000_000 = 0.03
        cost = compute_cost_per_query(results, model="opus")
        assert abs(cost - 0.03) < 1e-9

    def test_empty_results(self):
        assert compute_cost_per_query([], model="sonnet") == 0.0


class TestCategoryBreakdown:
    def test_breakdown_by_category(self):
        results = [
            _make_result("q01", recall_1=True, citation_correct=True),
            _make_result("q02", recall_1=False, citation_correct=False),
            _make_result("q03", recall_1=True, citation_correct=True),
        ]
        gt_map = {"q01": "chiffre_exact", "q02": "chiffre_exact", "q03": "tendance"}
        breakdown = compute_category_breakdown(results, gt_map)

        assert "chiffre_exact" in breakdown
        assert "tendance" in breakdown
        assert breakdown["chiffre_exact"]["count"] == 2.0
        assert breakdown["chiffre_exact"]["recall_at_1"] == 0.5
        assert breakdown["tendance"]["count"] == 1.0
        assert breakdown["tendance"]["recall_at_1"] == 1.0


class TestBuildReport:
    def test_has_all_metrics(self):
        results = [
            _make_result("q01", recall_1=True, recall_3=True, recall_5=True, citation_correct=True),
            _make_result("q02", recall_1=False, recall_3=True, recall_5=True, citation_correct=False),
            _make_result("q03", should_abstain=True, did_abstain=True),
        ]
        gt_map = {"q01": "chiffre_exact", "q02": "tendance", "q03": "abstention"}
        report = build_report(results, gt_map, model="sonnet")

        assert report.total_questions == 3
        assert report.recall_at_1 == pytest.approx(1 / 3)
        assert report.recall_at_3 == pytest.approx(2 / 3)
        assert report.recall_at_5 == pytest.approx(2 / 3)
        assert report.abstention_precision == 1.0
        assert report.abstention_recall == 1.0
        assert report.cost_per_query_usd > 0
        assert "chiffre_exact" in report.by_category
        assert "tendance" in report.by_category
        assert "abstention" in report.by_category
        assert len(report.results) == 3


# ── Runner ──────────────────────────────────────────────────────────


@dataclass
class FakeRetrievedPage:
    """Minimal stand-in for RetrievedPage in evaluation tests."""

    page_number: int
    source_filename: str = "LVMH_DEU_2024.pdf"
    score: float = 0.95


class FakeQueryResult:
    """Minimal stand-in for QueryResult."""

    def __init__(self, pages, answer="", citations=None, verification=None, generation_tokens=None):
        self.pages = pages
        self.answer = answer
        self.citations = citations or []
        self.verification = verification or {}
        self.generation_tokens = generation_tokens or {"input_tokens": 500, "output_tokens": 100}


def _make_pipeline_mock(pages, answer="Réponse test [Page 12]", citations=None, verification=None):
    """Create a mock pipeline that returns fixed results."""
    citations = citations or [{"page": 12}]
    result = FakeQueryResult(
        pages=pages,
        answer=answer,
        citations=citations,
        verification=verification or {},
    )
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=result)
    return pipeline


class TestEvaluateSingle:
    @pytest.mark.asyncio
    async def test_correct_retrieval(self):
        pages = [FakeRetrievedPage(page_number=12), FakeRetrievedPage(page_number=5)]
        pipeline = _make_pipeline_mock(pages)
        item = GroundTruthItem(
            id="q01",
            question="Quel est le CA de LVMH ?",
            source_pages=[12],
            category="chiffre_exact",
        )

        result = await evaluate_single(pipeline, item)

        assert result.question_id == "q01"
        assert result.recall_at_k[1] is True  # page 12 is top-1
        assert result.recall_at_k[5] is True
        assert result.retrieved_pages == [12, 5]
        assert result.citation_correct is True
        assert result.should_abstain is False
        assert result.did_abstain is False

    @pytest.mark.asyncio
    async def test_abstention_detected(self):
        pages = [FakeRetrievedPage(page_number=3)]
        pipeline = _make_pipeline_mock(
            pages,
            answer="Cette information n'apparaît pas dans les documents fournis.",
            citations=[],
        )
        item = GroundTruthItem(
            id="q50",
            question="Quel est le PIB du Japon ?",
            category="abstention",
        )

        result = await evaluate_single(pipeline, item)

        assert result.should_abstain is True
        assert result.did_abstain is True

    @pytest.mark.asyncio
    async def test_citation_matching(self):
        pages = [
            FakeRetrievedPage(page_number=5),
            FakeRetrievedPage(page_number=12),
            FakeRetrievedPage(page_number=23),
        ]
        pipeline = _make_pipeline_mock(
            pages,
            answer="Le CA est de 86,2 Md€ [Page 5] [Page 12]",
            citations=[{"page": 5}, {"page": 12}],
        )
        item = GroundTruthItem(
            id="q01",
            question="Test?",
            source_pages=[5, 12],
            category="chiffre_exact",
        )

        result = await evaluate_single(pipeline, item)

        assert result.cited_pages == [5, 12]
        assert result.citation_correct is True

    @pytest.mark.asyncio
    async def test_skip_verification(self):
        pages = [FakeRetrievedPage(page_number=12)]
        pipeline = _make_pipeline_mock(pages)
        item = GroundTruthItem(id="q01", question="Test?", source_pages=[12], category="chiffre_exact")

        result = await evaluate_single(pipeline, item, skip_verification=True)

        # Pipeline should have been called with skip_verification=True
        pipeline.query.assert_called_once_with(item.question, skip_verification=True)
        assert result.faithfulness_score is None

    @pytest.mark.asyncio
    async def test_retrieval_only_no_answer(self):
        pages = [FakeRetrievedPage(page_number=12)]
        pipeline = _make_pipeline_mock(pages)
        item = GroundTruthItem(id="q01", question="Test?", source_pages=[12], category="chiffre_exact")

        result = await evaluate_single(pipeline, item, retrieval_only=True)

        assert result.generated_answer == ""
        # skip_verification=True when retrieval_only
        pipeline.query.assert_called_once_with(item.question, skip_verification=True)


class TestLoadGroundTruth:
    def test_load_valid_file(self, tmp_path):
        gt_data = [
            {
                "id": "q01",
                "question": "Test?",
                "category": "chiffre_exact",
                "source_pages": [12],
                "expected_answer": "42",
                "source_document": "test.pdf",
            }
        ]
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(json.dumps(gt_data))

        items = load_ground_truth(gt_file)
        assert len(items) == 1
        assert items[0].id == "q01"

    def test_duplicate_ids_rejected(self, tmp_path):
        gt_data = [
            {"id": "q01", "question": "Q1?", "category": "chiffre_exact"},
            {"id": "q01", "question": "Q2?", "category": "tendance"},
        ]
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(json.dumps(gt_data))

        with pytest.raises(ValueError, match="Duplicate"):
            load_ground_truth(gt_file)


class TestGroundTruthStructure:
    """Structural validation of the actual ground_truth.json file."""

    @pytest.fixture()
    def ground_truth_items(self):
        from pathlib import Path

        gt_path = Path(__file__).parent.parent / "evaluation" / "ground_truth.json"
        return load_ground_truth(gt_path)

    def test_has_50_questions(self, ground_truth_items):
        assert len(ground_truth_items) == 50

    def test_ids_sequential(self, ground_truth_items):
        ids = [item.id for item in ground_truth_items]
        expected = [f"q{i:02d}" for i in range(1, 51)]
        assert ids == expected

    def test_non_abstention_have_source_pages(self, ground_truth_items):
        for item in ground_truth_items:
            if item.category.value != "abstention":
                assert item.source_pages, f"{item.id} missing source_pages"
                assert len(item.source_pages) <= 3, f"{item.id} has {len(item.source_pages)} source_pages (max 3)"

    def test_non_abstention_have_source_document(self, ground_truth_items):
        for item in ground_truth_items:
            if item.category.value != "abstention":
                assert item.source_document, f"{item.id} missing source_document"
                assert item.source_document.endswith(".pdf"), f"{item.id} source_document should be a PDF"

    def test_non_abstention_have_expected_answer(self, ground_truth_items):
        for item in ground_truth_items:
            if item.category.value != "abstention":
                assert item.expected_answer, f"{item.id} missing expected_answer"

    def test_abstention_have_no_source(self, ground_truth_items):
        for item in ground_truth_items:
            if item.category.value == "abstention":
                assert not item.source_pages, f"{item.id} is abstention but has source_pages"
                assert not item.source_document, f"{item.id} is abstention but has source_document"
                assert not item.expected_answer, f"{item.id} is abstention but has expected_answer"

    def test_source_documents_match_known_pdfs(self, ground_truth_items):
        known_pdfs = {
            "LVMH_DEU_2024.pdf",
            "LOreal_DEU_2024.pdf",
            "BNP_Paribas_DEU_2024.pdf",
            "Danone_DEU_2024.pdf",
            "SocieteGenerale_DEU_2024.pdf",
            "Sanofi_DEU_2024.pdf",
            "Carrefour_DEU_2024.pdf",
            "TotalEnergies_DEU_2024.pdf",
            "SchneiderElectric_DEU_2024.pdf",
            "Airbus_Annual_Report_2024.pdf",
        }
        for item in ground_truth_items:
            if item.source_document:
                assert item.source_document in known_pdfs, (
                    f"{item.id} has unknown source_document: {item.source_document}"
                )
