"""Data models for ground truth and evaluation results."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class QuestionCategory(StrEnum):
    """Category of evaluation question."""

    chiffre_exact = "chiffre_exact"
    tendance = "tendance"
    comparaison = "comparaison"
    tableau = "tableau"
    graphique = "graphique"
    abstention = "abstention"


class GroundTruthItem(BaseModel):
    """A single ground truth question with expected answer and sources."""

    id: str
    question: str
    expected_answer: str = ""
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)
    category: QuestionCategory

    @field_validator("source_pages")
    @classmethod
    def pages_must_be_positive(cls, v: list[int]) -> list[int]:
        for page in v:
            if page < 1:
                raise ValueError(f"Page numbers must be >= 1, got {page}")
        return v


class RetrievedSource(BaseModel):
    """A retrieved (document, page) pair for evaluation."""

    document: str
    page: int


class EvaluationResult(BaseModel):
    """Result of evaluating a single question."""

    question_id: str
    retrieved_pages: list[int] = Field(default_factory=list)
    retrieved_sources: list[RetrievedSource] = Field(default_factory=list)
    recall_at_k: dict[int, bool] = Field(default_factory=dict)
    generated_answer: str = ""
    faithfulness_score: float | None = None
    cited_pages: list[int] = Field(default_factory=list)
    citation_correct: bool = False
    should_abstain: bool = False
    did_abstain: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class EvaluationReport(BaseModel):
    """Aggregate evaluation report."""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    avg_faithfulness: float | None = None
    citation_accuracy: float = 0.0
    abstention_precision: float = 0.0
    abstention_recall: float = 0.0
    cost_per_query_usd: float = 0.0
    total_questions: int = 0
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    results: list[EvaluationResult] = Field(default_factory=list)
