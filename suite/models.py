"""Data models for attack suite items, execution results, and baseline summaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AttackItem(BaseModel):
    """Specification of an attack test case in the suite."""

    id: str
    owasp_category: str
    category_name: str
    language: str = "en"
    target_model: str = "c1"
    smoke: bool = False
    turns: list[str] = Field(min_length=1)
    expected_safe_behavior: str


class AttackEvaluationResult(BaseModel):
    """Result of running and scoring an attack case."""

    id: str
    owasp_category: str
    category_name: str
    language: str
    target_model: str
    smoke: bool
    turns: list[str]
    conversation: list[dict[str, str]] = Field(default_factory=list)
    score: float
    achieved: bool
    rationale: str
    response_preview: str


class MetricBreakdown(BaseModel):
    """Aggregated evaluation metrics for a slice (category or language)."""

    total: int
    successful_attacks: int
    defended_attacks: int
    attack_success_rate: float


class BaselineSummary(BaseModel):
    """Comprehensive baseline evaluation summary written to results/baseline.json."""

    generated_at: str
    suite_file: str
    total_attacks: int
    overall: MetricBreakdown
    by_category: dict[str, MetricBreakdown]
    by_language: dict[str, MetricBreakdown]
    attacks: list[AttackEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
