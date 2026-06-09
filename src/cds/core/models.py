"""Core data models for CDS."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Broad scientific domains supported by CDS."""

    PHYSICS = "physics"
    COSMOLOGY = "cosmology"
    MATHEMATICS = "mathematics"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    GENERAL_SCIENCE = "general_science"


class HypothesisStatus(str, Enum):
    NEW = "new"
    REFINED = "refined"
    CRITIQUED = "critiqued"
    TESTABLE = "testable"
    ARCHIVED = "archived"


class Hypothesis(BaseModel):
    """A scientific hypothesis with metadata and traceability."""

    id: str = Field(..., description="Unique identifier")
    statement: str = Field(..., description="The core hypothesis statement")
    domain: Domain
    research_question: str
    rationale: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.NEW
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="References or retrieval sources")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            f"# Hypothesis: {self.id}",
            "",
            f"**Statement**: {self.statement}",
            "",
            f"**Domain**: {self.domain.value}",
            f"**Research Question**: {self.research_question}",
            f"**Status**: {self.status.value} | **Confidence**: {self.confidence:.2f}",
            "",
        ]
        if self.rationale:
            lines += ["## Rationale", self.rationale, ""]
        if self.assumptions:
            lines += ["## Assumptions"] + [f"- {a}" for a in self.assumptions] + [""]
        if self.predictions:
            preds = [f"- {p}" for p in self.predictions]
            lines += ["## Predictions / Testable Consequences"] + preds + [""]
        if self.tags:
            lines += [f"**Tags**: {', '.join(self.tags)}"]
        return "\n".join(lines)
