"""Hypothesis generation and lifecycle tracking."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class HypothesisStatus(Enum):
    DRAFT = "draft"
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    REVISED = "revised"


@dataclass
class Hypothesis:
    statement: str
    rationale: str
    variables: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.DRAFT
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.statement or not self.statement.strip():
            raise ValueError("statement must not be empty")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    def propose(self) -> None:
        if self.status != HypothesisStatus.DRAFT:
            raise InvalidTransitionError(self.status, HypothesisStatus.PROPOSED)
        self.status = HypothesisStatus.PROPOSED

    def start_testing(self) -> None:
        if self.status != HypothesisStatus.PROPOSED:
            raise InvalidTransitionError(self.status, HypothesisStatus.TESTING)
        self.status = HypothesisStatus.TESTING

    def support(self, confidence: float | None = None) -> None:
        if self.status != HypothesisStatus.TESTING:
            raise InvalidTransitionError(self.status, HypothesisStatus.SUPPORTED)
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")
            self.confidence = confidence
        self.status = HypothesisStatus.SUPPORTED

    def refute(self, confidence: float | None = None) -> None:
        if self.status != HypothesisStatus.TESTING:
            raise InvalidTransitionError(self.status, HypothesisStatus.REFUTED)
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")
            self.confidence = confidence
        self.status = HypothesisStatus.REFUTED

    def revise(self, new_statement: str, new_rationale: str | None = None) -> None:
        if not new_statement or not new_statement.strip():
            raise ValueError("new_statement must not be empty")
        self.statement = new_statement.strip()
        if new_rationale is not None:
            self.rationale = new_rationale.strip()
        self.status = HypothesisStatus.REVISED

    def add_prediction(self, prediction: str) -> None:
        if not prediction or not prediction.strip():
            raise ValueError("prediction must not be empty")
        self.predictions.append(prediction.strip())

    def summary(self) -> str:
        return f"[{self.status.value}] {self.statement} (confidence={self.confidence:.2f})"


class InvalidTransitionError(Exception):
    def __init__(self, from_status: HypothesisStatus, to_status: HypothesisStatus) -> None:
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Cannot transition from {from_status.value!r} to {to_status.value!r}"
        )


class HypothesisStore:
    """Simple in-memory store for hypotheses."""

    def __init__(self) -> None:
        self._items: dict[str, Hypothesis] = {}

    def add(self, hypothesis: Hypothesis) -> str:
        if hypothesis.id in self._items:
            raise ValueError(f"Hypothesis {hypothesis.id!r} already exists")
        self._items[hypothesis.id] = hypothesis
        return hypothesis.id

    def get(self, hypothesis_id: str) -> Hypothesis:
        try:
            return self._items[hypothesis_id]
        except KeyError:
            raise KeyError(f"Hypothesis {hypothesis_id!r} not found") from None

    def remove(self, hypothesis_id: str) -> Hypothesis:
        try:
            return self._items.pop(hypothesis_id)
        except KeyError:
            raise KeyError(f"Hypothesis {hypothesis_id!r} not found") from None

    def list_all(self) -> list[Hypothesis]:
        return sorted(self._items.values(), key=lambda h: h.created_at)

    def filter_by_status(self, status: HypothesisStatus) -> list[Hypothesis]:
        return [h for h in self._items.values() if h.status == status]

    def filter_by_tag(self, tag: str) -> list[Hypothesis]:
        return [h for h in self._items.values() if tag in h.tags]

    def search(self, query: str) -> list[Hypothesis]:
        q = query.lower()
        return [
            h for h in self._items.values()
            if q in h.statement.lower() or q in h.rationale.lower()
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, hypothesis_id: str) -> bool:
        return hypothesis_id in self._items
