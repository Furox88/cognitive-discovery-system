"""Tests for the hypothesis generation and management module."""

import pytest

from cds.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    HypothesisStore,
    InvalidTransitionError,
)


# --- Hypothesis creation ---


class TestHypothesisCreation:
    def test_create_minimal(self):
        h = Hypothesis(statement="Water boils at 100C", rationale="Standard physics")
        assert h.status == HypothesisStatus.DRAFT
        assert h.confidence == 0.5
        assert h.id  # non-empty
        assert h.created_at is not None

    def test_create_with_all_fields(self):
        h = Hypothesis(
            statement="E=mc^2",
            rationale="Energy-mass equivalence",
            variables=["E", "m", "c"],
            predictions=["Energy release in fission"],
            confidence=0.9,
            tags=["physics", "relativity"],
        )
        assert h.variables == ["E", "m", "c"]
        assert h.confidence == 0.9
        assert "physics" in h.tags

    @pytest.mark.parametrize(
        "stmt, rationale",
        [("", "valid"), ("valid", ""), ("  ", "valid"), ("valid", "   ")],
    )
    def test_empty_fields_rejected(self, stmt, rationale):
        with pytest.raises(ValueError):
            Hypothesis(statement=stmt, rationale=rationale)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            Hypothesis(statement="test", rationale="test", confidence=1.5)
        with pytest.raises(ValueError):
            Hypothesis(statement="test", rationale="test", confidence=-0.1)


# --- Hypothesis lifecycle ---


class TestHypothesisLifecycle:
    def _make(self) -> Hypothesis:
        return Hypothesis(statement="Plants grow toward light", rationale="Phototropism")

    def test_full_lifecycle_support(self):
        h = self._make()
        h.propose()
        assert h.status == HypothesisStatus.PROPOSED
        h.start_testing()
        assert h.status == HypothesisStatus.TESTING
        h.support(confidence=0.95)
        assert h.status == HypothesisStatus.SUPPORTED
        assert h.confidence == 0.95

    def test_full_lifecycle_refute(self):
        h = self._make()
        h.propose()
        h.start_testing()
        h.refute(confidence=0.1)
        assert h.status == HypothesisStatus.REFUTED
        assert h.confidence == 0.1

    def test_invalid_transition_draft_to_testing(self):
        h = self._make()
        with pytest.raises(InvalidTransitionError):
            h.start_testing()

    def test_invalid_transition_proposed_to_supported(self):
        h = self._make()
        h.propose()
        with pytest.raises(InvalidTransitionError):
            h.support()

    def test_revise(self):
        h = self._make()
        h.revise("Updated statement", "Updated rationale")
        assert h.statement == "Updated statement"
        assert h.rationale == "Updated rationale"
        assert h.status == HypothesisStatus.REVISED

    def test_revise_empty_rejected(self):
        h = self._make()
        with pytest.raises(ValueError):
            h.revise("")

    def test_add_prediction(self):
        h = self._make()
        h.add_prediction("Leaves will bend toward window")
        assert "Leaves will bend toward window" in h.predictions

    def test_add_prediction_empty_rejected(self):
        h = self._make()
        with pytest.raises(ValueError):
            h.add_prediction("")

    def test_summary(self):
        h = self._make()
        s = h.summary()
        assert "[draft]" in s
        assert "Plants grow toward light" in s

    def test_support_invalid_confidence(self):
        h = self._make()
        h.propose()
        h.start_testing()
        with pytest.raises(ValueError):
            h.support(confidence=2.0)

    def test_refute_invalid_confidence(self):
        h = self._make()
        h.propose()
        h.start_testing()
        with pytest.raises(ValueError):
            h.refute(confidence=-0.5)


# --- HypothesisStore ---


class TestHypothesisStore:
    def _store_with_items(self) -> tuple[HypothesisStore, Hypothesis, Hypothesis]:
        store = HypothesisStore()
        h1 = Hypothesis(statement="H1", rationale="R1", tags=["bio"])
        h2 = Hypothesis(statement="H2", rationale="R2", tags=["physics"])
        store.add(h1)
        store.add(h2)
        return store, h1, h2

    def test_add_and_get(self):
        store, h1, _ = self._store_with_items()
        assert store.get(h1.id) is h1

    def test_add_duplicate_rejected(self):
        store, h1, _ = self._store_with_items()
        with pytest.raises(ValueError):
            store.add(h1)

    def test_get_missing_raises(self):
        store = HypothesisStore()
        with pytest.raises(KeyError):
            store.get("nonexistent")

    def test_remove(self):
        store, h1, _ = self._store_with_items()
        removed = store.remove(h1.id)
        assert removed is h1
        assert h1.id not in store

    def test_remove_missing_raises(self):
        store = HypothesisStore()
        with pytest.raises(KeyError):
            store.remove("nonexistent")

    def test_list_all(self):
        store, _, _ = self._store_with_items()
        assert len(store.list_all()) == 2

    def test_filter_by_status(self):
        store, h1, _ = self._store_with_items()
        h1.propose()
        result = store.filter_by_status(HypothesisStatus.PROPOSED)
        assert len(result) == 1
        assert result[0] is h1

    def test_filter_by_tag(self):
        store, _, h2 = self._store_with_items()
        result = store.filter_by_tag("physics")
        assert len(result) == 1
        assert result[0] is h2

    def test_search(self):
        store, h1, _ = self._store_with_items()
        assert len(store.search("H1")) == 1

    def test_len_and_contains(self):
        store, h1, _ = self._store_with_items()
        assert len(store) == 2
        assert h1.id in store
