import pytest
from cds.hypothesis import (
    Hypothesis, HypothesisStatus, HypothesisStore, InvalidTransitionError,
)


class TestHypothesis:
    def test_create(self):
        h = Hypothesis(statement="Water boils at 100C", rationale="Standard physics")
        assert h.status == HypothesisStatus.DRAFT
        assert h.confidence == 0.5

    def test_create_with_fields(self):
        h = Hypothesis(
            statement="E=mc^2", rationale="Energy-mass equivalence",
            variables=["E", "m", "c"], confidence=0.9, tags=["physics"],
        )
        assert h.confidence == 0.9
        assert "physics" in h.tags

    @pytest.mark.parametrize("stmt,rat", [("", "ok"), ("ok", ""), ("  ", "ok")])
    def test_rejects_empty(self, stmt, rat):
        with pytest.raises(ValueError):
            Hypothesis(statement=stmt, rationale=rat)

    def test_bad_confidence(self):
        with pytest.raises(ValueError):
            Hypothesis(statement="x", rationale="y", confidence=1.5)

    def test_lifecycle_support(self):
        h = Hypothesis(statement="test", rationale="test")
        h.propose()
        assert h.status == HypothesisStatus.PROPOSED
        h.start_testing()
        h.support(confidence=0.95)
        assert h.status == HypothesisStatus.SUPPORTED

    def test_lifecycle_refute(self):
        h = Hypothesis(statement="test", rationale="test")
        h.propose()
        h.start_testing()
        h.refute(confidence=0.1)
        assert h.status == HypothesisStatus.REFUTED

    def test_invalid_transition(self):
        h = Hypothesis(statement="test", rationale="test")
        with pytest.raises(InvalidTransitionError):
            h.start_testing()  # can't skip PROPOSED

    def test_revise(self):
        h = Hypothesis(statement="old", rationale="old")
        h.revise("new statement", "new rationale")
        assert h.statement == "new statement"
        assert h.status == HypothesisStatus.REVISED

    def test_revise_empty(self):
        h = Hypothesis(statement="x", rationale="y")
        with pytest.raises(ValueError):
            h.revise("")

    def test_prediction(self):
        h = Hypothesis(statement="x", rationale="y")
        h.add_prediction("something happens")
        assert "something happens" in h.predictions

    def test_prediction_empty(self):
        h = Hypothesis(statement="x", rationale="y")
        with pytest.raises(ValueError):
            h.add_prediction("")

    def test_summary(self):
        h = Hypothesis(statement="test stmt", rationale="r")
        assert "[draft]" in h.summary()
        assert "test stmt" in h.summary()

    def test_support_bad_confidence(self):
        h = Hypothesis(statement="x", rationale="y")
        h.propose()
        h.start_testing()
        with pytest.raises(ValueError):
            h.support(confidence=2.0)

    def test_refute_bad_confidence(self):
        h = Hypothesis(statement="x", rationale="y")
        h.propose()
        h.start_testing()
        with pytest.raises(ValueError):
            h.refute(confidence=-0.5)

    def test_proposed_cant_support(self):
        h = Hypothesis(statement="x", rationale="y")
        h.propose()
        with pytest.raises(InvalidTransitionError):
            h.support()


class TestHypothesisStore:
    def _make_store(self):
        store = HypothesisStore()
        h1 = Hypothesis(statement="H1", rationale="R1", tags=["bio"])
        h2 = Hypothesis(statement="H2", rationale="R2", tags=["physics"])
        store.add(h1)
        store.add(h2)
        return store, h1, h2

    def test_add_get(self):
        store, h1, _ = self._make_store()
        assert store.get(h1.id) is h1

    def test_duplicate(self):
        store, h1, _ = self._make_store()
        with pytest.raises(ValueError):
            store.add(h1)

    def test_missing(self):
        store = HypothesisStore()
        with pytest.raises(KeyError):
            store.get("nope")

    def test_remove(self):
        store, h1, _ = self._make_store()
        assert store.remove(h1.id) is h1
        assert h1.id not in store

    def test_remove_missing(self):
        with pytest.raises(KeyError):
            HypothesisStore().remove("nope")

    def test_list_all(self):
        store, _, _ = self._make_store()
        assert len(store.list_all()) == 2

    def test_filter_status(self):
        store, h1, _ = self._make_store()
        h1.propose()
        assert len(store.filter_by_status(HypothesisStatus.PROPOSED)) == 1

    def test_filter_tag(self):
        store, _, h2 = self._make_store()
        assert store.filter_by_tag("physics")[0] is h2

    def test_search(self):
        store, _, _ = self._make_store()
        assert len(store.search("H1")) == 1

    def test_len(self):
        store, h1, _ = self._make_store()
        assert len(store) == 2
        assert h1.id in store
