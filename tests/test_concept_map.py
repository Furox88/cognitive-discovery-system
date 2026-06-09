import pytest
from cds.concept_map import Concept, ConceptMap, Relation


class TestConcept:
    def test_create(self):
        c = Concept(name="Gravity", description="Fundamental force")
        assert c.name == "Gravity"

    def test_strips(self):
        assert Concept(name="  Gravity  ").name == "Gravity"

    def test_empty(self):
        with pytest.raises(ValueError):
            Concept(name="")

    def test_eq(self):
        assert Concept("Gravity") == Concept("Gravity")
        assert Concept("Gravity") != Concept("Magnetism")

    def test_hashable(self):
        assert len({Concept("Gravity"), Concept("Gravity")}) == 1


class TestRelation:
    def test_create(self):
        r = Relation(source="A", target="B", label="causes", weight=0.8)
        assert r.weight == 0.8

    def test_empty_source(self):
        with pytest.raises(ValueError):
            Relation(source="", target="B")

    def test_empty_target(self):
        with pytest.raises(ValueError):
            Relation(source="A", target="")

    def test_neg_weight(self):
        with pytest.raises(ValueError):
            Relation(source="A", target="B", weight=-1)


class TestConceptMap:
    def _build(self):
        cm = ConceptMap()
        cm.add_concept(Concept("Force", "A push or pull"))
        cm.add_concept(Concept("Mass", "Amount of matter"))
        cm.add_concept(Concept("Acceleration", "Rate of velocity change"))
        cm.add_relation(Relation("Force", "Mass", label="depends_on"))
        cm.add_relation(Relation("Force", "Acceleration", label="depends_on"))
        return cm

    def test_add_get(self):
        cm = self._build()
        assert cm.get_concept("Force").name == "Force"

    def test_duplicate(self):
        cm = self._build()
        with pytest.raises(ValueError):
            cm.add_concept(Concept("Force"))

    def test_missing(self):
        with pytest.raises(KeyError):
            ConceptMap().get_concept("Nope")

    def test_remove(self):
        cm = self._build()
        cm.remove_concept("Mass")
        assert cm.concept_count == 2
        assert all(r.target != "Mass" for r in cm.get_relations("Force"))

    def test_remove_missing(self):
        with pytest.raises(KeyError):
            ConceptMap().remove_concept("Nope")

    def test_sorted(self):
        cm = self._build()
        names = [c.name for c in cm.list_concepts()]
        assert names == sorted(names)

    def test_relation_missing_source(self):
        cm = ConceptMap()
        cm.add_concept(Concept("A"))
        with pytest.raises(KeyError):
            cm.add_relation(Relation("B", "A"))

    def test_relation_missing_target(self):
        cm = ConceptMap()
        cm.add_concept(Concept("A"))
        with pytest.raises(KeyError):
            cm.add_relation(Relation("A", "B"))

    def test_relations(self):
        cm = self._build()
        assert len(cm.get_relations("Force")) == 2

    def test_incoming(self):
        cm = self._build()
        assert cm.get_incoming("Mass")[0].source == "Force"

    def test_neighbors(self):
        cm = self._build()
        assert set(cm.neighbors("Force")) == {"Mass", "Acceleration"}

    def test_ancestors(self):
        cm = self._build()
        assert cm.ancestors("Mass") == ["Force"]

    def test_find_path(self):
        cm = self._build()
        cm.add_relation(Relation("Mass", "Acceleration"))
        path = cm.find_path("Force", "Acceleration")
        assert path[0] == "Force" and path[-1] == "Acceleration"

    def test_find_path_self(self):
        cm = self._build()
        assert cm.find_path("Force", "Force") == ["Force"]

    def test_find_path_none(self):
        cm = self._build()
        assert cm.find_path("Mass", "Force") is None

    def test_find_path_missing(self):
        cm = self._build()
        with pytest.raises(KeyError):
            cm.find_path("Nope", "Force")

    def test_search(self):
        cm = self._build()
        assert cm.search("force")[0].name == "Force"

    def test_counts(self):
        cm = self._build()
        assert cm.concept_count == 3
        assert cm.relation_count == 2
