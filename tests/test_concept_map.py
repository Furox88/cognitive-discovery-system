"""Tests for the concept relationship mapping module."""

import pytest

from cds.concept_map import Concept, ConceptMap, Relation


class TestConcept:
    def test_create(self):
        c = Concept(name="Gravity", description="Fundamental force")
        assert c.name == "Gravity"

    def test_create_strips_whitespace(self):
        c = Concept(name="  Gravity  ")
        assert c.name == "Gravity"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            Concept(name="")

    def test_equality_by_name(self):
        assert Concept(name="Gravity") == Concept(name="Gravity")
        assert Concept(name="Gravity") != Concept(name="Magnetism")

    def test_hashable(self):
        s = {Concept(name="Gravity"), Concept(name="Gravity")}
        assert len(s) == 1


class TestRelation:
    def test_create(self):
        r = Relation(source="A", target="B", label="causes", weight=0.8)
        assert r.source == "A"
        assert r.weight == 0.8

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError):
            Relation(source="", target="B")

    def test_empty_target_rejected(self):
        with pytest.raises(ValueError):
            Relation(source="A", target="")

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError):
            Relation(source="A", target="B", weight=-1)


class TestConceptMap:
    def _build_map(self) -> ConceptMap:
        cm = ConceptMap()
        cm.add_concept(Concept(name="Force", description="A push or pull"))
        cm.add_concept(Concept(name="Mass", description="Amount of matter"))
        cm.add_concept(Concept(name="Acceleration", description="Rate of velocity change"))
        cm.add_relation(Relation(source="Force", target="Mass", label="depends_on"))
        cm.add_relation(Relation(source="Force", target="Acceleration", label="depends_on"))
        return cm

    def test_add_and_get_concept(self):
        cm = self._build_map()
        c = cm.get_concept("Force")
        assert c.name == "Force"

    def test_add_duplicate_concept_rejected(self):
        cm = self._build_map()
        with pytest.raises(ValueError):
            cm.add_concept(Concept(name="Force"))

    def test_get_concept_missing(self):
        cm = ConceptMap()
        with pytest.raises(KeyError):
            cm.get_concept("Nope")

    def test_remove_concept(self):
        cm = self._build_map()
        removed = cm.remove_concept("Mass")
        assert removed.name == "Mass"
        assert cm.concept_count == 2
        # relation from Force→Mass should be gone
        rels = cm.get_relations("Force")
        assert all(r.target != "Mass" for r in rels)

    def test_remove_concept_missing(self):
        cm = ConceptMap()
        with pytest.raises(KeyError):
            cm.remove_concept("Nope")

    def test_list_concepts_sorted(self):
        cm = self._build_map()
        names = [c.name for c in cm.list_concepts()]
        assert names == sorted(names)

    def test_add_relation_missing_source(self):
        cm = ConceptMap()
        cm.add_concept(Concept(name="A"))
        with pytest.raises(KeyError):
            cm.add_relation(Relation(source="B", target="A"))

    def test_add_relation_missing_target(self):
        cm = ConceptMap()
        cm.add_concept(Concept(name="A"))
        with pytest.raises(KeyError):
            cm.add_relation(Relation(source="A", target="B"))

    def test_get_relations(self):
        cm = self._build_map()
        rels = cm.get_relations("Force")
        assert len(rels) == 2

    def test_get_incoming(self):
        cm = self._build_map()
        inc = cm.get_incoming("Mass")
        assert len(inc) == 1
        assert inc[0].source == "Force"

    def test_neighbors(self):
        cm = self._build_map()
        nbrs = cm.neighbors("Force")
        assert set(nbrs) == {"Mass", "Acceleration"}

    def test_ancestors(self):
        cm = self._build_map()
        anc = cm.ancestors("Mass")
        assert anc == ["Force"]

    def test_find_path(self):
        cm = self._build_map()
        # Add a chain: Mass → Acceleration
        cm.add_relation(Relation(source="Mass", target="Acceleration"))
        path = cm.find_path("Force", "Acceleration")
        assert path is not None
        assert path[0] == "Force"
        assert path[-1] == "Acceleration"

    def test_find_path_same_node(self):
        cm = self._build_map()
        assert cm.find_path("Force", "Force") == ["Force"]

    def test_find_path_no_path(self):
        cm = self._build_map()
        assert cm.find_path("Mass", "Force") is None

    def test_find_path_missing_concept(self):
        cm = self._build_map()
        with pytest.raises(KeyError):
            cm.find_path("Nope", "Force")

    def test_search(self):
        cm = self._build_map()
        result = cm.search("force")
        assert len(result) == 1
        assert result[0].name == "Force"

    def test_counts(self):
        cm = self._build_map()
        assert cm.concept_count == 3
        assert cm.relation_count == 2
