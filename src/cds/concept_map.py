"""Directed graph for mapping concept relationships."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Concept:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Concept name must not be empty")
        self.name = self.name.strip()

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Concept):
            return NotImplemented
        return self.name == other.name


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    label: str = "related_to"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.source.strip():
            raise ValueError("source must not be empty")
        if not self.target or not self.target.strip():
            raise ValueError("target must not be empty")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


class ConceptMap:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._relations: list[Relation] = []

    def add_concept(self, concept: Concept) -> None:
        if concept.name in self._concepts:
            raise ValueError(f"Concept {concept.name!r} already exists")
        self._concepts[concept.name] = concept

    def get_concept(self, name: str) -> Concept:
        try:
            return self._concepts[name]
        except KeyError:
            raise KeyError(f"Concept {name!r} not found") from None

    def remove_concept(self, name: str) -> Concept:
        try:
            concept = self._concepts.pop(name)
        except KeyError:
            raise KeyError(f"Concept {name!r} not found") from None
        self._relations = [
            r for r in self._relations if r.source != name and r.target != name
        ]
        return concept

    def list_concepts(self) -> list[Concept]:
        return sorted(self._concepts.values(), key=lambda c: c.name)

    def add_relation(self, relation: Relation) -> None:
        if relation.source not in self._concepts:
            raise KeyError(f"Source concept {relation.source!r} not found")
        if relation.target not in self._concepts:
            raise KeyError(f"Target concept {relation.target!r} not found")
        self._relations.append(relation)

    def get_relations(self, concept_name: str) -> list[Relation]:
        return [r for r in self._relations if r.source == concept_name]

    def get_incoming(self, concept_name: str) -> list[Relation]:
        return [r for r in self._relations if r.target == concept_name]

    def neighbors(self, concept_name: str) -> list[str]:
        return [r.target for r in self._relations if r.source == concept_name]

    def ancestors(self, concept_name: str) -> list[str]:
        return [r.source for r in self._relations if r.target == concept_name]

    def find_path(self, source: str, target: str) -> list[str] | None:
        """BFS shortest path. Returns None if no path exists."""
        if source not in self._concepts:
            raise KeyError(f"Concept {source!r} not found")
        if target not in self._concepts:
            raise KeyError(f"Concept {target!r} not found")
        if source == target:
            return [source]

        visited: set[str] = {source}
        queue: list[list[str]] = [[source]]
        while queue:
            path = queue.pop(0)
            for neighbor in self.neighbors(path[-1]):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def search(self, query: str) -> list[Concept]:
        q = query.lower()
        return [
            c for c in self._concepts.values()
            if q in c.name.lower() or q in c.description.lower()
        ]

    @property
    def concept_count(self) -> int:
        return len(self._concepts)

    @property
    def relation_count(self) -> int:
        return len(self._relations)
