"""Concept relationship mapping.

A directed graph for representing relationships between scientific
concepts, supporting traversal, search, and basic graph analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence


@dataclass
class Concept:
    """A node in the concept map."""

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
    """A directed edge between two concepts."""

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
    """A directed graph of concepts and their relationships."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._relations: list[Relation] = []

    # --- Concept CRUD ---

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the map."""
        if concept.name in self._concepts:
            raise ValueError(f"Concept {concept.name!r} already exists")
        self._concepts[concept.name] = concept

    def get_concept(self, name: str) -> Concept:
        """Retrieve a concept by name."""
        try:
            return self._concepts[name]
        except KeyError:
            raise KeyError(f"Concept {name!r} not found") from None

    def remove_concept(self, name: str) -> Concept:
        """Remove a concept and all its relations."""
        try:
            concept = self._concepts.pop(name)
        except KeyError:
            raise KeyError(f"Concept {name!r} not found") from None
        self._relations = [
            r for r in self._relations if r.source != name and r.target != name
        ]
        return concept

    def list_concepts(self) -> list[Concept]:
        """Return all concepts sorted by name."""
        return sorted(self._concepts.values(), key=lambda c: c.name)

    # --- Relation CRUD ---

    def add_relation(self, relation: Relation) -> None:
        """Add a directed relation between two existing concepts."""
        if relation.source not in self._concepts:
            raise KeyError(f"Source concept {relation.source!r} not found")
        if relation.target not in self._concepts:
            raise KeyError(f"Target concept {relation.target!r} not found")
        self._relations.append(relation)

    def get_relations(self, concept_name: str) -> list[Relation]:
        """Return all relations originating from the given concept."""
        return [r for r in self._relations if r.source == concept_name]

    def get_incoming(self, concept_name: str) -> list[Relation]:
        """Return all relations targeting the given concept."""
        return [r for r in self._relations if r.target == concept_name]

    # --- Graph queries ---

    def neighbors(self, concept_name: str) -> list[str]:
        """Return names of concepts directly connected from the given concept."""
        return [r.target for r in self._relations if r.source == concept_name]

    def ancestors(self, concept_name: str) -> list[str]:
        """Return names of concepts that have a relation targeting this concept."""
        return [r.source for r in self._relations if r.target == concept_name]

    def find_path(self, source: str, target: str) -> list[str] | None:
        """BFS to find the shortest path between two concepts.

        Returns the path as a list of concept names, or None if no path exists.
        """
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
            current = path[-1]
            for neighbor in self.neighbors(current):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def search(self, query: str) -> list[Concept]:
        """Search concepts by name or description."""
        q = query.lower()
        return [
            c
            for c in self._concepts.values()
            if q in c.name.lower() or q in c.description.lower()
        ]

    @property
    def concept_count(self) -> int:
        return len(self._concepts)

    @property
    def relation_count(self) -> int:
        return len(self._relations)
