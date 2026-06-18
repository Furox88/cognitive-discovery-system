"""Knowledge graph — typed relationships between named concepts.

The core of :mod:`cds.knowledge`. A :class:`KnowledgeGraph` stores named
:class:`Concept` nodes (each with optional description, tags, and free-form
metadata) connected by typed, directed :class:`Relation` edges
(``"is-a"``, ``"depends-on"``, ``"related-to"``, …).

This is deliberately a *purpose-built* structure rather than a wrapper
around :mod:`cds.graph`. The latter is hard-wired to dense integer vertex
IDs and carries only an untyped float weight per edge, whereas a knowledge
graph's value is exactly string-keyed nodes with rich metadata and typed
relationships. The traversal here (BFS path-finding, transitive closure,
cycle detection) is small and self-contained.

Design choices:

* Relations are **directed** (``source -> target``): ``"A depends-on B"``
  is not symmetric. For convenience, :meth:`KnowledgeGraph.neighbors`
  reports both directions, while :meth:`neighbors_in` / :meth:`neighbors_out`
  expose the direction when you need it.
* Persistence mirrors the :class:`cds.nlp.bpe.BPEMerge` idiom: every
  entity serializes via :meth:`to_dict`/``from_dict``, and the owning
  :class:`KnowledgeGraph` orchestrates :meth:`save`/:meth:`load` with the
  stdlib :mod:`json` module.

References:
    - Sowa, J.F. (1976). "Conceptual Graphs for a Data Base Interface"
      — the typed-relationship convention used by :class:`Relation`.
    - Cormen et al., CLRS §22 — the BFS shortest-path / reachability
      algorithms used by :meth:`find_path` and :meth:`reachable`.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Concept:
    """A named node in the knowledge graph.

    Attributes:
        name: the unique, human-readable concept identifier (also its
            dictionary key inside a :class:`KnowledgeGraph`).
        description: optional one- or two-line summary of the concept.
        tags: free-form labels for grouping and retrieval (e.g.
            ``["physics", "mechanics"]``).
        metadata: additional string-valued properties (e.g. source URLs,
            units) that don't fit the structured fields above.
    """

    name: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Concept) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("Concept", self.name))

    def to_dict(self) -> dict[str, object]:
        """Serialize this concept to a JSON-friendly dict."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Concept:
        """Reconstruct a :class:`Concept` from :meth:`to_dict` output.

        Raises:
            ValueError: if ``data`` is missing keys or has the wrong types.
        """
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError(f"Invalid concept name: {name!r}")
        description = data["description"]
        if description is not None and not isinstance(description, str):
            raise ValueError(f"Invalid concept description: {description!r}")
        tags_raw = data["tags"]
        if not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw):
            raise ValueError(f"Invalid concept tags: {tags_raw!r}")
        metadata_raw = data["metadata"]
        if not isinstance(metadata_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata_raw.items()
        ):
            raise ValueError(f"Invalid concept metadata: {metadata_raw!r}")
        return cls(
            name=name,
            description=description,
            tags=list(tags_raw),
            metadata=dict(metadata_raw),
        )


@dataclass
class Relation:
    """A typed, directed edge ``source -> target`` between two concepts.

    Attributes:
        source: name of the origin concept.
        target: name of the destination concept.
        kind: the relationship type (e.g. ``"is-a"``, ``"depends-on"``,
            ``"related-to"``). Semantics are caller-defined; the graph does
            not interpret kinds beyond using them for filtering.
        weight: optional numeric strength (default 1.0). Higher is stronger;
            used by callers for ranking, not by the core traversal.
    """

    source: str
    target: str
    kind: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        """Serialize this relation to a JSON-friendly dict."""
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Relation:
        """Reconstruct a :class:`Relation` from :meth:`to_dict` output.

        Raises:
            ValueError: if ``data`` is missing keys or has the wrong types.
        """
        source = data["source"]
        target = data["target"]
        kind = data["kind"]
        if not isinstance(source, str):
            raise ValueError(f"Invalid relation source: {source!r}")
        if not isinstance(target, str):
            raise ValueError(f"Invalid relation target: {target!r}")
        if not isinstance(kind, str):
            raise ValueError(f"Invalid relation kind: {kind!r}")
        weight = data["weight"]
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise ValueError(f"Invalid relation weight: {weight!r}")
        return cls(source=source, target=target, kind=kind, weight=float(weight))


@dataclass
class KnowledgeGraph:
    """A knowledge graph of named concepts and typed relations.

    Attributes:
        name: human-readable graph title (used in :meth:`to_markdown`).
        concepts: mapping of concept name to :class:`Concept` node.
        relations: ordered list of :class:`Relation` edges.
    """

    name: str
    concepts: dict[str, Concept] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def add_concept(
        self,
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Concept:
        """Add a concept, returning the stored node.

        If ``name`` already exists, the existing concept is returned
        unchanged (idempotent) rather than overwritten.

        Args:
            name: unique concept identifier.
            description: optional summary.
            tags: optional grouping labels.
            metadata: optional string-valued properties.

        Returns:
            the stored :class:`Concept` (newly created or the pre-existing one).
        """
        if name in self.concepts:
            return self.concepts[name]
        concept = Concept(
            name=name,
            description=description,
            tags=list(tags) if tags else [],
            metadata=dict(metadata) if metadata else {},
        )
        self.concepts[name] = concept
        return concept

    def add_relation(
        self,
        source: str,
        target: str,
        kind: str,
        weight: float = 1.0,
    ) -> Relation:
        """Add a typed, directed relation ``source -> target``.

        Both endpoints must already exist as concepts (use
        :meth:`link_concepts` to auto-create them).

        Raises:
            KeyError: if ``source`` or ``target`` is not a known concept.
        """
        for endpoint, label in ((source, "source"), (target, "target")):
            if endpoint not in self.concepts:
                raise KeyError(f"unknown {label} concept: {endpoint!r}")
        relation = Relation(source=source, target=target, kind=kind, weight=weight)
        self.relations.append(relation)
        return relation

    def link_concepts(
        self,
        source: str,
        target: str,
        kind: str,
        weight: float = 1.0,
    ) -> Relation:
        """Auto-create both concepts (if missing) and add a relation between them."""
        self.add_concept(source)
        self.add_concept(target)
        return self.add_relation(source, target, kind, weight)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def neighbors(self, name: str, kind: str | None = None) -> list[str]:
        """Undirected neighbors of ``name`` — every concept directly linked.

        A relation touching ``name`` at either endpoint contributes its
        *other* endpoint. ``kind`` optionally restricts to one relation type.

        Raises:
            KeyError: if ``name`` is not a known concept.
        """
        self._require_concept(name)
        found: list[str] = []
        for relation in self.relations:
            if kind is not None and relation.kind != kind:
                continue
            if relation.source == name and relation.target not in found:
                found.append(relation.target)
            elif relation.target == name and relation.source not in found:
                found.append(relation.source)
        return found

    def neighbors_out(self, name: str, kind: str | None = None) -> list[str]:
        """Concepts that ``name`` points to via outgoing relations.

        Raises:
            KeyError: if ``name`` is not a known concept.
        """
        self._require_concept(name)
        found: list[str] = []
        for relation in self.relations:
            if kind is not None and relation.kind != kind:
                continue
            if relation.source == name and relation.target not in found:
                found.append(relation.target)
        return found

    def neighbors_in(self, name: str, kind: str | None = None) -> list[str]:
        """Concepts that point at ``name`` via incoming relations.

        Raises:
            KeyError: if ``name`` is not a known concept.
        """
        self._require_concept(name)
        found: list[str] = []
        for relation in self.relations:
            if kind is not None and relation.kind != kind:
                continue
            if relation.target == name and relation.source not in found:
                found.append(relation.source)
        return found

    def find_path(self, source: str, target: str) -> list[str] | None:
        """Shortest undirected path (by hop count) from ``source`` to ``target``.

        Returns the sequence of concept names ``[source, ..., target]``, or
        ``None`` if no path exists or the endpoints are unknown. A path of
        length 1 (``source == target``) returns ``[source]``.

        Uses BFS following the edges in either direction, so the returned
        path may traverse relations against their direction.
        """
        if source not in self.concepts or target not in self.concepts:
            return None
        if source == target:
            return [source]
        predecessor: dict[str, str] = {source: source}
        queue: deque[str] = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in self.neighbors(node):
                if neighbor in predecessor:
                    continue
                predecessor[neighbor] = node
                if neighbor == target:
                    return _reconstruct_path(predecessor, target)
                queue.append(neighbor)
        return None

    def reachable(self, start: str) -> set[str]:
        """All concepts reachable from ``start`` over undirected edges (incl. itself).

        Returns an empty set if ``start`` is not a known concept.
        """
        if start not in self.concepts:
            return set()
        seen: set[str] = set()
        queue: deque[str] = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in self.neighbors(node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return seen

    def find_cycles(self) -> list[list[str]]:
        """Return every directed cycle in the graph as a list of concept names.

        Each cycle is reported once, normalized to start at its
        lexicographically smallest member so the same cycle is not reported
        from every starting rotation. Self-loops (a relation whose source and
        target are equal) are returned as ``[name]``.

        Uses DFS back-edge detection with an explicit recursion-emulating
        stack so deep graphs do not hit Python's recursion limit.
        """
        adj: dict[str, list[str]] = {name: [] for name in self.concepts}
        for relation in self.relations:
            adj[relation.source].append(relation.target)

        found: set[tuple[str, ...]] = set()
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in self.concepts}

        for root in sorted(self.concepts):
            if color[root] != WHITE:
                continue
            # Each stack frame: the node plus an iterator position over its successors.
            stack: list[tuple[str, list[str]]] = [(root, list(adj[root]))]
            color[root] = GRAY
            path: list[str] = [root]
            while stack:
                node, succs = stack[-1]
                advanced = False
                while succs:
                    nxt = succs.pop()
                    if color[nxt] == GRAY:
                        # Back edge: a cycle from nxt back along the current
                        # DFS path. A node is GRAY iff it is on ``path`` (we
                        # always append to ``path`` in lockstep with marking
                        # GRAY below), so ``nxt`` is guaranteed to be present.
                        cycle = path[path.index(nxt) :]
                        found.add(_normalize_cycle(cycle))
                        # Do not descend into the gray node; keep scanning successors.
                        continue
                    # The only remaining color is WHITE: descend into it.
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append((nxt, list(adj[nxt])))
                    advanced = True
                    break
                if not advanced:
                    # Exhausted this node's successors: mark black and pop.
                    color[node] = BLACK
                    path.pop()
                    stack.pop()
        return [list(cycle) for cycle in sorted(found)]

    def _require_concept(self, name: str) -> None:
        if name not in self.concepts:
            raise KeyError(f"unknown concept: {name!r}")

    # ------------------------------------------------------------------ #
    # Rendering & serialization
    # ------------------------------------------------------------------ #
    def to_markdown(self) -> str:
        """Render this graph as a structured Markdown document."""
        lines: list[str] = [f"# Knowledge Graph: {self.name}", ""]
        if not self.concepts:
            lines += ["_No concepts._", ""]
        else:
            lines += ["## Concepts", ""]
            for name in sorted(self.concepts):
                lines.append(f"- **{name}**")
            lines.append("")
        if not self.relations:
            lines += ["_No relations._", ""]
        else:
            lines += ["## Relations", ""]
            for relation in self.relations:
                lines.append(f"- `{relation.source}` --{relation.kind}--> `{relation.target}`")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialize the whole graph to a JSON-friendly dict."""
        return {
            "name": self.name,
            "concepts": [concept.to_dict() for concept in self.concepts.values()],
            "relations": [relation.to_dict() for relation in self.relations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> KnowledgeGraph:
        """Reconstruct a :class:`KnowledgeGraph` from :meth:`to_dict` output.

        Raises:
            ValueError: if ``data`` is missing keys or has the wrong types.
        """
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError(f"Invalid graph name: {name!r}")
        concepts_raw = data["concepts"]
        relations_raw = data["relations"]
        if not isinstance(concepts_raw, list):
            raise ValueError(f"Invalid concepts list: {concepts_raw!r}")
        if not isinstance(relations_raw, list):
            raise ValueError(f"Invalid relations list: {relations_raw!r}")
        graph = cls(name=name)
        for item in concepts_raw:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid concept entry: {item!r}")
            concept = Concept.from_dict(item)
            graph.concepts[concept.name] = concept
        for item in relations_raw:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid relation entry: {item!r}")
            graph.relations.append(Relation.from_dict(item))
        return graph

    def save(self, path: str | Path) -> None:
        """Write this graph to ``path`` as indented UTF-8 JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        """Read a graph previously written by :meth:`save`.

        Raises:
            ValueError: if the file does not contain valid graph JSON.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid graph file (expected object): {data!r}")
        return cls.from_dict(data)


def _reconstruct_path(predecessor: dict[str, str], target: str) -> list[str]:
    """Walk the ``predecessor`` map backwards from ``target`` to the root."""
    path: list[str] = [target]
    node = target
    while predecessor[node] != node:
        node = predecessor[node]
        path.append(node)
    path.reverse()
    return path


def _normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
    """Return ``cycle`` rotated to start at its smallest member (as a tuple)."""
    if not cycle:
        return ()
    pivot = cycle.index(min(cycle))
    rotated = cycle[pivot:] + cycle[:pivot]
    return tuple(rotated)
