"""Structured retrieval across a knowledge graph and its notes.

Given a free-text ``query`` and an optional ``tag`` filter, :func:`search`
ranks matching concepts and notes by relevance. The scoring is deliberately
simple and deterministic — no statistics, no tokenization beyond case-folding
— so results are easy to explain and test:

* An exact name/tag match scores ``1.0`` (highest confidence).
* A substring match in the description/body/title scores ``0.5``.
* Results are sorted by descending score, then by name/id alphabetically
  so ties resolve deterministically.

This is the *retrieval* half of Issue #3's four goals; the graph, concept
mapping, and note-management halves live in :mod:`cds.knowledge.graph` and
:mod:`cds.knowledge.notes`.
"""

from __future__ import annotations

from dataclasses import dataclass

from cds.knowledge.graph import KnowledgeGraph
from cds.knowledge.notes import Notebook

# Score constants — exposed as module attributes for clarity in assertions.
NAME_TAG_SCORE = 1.0
SUBSTRING_SCORE = 0.5


@dataclass
class SearchResult:
    """A single ranked retrieval hit.

    Attributes:
        concept_name: the matched concept name, if the hit is a concept;
            ``None`` otherwise.
        note_id: the matched note id, if the hit is a note; ``None`` otherwise.
        score: relevance in ``[0, 1]`` — higher is better.
        matched_on: short label of the field that matched
            (e.g. ``"name"``, ``"description"``, ``"title"``).
    """

    concept_name: str | None
    note_id: str | None
    score: float
    matched_on: str


def search_concepts(
    graph: KnowledgeGraph,
    query: str,
    tag: str | None = None,
) -> list[SearchResult]:
    """Find concepts in ``graph`` matching ``query``.

    A concept matches if its name matches the query exactly (score 1.0) or
    its name or description contains the query as a substring (score 0.5).
    When ``tag`` is given, only concepts carrying that tag are considered.

    Args:
        graph: the :class:`KnowledgeGraph` to search.
        query: case-insensitive search text.
        tag: optional tag filter; ``None`` disables filtering.

    Returns:
        ranked :class:`SearchResult` list (best first, ties alphabetical).
    """
    needle = query.casefold()
    results: list[SearchResult] = []
    for name in sorted(graph.concepts):
        concept = graph.concepts[name]
        if tag is not None and tag not in concept.tags:
            continue
        name_folded = name.casefold()
        if name_folded == needle:
            results.append(
                SearchResult(
                    concept_name=name, note_id=None, score=NAME_TAG_SCORE, matched_on="name"
                )
            )
        elif needle in name_folded:
            results.append(
                SearchResult(
                    concept_name=name, note_id=None, score=SUBSTRING_SCORE, matched_on="name"
                )
            )
        elif concept.description is not None and needle in concept.description.casefold():
            results.append(
                SearchResult(
                    concept_name=name, note_id=None, score=SUBSTRING_SCORE, matched_on="description"
                )
            )
    results.sort(key=lambda r: (-r.score, r.concept_name or ""))
    return results


def search_notes(
    notebook: Notebook,
    query: str,
    tag: str | None = None,
) -> list[SearchResult]:
    """Find notes in ``notebook`` matching ``query``.

    A note matches if its title matches exactly (score 1.0) or its title or
    body contains the query as a substring (score 0.5). When ``tag`` is
    given, only notes carrying that tag are considered.

    Args:
        notebook: the :class:`Notebook` to search.
        query: case-insensitive search text.
        tag: optional tag filter; ``None`` disables filtering.

    Returns:
        ranked :class:`SearchResult` list (best first, ties alphabetical).
    """
    needle = query.casefold()
    results: list[SearchResult] = []
    for note_id in sorted(notebook.notes):
        note = notebook.notes[note_id]
        if tag is not None and tag not in note.tags:
            continue
        title_folded = note.title.casefold()
        if title_folded == needle:
            results.append(
                SearchResult(
                    concept_name=None, note_id=note_id, score=NAME_TAG_SCORE, matched_on="title"
                )
            )
        elif needle in title_folded:
            results.append(
                SearchResult(
                    concept_name=None, note_id=note_id, score=SUBSTRING_SCORE, matched_on="title"
                )
            )
        elif needle in note.body.casefold():
            results.append(
                SearchResult(
                    concept_name=None, note_id=note_id, score=SUBSTRING_SCORE, matched_on="body"
                )
            )
    results.sort(key=lambda r: (-r.score, r.note_id or ""))
    return results


def search(
    graph: KnowledgeGraph,
    notebook: Notebook,
    query: str,
    tag: str | None = None,
) -> list[SearchResult]:
    """Combined ranked search over both a graph's concepts and a notebook's notes.

    Results from :func:`search_concepts` and :func:`search_notes` are merged
    and re-ranked by score (desc) then by identifier (asc).

    Args:
        graph: the :class:`KnowledgeGraph` whose concepts to search.
        notebook: the :class:`Notebook` whose notes to search.
        query: case-insensitive search text.
        tag: optional tag filter applied to both concepts and notes.

    Returns:
        ranked :class:`SearchResult` list (best first, ties alphabetical).
    """
    combined = search_concepts(graph, query, tag) + search_notes(notebook, query, tag)
    combined.sort(key=lambda r: (-r.score, r.concept_name or r.note_id or ""))
    return combined
