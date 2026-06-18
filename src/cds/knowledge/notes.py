"""Research notes linked to concepts — :class:`Note` and :class:`Notebook`.

A :class:`Note` is a free-form record (title + body + tags) that can be
linked to one or more concept names in a :class:`KnowledgeGraph`. A
:class:`Notebook` is the collection that owns notes by id and answers
"which notes mention this concept?" and "which notes carry this tag?".

Like :mod:`cds.knowledge.graph`, persistence mirrors the
:class:`cds.nlp.bpe.BPEMerge` idiom: :meth:`to_dict`/``from_dict`` per
entity, :meth:`Notebook.save`/:meth:`Notebook.load` for file round-trips.

Notes are deliberately decoupled from the graph: a note may reference a
concept name that is not (yet) in any graph, and adding a concept never
requires touching notes. The link is a string, not a reference, mirroring
how a citation works in a lab notebook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Note:
    """A single research note linked to zero or more concept names.

    Attributes:
        id: unique note identifier within a :class:`Notebook`.
        title: short human-readable heading.
        body: the note's free-form text content.
        tags: grouping labels (e.g. ``["experiment", "failed"]``).
        linked_concepts: names of concepts this note references. These are
            plain strings, not references — they need not exist in any
            particular :class:`KnowledgeGraph`.
        created: optional ISO-8601 timestamp (or any caller-defined marker).
    """

    id: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    linked_concepts: list[str] = field(default_factory=list)
    created: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize this note to a JSON-friendly dict."""
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "linked_concepts": list(self.linked_concepts),
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Note:
        """Reconstruct a :class:`Note` from :meth:`to_dict` output.

        Raises:
            ValueError: if ``data`` is missing keys or has the wrong types.
        """
        note_id = data["id"]
        title = data["title"]
        body = data["body"]
        if not isinstance(note_id, str):
            raise ValueError(f"Invalid note id: {note_id!r}")
        if not isinstance(title, str):
            raise ValueError(f"Invalid note title: {title!r}")
        if not isinstance(body, str):
            raise ValueError(f"Invalid note body: {body!r}")
        tags_raw = data["tags"]
        if not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw):
            raise ValueError(f"Invalid note tags: {tags_raw!r}")
        linked_raw = data["linked_concepts"]
        if not isinstance(linked_raw, list) or not all(isinstance(c, str) for c in linked_raw):
            raise ValueError(f"Invalid note linked_concepts: {linked_raw!r}")
        created = data["created"]
        if created is not None and not isinstance(created, str):
            raise ValueError(f"Invalid note created: {created!r}")
        return cls(
            id=note_id,
            title=title,
            body=body,
            tags=list(tags_raw),
            linked_concepts=list(linked_raw),
            created=created,
        )

    def to_markdown(self) -> str:
        """Render this note as a self-contained Markdown document."""
        lines: list[str] = [f"# {self.title}", ""]
        if self.created:
            lines += [f"_Created: {self.created}_", ""]
        if self.tags:
            lines += ["Tags: " + ", ".join(f"`{tag}`" for tag in self.tags), ""]
        lines += [self.body, ""]
        if self.linked_concepts:
            lines += ["## Linked concepts", ""]
            for concept in self.linked_concepts:
                lines.append(f"- `{concept}`")
            lines.append("")
        return "\n".join(lines)


@dataclass
class Notebook:
    """An ordered collection of research notes keyed by id.

    Attributes:
        name: human-readable notebook title (used in :meth:`to_markdown`).
        notes: mapping of note id to :class:`Note`.
    """

    name: str
    notes: dict[str, Note] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction & lookup
    # ------------------------------------------------------------------ #
    def add_note(
        self,
        note_id: str,
        title: str,
        body: str,
        tags: list[str] | None = None,
        linked_concepts: list[str] | None = None,
        created: str | None = None,
    ) -> Note:
        """Add a note, returning the stored :class:`Note`.

        If ``note_id`` already exists it is overwritten (last-write-wins),
        matching how a researcher edits a numbered entry in place.

        Returns:
            the stored :class:`Note`.
        """
        note = Note(
            id=note_id,
            title=title,
            body=body,
            tags=list(tags) if tags else [],
            linked_concepts=list(linked_concepts) if linked_concepts else [],
            created=created,
        )
        self.notes[note_id] = note
        return note

    def get_note(self, note_id: str) -> Note:
        """Return the note with ``note_id``.

        Raises:
            KeyError: if ``note_id`` is not in this notebook.
        """
        if note_id not in self.notes:
            raise KeyError(f"unknown note: {note_id!r}")
        return self.notes[note_id]

    def notes_for_concept(self, concept: str) -> list[Note]:
        """All notes that reference ``concept`` (by linked_concepts membership)."""
        return [note for note in self.notes.values() if concept in note.linked_concepts]

    def notes_by_tag(self, tag: str) -> list[Note]:
        """All notes carrying ``tag``."""
        return [note for note in self.notes.values() if tag in note.tags]

    # ------------------------------------------------------------------ #
    # Rendering & serialization
    # ------------------------------------------------------------------ #
    def to_markdown(self) -> str:
        """Render a compact index of this notebook's notes as Markdown."""
        lines: list[str] = [f"# Notebook: {self.name}", ""]
        if not self.notes:
            lines += ["_No notes._", ""]
        else:
            lines += ["## Notes", ""]
            for note_id in sorted(self.notes):
                note = self.notes[note_id]
                tags = f" ({', '.join(note.tags)})" if note.tags else ""
                lines.append(f"- **{note_id}**: {note.title}{tags}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialize the notebook to a JSON-friendly dict."""
        return {
            "name": self.name,
            "notes": [note.to_dict() for note in self.notes.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Notebook:
        """Reconstruct a :class:`Notebook` from :meth:`to_dict` output.

        Raises:
            ValueError: if ``data`` is missing keys or has the wrong types.
        """
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError(f"Invalid notebook name: {name!r}")
        notes_raw = data["notes"]
        if not isinstance(notes_raw, list):
            raise ValueError(f"Invalid notes list: {notes_raw!r}")
        notebook = cls(name=name)
        for item in notes_raw:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid note entry: {item!r}")
            note = Note.from_dict(item)
            notebook.notes[note.id] = note
        return notebook

    def save(self, path: str | Path) -> None:
        """Write this notebook to ``path`` as indented UTF-8 JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Notebook:
        """Read a notebook previously written by :meth:`save`.

        Raises:
            ValueError: if the file does not contain valid notebook JSON.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid notebook file (expected object): {data!r}")
        return cls.from_dict(data)
