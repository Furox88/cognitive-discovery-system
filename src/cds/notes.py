"""Scientific note management.

A lightweight system for capturing, tagging, and querying research notes
with full-text search and cross-referencing support.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence


@dataclass
class Note:
    """A single research note with metadata."""

    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.body or not self.body.strip():
            raise ValueError("body must not be empty")
        self.title = self.title.strip()
        self.body = self.body.strip()

    def update(self, title: str | None = None, body: str | None = None) -> None:
        """Update the note's title and/or body."""
        if title is not None:
            if not title.strip():
                raise ValueError("title must not be empty")
            self.title = title.strip()
        if body is not None:
            if not body.strip():
                raise ValueError("body must not be empty")
            self.body = body.strip()
        self.updated_at = datetime.now(timezone.utc)

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        tag = tag.strip().lower()
        if not tag:
            raise ValueError("tag must not be empty")
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag."""
        tag = tag.strip().lower()
        try:
            self.tags.remove(tag)
        except ValueError:
            raise ValueError(f"Tag {tag!r} not found on this note") from None

    def add_reference(self, ref_id: str) -> None:
        """Add a cross-reference to another note."""
        if not ref_id or not ref_id.strip():
            raise ValueError("ref_id must not be empty")
        ref_id = ref_id.strip()
        if ref_id not in self.references:
            self.references.append(ref_id)

    def snippet(self, length: int = 80) -> str:
        """Return a truncated preview of the body."""
        if len(self.body) <= length:
            return self.body
        return self.body[:length].rstrip() + "..."


class Notebook:
    """An in-memory collection of research notes with search capabilities."""

    def __init__(self) -> None:
        self._notes: dict[str, Note] = {}

    def add(self, note: Note) -> str:
        """Add a note and return its id."""
        if note.id in self._notes:
            raise ValueError(f"Note {note.id!r} already exists")
        self._notes[note.id] = note
        return note.id

    def get(self, note_id: str) -> Note:
        """Retrieve a note by id."""
        try:
            return self._notes[note_id]
        except KeyError:
            raise KeyError(f"Note {note_id!r} not found") from None

    def remove(self, note_id: str) -> Note:
        """Remove and return a note by id."""
        try:
            return self._notes.pop(note_id)
        except KeyError:
            raise KeyError(f"Note {note_id!r} not found") from None

    def list_all(self) -> list[Note]:
        """Return all notes ordered by last update."""
        return sorted(self._notes.values(), key=lambda n: n.updated_at, reverse=True)

    def filter_by_tag(self, tag: str) -> list[Note]:
        """Return notes that contain the given tag."""
        tag = tag.strip().lower()
        return [n for n in self._notes.values() if tag in n.tags]

    def search(self, query: str) -> list[Note]:
        """Full-text keyword search across title and body."""
        q = query.lower()
        return [
            n
            for n in self._notes.values()
            if q in n.title.lower() or q in n.body.lower()
        ]

    def referenced_by(self, note_id: str) -> list[Note]:
        """Return notes that reference the given note id."""
        return [n for n in self._notes.values() if note_id in n.references]

    def __len__(self) -> int:
        return len(self._notes)

    def __contains__(self, note_id: str) -> bool:
        return note_id in self._notes
