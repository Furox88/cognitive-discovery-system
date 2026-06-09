"""Tests for the scientific note management module."""

import pytest

from cds.notes import Note, Notebook


class TestNoteCreation:
    def test_create_minimal(self):
        n = Note(title="Experiment log", body="Observed the reaction.")
        assert n.title == "Experiment log"
        assert n.body == "Observed the reaction."
        assert n.id
        assert n.created_at is not None

    def test_create_strips_whitespace(self):
        n = Note(title="  Padded  ", body="  Content  ")
        assert n.title == "Padded"
        assert n.body == "Content"

    @pytest.mark.parametrize(
        "title, body",
        [("", "valid"), ("valid", ""), ("  ", "valid"), ("valid", "   ")],
    )
    def test_empty_fields_rejected(self, title, body):
        with pytest.raises(ValueError):
            Note(title=title, body=body)


class TestNoteOperations:
    def _make(self) -> Note:
        return Note(title="Original", body="Original body")

    def test_update_title(self):
        n = self._make()
        n.update(title="Updated")
        assert n.title == "Updated"

    def test_update_body(self):
        n = self._make()
        n.update(body="New body")
        assert n.body == "New body"

    def test_update_empty_title_rejected(self):
        n = self._make()
        with pytest.raises(ValueError):
            n.update(title="")

    def test_update_empty_body_rejected(self):
        n = self._make()
        with pytest.raises(ValueError):
            n.update(body="  ")

    def test_add_tag(self):
        n = self._make()
        n.add_tag("physics")
        assert "physics" in n.tags

    def test_add_tag_normalizes(self):
        n = self._make()
        n.add_tag("  Physics  ")
        assert "physics" in n.tags

    def test_add_tag_deduplicates(self):
        n = self._make()
        n.add_tag("physics")
        n.add_tag("physics")
        assert n.tags.count("physics") == 1

    def test_add_tag_empty_rejected(self):
        n = self._make()
        with pytest.raises(ValueError):
            n.add_tag("")

    def test_remove_tag(self):
        n = self._make()
        n.add_tag("physics")
        n.remove_tag("physics")
        assert "physics" not in n.tags

    def test_remove_tag_missing(self):
        n = self._make()
        with pytest.raises(ValueError):
            n.remove_tag("nonexistent")

    def test_add_reference(self):
        n = self._make()
        n.add_reference("abc123")
        assert "abc123" in n.references

    def test_add_reference_deduplicates(self):
        n = self._make()
        n.add_reference("abc123")
        n.add_reference("abc123")
        assert n.references.count("abc123") == 1

    def test_add_reference_empty_rejected(self):
        n = self._make()
        with pytest.raises(ValueError):
            n.add_reference("")

    def test_snippet_short(self):
        n = Note(title="Short", body="Short body")
        assert n.snippet() == "Short body"

    def test_snippet_long(self):
        n = Note(title="Long", body="A" * 200)
        s = n.snippet(length=80)
        assert len(s) <= 83  # 80 + "..."
        assert s.endswith("...")


class TestNotebook:
    def _notebook_with_items(self) -> tuple[Notebook, Note, Note]:
        nb = Notebook()
        n1 = Note(title="Quantum mechanics", body="Wave-particle duality")
        n1.add_tag("physics")
        n2 = Note(title="Cell biology", body="Mitosis and meiosis")
        n2.add_tag("biology")
        nb.add(n1)
        nb.add(n2)
        return nb, n1, n2

    def test_add_and_get(self):
        nb, n1, _ = self._notebook_with_items()
        assert nb.get(n1.id) is n1

    def test_add_duplicate_rejected(self):
        nb, n1, _ = self._notebook_with_items()
        with pytest.raises(ValueError):
            nb.add(n1)

    def test_get_missing_raises(self):
        nb = Notebook()
        with pytest.raises(KeyError):
            nb.get("nope")

    def test_remove(self):
        nb, n1, _ = self._notebook_with_items()
        removed = nb.remove(n1.id)
        assert removed is n1
        assert n1.id not in nb

    def test_remove_missing_raises(self):
        nb = Notebook()
        with pytest.raises(KeyError):
            nb.remove("nope")

    def test_list_all(self):
        nb, _, _ = self._notebook_with_items()
        assert len(nb.list_all()) == 2

    def test_filter_by_tag(self):
        nb, _, n2 = self._notebook_with_items()
        result = nb.filter_by_tag("biology")
        assert len(result) == 1
        assert result[0] is n2

    def test_search(self):
        nb, n1, _ = self._notebook_with_items()
        result = nb.search("quantum")
        assert len(result) == 1
        assert result[0] is n1

    def test_referenced_by(self):
        nb, n1, n2 = self._notebook_with_items()
        n2.add_reference(n1.id)
        refs = nb.referenced_by(n1.id)
        assert len(refs) == 1
        assert refs[0] is n2

    def test_len_and_contains(self):
        nb, n1, _ = self._notebook_with_items()
        assert len(nb) == 2
        assert n1.id in nb
