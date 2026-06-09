import pytest
from cds.notes import Note, Notebook


class TestNote:
    def test_create(self):
        n = Note(title="Experiment log", body="Observed the reaction.")
        assert n.title == "Experiment log"
        assert n.id

    def test_strips_whitespace(self):
        n = Note(title="  Padded  ", body="  Content  ")
        assert n.title == "Padded"

    @pytest.mark.parametrize("t,b", [("", "ok"), ("ok", ""), ("  ", "ok")])
    def test_rejects_empty(self, t, b):
        with pytest.raises(ValueError):
            Note(title=t, body=b)

    def test_update(self):
        n = Note(title="Old", body="Old body")
        n.update(title="New", body="New body")
        assert n.title == "New"

    def test_update_empty_title(self):
        n = Note(title="x", body="y")
        with pytest.raises(ValueError):
            n.update(title="")

    def test_update_empty_body(self):
        n = Note(title="x", body="y")
        with pytest.raises(ValueError):
            n.update(body="  ")

    def test_tags(self):
        n = Note(title="x", body="y")
        n.add_tag("physics")
        n.add_tag("  Physics  ")  # should deduplicate
        assert n.tags.count("physics") == 1

    def test_tag_empty(self):
        n = Note(title="x", body="y")
        with pytest.raises(ValueError):
            n.add_tag("")

    def test_remove_tag(self):
        n = Note(title="x", body="y")
        n.add_tag("physics")
        n.remove_tag("physics")
        assert "physics" not in n.tags

    def test_remove_tag_missing(self):
        n = Note(title="x", body="y")
        with pytest.raises(ValueError):
            n.remove_tag("nope")

    def test_references(self):
        n = Note(title="x", body="y")
        n.add_reference("abc")
        n.add_reference("abc")  # dedup
        assert n.references.count("abc") == 1

    def test_ref_empty(self):
        n = Note(title="x", body="y")
        with pytest.raises(ValueError):
            n.add_reference("")

    def test_snippet(self):
        n = Note(title="x", body="A" * 200)
        s = n.snippet(80)
        assert s.endswith("...")
        assert len(s) <= 83

    def test_snippet_short(self):
        n = Note(title="x", body="short")
        assert n.snippet() == "short"


class TestNotebook:
    def _make(self):
        nb = Notebook()
        n1 = Note(title="Quantum", body="Wave-particle duality")
        n1.add_tag("physics")
        n2 = Note(title="Biology", body="Mitosis stuff")
        n2.add_tag("bio")
        nb.add(n1)
        nb.add(n2)
        return nb, n1, n2

    def test_add_get(self):
        nb, n1, _ = self._make()
        assert nb.get(n1.id) is n1

    def test_duplicate(self):
        nb, n1, _ = self._make()
        with pytest.raises(ValueError):
            nb.add(n1)

    def test_missing(self):
        with pytest.raises(KeyError):
            Notebook().get("nope")

    def test_remove(self):
        nb, n1, _ = self._make()
        assert nb.remove(n1.id) is n1
        assert n1.id not in nb

    def test_remove_missing(self):
        with pytest.raises(KeyError):
            Notebook().remove("nope")

    def test_list_all(self):
        nb, _, _ = self._make()
        assert len(nb.list_all()) == 2

    def test_filter_tag(self):
        nb, _, n2 = self._make()
        assert nb.filter_by_tag("bio")[0] is n2

    def test_search(self):
        nb, n1, _ = self._make()
        assert nb.search("quantum")[0] is n1

    def test_referenced_by(self):
        nb, n1, n2 = self._make()
        n2.add_reference(n1.id)
        assert nb.referenced_by(n1.id)[0] is n2

    def test_len(self):
        nb, n1, _ = self._make()
        assert len(nb) == 2
        assert n1.id in nb
