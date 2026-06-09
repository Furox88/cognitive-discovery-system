"""Tests for the CLI module."""

import pytest

from cds.cli import main


class TestCLI:
    def test_info_command(self, capsys):
        rc = main(["info"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Cognitive Discovery System" in out
        assert "hypothesis" in out

    def test_hypothesis_new(self, capsys):
        rc = main(["hypothesis", "new", "Water boils at 100C", "Standard physics"])
        assert rc == 0
        assert "Created hypothesis" in capsys.readouterr().out

    def test_note_new(self, capsys):
        rc = main(["note", "new", "Lab Notes", "Observed reaction."])
        assert rc == 0
        assert "Created note" in capsys.readouterr().out

    def test_concept_new(self, capsys):
        rc = main(["concept", "new", "Gravity", "--description", "A force"])
        assert rc == 0
        assert "Created concept" in capsys.readouterr().out

    def test_no_command_shows_help(self, capsys):
        rc = main([])
        assert rc == 0

    def test_hypothesis_no_action_shows_help(self, capsys):
        rc = main(["hypothesis"])
        assert rc == 0
