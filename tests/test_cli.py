from cds.cli import main


class TestCLI:
    def test_info(self, capsys):
        assert main(["info"]) == 0
        assert "Cognitive Discovery System" in capsys.readouterr().out

    def test_hypothesis(self, capsys):
        assert main(["hypothesis", "new", "Water boils at 100C", "Physics"]) == 0
        assert "Created hypothesis" in capsys.readouterr().out

    def test_note(self, capsys):
        assert main(["note", "new", "Lab Notes", "Observed reaction."]) == 0
        assert "Created note" in capsys.readouterr().out

    def test_concept(self, capsys):
        assert main(["concept", "new", "Gravity", "--description", "A force"]) == 0
        assert "Created concept" in capsys.readouterr().out

    def test_no_command(self, capsys):
        assert main([]) == 0

    def test_hypothesis_no_action(self, capsys):
        assert main(["hypothesis"]) == 0
