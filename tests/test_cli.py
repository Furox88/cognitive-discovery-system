"""Basic smoke tests for the CDS CLI (no heavy output parsing)."""

from typer.testing import CliRunner

from cds.cli import app

runner = CliRunner()


def test_help_shows_main_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "modules" in result.output
    assert "hypothesis" in result.output
    assert "hypothesize" in result.output


def test_modules_runs():
    result = runner.invoke(app, ["modules"])
    assert result.exit_code == 0
    assert "CDS Scientific Modules" in result.output
    assert "cds.hypothesis" in result.output


def test_hypothesis_runs_with_default():
    result = runner.invoke(app, ["hypothesis"])
    assert result.exit_code == 0
    assert "Generating hypotheses for:" in result.output


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "CDS" in result.output and "version" in result.output.lower()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "CDS" in result.output and "version" in result.output.lower()
