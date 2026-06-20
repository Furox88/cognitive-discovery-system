"""Extended tests for the CLI to achieve high coverage."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cds.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()


def test_cli_info() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "System" in result.stdout
    assert "Architecture" in result.stdout


def test_cli_modules() -> None:
    result = runner.invoke(app, ["modules"])
    assert result.exit_code == 0
    assert "Scientific Modules" in result.stdout
    assert "cds.quantum" in result.stdout


def test_cli_constants() -> None:
    result = runner.invoke(app, ["constants"])
    assert result.exit_code == 0
    assert "Physical Constants" in result.stdout
    assert "c" in result.stdout


def test_cli_hypothesize_dry_run() -> None:
    result = runner.invoke(app, ["hypothesis", "Test question", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run mode" in result.stdout


def test_cli_prompt() -> None:
    result = runner.invoke(app, ["prompt", "How to fix gravity?"])
    assert result.exit_code == 0
    assert "Research Question: How to fix gravity?" in result.stdout


def test_cli_hypothesize_output(tmp_path: Path) -> None:
    output_file = tmp_path / "hypo.json"
    result = runner.invoke(app, ["hypothesis", "Test", "--num", "1", "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)
        assert len(data) == 1
        assert "statement" in data[0]


def test_cli_calc_ke(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock user input for mass and velocity
    inputs = iter(["10", "5"])
    monkeypatch.setattr("typer.prompt", lambda _: next(inputs))

    result = runner.invoke(app, ["calc", "ke"])
    assert result.exit_code == 0
    assert "Kinetic Energy = 125.0000 J" in result.stdout


def test_cli_benchmark() -> None:
    result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 0
    assert "Benchmarking System" in result.stdout


def test_cli_hypothesis_basic() -> None:
    result = runner.invoke(app, ["hypothesis", "Why is the sky blue?"])
    assert result.exit_code == 0
    assert "Generating hypotheses for" in result.stdout
    assert "H-" in result.stdout  # Panel title contains H-ID


def test_cli_calc_gravity(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["5.97e24", "7.35e22", "3.84e8"])
    monkeypatch.setattr("typer.prompt", lambda _: next(inputs))
    result = runner.invoke(app, ["calc", "gravity"])
    assert result.exit_code == 0
    assert "Force =" in result.stdout


def test_cli_calc_wave(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("typer.prompt", lambda _: "500e-9")
    result = runner.invoke(app, ["calc", "wave"])
    assert result.exit_code == 0
    assert "Frequency =" in result.stdout


def test_cli_calc_gas(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["1", "300", "0.024"])
    monkeypatch.setattr("typer.prompt", lambda _: next(inputs))
    result = runner.invoke(app, ["calc", "gas"])
    assert result.exit_code == 0
    assert "Pressure =" in result.stdout


def test_cli_calc_invalid() -> None:
    result = runner.invoke(app, ["calc", "unknown"])
    assert result.exit_code == 0
    assert "Unknown formula" in result.stdout


def test_cli_hypothesis_zero_num_omits_detail_panel() -> None:
    # --num 0 makes generate() return an empty list, so the `if hypos:` guard
    # is False and the "Detailed view of first hypothesis" panel is skipped
    # (cli.py 130 -> 134 edge). The command still succeeds and prints the
    # (empty) table title.
    result = runner.invoke(app, ["hypothesis", "anything", "--num", "0"])
    assert result.exit_code == 0
    assert "Detailed view" not in result.stdout
