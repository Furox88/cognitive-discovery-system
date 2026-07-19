"""Extended tests for the CLI to achieve high coverage.

Drives :func:`cds.cli.main` with an explicit ``argv`` and captures output via
``capsys``. The ``calc`` subcommand reads interactive prompts through
:func:`builtins.input`, which tests monkeypatch to feed scripted answers —
the previous approach patched ``typer.prompt`` instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cds.cli import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "version" in out.lower()


def test_cli_info(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "System" in out
    assert "Architecture" in out


def test_cli_modules(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["modules"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Scientific Modules" in out
    assert "cds.quantum" in out


def test_cli_constants(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["constants"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Physical Constants" in out
    assert "c" in out


def test_cli_hypothesize_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["hypothesis", "Test question", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Dry run mode" in out


def test_cli_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["prompt", "How to fix gravity?"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "How to fix gravity?" in out


def test_cli_hypothesize_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_file = tmp_path / "hypo.json"
    rc = main(["hypothesis", "Test", "--num", "1", "--output", str(output_file)])
    capsys.readouterr()
    assert rc == 0
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)
        assert len(data) == 1
        assert "statement" in data[0]


def test_cli_calc_ke(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Mock user input for mass and velocity
    inputs = iter(["10", "5"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    rc = main(["calc", "ke"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Kinetic Energy = 125.0000 J" in out


def test_cli_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["benchmark"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Benchmarking System" in out


def test_cli_hypothesis_basic(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["hypothesis", "Why is the sky blue?"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Generating hypotheses for" in out
    assert "H-" in out  # Table row contains H-ID


def test_cli_calc_gravity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = iter(["5.97e24", "7.35e22", "3.84e8"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    rc = main(["calc", "gravity"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Force =" in out


def test_cli_calc_wave(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "500e-9")
    rc = main(["calc", "wave"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Frequency =" in out


def test_cli_calc_gas(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    inputs = iter(["1", "300", "0.024"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    rc = main(["calc", "gas"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Pressure =" in out


def test_cli_calc_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["calc", "unknown"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Unknown formula" in out


def test_cli_hypothesis_zero_num_omits_detail_panel(capsys: pytest.CaptureFixture[str]) -> None:
    # --num 0 makes generate() return an empty list, so the `if hypos:` guard
    # is False and the "Detailed view of first hypothesis" line is skipped.
    # The command still succeeds and prints the (empty) table title.
    rc = main(["hypothesis", "anything", "--num", "0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Detailed view" not in out


def test_cli_stats(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["stats", "1,2,3,4,5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mean" in out
    assert "3" in out


def test_cli_sample_gaussian(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["sample", "gaussian", "-n", "3", "--seed", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert len(out.strip().split(",")) == 3


def test_cli_sample_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["sample", "uniform", "-n", "2", "--a", "1", "--b", "0"])
    assert rc == 1


def test_cli_stats_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["stats", "1,x,3"])
    assert rc == 1


def test_cli_stats_single(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["stats", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mean" in out
    assert "stdev" not in out  # n==1 skips stdev rows


def test_cli_sample_all_dists(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    for dist in ("uniform", "exponential", "poisson"):
        rc = main(["sample", dist, "-n", "2", "--seed", "0"])
        assert rc == 0
        assert capsys.readouterr().out.strip()


def test_cli_plot_file_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--kind acf with too-short series surfaces ValueError."""
    import cds.plot as plot_mod

    def boom(*_a: object, **_k: object) -> object:
        raise ValueError("data must have at least 2 observations")

    monkeypatch.setattr(plot_mod, "plot_acf", boom)
    path = tmp_path / "x.png"
    rc = main(["plot", "1", "--kind", "acf", "--file", str(path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out


def test_cli_integrate(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["integrate", "x2", "--a", "0", "--b", "1", "-n", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "≈" in out or "0.33" in out


def test_cli_integrate_trap(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["integrate", "unit", "--a", "0", "--b", "2", "--method", "trap", "-n", "10"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2" in out


def test_cli_integrate_bad_n(capsys: pytest.CaptureFixture[str]) -> None:
    from cds.cli import main

    rc = main(["integrate", "sin", "-n", "0"])
    assert rc == 1
