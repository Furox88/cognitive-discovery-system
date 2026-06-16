"""Tests covering previously untested code paths.

Targets:
- cds.quantum.simulator.measure (state collapse)
- cds.__main__ entry point
- cds.hypothesis.evaluator new dispatch paths
"""
import math
import subprocess
import sys

from cds import __version__

from typer.testing import CliRunner

import cds.cli as cli_mod
from cds.cli import app
from cds.hypothesis import Domain, HypothesisEvaluator, HypothesisStatus, generate_hypotheses
from cds.montecarlo import buffon_needle, estimate_pi
from cds.montecarlo.methods import _pi_worker
from cds.quantum import QuantumCircuit, hadamard, pauli_x
from cds.quantum.circuit import Qubit
from cds.quantum.simulator import measure, simulate

_runner = CliRunner()


# ---------------------------------------------------------------------------
# Quantum simulator: measure() state collapse
# ---------------------------------------------------------------------------
def test_measure_collapse_to_zero():
    """A qubit in |0> collapses to 0 and stays |0>."""
    q = Qubit(alpha=1.0 + 0j, beta=0.0 + 0j)
    outcome = measure(q)
    assert outcome == 0
    assert q.alpha == 1.0
    assert q.beta == 0.0


def test_measure_collapse_to_one():
    """A qubit in |1> collapses to 1 and stays |1>."""
    q = Qubit(alpha=0.0 + 0j, beta=1.0 + 0j)
    outcome = measure(q)
    assert outcome == 1
    assert q.alpha == 0.0
    assert q.beta == 1.0


def test_measure_superposition_distribution():
    """A |+> qubit collapses ~50/50 over many samples."""
    inv = 1.0 / math.sqrt(2)
    counts = {0: 0, 1: 0}
    for _ in range(2000):
        q = Qubit(alpha=complex(inv), beta=complex(inv))
        counts[measure(q)] += 1
    # Both outcomes should occur in 2000 shots
    assert counts[0] > 0 and counts[1] > 0


def test_simulate_seed_reproducible():
    """simulate() with the same seed gives identical counts."""
    circuit = QuantumCircuit().add(hadamard()).add(pauli_x())
    a = simulate(circuit, shots=500, seed=123)
    b = simulate(circuit, shots=500, seed=123)
    assert a == b


# ---------------------------------------------------------------------------
# __main__ entry point (coverage for cds.__main__)
# ---------------------------------------------------------------------------
def test_main_module_runs():
    """`python -m cds --version` exits 0 and prints the version."""
    result = subprocess.run(
        [sys.executable, "-m", "cds", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


# ---------------------------------------------------------------------------
# Evaluator: new dispatch paths
# ---------------------------------------------------------------------------
def _make_hypothesis():
    hypos = generate_hypotheses("Test question?", Domain.GENERAL_SCIENCE, n=1)
    return hypos[0]


def test_evaluator_one_sample_significant():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    # Sample mean well above the reference mean
    result = evaluator.evaluate(
        hypo,
        {"one_sample": [10.2, 10.5, 9.8, 11.0, 10.4], "popmean": 5.0},
    )
    assert result.test_name == "One-sample t-test"
    assert result.is_significant
    assert hypo.status == HypothesisStatus.VALIDATED


def test_evaluator_one_sample_not_significant():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    # Sample mean close to the reference mean
    result = evaluator.evaluate(
        hypo,
        {"one_sample": [10.0, 10.1, 9.9, 10.0, 10.1], "popmean": 10.0},
    )
    assert not result.is_significant
    assert hypo.status == HypothesisStatus.REJECTED


def test_evaluator_chi_square_gof():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    # Strongly skewed observed vs uniform expected -> significant
    result = evaluator.evaluate(
        hypo,
        {"chi_square_gof": {"observed": [50, 10, 10, 30]}},
    )
    assert result.test_name == "Chi-square goodness-of-fit"
    assert result.is_significant


def test_evaluator_chi_square_independence():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    # Strong association between rows and columns
    table = [[100, 10], [10, 100]]
    result = evaluator.evaluate(hypo, {"chi_square_independence": table})
    assert result.test_name == "Chi-square independence"
    assert result.is_significant


def test_evaluator_paired():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        hypo,
        {"paired": ([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0])},
    )
    assert result.test_name == "Two-sample t-test"
    assert result.is_significant


def test_evaluator_unsupported_format_raises():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.evaluate(hypo, {"unknown_key": [1, 2, 3]})
        raise AssertionError("Expected ValueError for unsupported data format")
    except ValueError as exc:
        assert "Unsupported data format" in str(exc)


def test_evaluator_compare_groups_too_few_raises():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.compare_groups(hypo, [[1.0, 2.0]])
        raise AssertionError("Expected ValueError for single group")
    except ValueError as exc:
        assert "2 groups" in str(exc)


def test_evaluator_goodness_of_fit_too_few_raises():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.goodness_of_fit(hypo, [5.0])
        raise AssertionError("Expected ValueError for single category")
    except ValueError as exc:
        assert "2 categories" in str(exc)


def test_evaluator_independence_bad_table_raises():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.test_independence(hypo, [[1.0]])
        raise AssertionError("Expected ValueError for bad contingency table")
    except ValueError as exc:
        assert "contingency" in str(exc)


def test_evaluator_one_sample_too_few_raises():
    hypo = _make_hypothesis()
    evaluator = HypothesisEvaluator()
    try:
        evaluator.compare_to_reference(hypo, [5.0], popmean=4.0)
        raise AssertionError("Expected ValueError for single observation")
    except ValueError as exc:
        assert "2 observations" in str(exc)


# ---------------------------------------------------------------------------
# CLI commands: plot, constants, dashboard error path
# ---------------------------------------------------------------------------
def test_cli_plot_valid():
    result = _runner.invoke(app, ["plot", "1,5,3,8", "--title", "Data"])
    assert result.exit_code == 0
    assert "Data" in result.stdout


def test_cli_plot_invalid():
    result = _runner.invoke(app, ["plot", "1,abc,3"])
    assert result.exit_code == 0  # CLI catches the error, prints message
    assert "Error" in result.stdout


def test_cli_constants():
    result = _runner.invoke(app, ["constants"])
    assert result.exit_code == 0
    assert "Physical Constants" in result.stdout


def test_cli_dashboard_missing_file(monkeypatch):
    """Dashboard command reports an error when the app file is absent."""
    monkeypatch.setattr(cli_mod.Path, "exists", lambda self: False)
    result = _runner.invoke(app, ["dashboard"])
    assert result.exit_code == 0
    assert "not found" in result.stdout.lower()


def test_cli_calc_unknown_formula():
    result = _runner.invoke(app, ["calc", "unknown"])
    assert result.exit_code == 0
    assert "Unknown formula" in result.stdout


def test_cli_no_command_shows_help():
    """Invoking the CLI with no subcommand prints help."""
    result = _runner.invoke(app, [])
    assert "help" in result.stdout.lower() or "Usage" in result.stdout


def test_cli_hypothesis_show_prompt():
    result = _runner.invoke(app, ["hypothesis", "Test question", "--show-prompt"])
    assert result.exit_code == 0
    assert "Prompt Template" in result.stdout


def test_cli_calc_input_error(monkeypatch):
    """calc command with non-numeric input prints error."""
    monkeypatch.setattr("typer.prompt", lambda _: "not_a_number")
    result = _runner.invoke(app, ["calc", "ke"])
    assert result.exit_code == 0
    assert "Error" in result.stdout


def test_cli_calc_generic_exception(monkeypatch):
    """calc command with an unexpected exception prints error."""

    def bad_prompt(_):
        raise RuntimeError("surprise")

    monkeypatch.setattr("typer.prompt", bad_prompt)
    result = _runner.invoke(app, ["calc", "ke"])
    assert result.exit_code == 0
    assert "Error" in result.stdout


def test_cli_dashboard_launch(monkeypatch):
    """Dashboard command: mock subprocess so streamlit is not actually launched."""

    def fake_run(cmd, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.Path, "exists", lambda self: True)
    result = _runner.invoke(app, ["dashboard"])
    assert "Dashboard stopped" in result.stdout


def test_cli_dashboard_streamlit_missing(monkeypatch):
    """Dashboard command: FileNotFoundError when streamlit is not installed."""

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("streamlit not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.Path, "exists", lambda self: True)
    result = _runner.invoke(app, ["dashboard"])
    assert "Streamlit not found" in result.stdout


def test_cli_modules_end_line():
    """modules command: verify the final lines are printed."""
    result = _runner.invoke(app, ["modules"])
    assert result.exit_code == 0
    assert "See examples/" in result.stdout


def test_import_main():
    """Ensure cds.__main__ can be imported without side effects."""
    import cds.__main__  # noqa: F401


# ---------------------------------------------------------------------------
# Monte Carlo: _pi_worker (parallel worker) and edge cases
# ---------------------------------------------------------------------------
def test_pi_worker_counts_inside():
    """_pi_worker counts points inside the unit quarter-circle."""
    inside = _pi_worker((1000, 42))
    # With 1000 samples roughly 785 +/- 30 should be inside (pi/4 * 1000)
    assert 700 < inside < 850


def test_pi_worker_no_seed():
    """_pi_worker runs without a seed (random seed)."""
    inside = _pi_worker((500, None))
    assert 0 <= inside <= 500


def test_estimate_pi_zero_samples():
    """estimate_pi with 0 samples returns a zero result."""
    result = estimate_pi(n_samples=0)
    assert result.estimate == 0.0
    assert result.samples == 0
    assert result.std_error == 0.0


def test_buffon_needle_no_crossings():
    """buffon_needle with a very short needle can produce 0 crossings."""
    # Short needle, large spacing -> few/no crossings. Guard against div-by-zero.
    result = buffon_needle(needle_length=0.01, line_spacing=2.0, n_throws=5, seed=1)
    assert result.samples == 5
