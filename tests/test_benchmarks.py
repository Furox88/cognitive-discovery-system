"""Smoke test for the benchmark suite.

Verifies structure (return types, keys) without timing the actual functions,
so CI stays fast. Timing correctness is validated manually via
``python benchmarks/run_benchmarks.py``.
"""

from collections import OrderedDict
from typing import Any, cast

import pytest


class TestBenchmarkStructure:
    """Each bench_* function must return OrderedDict[str, str]."""

    def _import_and_call(self, func_name: str, module_path: str) -> OrderedDict[str, str]:
        import importlib

        mod = importlib.import_module(module_path)
        return cast(OrderedDict[str, str], getattr(mod, func_name)())

    @pytest.mark.parametrize(
        "func, module",
        [
            ("bench_linalg", "benchmarks.run_benchmarks"),
            ("bench_linalg_intelligence", "benchmarks.run_benchmarks"),
            ("bench_montecarlo", "benchmarks.run_benchmarks"),
            ("bench_quantum", "benchmarks.run_benchmarks"),
            ("bench_signals", "benchmarks.run_benchmarks"),
            ("bench_numerical_integration", "benchmarks.run_benchmarks"),
        ],
    )
    def test_returns_ordered_dict_of_strings(self, func: Any, module: Any) -> None:
        result = self._import_and_call(func, module)
        assert isinstance(result, OrderedDict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    @pytest.mark.parametrize(
        "func, module, required_key",
        [
            ("bench_linalg", "benchmarks.run_benchmarks", "CDS Matrix Mul (100x100)"),
            ("bench_montecarlo", "benchmarks.run_benchmarks", "Parallel Pi (100k samples)"),
            ("bench_quantum", "benchmarks.run_benchmarks", "Intelligence Speedup"),
            ("bench_signals", "benchmarks.run_benchmarks", "Algorithmic speedup"),
            ("bench_numerical_integration", "benchmarks.run_benchmarks", "Romberg (auto tol)"),
        ],
    )
    def test_contains_expected_keys(self, func: Any, module: Any, required_key: Any) -> None:
        result = self._import_and_call(func, module)
        assert required_key in result

    def test_run_all_generates_report(self, tmp_path: Any, monkeypatch: Any) -> None:
        """run_all writes a valid markdown report with all six sections."""
        import importlib

        mod = importlib.import_module("benchmarks.run_benchmarks")
        monkeypatch.setattr(mod, "_bench", lambda _func, number=1, repeat=1: 0.0)
        monkeypatch.chdir(tmp_path)
        mod.run_all()
        report = (tmp_path / "docs" / "benchmarks.md").read_text(encoding="utf-8")
        assert "# CDS Performance & Intelligence Report" in report
        assert "Signal Processing" in report
        assert "Numerical Integration" in report
        assert "Algorithmic Intelligence" in report
        assert "Quantum Intelligence" in report
        assert "Convergence" in report
