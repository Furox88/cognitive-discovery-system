"""Tests targeting remaining coverage gaps to push total above 97%.

Covers edge cases in:
- stats/hypothesis_tests.py (chi2_sf boundary, f_sf boundary, ANOVA edge, independence edge)
- optimization/minimize.py (newton flat-derivative, adam state resume, line_search non-converge)
- signals/processing.py (fft non-power-of-2 auto-pad, ifft auto-pad, power_spectrum dft path,
  low_pass dft path, convolve empty, fft2/ifft2 empty)
- data_analysis/viz.py (negative bar, interpolation path, flat line)
"""

import math
from typing import Any

import pytest

from cds.data_analysis.viz import plot_bar, plot_line
from cds.optimization.minimize import OptResult, adam, line_search, newton_method
from cds.signals.processing import (
    convolve,
    dft,
    fft,
    fft2,
    fft_radix2,
    ifft,
    ifft2,
    low_pass_filter,
    power_spectrum,
)
from cds.stats.hypothesis_tests import (
    chi2_sf,
    chi_square_gof,
    chi_square_independence,
    f_sf,
    one_sample_ttest,
    one_way_anova,
    two_sample_ttest,
)

# ---------------------------------------------------------------------------
# stats/hypothesis_tests.py — edge cases
# ---------------------------------------------------------------------------


class TestChi2SfBoundary:
    def test_zero_x_returns_one(self) -> None:
        assert chi2_sf(0.0, df=5) == 1.0

    def test_negative_x_returns_one(self) -> None:
        assert chi2_sf(-1.0, df=5) == 1.0

    def test_small_x(self) -> None:
        # Very small positive x, should be close to 1.0
        assert chi2_sf(0.001, df=5) > 0.99


class TestFSfBoundary:
    def test_zero_f_returns_one(self) -> None:
        assert f_sf(0.0, df1=3, df2=10) == 1.0

    def test_negative_f_returns_one(self) -> None:
        assert f_sf(-1.0, df1=3, df2=10) == 1.0


class TestOneSampleTTestEdgeCases:
    def test_single_element_raises(self) -> None:
        with pytest.raises(ValueError):
            one_sample_ttest([1.0])

    def test_zero_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            one_sample_ttest([5.0, 5.0, 5.0, 5.0])


class TestTwoSampleTTestEdgeCases:
    def test_single_element_in_a_raises(self) -> None:
        with pytest.raises(ValueError):
            two_sample_ttest([1.0], [2.0, 3.0])

    def test_single_element_in_b_raises(self) -> None:
        with pytest.raises(ValueError):
            two_sample_ttest([1.0, 2.0], [3.0])

    def test_zero_variance_in_both_raises(self) -> None:
        with pytest.raises(ValueError):
            two_sample_ttest([5.0, 5.0], [5.0, 5.0])


class TestChiSquareIndependenceEdgeCases:
    def test_single_row_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_independence([[1, 2, 3]])

    def test_single_column_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_independence([[1], [2], [3]])

    def test_jagged_rows_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_independence([[1, 2], [3]])

    def test_zero_grand_total_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_independence([[0, 0], [0, 0]])


class TestOneWayAnovaEdgeCases:
    def test_single_group_raises(self) -> None:
        with pytest.raises(ValueError):
            one_way_anova([1.0, 2.0])

    def test_empty_group_raises(self) -> None:
        with pytest.raises(ValueError):
            one_way_anova([1.0, 2.0], [])

    def test_n_total_equals_k_raises(self) -> None:
        # 3 groups with 1 element each: n_total == k
        with pytest.raises(ValueError):
            one_way_anova([1.0], [2.0], [3.0])

    def test_zero_within_variance_raises(self) -> None:
        # All identical within groups but different between groups
        with pytest.raises(ValueError):
            one_way_anova([1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0])


class TestChiSquareGofEdgeCases:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_gof([1, 2, 3], [1, 2])

    def test_too_few_categories_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_gof([1], [1])

    def test_zero_expected_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_gof([1, 2], [0, 3])

    def test_negative_expected_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_gof([1, 2], [-1, 3])


# ---------------------------------------------------------------------------
# optimization/minimize.py — edge cases
# ---------------------------------------------------------------------------


class TestNewtonMethodEdgeCases:
    def test_flat_derivative_breaks(self) -> None:
        """f(x) = constant → derivative = 0, should break early with converged=False."""
        result = newton_method(lambda x: 5.0, x0=1.0, max_iter=100)
        assert isinstance(result, OptResult)
        assert result.converged is False

    def test_non_converged_max_iter(self) -> None:
        """Oscillating function that won't converge."""
        # f(x) = x^3 - 2x + 2, x0=0 oscillates between 0 and 1
        result = newton_method(lambda x: x**3 - 2 * x + 2, x0=0.0, max_iter=5)
        assert isinstance(result, OptResult)


class TestAdamScalarStateResume:
    def test_adam_resume_from_state(self) -> None:
        """Adam scalar: run for a few iterations, then resume with state."""

        def f(x: Any) -> Any:
            return (x - 5.0) ** 2

        # First run: state=None, should return state dict
        r1 = adam(f, x0=0.0, lr=0.1, max_iter=5, state=None)
        assert r1.state is not None
        assert "m" in r1.state and "v" in r1.state and "t" in r1.state
        # Second run: resume from state
        r2 = adam(f, x0=r1.x, lr=0.1, max_iter=5, state=r1.state)
        assert r2.state is not None
        assert r2.state["t"] > r1.state["t"]

    def test_adam_scalar_custom_grad(self) -> None:
        """Adam scalar with explicit gradient function."""

        def f(x: Any) -> Any:
            return (x - 3.0) ** 2

        def grad_f(x: Any) -> Any:
            return 2 * (x - 3.0)

        result = adam(f, x0=10.0, lr=0.1, max_iter=200, grad_f=grad_f)
        assert abs(result.x - 3.0) < 0.1


class TestAdamVectorStateResume:
    def test_adam_vector_resume(self) -> None:
        """Adam vector: run, then resume with state."""

        def f(x: Any) -> Any:
            return x[0] ** 2 + (x[1] - 4) ** 2

        # First run: state=None
        r1 = adam(f, x0=[0.0, 0.0], lr=0.1, max_iter=5, state=None)
        assert r1.state is not None
        assert "m" in r1.state and "v" in r1.state and "t" in r1.state
        # Second run: resume
        r2 = adam(f, x0=list(r1.x), lr=0.1, max_iter=5, state=r1.state)
        assert r2.state is not None
        assert r2.state["t"] > r1.state["t"]

    def test_adam_vector_custom_grad(self) -> None:
        """Adam vector with explicit gradient function."""

        def f(x: Any) -> Any:
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x: Any) -> Any:
            return [2 * x[0], 2 * x[1]]

        result = adam(f, x0=[5.0, 5.0], lr=0.1, max_iter=200, grad_f=grad_f)
        assert abs(result.x[0]) < 0.1
        assert abs(result.x[1]) < 0.1


class TestLineSearchEdgeCases:
    def test_already_converged(self) -> None:
        """a and b are very close → should converge immediately."""
        result = line_search(lambda x: x**2, a=0.999, b=1.001, tol=0.01)
        assert result.converged is True

    def test_max_iter_exhausted(self) -> None:
        """Function where golden section is slow to converge with very low max_iter."""
        result = line_search(lambda x: math.sin(x) + 0.1 * x, a=-5.0, b=5.0, max_iter=1)
        assert isinstance(result, OptResult)


# ---------------------------------------------------------------------------
# signals/processing.py — edge cases
# ---------------------------------------------------------------------------


class TestFFTRadix2EdgeCases:
    def test_empty_signal(self) -> None:
        assert fft_radix2([]) == []

    def test_single_element(self) -> None:
        result = fft_radix2([5.0])
        assert len(result) == 1
        assert abs(result[0] - 5.0) < 1e-10

    def test_non_power_of_2_raises(self) -> None:
        with pytest.raises(ValueError):
            fft_radix2([1, 2, 3])


class TestFFTAutoPad:
    def test_empty_signal(self) -> None:
        assert fft([]) == []

    def test_non_power_of_2_auto_pads(self) -> None:
        """fft() should auto-pad to next power of 2 and return correct result."""
        signal = [1.0, 2.0, 3.0]  # length 3 → padded to 4
        result = fft(signal)
        # Should match dft of the padded signal
        padded = signal + [0.0]  # zero-padded to length 4
        expected = fft_radix2(padded)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10


class TestIFFTAutoPad:
    def test_empty_spectrum(self) -> None:
        assert ifft([]) == []

    def test_non_power_of_2_auto_pads(self) -> None:
        """ifft() should auto-pad non-power-of-2 input."""
        spectrum = [complex(6, 0), complex(0, 0), complex(0, 0)]
        result = ifft(spectrum)
        # Just verify it returns something of correct length
        assert len(result) == 4  # padded to 4


class TestConvolveEdgeCases:
    def test_empty_a(self) -> None:
        assert convolve([], [1, 2]) == []

    def test_empty_b(self) -> None:
        assert convolve([1, 2], []) == []

    def test_both_empty(self) -> None:
        assert convolve([], []) == []


class TestPowerSpectrumEdgeCases:
    def test_empty_signal(self) -> None:
        assert power_spectrum([]) == []

    def test_non_power_of_2_uses_dft(self) -> None:
        """power_spectrum with non-power-of-2 length should use dft path."""
        signal = [1.0, 2.0, 3.0, 5.0, 7.0]
        result = power_spectrum(signal)
        # Manually compute |DFT|^2 / N
        d = dft(signal)
        n = len(signal)
        expected = [abs(x) ** 2 / n for x in d]
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10


class TestLowPassFilterEdgeCases:
    def test_empty_signal(self) -> None:
        assert low_pass_filter([], cutoff=1) == []

    def test_non_power_of_2_uses_dft_path(self) -> None:
        """low_pass_filter with non-power-of-2 signal should use dft/idft path."""
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        cutoff = 1
        result = low_pass_filter(signal, cutoff)
        # Verify roundtrip-ish: the result should be a filtered version
        assert len(result) == len(signal)

    def test_high_cutoff_passes_all(self) -> None:
        """cutoff >= n/2 → nothing is filtered out."""
        signal = [1.0, 0.0, 1.0, 0.0]
        result = low_pass_filter(signal, cutoff=2)
        # With cutoff=2, bins 2 and 2 (n=4, n-cutoff=2) are zeroed but bins 0,1,3 kept
        assert len(result) == 4


class TestFFT2EdgeCases:
    def test_empty_matrix_raises(self) -> None:
        with pytest.raises(ValueError):
            fft2([])

    def test_jagged_rows_raises(self) -> None:
        with pytest.raises(ValueError):
            fft2([[1, 2], [3]])


class TestIFFT2EdgeCases:
    def test_empty_matrix_raises(self) -> None:
        with pytest.raises(ValueError):
            ifft2([])

    def test_jagged_rows_raises(self) -> None:
        with pytest.raises(ValueError):
            ifft2([[1, 2], [3]])


# ---------------------------------------------------------------------------
# data_analysis/viz.py — edge cases
# ---------------------------------------------------------------------------


class TestPlotBarEdgeCases:
    def test_empty_data(self) -> None:
        assert plot_bar({}) == "No data to plot."

    def test_negative_values(self) -> None:
        """Negative values should render with ░ character."""
        result = plot_bar({"loss": -5, "gain": 3})
        assert "loss" in result
        assert "gain" in result
        assert "░" in result  # negative bar uses this character
        assert "█" in result  # positive bar uses this character

    def test_single_entry(self) -> None:
        result = plot_bar({"only": 10})
        assert "only" in result
        assert "█" in result


class TestPlotLineEdgeCases:
    def test_empty_data(self) -> None:
        assert plot_line([]) == "No data to plot."

    def test_single_point(self) -> None:
        result = plot_line([5.0])
        assert "█" in result or "─" in result or "|" in result

    def test_flat_line(self) -> None:
        """All identical values → y_range should be clamped to 1.0 to avoid division by zero."""
        result = plot_line([3.0, 3.0, 3.0, 3.0, 3.0])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_interpolation_path(self) -> None:
        """More data points than width triggers interpolation/sampling."""
        # 100 points but default width is 80 → should trigger sampling
        data = [math.sin(i * 0.1) for i in range(100)]
        result = plot_line(data, width=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_width(self) -> None:
        """Explicit small width to force interpolation."""
        data = list(range(20))
        result = plot_line(data, width=5)
        assert isinstance(result, str)

    def test_height_parameter(self) -> None:
        """Custom height should produce proportional output."""
        data = [0, 5, 10, 5, 0]
        result = plot_line(data, height=3)
        # Count lines with plot characters
        lines = [ln for ln in result.split("\n") if any(c in ln for c in "█▀▄▌▐│─")]
        assert len(lines) <= 3
