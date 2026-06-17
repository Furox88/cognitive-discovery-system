from typing import Any, cast

import pytest

from cds.signals import fft2, ifft2


def _max_err(a: Any, b: Any) -> float:
    return cast(float, max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0]))))


class TestFFT2:
    def test_constant_matrix_dc_component(self) -> None:
        m: list[list[float | complex]] = [[1.0, 1.0], [1.0, 1.0]]
        out = fft2(m)
        # DC term equals sum of all entries
        assert abs(out[0][0] - 4.0) < 1e-12
        # all other terms zero
        assert abs(out[0][1]) < 1e-12
        assert abs(out[1][0]) < 1e-12
        assert abs(out[1][1]) < 1e-12

    def test_roundtrip_identity(self) -> None:
        m: list[list[float | complex]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
        recovered = ifft2(fft2(m))
        for i in range(4):
            for j in range(4):
                assert abs(recovered[i][j].real - m[i][j]) < 1e-9
                assert abs(recovered[i][j].imag) < 1e-9

    def test_separability_matches_manual(self) -> None:
        m: list[list[float | complex]] = [[1.0, 0.0], [0.0, 0.0]]
        out = fft2(m)
        # impulse at (0,0): all frequency components equal 1
        for i in range(2):
            for j in range(2):
                assert abs(out[i][j] - 1.0) < 1e-12

    def test_complex_input_roundtrip(self) -> None:
        m = [[1 + 1j, 2 - 1j], [0 + 0j, 3 + 2j]]
        recovered = ifft2(fft2(m))
        assert _max_err(recovered, m) < 1e-9

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            fft2([])
        with pytest.raises(ValueError):
            ifft2([])

    def test_ragged_raises(self) -> None:
        with pytest.raises(ValueError):
            fft2([[1.0, 2.0], [3.0]])

    def test_parseval_energy(self) -> None:
        m: list[list[float | complex]] = [[1.0, 2.0], [3.0, 4.0]]
        out = fft2(m)
        n = 4
        space_energy = sum(m[i][j] ** 2 for i in range(2) for j in range(2))
        freq_energy = sum(abs(out[i][j]) ** 2 for i in range(2) for j in range(2)) / n
        assert abs(space_energy - freq_energy) < 1e-9
