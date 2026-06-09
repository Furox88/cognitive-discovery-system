"""Discrete Fourier Transform and signal processing utilities."""
from __future__ import annotations

import cmath
import math


def dft(signal: list[complex]) -> list[complex]:
    """Discrete Fourier Transform (direct computation).

    Args:
        signal: input signal of length N

    Returns:
        list of N complex frequency components
    """
    n = len(signal)
    result = []
    for k in range(n):
        s = 0 + 0j
        for t in range(n):
            angle = -2 * math.pi * k * t / n
            s += signal[t] * cmath.exp(1j * angle)
        result.append(s)
    return result


def idft(spectrum: list[complex]) -> list[complex]:
    """Inverse Discrete Fourier Transform.

    Args:
        spectrum: frequency-domain signal of length N

    Returns:
        list of N complex time-domain samples
    """
    n = len(spectrum)
    result = []
    for t in range(n):
        s = 0 + 0j
        for k in range(n):
            angle = 2 * math.pi * k * t / n
            s += spectrum[k] * cmath.exp(1j * angle)
        result.append(s / n)
    return result


def fft_radix2(signal: list[complex]) -> list[complex]:
    """Cooley-Tukey radix-2 FFT. Input length must be a power of 2.

    Args:
        signal: input signal (length must be power of 2)

    Returns:
        list of complex frequency components

    Raises:
        ValueError: if length is not a power of 2
    """
    n = len(signal)
    if n == 0:
        return []
    if n & (n - 1) != 0:
        raise ValueError("length must be a power of 2")
    if n == 1:
        return list(signal)

    even = fft_radix2(signal[0::2])
    odd = fft_radix2(signal[1::2])

    result = [0 + 0j] * n
    for k in range(n // 2):
        w = cmath.exp(-2j * math.pi * k / n)
        result[k] = even[k] + w * odd[k]
        result[k + n // 2] = even[k] - w * odd[k]
    return result


def fft2(matrix: list[list[complex]]) -> list[list[complex]]:
    """2-D Discrete Fourier Transform via the row-column algorithm.

    Applies a 1-D FFT to each row, then a 1-D FFT to each column of the
    result. Both dimensions must be powers of two so the radix-2 FFT applies.

    Reference:
        Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine
        calculation of complex Fourier series." Mathematics of Computation,
        19(90), 297-301. The separability of the multidimensional DFT into
        successive 1-D transforms is the standard row-column method (see also
        Gonzalez & Woods, Digital Image Processing, §4).

    Args:
        matrix: 2-D input (rows x cols), each dimension a power of two

    Returns:
        2-D list of complex frequency components

    Raises:
        ValueError: if the matrix is empty or rows have unequal length
    """
    rows = len(matrix)
    if rows == 0:
        raise ValueError("matrix must be non-empty")
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("all rows must have equal length")

    row_fft = [fft_radix2(list(row)) for row in matrix]

    result = [[0 + 0j] * cols for _ in range(rows)]
    for j in range(cols):
        column = [row_fft[i][j] for i in range(rows)]
        col_fft = fft_radix2(column)
        for i in range(rows):
            result[i][j] = col_fft[i]
    return result


def ifft2(spectrum: list[list[complex]]) -> list[list[complex]]:
    """Inverse 2-D DFT via the row-column algorithm.

    Inverts :func:`fft2` by applying the inverse 1-D DFT along columns then
    rows. Uses the separability of the multidimensional inverse transform.

    Reference:
        Cooley, J. W., & Tukey, J. W. (1965). Mathematics of Computation,
        19(90), 297-301; row-column decomposition per Gonzalez & Woods,
        Digital Image Processing, §4.

    Args:
        spectrum: 2-D frequency-domain input (rows x cols)

    Returns:
        2-D list of complex time/space-domain samples

    Raises:
        ValueError: if the matrix is empty or rows have unequal length
    """
    rows = len(spectrum)
    if rows == 0:
        raise ValueError("matrix must be non-empty")
    cols = len(spectrum[0])
    if any(len(row) != cols for row in spectrum):
        raise ValueError("all rows must have equal length")

    col_inv = [[0 + 0j] * cols for _ in range(rows)]
    for j in range(cols):
        column = [spectrum[i][j] for i in range(rows)]
        inv = idft(column)
        for i in range(rows):
            col_inv[i][j] = inv[i]

    result = [idft(col_inv[i]) for i in range(rows)]
    return result


def convolve(a: list[float], b: list[float]) -> list[float]:
    """Linear convolution of two real sequences.

    Args:
        a: first signal
        b: second signal (kernel)

    Returns:
        convolved signal of length len(a) + len(b) - 1
    """
    na, nb = len(a), len(b)
    out_len = na + nb - 1
    result = [0.0] * out_len
    for i in range(na):
        for j in range(nb):
            result[i + j] += a[i] * b[j]
    return result


def power_spectrum(signal: list[complex]) -> list[float]:
    """Compute the power spectrum |X[k]|^2 / N.

    Args:
        signal: input signal

    Returns:
        list of power values for each frequency bin
    """
    spectrum = dft(signal)
    n = len(spectrum)
    return [abs(x) ** 2 / n for x in spectrum]


def low_pass_filter(
    signal: list[complex], cutoff: int,
) -> list[complex]:
    """Simple frequency-domain low-pass filter.

    Zeroes out all frequency components above the cutoff index.

    Args:
        signal: input signal
        cutoff: keep frequencies 0..cutoff-1 (and mirror)

    Returns:
        filtered signal (real parts)
    """
    spectrum = dft(signal)
    n = len(spectrum)
    for k in range(n):
        if cutoff <= k <= n - cutoff:
            spectrum[k] = 0 + 0j
    return idft(spectrum)
