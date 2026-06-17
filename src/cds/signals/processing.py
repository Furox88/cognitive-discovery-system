"""Discrete Fourier Transform and signal processing utilities."""

from __future__ import annotations

import cmath
import math


def dft(signal: list[float | complex]) -> list[complex]:
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


def idft(spectrum: list[float | complex]) -> list[complex]:
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


def fft_radix2(signal: list[float | complex]) -> list[complex]:
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
        raise ValueError(
            f"signal length must be a power of 2 for FFT (got {n}); pad with zeros or use dft() for arbitrary lengths"
        )
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


def fft(signal: list[float | complex]) -> list[complex]:
    """Compute 1D FFT of any length using radix-2 and zero-padding.

    Automatically pads input to next power of 2 for O(N log N) speed.
    """
    n = len(signal)
    if n == 0:
        return []

    # Next power of 2
    padded_len = 1 << (n - 1).bit_length()
    padded_signal = list(signal) + [0j] * (padded_len - n)

    # Compute radix-2
    full_spectrum = fft_radix2(padded_signal)

    return full_spectrum


def ifft(spectrum: list[complex]) -> list[complex]:
    """Compute 1D Inverse FFT of any length using radix-2 and padding.

    Args:
        spectrum: frequency-domain signal

    Returns:
        list of complex time-domain samples
    """
    n = len(spectrum)
    if n == 0:
        return []

    # If not a power of 2, pad it to next power of 2 (matches fft behavior)
    if n & (n - 1) != 0:
        padded_len = 1 << (n - 1).bit_length()
        padded_spectrum = list(spectrum) + [0j] * (padded_len - n)
    else:
        padded_spectrum = spectrum
        padded_len = n

    # The property: IFFT(x) = conj(FFT(conj(x))) / N
    conj_spectrum = [x.conjugate() for x in padded_spectrum]
    forward_fft = fft_radix2(conj_spectrum)
    return [x.conjugate() / padded_len for x in forward_fft]


def convolve(a: list[float], b: list[float]) -> list[float]:
    """Linear convolution using the FFT Theorem (O(N log N))."""
    if not a or not b:
        return []

    na, nb = len(a), len(b)
    n_out = na + nb - 1

    # Next power of 2 for FFT speed
    n_fft = 1 << (n_out - 1).bit_length()

    # Transform to frequency domain
    fa = fft(list(a) + [0j] * (n_fft - na))
    fb = fft(list(b) + [0j] * (n_fft - nb))

    # Multiplication in frequency domain
    fc = [xa * xb for xa, xb in zip(fa, fb)]

    # Inverse transform
    full_conv = ifft(fc)

    # Return truncated to correct length
    return [x.real for x in full_conv[:n_out]]


def power_spectrum(signal: list[float | complex]) -> list[float]:
    """Compute the power spectrum |X[k]|^2 / N."""
    n = len(signal)
    if n == 0:
        return []

    # Use FFT if possible (O(N log N))
    if (n & (n - 1) == 0) and n > 0:
        spectrum = fft_radix2(signal)
    else:
        spectrum = dft(signal)

    return [abs(x) ** 2 / n for x in spectrum]


def low_pass_filter(signal: list[float | complex], cutoff: int) -> list[complex]:
    """Simple frequency-domain low-pass filter."""
    n = len(signal)
    if n == 0:
        return []

    # Choose best transform
    if (n & (n - 1) == 0) and n > 0:
        spectrum = fft_radix2(signal)
        inv_func = ifft
    else:
        spectrum = dft(signal)
        inv_func = idft

    for k in range(n):
        if cutoff <= k <= n - cutoff:
            spectrum[k] = 0 + 0j

    return inv_func(spectrum)


def fft2(matrix: list[list[float | complex]]) -> list[list[complex]]:
    """2-D Discrete Fourier Transform (O(N log N))."""
    rows = len(matrix)
    if rows == 0:
        raise ValueError("matrix must be non-empty")
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("all rows must have the same length (ragged matrix detected)")

    # Row-wise FFT
    row_fft = [fft(list(row)) for row in matrix]

    # Column-wise FFT
    transposed = list(zip(*row_fft))
    col_fft = [fft(list(col)) for col in transposed]

    return [list(row) for row in zip(*col_fft)]


def ifft2(spectrum: list[list[float | complex]]) -> list[list[complex]]:
    """Inverse 2-D DFT (O(N log N))."""
    rows = len(spectrum)
    if rows == 0:
        raise ValueError("matrix must be non-empty")
    cols = len(spectrum[0])
    if any(len(row) != cols for row in spectrum):
        raise ValueError("all rows must have the same length (ragged matrix detected)")

    # Row-wise IFFT
    row_inv = [ifft(list(row)) for row in spectrum]

    # Column-wise IFFT
    transposed = list(zip(*row_inv))
    col_inv = [ifft(list(col)) for col in transposed]

    return [list(row) for row in zip(*col_inv)]
