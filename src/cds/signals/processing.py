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
