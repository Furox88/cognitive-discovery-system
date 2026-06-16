"""Tests for signal processing module."""
import math

import pytest

from cds.signals.processing import (
    convolve,
    dft,
    fft_radix2,
    idft,
    low_pass_filter,
    power_spectrum,
)

# --- DFT / IDFT ---

def test_dft_dc_signal():
    # constant signal -> all energy at DC (bin 0)
    signal = [1 + 0j] * 4
    result = dft(signal)
    assert abs(result[0] - 4) < 1e-9
    for k in range(1, 4):
        assert abs(result[k]) < 1e-9


def test_dft_single_frequency():
    # pure cosine at frequency 1
    n = 8
    signal = [complex(math.cos(2 * math.pi * k / n)) for k in range(n)]
    result = dft(signal)
    # energy at bin 1 and bin 7 (mirror)
    assert abs(result[1]) > 3.0
    assert abs(result[7]) > 3.0


def test_idft_roundtrip():
    signal = [1 + 0j, 2 + 1j, 0 - 1j, 3 + 0j]
    spectrum = dft(signal)
    recovered = idft(spectrum)
    for s, r in zip(signal, recovered):
        assert abs(s - r) < 1e-9


def test_dft_empty():
    assert dft([]) == []


def test_idft_single():
    signal = [5 + 0j]
    assert abs(idft(dft(signal))[0] - 5) < 1e-9


# --- FFT radix-2 ---

def test_fft_matches_dft():
    signal = [complex(i) for i in range(8)]
    dft_result = dft(signal)
    fft_result = fft_radix2(signal)
    for d, f in zip(dft_result, fft_result):
        assert abs(d - f) < 1e-9


def test_fft_power_of_2_required():
    with pytest.raises(ValueError):
        fft_radix2([1 + 0j, 2 + 0j, 3 + 0j])


def test_fft_empty():
    assert fft_radix2([]) == []


def test_fft_single():
    assert abs(fft_radix2([42 + 0j])[0] - 42) < 1e-9


def test_fft_two_elements():
    result = fft_radix2([1 + 0j, -1 + 0j])
    assert abs(result[0]) < 1e-9
    assert abs(result[1] - 2) < 1e-9


# --- Convolution ---

def test_convolve_impulse():
    # convolve with delta function
    a = [1.0, 2.0, 3.0]
    b = [1.0]
    assert convolve(a, b) == [1.0, 2.0, 3.0]


def test_convolve_simple():
    a = [1.0, 1.0]
    b = [1.0, 1.0]
    result = convolve(a, b)
    assert len(result) == 3
    assert abs(result[0] - 1) < 1e-9
    assert abs(result[1] - 2) < 1e-9
    assert abs(result[2] - 1) < 1e-9


def test_convolve_length():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0]
    result = convolve(a, b)
    assert len(result) == 4  # 3 + 2 - 1


# --- Power spectrum ---

def test_power_spectrum_dc():
    signal = [2 + 0j] * 4
    ps = power_spectrum(signal)
    # all energy at DC
    assert ps[0] > 0
    for k in range(1, 4):
        assert abs(ps[k]) < 1e-9


def test_power_spectrum_length():
    signal = [complex(i) for i in range(8)]
    assert len(power_spectrum(signal)) == 8


# --- Low-pass filter ---

def test_low_pass_preserves_dc():
    signal = [3 + 0j] * 8
    filtered = low_pass_filter(signal, cutoff=2)
    for s in filtered:
        assert abs(s.real - 3) < 1e-9


def test_low_pass_removes_high_freq():
    n = 16
    # signal = DC(1) + high-freq noise
    signal = [
        1 + 0j + 0.5 * complex(math.cos(2 * math.pi * 7 * k / n))
        for k in range(n)
    ]
    filtered = low_pass_filter(signal, cutoff=3)
    # after filtering, should be close to DC=1
    for s in filtered:
        assert abs(s.real - 1) < 0.15
