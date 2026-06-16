"""Signal processing demo — FFT and filtering."""

import math

from cds.signals import convolve, dft, fft_radix2, low_pass_filter

# --- DFT of a simple signal ---
print("=== DFT of Cosine Signal ===")
n = 16
signal = [complex(math.cos(2 * math.pi * 2 * k / n)) for k in range(n)]
spectrum = dft(signal)
print("Frequency magnitudes:")
for k, x in enumerate(spectrum):
    mag = abs(x)
    if mag > 0.5:
        print(f"  Bin {k}: {mag:.2f}")

# --- FFT comparison ---
print("\n=== FFT (radix-2) ===")
fft_result = fft_radix2(signal)
print(f"FFT matches DFT: {all(abs(a - b) < 1e-9 for a, b in zip(spectrum, fft_result))}")

# --- Convolution ---
print("\n=== Convolution ===")
a = [1.0, 2.0, 3.0, 4.0]
b = [0.5, 0.5]  # averaging kernel
result = convolve(a, b)
print(f"{a} * {b} = {[round(x, 1) for x in result]}")

# --- Low-pass filter ---
print("\n=== Low-Pass Filter ===")
noisy = [1 + 0.3 * math.sin(2 * math.pi * 7 * k / n) for k in range(n)]
filtered = low_pass_filter([complex(x) for x in noisy], cutoff=3)
print("Original (first 8):", [f"{x:.2f}" for x in noisy[:8]])
print("Filtered (first 8):", [f"{x.real:.2f}" for x in filtered[:8]])
