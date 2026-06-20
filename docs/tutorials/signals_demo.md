# Signal Processing Tutorial

`cds.signals` provides a from-scratch Fourier toolkit (DFT/FFT, forward and
inverse), convolution, power-spectrum estimation, and a frequency-domain
low-pass filter — all pure Python with no NumPy dependency.

## 1. DFT and inverse

Compute the discrete Fourier transform of a complex signal; `idft` inverts it.

```python
import math
from cds.signals import dft, idft

n = 16
signal = [complex(math.cos(2 * math.pi * 2 * k / n)) for k in range(n)]  # 2 Hz cosine
spectrum = dft(signal)

print("Frequency magnitudes:")
for k, x in enumerate(spectrum):
    if abs(x) > 0.5:
        print(f"  Bin {k}: {abs(x):.2f}")
#   Bin 2: 8.00
#   Bin 14: 8.00   (negative-frequency mirror)
```

## 2. Radix-2 FFT and 2-D FFT

`fft_radix2` is the fast path for power-of-two lengths; `fft2` / `ifft2` handle
2-D signals (matrices).

```python
from cds.signals import fft_radix2

fft_result = fft_radix2(signal)
print(all(abs(a - b) < 1e-9 for a, b in zip(spectrum, fft_result)))  # True
```

## 3. Convolution

Linear convolution of two real sequences.

```python
from cds.signals import convolve

a = [1.0, 2.0, 3.0, 4.0]
b = [0.5, 0.5]  # 2-point averaging kernel
print(convolve(a, b))  # [0.5, 1.5, 2.5, 3.5, 2.0]
```

## 4. Power spectrum and low-pass filtering

`power_spectrum` returns the magnitude at each frequency bin. `low_pass_filter`
zeroes out bins above a cutoff frequency to smooth a noisy signal.

```python
import math
from cds.signals import low_pass_filter

noisy = [1 + 0.3 * math.sin(2 * math.pi * 7 * k / n) for k in range(n)]  # high-freq noise on DC
filtered = low_pass_filter([complex(x) for x in noisy], cutoff=3)
print("Original (first 4):", [f"{x:.2f}" for x in noisy[:4]])
print("Filtered (first 4):", [f"{x.real:.2f}" for x in filtered[:4]])
# Original (first 4): ['1.00', '1.11', '0.79', '1.28']
# Filtered (first 4):  ['1.00', '1.00', '1.00', '1.00']  — noise removed
```

---

Run the full demo with `python examples/signals_demo.py`.
