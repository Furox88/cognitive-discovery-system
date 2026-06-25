"""Signal processing demo — FFT and filtering."""

import math

from cds.signals import (
    apply_filter,
    butter_bandpass,
    butter_lowpass,
    convolve,
    dft,
    fft_radix2,
    low_pass_filter,
    moving_median,
)

# --- DFT of a simple signal ---


def main() -> None:
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

    # --- Butterworth IIR low-pass ---
    print("\n=== Butterworth Low-Pass (IIR) ===")
    m = 512
    # slow trend at 2% of Nyquist (well inside the passband), fast jitter at 90%.
    clean = [math.sin(math.pi * 0.02 * k) for k in range(m)]
    jitter = [0.5 * math.sin(math.pi * 0.9 * k) for k in range(m)]
    mixed = [a + c for a, c in zip(clean, jitter)]

    coef = butter_lowpass(order=4, cutoff=0.2)
    smoothed = apply_filter(mixed, coef)
    tail = slice(m // 2, m)
    in_amp = (max(mixed[tail]) - min(mixed[tail])) / 2
    out_amp = (max(smoothed[tail]) - min(smoothed[tail])) / 2
    print(f"input amplitude : {in_amp:.3f}  (trend + near-Nyquist jitter)")
    print(f"output amplitude: {out_amp:.3f}  (~1.0: jitter gone, trend kept)")

    # --- Band-pass isolate a single tone ---
    print("\n=== Band-Pass Isolation ===")
    tones = [math.sin(math.pi * w * k) for w, k in zip([0.1] * m, range(m))]
    for k in range(m):  # add a second in-band tone plus an out-of-band one
        tones[k] += 0.8 * math.sin(math.pi * 0.35 * k)
        tones[k] += 0.6 * math.sin(math.pi * 0.8 * k)
    bp = butter_bandpass(order=4, low=0.2, high=0.5)
    isolated = apply_filter(tones, bp)
    iso_amp = (max(isolated[tail]) - min(isolated[tail])) / 2
    print(f"band-pass amplitude (0.35 tone): {iso_amp:.3f}  (~0.8 expected)")

    # --- Moving-median spike removal ---
    print("\n=== Moving-Median Denoising ===")
    base = [1.0] * 8
    spiked = base[:3] + [25.0] + base[3:6] + [-18.0] + base[6:]
    restored = moving_median(spiked, window=3)
    print(f"input :   {[round(x, 1) for x in spiked]}")
    print(f"denoised: {[round(x, 1) for x in restored]}")


if __name__ == "__main__":
    main()
