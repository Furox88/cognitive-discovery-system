#!/usr/bin/env python3
"""Demo: optional ``cds[plot]`` matplotlib helpers.

Requires: ``pip install cognitive-discovery-system[plot]``

Saves figures under ``examples/_plot_out/`` (gitignored-friendly names).
"""

from __future__ import annotations

import math
from pathlib import Path

from cds.plot import (
    plot_acf,
    plot_histogram,
    plot_optimization_path,
    plot_series,
    plot_spectrum,
    plot_waveform,
)
from cds.signals import power_spectrum


def main() -> None:
    out = Path(__file__).resolve().parent / "_plot_out"
    out.mkdir(exist_ok=True)

    # 1) series + histogram
    y = [math.sin(2 * math.pi * i / 40) + 0.1 * math.sin(2 * math.pi * i / 7) for i in range(120)]
    plot_series(y, title="Noisy sine").savefig(out / "series.png", dpi=120)
    plot_histogram(y, bins=24, title="Amplitude histogram").savefig(out / "hist.png", dpi=120)

    # 2) waveform + spectrum (CDS power_spectrum uses FFT under the hood)
    n = 128
    fs = 128.0
    signal = [math.sin(2 * math.pi * 8 * t / fs) for t in range(n)]
    plot_waveform(signal, sample_rate=fs, title="8 Hz tone").savefig(out / "wave.png", dpi=120)
    power = power_spectrum(signal)
    half = [math.sqrt(max(0.0, p)) for p in power[: n // 2]]
    plot_spectrum(half, sample_rate=fs / 2, title="Magnitude spectrum").savefig(
        out / "spectrum.png", dpi=120
    )

    # 3) ACF of AR-like series
    ar = [0.0]
    for i in range(1, 100):
        ar.append(0.75 * ar[-1] + (0.2 if i % 3 == 0 else -0.05))
    plot_acf(ar, max_lag=20, title="ACF of AR(1)-like series").savefig(out / "acf.png", dpi=120)

    # 4) fake optimizer path
    path = [(math.cos(t / 8) * (1 - t / 40), math.sin(t / 8) * (1 - t / 40)) for t in range(40)]
    plot_optimization_path(path, title="Spiral descent").savefig(out / "opt_path.png", dpi=120)

    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
