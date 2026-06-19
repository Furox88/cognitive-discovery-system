"""2-D FFT demo (row-column Cooley-Tukey)."""

from cds.signals import fft2, ifft2

# A small 4x4 "image"


def main() -> None:
    image: list[list[float | complex]] = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
    ]

    print("=== 2-D FFT ===")
    spectrum = fft2(image)
    print("DC component (sum of all pixels):")
    print(f"  F[0][0] = {spectrum[0][0].real:.1f}  (expected {sum(sum(r) for r in image):.1f})")

    print("\nMagnitude spectrum |F[u][v]|:")
    for row in spectrum:
        print("  " + "  ".join(f"{abs(x):6.2f}" for x in row))

    print("\n=== Inverse 2-D FFT (round-trip) ===")
    recovered = ifft2(spectrum)
    max_err = max(abs(recovered[i][j].real - image[i][j]) for i in range(4) for j in range(4))
    print(f"Max reconstruction error: {max_err:.2e}")


if __name__ == "__main__":
    main()
