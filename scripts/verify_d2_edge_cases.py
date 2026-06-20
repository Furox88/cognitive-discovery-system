"""D2 — Edge-case Big-O / sınır koşulları doğrulama.

Linalg, FFT, matrix işlemlerinin sınır koşullarını test eder:
  - N=0, N=1, N=2 matrisler
  - Singular / nearly-singular / zero matris
  - Simetrik olmayan / negative-definite (Cholesky)
  - FFT power-of-2 olmayan uzunluk (zero-padding davranışı)
  - Boş liste / tek eleman
  - Boyut uyumsuzluğu (ValueError beklenir)

Çalıştırma: python scripts/verify_d2_edge_cases.py
"""

from __future__ import annotations

import math
import sys

from cds.math_utils.linalg import (
    cholesky,
    determinant,
    dot,
    gram_schmidt,
    identity,
    lu_decomposition,
    mat_mul,
    matrix_inverse,
    power_iteration,
    qr_decomposition,
    solve_linear,
    transpose,
)
from cds.signals.processing import dft, fft, fft2, fft_radix2, ifft, ifft2

FAIL = "\033[91m"
PASS = "\033[92m"
WARN = "\033[93m"
END = "\033[0m"

failures: list[str] = []
warnings: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = f"{PASS}PASS{END}" if cond else f"{FAIL}FAIL{END}"
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(name + (f": {detail}" if detail else ""))


def warn(name: str, detail: str) -> None:
    print(f"  [{WARN}WARN{END}] {name} — {detail}")
    warnings.append(name + f": {detail}")


# ---------------------------------------------------------------------------
# D2.1 — Linalg N=0, N=1, N=2
# ---------------------------------------------------------------------------


def test_linalg_small_n() -> None:
    print("\n=== D2.1 Linalg küçük N (0, 1, 2) ===")

    # N=0: empty matrix
    try:
        d = determinant([])
        check("determinant([]) == 1.0 (boş çarpım kuralı)", d == 1.0, f"got {d}")
    except Exception as e:
        check("determinant([]) tanımlı", False, f"{type(e).__name__}: {e}")

    try:
        i0 = identity(0)
        check("identity(0) == []", i0 == [])
    except Exception as e:
        check("identity(0) tanımlı", False, f"{type(e).__name__}: {e}")

    # N=1: 1x1 matrix
    m1 = [[5.0]]
    check("determinant([[5]]) == 5.0", determinant(m1) == 5.0)
    check("determinant([[0]]) == 0.0", determinant([[0.0]]) == 0.0)
    check("transpose([[5]]) == [[5]]", transpose(m1) == [[5.0]])

    # LU on 1x1
    try:
        P, L, U = lu_decomposition([[5.0]])
        check("lu_decomposition([[5]]) works", True, f"P={P} L={L} U={U}")
        check("L == [[1]]", L == [[1.0]])
        check("U == [[5]]", U == [[5.0]])
    except Exception as e:
        check("lu_decomposition([[5]]) tanımlı", False, f"{type(e).__name__}: {e}")

    # matrix_inverse on 1x1
    inv = matrix_inverse([[4.0]])
    check("matrix_inverse([[4]]) == [[0.25]]", inv == [[0.25]], f"got {inv}")

    # QR on 1x1
    Q, R = qr_decomposition([[3.0]])
    check("QR of [[3]]: Q==[[1]] or [[-1]]", Q == [[1.0]] or Q == [[-1.0]])
    check("QR of [[3]]: |R| == [[3]]", abs(R[0][0]) == 3.0)

    # solve_linear 1x1
    x = solve_linear([[2.0]], [6.0])
    check("solve_linear([[2]], [6]) == [3]", x == [3.0])

    # N=2: 2x2 matrix
    m2 = [[1.0, 2.0], [3.0, 4.0]]
    check("determinant 2x2 = -2", abs(determinant(m2) - (-2.0)) < 1e-12, f"got {determinant(m2)}")

    # mat_mul boyut kontrolü
    try:
        mat_mul([[1, 2, 3]], [[1, 2]])
        check("mat_mul incompatible shapes raises", False, "no exception")
    except ValueError:
        check("mat_mul incompatible shapes raises ValueError", True)
    except Exception as e:
        check("mat_mul incompatible shapes raises ValueError", False, f"got {type(e).__name__}")

    # dot boyut kontrolü
    try:
        dot([1.0, 2.0], [1.0])
        check("dot unequal lengths raises", False, "no exception")
    except ValueError:
        check("dot unequal lengths raises ValueError", True)


# ---------------------------------------------------------------------------
# D2.2 — Singular / nearly-singular / zero matrix
# ---------------------------------------------------------------------------


def test_singular_matrices() -> None:
    print("\n=== D2.2 Singular / zero / nearly-singular matris ===")

    # Tam singular: [[1,2],[2,4]] rank 1
    singular = [[1.0, 2.0], [2.0, 4.0]]
    try:
        lu_decomposition(singular)
        check("lu_decomposition singular raises", False, "no exception")
    except ValueError as e:
        check("lu_decomposition singular raises ValueError", True, str(e)[:60])

    # determinant singular → 0.0 (ValueError yakalanıp 0 dönüyor)
    d = determinant(singular)
    check("determinant of singular matrix == 0.0", d == 0.0, f"got {d}")

    # matrix_inverse singular → ValueError
    try:
        matrix_inverse(singular)
        check("matrix_inverse singular raises", False, "no exception")
    except ValueError:
        check("matrix_inverse singular raises ValueError", True)

    # solve_linear singular → ValueError
    try:
        solve_linear(singular, [1.0, 2.0])
        check("solve_linear singular raises", False, "no exception")
    except ValueError:
        check("solve_linear singular raises ValueError", True)

    # Zero matrix: [[0,0],[0,0]]
    zero = [[0.0, 0.0], [0.0, 0.0]]
    try:
        lu_decomposition(zero)
        check("lu_decomposition zero matrix raises", False, "no exception")
    except ValueError:
        check("lu_decomposition zero matrix raises ValueError", True)
    check("determinant of zero matrix == 0.0", determinant(zero) == 0.0)

    # Nearly singular: küçük köşegen
    nearly = [[1e-15, 0.0], [0.0, 1.0]]
    # Bu NEAR_ZERO eşiğine göre singular sayılabilir
    try:
        lu_decomposition(nearly)
        # Eğer geçerse, determinant makul olmalı
        d = determinant(nearly)
        warn("nearly singular 1e-15 pivot", f"NEAR_ZERO eşiği altında değil, d={d}")
    except ValueError:
        check("lu_decomposition nearly-singular (1e-15) raises", True)

    # Upper-triangular singular (backward substitution'da yakalanmalı)
    ut_singular = [[1.0, 1.0], [0.0, 0.0]]
    try:
        solve_linear(ut_singular, [1.0, 0.0])
        check("solve_linear UT-singular raises", False, "no exception")
    except ValueError:
        check("solve_linear UT-singular raises ValueError", True)


# ---------------------------------------------------------------------------
# D2.3 — Cholesky: non-SPD matrisler
# ---------------------------------------------------------------------------


def test_cholesky_non_spd() -> None:
    print("\n=== D2.3 Cholesky non-SPD matrisler ===")

    # Negative definite: tüm özdeğerler negatif
    neg_def = [[-2.0, 0.0], [0.0, -3.0]]
    try:
        cholesky(neg_def)
        check("cholesky negative-definite raises", False, "no exception")
    except ValueError:
        check("cholesky negative-definite raises ValueError", True)

    # Indefinite: karışık işaretli özdeğerler
    indef = [[1.0, 0.0], [0.0, -1.0]]
    try:
        cholesky(indef)
        check("cholesky indefinite raises", False, "no exception")
    except ValueError:
        check("cholesky indefinite raises ValueError", True)

    # Simetrik ama singular: [[1,1],[1,1]]
    try:
        cholesky([[1.0, 1.0], [1.0, 1.0]])
        check("cholesky singular raises", False, "no exception")
    except ValueError:
        check("cholesky singular raises ValueError", True)

    # SPD: doğrulama
    spd = [[4.0, 2.0], [2.0, 3.0]]
    L = cholesky(spd)
    # A == L L^T ?
    LLt = mat_mul(L, transpose(L))
    err = max(abs(LLt[i][j] - spd[i][j]) for i in range(2) for j in range(2))
    check("cholesky SPD reconstructs A = L L^T", err < 1e-12, f"err={err:.2e}")

    # 1x1 SPD
    L1 = cholesky([[9.0]])
    check("cholesky 1x1 [[9]] = [[3]]", L1 == [[3.0]], f"got {L1}")

    # 1x1 negatif
    try:
        cholesky([[-9.0]])
        check("cholesky 1x1 negative raises", False, "no exception")
    except ValueError:
        check("cholesky 1x1 negative raises ValueError", True)


# ---------------------------------------------------------------------------
# D2.4 — Power iteration & Gram-Schmidt edge cases
# ---------------------------------------------------------------------------


def test_eigen_orthonormal_edges() -> None:
    print("\n=== D2.4 Power iteration & Gram-Schmidt edge cases ===")

    # 1x1 power iteration
    eig, vec = power_iteration([[7.0]])
    check("power_iteration 1x1 eig == 7", abs(eig - 7.0) < 1e-9, f"eig={eig}")
    check("power_iteration 1x1 vec normalized", abs(abs(vec[0]) - 1.0) < 1e-9, f"vec={vec}")

    # Zero matrix power iteration: norm→0, çıkış
    eig, vec = power_iteration([[0.0, 0.0], [0.0, 0.0]], max_iter=10)
    check("power_iteration zero matrix returns 0 eig", eig == 0.0, f"eig={eig}")

    # Gram-Schmidt: linearly dependent vectors
    dep = [[1.0, 0.0], [2.0, 0.0]]  # paralel
    ortho = gram_schmidt(dep)
    # ikinci vektör 0'a düşer, atlanır
    check("gram_schmidt drops dependent vectors", len(ortho) == 1, f"got {len(ortho)} vectors")

    # Gram-Schmidt: tek vektör
    ortho = gram_schmidt([[3.0, 4.0]])
    check(
        "gram_schmidt single vector normalized",
        abs(ortho[0][0] - 0.6) < 1e-9 and abs(ortho[0][1] - 0.8) < 1e-9,
    )

    # Gram-Schmidt: boş input
    ortho = gram_schmidt([])
    check("gram_schmidt([]) == []", ortho == [])

    # Gram-Schmidt: zero vector
    ortho = gram_schmidt([[0.0, 0.0]])
    check("gram_schmidt zero vector → empty", ortho == [])


# ---------------------------------------------------------------------------
# D2.5 — FFT power-of-2 olmayan uzunluk
# ---------------------------------------------------------------------------


def test_fft_non_power_of_2() -> None:
    print("\n=== D2.5 FFT power-of-2 olmayan uzunluk ===")

    # fft_radix2 explicitly rejects non-power-of-2
    for n in (3, 5, 6, 7, 9, 10, 12, 15):
        try:
            fft_radix2([1.0] * n)
            check(f"fft_radix2 len={n} raises", False, "no exception")
        except ValueError:
            check(f"fft_radix2 len={n} raises ValueError", True)

    # fft() accepts arbitrary length via zero-padding
    # Bu sessizce zero-pad yapar — dökümante edilmiş davranış
    sig3 = [1.0, 2.0, 3.0]
    spec = fft(sig3)
    check("fft len=3 zero-pads to len=4", len(spec) == 4, f"got len={len(spec)}")

    sig5 = [1.0, 2.0, 3.0, 4.0, 5.0]
    spec = fft(sig5)
    check("fft len=5 zero-pads to len=8", len(spec) == 8, f"got len={len(spec)}")

    # fft len=0: empty
    check("fft([]) == []", fft([]) == [])

    # fft len=1: identity
    spec = fft([5.0])
    check("fft len=1 == [5+0j]", abs(spec[0] - 5.0) < 1e-12, f"got {spec[0]}")

    # fft power-of-2 uzunluk: zero-padding yok
    sig4 = [1.0, 1.0, 1.0, 1.0]
    spec = fft(sig4)
    check("fft len=4 (power-of-2) keeps length", len(spec) == 4)
    # DC component = sum, others 0
    check("fft [1,1,1,1] DC = 4", abs(spec[0] - 4.0) < 1e-9, f"DC={spec[0]}")
    check("fft [1,1,1,1] k=1 ~ 0", abs(spec[1]) < 1e-9, f"k=1={spec[1]}")

    # FFT vs DFT consistency (same length power-of-2)
    sig8 = [math.sin(2 * math.pi * k / 8) for k in range(8)]
    fft_res = fft(sig8)
    dft_res = dft(sig8)
    max_err = max(abs(a - b) for a, b in zip(fft_res, dft_res, strict=True))
    check("fft vs dft agree on power-of-2 length", max_err < 1e-9, f"max err={max_err:.2e}")

    # CRITICAL: fft() zero-pads len=3 to len=4. This changes the spectrum!
    # Padding changes frequencies. User must be aware.
    # Verify round-trip: ifft(fft(x)) recovers x *only* if lengths match
    spec3 = fft([1.0, 2.0, 3.0])  # len 4
    rec = ifft(spec3)  # len 4
    # only first 3 should match original
    err_orig = max(abs(rec[i] - [1.0, 2.0, 3.0][i]) for i in range(3))
    check(
        "ifft(fft(x)) round-trips on padded tail (first N samples)",
        err_orig < 1e-9,
        f"err={err_orig:.2e}",
    )

    # fft2 with non-square or odd dimensions
    m2x3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    spec = fft2(m2x3)
    # Each row len=3 padded to 4, num rows=2 stays (no row padding for fft2)
    check(
        "fft2 2x3 pads each row to len=4",
        all(len(r) == 4 for r in spec) and len(spec) == 2,
        f"shape={[len(r) for r in spec]} x {len(spec)}",
    )

    # ifft2 round-trip
    rec = ifft2(spec)
    err = max(abs(rec[i][j] - m2x3[i][j]) for i in range(2) for j in range(3))
    check("ifft2(fft2(m)) round-trips (first NxM block)", err < 1e-9, f"err={err:.2e}")

    # fft2 of empty: raises ValueError (makul — boş matris tanımsız)
    try:
        fft2([])
        check("fft2([]) raises ValueError", False, "no exception")
    except ValueError:
        check("fft2([]) raises ValueError", True)


# ---------------------------------------------------------------------------
# D2.6 — Big-O sanity: PLU vs determinant expansion
# ---------------------------------------------------------------------------


def test_big_o_sanity() -> None:
    print("\n=== D2.6 Big-O sanity (timing PLU on growing N) ===")
    import random

    random.seed(42)
    # Start at N=8 so per-call fixed overhead is negligible relative to the
    # O(N^3) work; at N=4 the determinant runs in ~0.01ms and timing noise
    # dominates the scaling ratio, which previously masked the true ~O(N^3)
    # behavior (it read as 70x for an 8x size increase instead of ~512x).
    sizes = [8, 16, 32, 64, 128]
    times = []
    for n in sizes:
        # Random non-singular matrix
        m = []
        for _ in range(n):
            row = [random.gauss(0, 1) for _ in range(n)]
            m.append(row)
        # ensure non-singular by adding n*I
        for i in range(n):
            m[i][i] += n

        import time

        t0 = time.perf_counter()
        d = determinant(m)
        dt = time.perf_counter() - t0
        times.append(dt)
        check(
            f"determinant N={n} returns finite",
            math.isfinite(d),
            f"d={d:.3e}, time={dt * 1000:.1f}ms",
        )

    # O(N^3) için N 2 katına çıkınca zaman ~8 kat olmalı. Use the largest two
    # sizes (least contaminated by fixed overhead) for the scaling assertion.
    if times[-1] > 0 and times[-2] > 0:
        ratio = times[-1] / times[-2]
        # O(n^3): doubling N should give ~8x. Accept 4x–20x to leave room for
        # cache effects and timing jitter while still ruling out O(n^2)/O(n^4).
        check(
            f"determinant scaling ~ O(N³) (ratio {ratio:.1f}x for N doubling {sizes[-2]}->{sizes[-1]})",
            4.0 < ratio < 20.0,
            f"got {ratio:.2f}x, O(N³) expects ~8x",
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    test_linalg_small_n()
    test_singular_matrices()
    test_cholesky_non_spd()
    test_eigen_orthonormal_edges()
    test_fft_non_power_of_2()
    test_big_o_sanity()

    print("\n" + "=" * 70)
    print(f"Toplam: {len(failures)} hata, {len(warnings)} uyarı")
    if failures:
        print(f"\n{FAIL}HATALAR:{END}")
        for f in failures:
            print(f"  - {f}")
    if warnings:
        print(f"\n{WARN}UYARILAR:{END}")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
