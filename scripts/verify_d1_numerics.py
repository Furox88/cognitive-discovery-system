"""D1 — Numerik doğruluk derinlemesine doğrulama.

Bu script bibliyografik referans değerlerle (Abramowitz & Stegun, NIST DLMF)
karşılaştırma yapar:
  - Gauss-Legendre düğüm/ağırlıkları
  - Romberg / Simpson / Trapezoid / Gauss referans integral değerleri
  - RK45 adaptif adım davranışı
  - ODE sistemleri (harmonik osilatör, coupled decay)
  - adaptive_simpson'un tolerance karşılaması

Çalıştırma: python scripts/verify_d1_numerics.py
"""

from __future__ import annotations

import math
import sys
import time

from cds.diffeq import euler_method, midpoint_method, rk4, rk45, solve_system
from cds.numerical_integration import (
    adaptive_simpson,
    gaussian_quadrature,
    romberg,
    simpson,
    simpson_38,
    trapezoid,
)
from cds.numerical_integration.quadrature import _gauss_legendre_nodes, _legendre

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

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
# D1.1 — Gauss-Legendre düğüm/ağırlık doğruluğu
#   Referans: Abramowitz & Stegun Table 25.4 / NIST DLMF §3.5.10
# ---------------------------------------------------------------------------

# n=5 için bilinen düğüm/ağırlık değerleri (A&S 25.4.29)
GL5_REF = [
    (0.0000000000000000, 0.5688888888888889),
    (0.5384693101056831, 0.4786286704993665),
    (-0.5384693101056831, 0.4786286704993665),
    (0.9061798459386640, 0.2369268850561891),
    (-0.9061798459386640, 0.2369268850561891),
]

# n=4 (A&S 25.4.28)
GL4_REF = [
    (0.3399810435848563, 0.6521451548625461),
    (-0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538),
    (-0.8611363115940526, 0.3478548451374538),
]


def test_gauss_legendre_nodes() -> None:
    print("\n=== D1.1 Gauss-Legendre nodes/weights (A&S reference) ===")

    # n=1: tek nokta, midpoint = 0, weight = 2
    nodes1 = _gauss_legendre_nodes(1)
    check("n=1 node count", len(nodes1) == 1)
    check("n=1 node is 0.0", abs(nodes1[0][0] - 0.0) < 1e-15)
    check("n=1 weight is 2.0", abs(nodes1[0][1] - 2.0) < 1e-15)

    # n=5
    nodes5 = sorted(_gauss_legendre_nodes(5), key=lambda t: t[0])
    ref5 = sorted(GL5_REF, key=lambda t: t[0])
    max_node_err = max(abs(a[0] - b[0]) for a, b in zip(nodes5, ref5, strict=True))
    max_weight_err = max(abs(a[1] - b[1]) for a, b in zip(nodes5, ref5, strict=True))
    check("n=5 nodes within 1e-12", max_node_err < 1e-12, f"max node err={max_node_err:.2e}")
    check("n=5 weights within 1e-12", max_weight_err < 1e-12, f"max weight err={max_weight_err:.2e}")

    # n=4
    nodes4 = sorted(_gauss_legendre_nodes(4), key=lambda t: t[0])
    ref4 = sorted(GL4_REF, key=lambda t: t[0])
    max_node_err4 = max(abs(a[0] - b[0]) for a, b in zip(nodes4, ref4, strict=True))
    max_weight_err4 = max(abs(a[1] - b[1]) for a, b in zip(nodes4, ref4, strict=True))
    check("n=4 nodes within 1e-12", max_node_err4 < 1e-12, f"max node err={max_node_err4:.2e}")
    check("n=4 weights within 1e-12", max_weight_err4 < 1e-12, f"max weight err={max_weight_err4:.2e}")

    # Symmetri kontrolü: x_i = -x_{n+1-i}, w_i = w_{n+1-i}
    for n in (2, 3, 5, 8, 16):
        nodes = sorted(_gauss_legendre_nodes(n), key=lambda t: t[0])
        sym_ok = all(
            abs(nodes[i][0] + nodes[n - 1 - i][0]) < 1e-13
            and abs(nodes[i][1] - nodes[n - 1 - i][1]) < 1e-13
            for i in range(n // 2)
        )
        check(f"n={n} symmetry (±x, equal weights)", sym_ok)

    # Ağırlıkların toplamı [-1,1] üzerinde = 2 (constant integrand exact)
    for n in (1, 2, 3, 5, 10, 20):
        wsum = sum(w for _, w in _gauss_legendre_nodes(n))
        check(f"n={n} sum of weights == 2.0", abs(wsum - 2.0) < 1e-12, f"sum={wsum:.15f}")

    # Düğümler gerçekten P_n kökü mü?
    for n in (3, 5, 8):
        for x, _w in _gauss_legendre_nodes(n):
            p, _dp = _legendre(n, x)
            check(f"n={n} node x={x:+.6f} is P_n root", abs(p) < 1e-10, f"P_n(x)={p:.2e}")

    # Polinom doğruluğu: n düğüm derece <= 2n-1 için kesin
    for n in (3, 4, 5):
        for deg in range(0, 2 * n):
            # ∫_{-1}^{1} x^deg dx: 0 for odd degree (odd integrand, symmetric
            # interval), otherwise 2/(deg+1). The earlier check always used
            # ``2/(deg+1)`` regardless of parity, which is wrong for odd ``deg``
            # (the true integral is 0) and produced spurious large errors.
            exact = 0.0 if deg % 2 == 1 else 2.0 / (deg + 1)
            got = gaussian_quadrature(lambda x: x**deg, -1.0, 1.0, n)
            err = abs(got - exact)
            check(
                f"n={n} exact for degree {deg} (<= {2 * n - 1})",
                err < 1e-10,
                f"err={err:.2e}",
            )


# ---------------------------------------------------------------------------
# D1.2 — Quadrature referans değerleri
# ---------------------------------------------------------------------------


def test_quadrature_reference_values() -> None:
    print("\n=== D1.2 Quadrature referans değerleri ===")

    # ∫_0^1 e^x dx = e - 1
    exact = math.e - 1
    f = math.exp

    val_trap = trapezoid(f, 0, 1, 10000)
    val_simp = simpson(f, 0, 1, 100)
    val_rom = romberg(f, 0, 1, tol=1e-14).value
    val_gauss = gaussian_quadrature(f, 0, 1, 10)
    val_adapt = adaptive_simpson(f, 0, 1, tol=1e-12).value

    check("trapezoid  ∫e^x err < 1e-8", abs(val_trap - exact) < 1e-8, f"err={abs(val_trap - exact):.2e}")
    # Simpson is O(h^4); at n=100 the expected error is ~1e-10, not 1e-12.
    # The earlier 1e-12 threshold was tighter than the method's order allows
    # for this panel count and produced a spurious FAIL at 9.55e-11.
    check("simpson    ∫e^x err < 1e-9", abs(val_simp - exact) < 1e-9, f"err={abs(val_simp - exact):.2e}")
    check("romberg    ∫e^x err < 1e-12", abs(val_rom - exact) < 1e-12, f"err={abs(val_rom - exact):.2e}")
    check("gauss      ∫e^x err < 1e-12", abs(val_gauss - exact) < 1e-12, f"err={abs(val_gauss - exact):.2e}")
    check("adaptive   ∫e^x err < 1e-10", abs(val_adapt - exact) < 1e-10, f"err={abs(val_adapt - exact):.2e}")

    # ∫_0^π sin(x) dx = 2
    sin_exact = 2.0
    val = gaussian_quadrature(math.sin, 0, math.pi, 8)
    check("gauss n=8  ∫sin err < 1e-12", abs(val - sin_exact) < 1e-12, f"err={abs(val - sin_exact):.2e}")

    # ∫_{-1}^{1} 1/(1+x²) dx = 2 atan(1) = π/2
    f_runge = lambda x: 1.0 / (1.0 + x * x)  # noqa: E731
    runge_exact = math.pi / 2
    val = romberg(f_runge, -1, 1, tol=1e-12).value
    check("romberg ∫1/(1+x²) err < 1e-10", abs(val - runge_exact) < 1e-10, f"err={abs(val - runge_exact):.2e}")

    # ∫_0^1 x⁴ dx = 1/5  (Simpson 1/3 is O(h⁴); at n=10 the expected error is
    # ~1e-5, not 1e-6. The earlier 1e-6 threshold was tighter than O(h⁴) at
    # n=10 allows and produced a spurious FAIL at 1.33e-05.)
    val = simpson(lambda x: x**4, 0, 1, 10)
    check("simpson ∫x⁴ err < 1e-4", abs(val - 0.2) < 1e-4, f"err={abs(val - 0.2):.2e}")

    # ∫_0^3 x³ dx = 81/4 = 20.25  (simpson_38 kesin)
    val = simpson_38(lambda x: x**3, 0, 3, 9)
    check("simpson_38 ∫x³ [0,3] kesin", abs(val - 20.25) < 1e-12, f"val={val:.15f}")

    # Gaussian bell: ∫_{-∞}^{∞} e^{-x²} = √π. [-4,4] truncation ile ~2e-7 hata.
    val = adaptive_simpson(lambda x: math.exp(-x * x), -4, 4, tol=1e-12).value
    check("adaptive ∫e^{-x²} [-4,4] ~ √π", abs(val - math.sqrt(math.pi)) < 1e-6, f"err={abs(val - math.sqrt(math.pi)):.2e}")

    # Reversed limits: sign flip
    val = simpson(lambda x: x**2, 1, 0, 10)
    check("simpson reversed limits sign flip", abs(val - (-1 / 3)) < 1e-12, f"val={val:.15f}")

    val = gaussian_quadrature(lambda x: x**2, 1, 0, 5)
    check("gauss reversed limits sign flip", abs(val - (-1 / 3)) < 1e-10, f"val={val:.15f}")


# ---------------------------------------------------------------------------
# D1.3 — Romberg yakınsama oranı (Richardson extrapolation)
# ---------------------------------------------------------------------------


def test_romberg_convergence_order() -> None:
    print("\n=== D1.3 Romberg yakınsama oranı ===")
    # ∫_0^1 sin(x) e^x dx. Antiderivative of sin(x)e^x is e^x(sin x - cos x)/2:
    #   = [e^x (sin x - cos x) / 2]_0^1
    #   = ((sin1 - cos1) e^1 - (sin0 - cos0) e^0) / 2
    # Note: the earlier check used ``(sin1 + cos1)`` here, which is the
    # antiderivative of cos(x)e^x — wrong integrand — and produced a spurious
    # err of 4.69e-01 against the true value 0.9093.
    exact = ((math.sin(1) - math.cos(1)) * math.e - (math.sin(0) - math.cos(0))) / 2
    f = lambda x: math.sin(x) * math.exp(x)  # noqa: E731

    # max_iter'i artırarak hata nasıl düşüyor?
    errs = []
    for mi in (3, 5, 7, 10):
        val = romberg(f, 0, 1, tol=1e-15, max_iter=mi).value
        errs.append(abs(val - exact))

    # Romberg tablosunun ilk satırları Richardson extrapolation'uyla
    # super-exponentially düşer, fakat ~1e-16'ya (makine epsilonu) ulaşınca
    # plateau yapar — bu noktadan sonra "bir öncekinin yarısı" beklentisi
    # karşılanamaz. Bu yüzden monotone düşüşü yalnızca makine hassasiyetine
    # ulaşmamış aralıklarda arıyoruz (errs > 1e-13 eşiği).
    MACHINE_PLATEAU = 1e-13
    pre_plateau = [e for e in errs if e > MACHINE_PLATEAU]
    monotone = all(
        pre_plateau[i + 1] < pre_plateau[i] * 0.5
        for i in range(len(pre_plateau) - 1)
    )
    check(
        "romberg hatası makine öncesi aralıkta monotone düşer",
        monotone,
        f"errs={[f'{e:.2e}' for e in errs]}",
    )
    # max_iter=10 ile makine hassasiyeti plateau'una ulaşmalı
    check(
        "romberg max_iter=10 makine hassasiyetine ulaşır (err < 1e-13)",
        errs[-1] < 1e-13,
        f"err={errs[-1]:.2e}",
    )


# ---------------------------------------------------------------------------
# D1.4 — RK45 adaptif adım davranışı
# ---------------------------------------------------------------------------


def test_rk45_adaptive() -> None:
    print("\n=== D1.4 RK45 adaptif adım ===")

    # 1) dy/dt = -y, y(0)=1  =>  y(2) = e^{-2}
    exact = math.exp(-2.0)
    sol = rk45(lambda t, y: -y, 0.0, 1.0, 2.0, dt=0.1, atol=1e-10, rtol=1e-10)
    err = abs(sol.y[-1] - exact)
    check("rk45 ∫ -y [0,2] err < 1e-6", err < 1e-6, f"err={err:.2e}, steps={sol.steps}")

    # 2) Sert problem: dy/dt = 100y  =>  y(0.1) = e^{10}
    exact_stiff = math.exp(10.0)
    sol = rk45(lambda t, y: 100.0 * y, 0.0, 1.0, 0.1, dt=0.1, atol=1e-12, rtol=1e-12)
    rel_err = abs(sol.y[-1] - exact_stiff) / exact_stiff
    check("rk45 sert dy/dt=100y rel err < 1e-3", rel_err < 1e-3, f"rel err={rel_err:.2e}, steps={sol.steps}")

    # 3) Adaptif adım sayısı sabit-adımdan az olmalı (verimlilik)
    sol_adaptive = rk45(lambda t, y: -y, 0.0, 1.0, 1.0, dt=0.1, atol=1e-10, rtol=1e-10)
    sol_fixed = rk4(lambda t, y: -y, 0.0, 1.0, 1.0, dt=0.001)
    exact1 = math.exp(-1.0)
    err_adapt = abs(sol_adaptive.y[-1] - exact1)
    err_fixed = abs(sol_fixed.y[-1] - exact1)
    # Aynı doğrulukta daha az adımda bitirmeli
    check(
        "rk45 fewer steps than rk4 at comparable accuracy",
        sol_adaptive.steps < sol_fixed.steps,
        f"adapt steps={sol_adaptive.steps} (err={err_adapt:.2e}), "
        f"fixed steps={sol_fixed.steps} (err={err_fixed:.2e})",
    )

    # 4) Adımların monoton artıp artmadığı (t doğal olarak artmalı)
    # Use tight tolerances: with the default atol=1e-6 / rtol=1e-3 the solver
    # takes only 5 coarse steps over [0, 2π] and lands ~1e-4 off sin(2π)=0,
    # which (correctly) exceeds the 1e-5 accuracy check. Tightening here is
    # exercising the accuracy guarantee, not the default-tol regime.
    sol = rk45(lambda t, y: math.cos(t), 0.0, 0.0, 2 * math.pi, dt=0.1, atol=1e-10, rtol=1e-10)
    t_mono = all(sol.t[i + 1] > sol.t[i] for i in range(len(sol.t) - 1))
    check("rk45 t values strictly increasing", t_mono)
    # Son değer sin(2π) = 0
    check(
        "rk45 ∫cos dt [0,2π] = sin(2π) ≈ 0",
        abs(sol.y[-1] - 0.0) < 1e-5,
        f"val={sol.y[-1]:.2e}",
    )

    # 5) Machine precision floor tetiklenmeli mi? Sonsuzluk doğru çalışıyor mu?
    # dy/dt = y², y(0)=1  =>  y(t) = 1/(1-t), t=1'de tekillik.
    # Tekilliği GEÇEN bir t_end kullan (t=2): solver adımları gittikçe küçülür
    # ve sonunda makine hassasiyeti floor'una (h_mag < eps_floor) düşerek
    # RuntimeError atar. NOT: t_end=1.0 (tam tekillik) veya t_end=1-1e-9
    # kullanılırsa, son adımda "kalan span < h_mag" kontrolü devreye girip
    # adımı direkt t_end'e snap'ler ve floor tetiklenmeden normal çıkış olur —
    # bu yüzden tekilliği aşan bir t_end seçmek şart.
    try:
        rk45(lambda t, y: y * y, 0.0, 1.0, 2.0, dt=0.01, atol=1e-12, rtol=1e-12)
        warn("rk45 singularity at t=1", "RuntimeError bekleniyordu ama atmadı")
    except RuntimeError as e:
        check("rk45 singularity raises RuntimeError", True, str(e)[:60])

    # 6) t_end == t0: boş döngü, sadece başlangıç değeri
    sol = rk45(lambda t, y: -y, 0.0, 1.0, 0.0)
    check("rk45 t_end==t0 returns just IC", len(sol.y) == 1 and sol.y[0] == 1.0)


# ---------------------------------------------------------------------------
# D1.5 — ODE sistemleri ve düşük dereceli metodlar
# ---------------------------------------------------------------------------


def test_ode_methods() -> None:
    print("\n=== D1.5 ODE metodları & sistemler ===")

    # Euler: 1. mertebe — y(1)=e^{-1}, dt=0.001 ile ~1e-3 hata
    sol = euler_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.001)
    err = abs(sol.y[-1] - math.exp(-1.0))
    check("euler dt=1e-3 err < 1e-2", err < 1e-2, f"err={err:.2e}")

    # Euler hata oranı O(dt): dt/2 → hata/2
    err1 = abs(euler_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.01).y[-1] - math.exp(-1.0))
    err2 = abs(euler_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.005).y[-1] - math.exp(-1.0))
    ratio = err2 / err1
    check("euler O(dt) — err halves when dt halves", 0.4 < ratio < 0.6, f"ratio={ratio:.3f}")

    # RK4: 4. mertebe — hata çok küçük
    sol = rk4(lambda t, y: -y, 0, 1.0, 1.0, dt=0.1)
    err = abs(sol.y[-1] - math.exp(-1.0))
    check("rk4 dt=0.1 err < 1e-5", err < 1e-5, f"err={err:.2e}")

    # midpoint: 2. mertebe
    sol = midpoint_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.01)
    err = abs(sol.y[-1] - math.exp(-1.0))
    check("midpoint dt=0.01 err < 1e-4", err < 1e-4, f"err={err:.2e}")

    # Harmonic oscillator (sistem): x''=-x, x(0)=1, v(0)=0
    def f_ho(t: float, y: list[float]) -> list[float]:
        return [y[1], -y[0]]

    t_vals, y_vals = solve_system(f_ho, 0, [1.0, 0.0], 2 * math.pi, dt=0.001)
    x_final = y_vals[-1][0]
    # cos(2π) = 1
    check(
        "solve_system harmonic osc x(2π) = cos(2π) = 1",
        abs(x_final - 1.0) < 1e-3,
        f"x_final={x_final:.6f}",
    )
    v_final = y_vals[-1][1]
    # sin(2π) = 0
    check(
        "solve_system harmonic osc v(2π) ≈ 0",
        abs(v_final - 0.0) < 1e-3,
        f"v_final={v_final:.6f}",
    )

    # coupled decay
    def f_cd(t: float, y: list[float]) -> list[float]:
        return [-y[0], -2.0 * y[1]]

    _, y_vals = solve_system(f_cd, 0, [1.0, 1.0], 1.0, dt=0.01)
    check(
        "solve_system coupled decay y1(1)=e^{-1}",
        abs(y_vals[-1][0] - math.exp(-1.0)) < 1e-6,
        f"err={abs(y_vals[-1][0] - math.exp(-1.0)):.2e}",
    )
    check(
        "solve_system coupled decay y2(1)=e^{-2}",
        abs(y_vals[-1][1] - math.exp(-2.0)) < 1e-5,
        f"err={abs(y_vals[-1][1] - math.exp(-2.0)):.2e}",
    )


# ---------------------------------------------------------------------------
# D1.6 — Performans/özgünlük tespiti: cache, negatif yön, tek nokta
# ---------------------------------------------------------------------------


def test_misc() -> None:
    print("\n=== D1.6 Misc: cache, yön, tek nokta ===")

    # Cache gerçekten çalışıyor mu? Aynı n iki kez çağrılınca aynı nesne dönmeli
    n1 = _gauss_legendre_nodes(7)
    n2 = _gauss_legendre_nodes(7)
    check("_gauss_legendre_nodes cache returns same object", n1 is n2)

    # Negatif dt/t_end yönü: t_end < t0 → geriye doğru entegrasyon
    sol = rk4(lambda t, y: 1.0, 0, 0.0, -1.0, dt=0.1)
    # y(t) = t, y(-1) = -1
    check("rk4 backward integration t_end<t0", abs(sol.y[-1] - (-1.0)) < 1e-9, f"y={sol.y[-1]:.6f}")

    # trapezoid tek panel
    val = trapezoid(lambda x: 2 * x, 0, 1, 1)
    # ∫_0^1 2x dx = 1, trapezoid exact for linear
    check("trapezoid n=1 exact for linear", abs(val - 1.0) < 1e-12, f"val={val:.6f}")

    # simpson min n=2
    val = simpson(lambda x: x**3, 0, 1, 2)
    check("simpson n=2 exact cubic", abs(val - 0.25) < 1e-12, f"val={val:.6f}")


# ---------------------------------------------------------------------------
# D1.7 — Performans: çok yüksek n'de Gauss-Legendre determinizmi
# ---------------------------------------------------------------------------


def test_high_n_performance() -> None:
    print("\n=== D1.7 Yüksek n Gauss-Legendre determinizmi ===")

    # n=64 büyük test — deterministik ve doğru olmalı
    start = time.perf_counter()
    nodes = _gauss_legendre_nodes(64)
    dt = time.perf_counter() - start
    # Ağırlık toplamı yine 2.0
    wsum = sum(w for _, w in nodes)
    check("n=64 sum weights == 2", abs(wsum - 2.0) < 1e-10, f"sum={wsum:.15f}")
    # Polinom derece 127'ye kadar kesin (2n-1 = 127 for n=64).
    # deg=127 is odd → ∫_{-1}^{1} x^127 dx = 0 (odd integrand over a symmetric
    # interval), NOT 2/128. The earlier check used ``2.0 / (deg + 1)`` here,
    # producing a spurious err of 1.56e-02 against the true value 0.0.
    deg = 127
    exact = 0.0 if deg % 2 == 1 else 2.0 / (deg + 1)
    val = gaussian_quadrature(lambda x: x**deg, -1, 1, 64)
    err = abs(val - exact)
    check("n=64 exact for degree 127", err < 1e-6, f"err={err:.2e}")
    if dt > 0.5:
        warn("n=64 compute time", f"{dt:.3f}s — functools.cache ilk çağrıda yavaş")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    test_gauss_legendre_nodes()
    test_quadrature_reference_values()
    test_romberg_convergence_order()
    test_rk45_adaptive()
    test_ode_methods()
    test_misc()
    test_high_n_performance()

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
