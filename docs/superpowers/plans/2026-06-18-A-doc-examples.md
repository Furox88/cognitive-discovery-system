# Documentation Examples & Tutorials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every public submodule a runnable `examples/<module>_demo.py` and a narrative `docs/tutorials/<module>_demo.md`.

**Architecture:** Pure additive — no production code changes. Each example mirrors the `examples/stats_demo.py` pattern: a `main()` function with section banners via `print`, `if __name__ == "__main__"` guard, deterministic outputs. Each tutorial mirrors `docs/tutorials/quick_start.md` (numbered sections, copyable snippets). Nav wiring is handled by Plan B (mkdocs.yml owner).

**Tech Stack:** Python stdlib only; `cds.*` imports; Markdown for docs.

**Spec reference:** `docs/superpowers/specs/2026-06-18-project-completion-design.md` §A

---

## File Structure

**Created (18 files):**
- `examples/core_demo.py` — Hypothesis/Domain data-model showcase
- `examples/data_analysis_demo.py` — CSV load, DataTable, normalize/z_score/moving_average, plot_bar/plot_line
- `examples/diffeq_demo.py` — euler/rk4/rk45/midpoint on dy/dt = -y; system solver
- `examples/graph_demo.py` — Graph construction, bfs/dfs/dijkstra/kruskal_mst/topological_sort/has_cycle
- `examples/math_utils_demo.py` — derivative/integral/gradient + linalg (lu, determinant, solve_linear, power_iteration, qr, cholesky)
- `examples/montecarlo_demo.py` — estimate_pi, mc_integrate, random_walk_1d/2d, buffon_needle
- `examples/numerical_integration_demo.py` — trapezoid/simpson/simpson_38/romberg/gaussian_quadrature/adaptive_simpson on ∫e^x
- `examples/probability_demo.py` — gaussian/uniform/exponential PDFs, binomial/poisson PMF, uniform_sample
- `examples/scientific_demo.py` — get_constant + kinetic_energy/gravitational_force/escape_velocity/etc.
- `docs/tutorials/core_demo.md`, `data_analysis_demo.md`, `diffeq_demo.md`, `graph_demo.md`, `math_utils_demo.md`, `montecarlo_demo.md`, `numerical_integration_demo.md`, `probability_demo.md`, `scientific_demo.md`

**Modified:** none (mkdocs.yml nav is Plan B's responsibility).

---

### Task 1: scientific_demo.py + tutorial

**Files:**
- Create: `examples/scientific_demo.py`
- Create: `docs/tutorials/scientific_demo.md`

- [ ] **Step 1: Write `examples/scientific_demo.py`**

```python
"""Scientific constants and physics formulas demo."""

from cds.scientific import (
    CONSTANTS,
    de_broglie_wavelength,
    escape_velocity,
    get_constant,
    gravitational_force,
    ideal_gas_pressure,
    kinetic_energy,
    photon_energy,
    schwarzschild_radius,
    wave_frequency,
)


def main() -> None:
    print("=== Physical Constants ===")
    c = get_constant("c")
    g = get_constant("g")
    print(f"Speed of light c = {c} m/s")
    print(f"Gravity g = {g} m/s^2")
    print(f"Available constants: {sorted(CONSTANTS)}")

    print("\n=== Mechanics ===")
    ke = kinetic_energy(mass=10.0, velocity=5.0)
    fg = gravitational_force(m1=5.0, m2=10.0, r=2.0)
    ev = escape_velocity(mass=5.972e24, radius=6.371e6)
    print(f"Kinetic energy (10kg @ 5m/s) = {ke:.2f} J")
    print(f"Gravitational force (5kg,10kg @ 2m) = {fg:.6e} N")
    print(f"Earth escape velocity = {ev:.1f} m/s")

    print("\n=== Waves & Photons ===")
    pe = photon_energy(frequency=5.0e14)
    wf = wave_frequency(wavelength=500e-9)
    dbw = de_broglie_wavelength(momentum=1.0e-24)
    print(f"Photon energy @ 5e14 Hz = {pe:.4e} J")
    print(f"Frequency of 500nm light = {wf:.4e} Hz")
    print(f"de Broglie wavelength (p=1e-24) = {dbw:.4e} m")

    print("\n=== Astrophysics & Thermo ===")
    rs = schwarzschild_radius(mass=1.989e30)
    pg = ideal_gas_pressure(n=1.0, temperature=300.0, volume=0.024)
    print(f"Sun Schwarzschild radius = {rs:.2f} m")
    print(f"Ideal gas pressure (n=1,T=300,V=0.024) = {pg:.2f} Pa")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example to verify it works**

Run: `python examples/scientific_demo.py`
Expected: prints four sections without error.

- [ ] **Step 3: Write `docs/tutorials/scientific_demo.md`**

```markdown
# 🧪 Scientific Constants & Formulas Tutorial

CDS bundles a curated set of physical constants and classical physics formulas — all pure Python.

## 1. Lookup a Constant

```python
from cds.scientific import get_constant, CONSTANTS

print(get_constant("c"))   # speed of light, m/s
print(sorted(CONSTANTS))   # all available names
```

## 2. Mechanics

```python
from cds.scientific import kinetic_energy, gravitational_force, escape_velocity

print(kinetic_energy(mass=10.0, velocity=5.0))            # Joules
print(gravitational_force(m1=5.0, m2=10.0, r=2.0))        # Newtons
print(escape_velocity(mass=5.972e24, radius=6.371e6))     # Earth, m/s
```

## 3. Waves, Photons & Relativity

```python
from cds.scientific import (
    photon_energy, wave_frequency, de_broglie_wavelength, schwarzschild_radius
)

print(photon_energy(frequency=5.0e14))           # green photon, J
print(wave_frequency(wavelength=500e-9))          # Hz
print(de_broglie_wavelength(momentum=1.0e-24))   # m
print(schwarzschild_radius(mass=1.989e30))       # Sun, m
```

## 4. Thermodynamics

```python
from cds.scientific import ideal_gas_pressure

print(ideal_gas_pressure(n=1.0, temperature=300.0, volume=0.024))  # Pascals
```
```

- [ ] **Step 4: Lint and format the example**

Run: `ruff check examples/scientific_demo.py && ruff format examples/scientific_demo.py`
Expected: clean / reformatted.

- [ ] **Step 5: Commit**

```bash
git add examples/scientific_demo.py docs/tutorials/scientific_demo.md
git commit -m "docs(examples): add scientific_demo + tutorial"
```

---

### Task 2: math_utils_demo.py + tutorial

**Files:**
- Create: `examples/math_utils_demo.py`
- Create: `docs/tutorials/math_utils_demo.md`

- [ ] **Step 1: Write `examples/math_utils_demo.py`**

```python
"""Calculus and linear algebra demo."""

import math

from cds.math_utils import (
    derivative,
    determinant,
    dot,
    gradient,
    integral,
    lu_decomposition,
    mat_mul,
    power_iteration,
    qr_decomposition,
    solve_linear,
    transpose,
)


def main() -> None:
    print("=== Calculus ===")
    d = derivative(lambda x: x**2, x=3.0)
    i = integral(lambda x: math.sin(x), a=0.0, b=math.pi)
    g = gradient(lambda v: v[0] ** 2 + v[1] ** 2, point=[1.0, 2.0])
    print(f"d/dx(x^2) @ x=3  = {d:.4f}  (expect 6)")
    print(f"∫_0^π sin(x) dx   = {i:.4f}  (expect 2)")
    print(f"grad(x^2+y^2)@[1,2] = {g}")

    print("\n=== Linear Algebra ===")
    A = [[2.0, 1.0], [1.0, 3.0]]
    B = [[1.0, 0.0], [0.0, 1.0]]
    print(f"mat_mul(A, I) = {mat_mul(A, B)}")
    print(f"transpose(A)  = {transpose(A)}")
    print(f"determinant(A)= {determinant(A):.4f}  (expect 5)")
    L, U, P = lu_decomposition(A)
    print(f"LU L = {L}")
    print(f"LU U = {U}")
    x = solve_linear(A, [3.0, 4.0])
    print(f"solve A·x=[3,4]: x = {x}")

    print("\n=== Decompositions & Spectral ===")
    Q, R = qr_decomposition(A)
    print(f"QR Q = {Q}")
    print(f"QR R = {R}")
    eigval, eigvec = power_iteration(A)
    print(f"power_iteration dominant eigenvalue = {eigval:.6f}")

    print("\n=== Vector ops ===")
    print(f"dot([1,2],[3,4]) = {dot([1.0, 2.0], [3.0, 4.0])}  (expect 11)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python examples/math_utils_demo.py`
Expected: prints all sections; numbers match the "expect" notes.

- [ ] **Step 3: Write `docs/tutorials/math_utils_demo.md`**

```markdown
# 📐 Math Utilities (Calculus & Linear Algebra) Tutorial

`cds.math_utils` provides pure-Python calculus primitives and a compact linear-algebra toolkit.

## 1. Calculus

```python
import math
from cds.math_utils import derivative, integral, gradient

print(derivative(lambda x: x**2, x=3.0))                # ≈ 6
print(integral(lambda x: math.sin(x), a=0.0, b=math.pi)) # ≈ 2
print(gradient(lambda v: v[0]**2 + v[1]**2, point=[1.0, 2.0]))  # [2, 4]
```

## 2. Linear Algebra Basics

```python
from cds.math_utils import mat_mul, transpose, determinant, solve_linear

A = [[2.0, 1.0], [1.0, 3.0]]
print(mat_mul(A, [[1,0],[0,1]]))   # back to A
print(transpose(A))                 # [[2,1],[1,3]]
print(determinant(A))               # 5.0
print(solve_linear(A, [3.0, 4.0]))  # solution of A·x = b
```

## 3. Decompositions

```python
from cds.math_utils import lu_decomposition, qr_decomposition, power_iteration

L, U, P = lu_decomposition(A)   # PLU factorisation
Q, R = qr_decomposition(A)      # QR factorisation
eigval, eigvec = power_iteration(A)  # dominant eigenpair via power iteration
```
```

- [ ] **Step 4: Lint and format**

Run: `ruff check examples/math_utils_demo.py && ruff format examples/math_utils_demo.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add examples/math_utils_demo.py docs/tutorials/math_utils_demo.md
git commit -m "docs(examples): add math_utils_demo + tutorial"
```

---

### Task 3: diffeq_demo.py + tutorial

**Files:**
- Create: `examples/diffeq_demo.py`
- Create: `docs/tutorials/diffeq_demo.md`

- [ ] **Step 1: Write `examples/diffeq_demo.py`**

```python
"""Ordinary differential equation solver demo."""

import math

from cds.diffeq import euler_method, midpoint_method, rk4, rk45, solve_system


def main() -> None:
    # dy/dt = -y  ->  exact solution y(t) = e^-t
    f = lambda t, y: -y
    t0, y0, t_end = 0.0, 1.0, 1.0

    print("=== Solving dy/dt = -y, y(0)=1, evaluate at t=1 (exact = e^-1 ≈ 0.3679) ===")
    sol_euler = euler_method(f, t0=t0, y0=y0, t_end=t_end, n=100)
    sol_mid = midpoint_method(f, t0=t0, y0=y0, t_end=t_end, n=100)
    sol_rk4 = rk4(f, t0=t0, y0=y0, t_end=t_end, n=10)
    sol_rk45 = rk45(f, t0=t0, y0=y0, t_end=t_end, rtol=1e-6)
    print(f"Euler   (n=100): {sol_euler.value:.6f}")
    print(f"Midpoint(n=100): {sol_mid.value:.6f}")
    print(f"RK4     (n=10) : {sol_rk4.value:.6f}")
    print(f"RK45 adaptive  : {sol_rk45.value:.6f}")

    print("\n=== System: predator-prey (Lotka-Volterra) ===")
    alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1

    def lotka(t: float, state: list[float]) -> list[float]:
        x, y = state
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]

    sys_sol = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=15.0, n=300)
    print(f"Final prey/predator state: {sys_sol.value}")
    print(f"Trajectory length: {len(sys_sol.ts)} points")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python examples/diffeq_demo.py`
Expected: RK45/RK4 close to 0.3679; system solution prints.

- [ ] **Step 3: Write `docs/tutorials/diffeq_demo.md`**

```markdown
# 🧬 Differential Equations Tutorial

`cds.diffeq` solves initial-value problems with classical schemes plus an adaptive RK45 integrator.

## 1. Single ODE

Solve `dy/dt = -y`, `y(0) = 1` whose exact value at `t=1` is `e^-1 ≈ 0.3679`:

```python
from cds.diffeq import euler_method, rk4, rk45

f = lambda t, y: -y
print(euler_method(f, t0=0.0, y0=1.0, t_end=1.0, n=100).value)
print(rk4(f, t0=0.0, y0=1.0, t_end=1.0, n=10).value)       # very accurate
print(rk45(f, t0=0.0, y0=1.0, t_end=1.0, rtol=1e-6).value) # adaptive
```

## 2. System of ODEs

```python
from cds.diffeq import solve_system

def lotka(t, state):
    x, y = state
    return [1.1*x - 0.4*x*y, 0.1*x*y - 0.4*y]   # Lotka-Volterra

sol = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=15.0, n=300)
print(sol.value)   # final [prey, predator]
print(len(sol.ts)) # trajectory points
```
```

- [ ] **Step 4: Lint and format**

Run: `ruff check examples/diffeq_demo.py && ruff format examples/diffeq_demo.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add examples/diffeq_demo.py docs/tutorials/diffeq_demo.md
git commit -m "docs(examples): add diffeq_demo + tutorial"
```

---

### Task 4: numerical_integration_demo.py + tutorial

**Files:**
- Create: `examples/numerical_integration_demo.py`
- Create: `docs/tutorials/numerical_integration_demo.md`

- [ ] **Step 1: Write `examples/numerical_integration_demo.py`**

```python
"""Deterministic quadrature demo — integrate e^x on [0,1]."""

import math

from cds.numerical_integration import (
    adaptive_simpson,
    gaussian_quadrature,
    romberg,
    simpson,
    simpson_38,
    trapezoid,
)


def main() -> None:
    f = math.exp
    exact = math.e - 1  # ≈ 1.718281828...

    print("=== ∫_0^1 e^x dx = e - 1 ≈ 1.718281828 ===\n")
    print(f"{'method':<28}{'value':>14}{'abs err':>14}")
    print("-" * 56)
    for name, v in [
        ("Trapezoid (n=1000)", trapezoid(f, 0, 1, 1000)),
        ("Simpson 1/3 (n=100)", simpson(f, 0, 1, 100)),
        ("Simpson 3/8 (n=99)", simpson_38(f, 0, 1, 99)),
        ("Gauss-Legendre (n=8)", gaussian_quadrature(f, 0, 1, 8)),
    ]:
        print(f"{name:<28}{v:>14.10f}{abs(v - exact):>14.2e}")

    r = romberg(f, 0, 1)
    print(f"{'Romberg (auto tol)':<28}{r.value:>14.10f}{abs(r.value - exact):>14.2e}")

    a = adaptive_simpson(f, 0, 1)
    print(f"{'Adaptive Simpson':<28}{a.value:>14.10f}{abs(a.value - exact):>14.2e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python examples/numerical_integration_demo.py`
Expected: errors shrink from trapezoid (~1e-7) to Gauss-Legendre (~1e-16).

- [ ] **Step 3: Write `docs/tutorials/numerical_integration_demo.md`**

```markdown
# 📈 Numerical Integration Tutorial

`cds.numerical_integration` provides deterministic quadrature rules. A canonical test: `∫_0^1 e^x dx = e - 1`.

## 1. Newton–Cotes Rules

```python
import math
from cds.numerical_integration import trapezoid, simpson, simpson_38

f = math.exp
print(trapezoid(f, 0, 1, 1000))  # O(h^2)
print(simpson(f, 0, 1, 100))      # O(h^4)
print(simpson_38(f, 0, 1, 99))    # 3/8 variant
```

## 2. Higher-Order & Adaptive

```python
from cds.numerical_integration import gaussian_quadrature, romberg, adaptive_simpson

print(gaussian_quadrature(f, 0, 1, 8).value)        # ~1e-16 error
print(romberg(f, 0, 1).value)                        # Richardson extrapolation
print(adaptive_simpson(f, 0, 1).value)               # tolerance-driven bisection
```

**Why it matters:** you can watch error drop from `1e-7` (trapezoid) to `1e-16` (Gauss-Legendre) on the same problem — a clear lesson in quadrature convergence.
```

- [ ] **Step 4: Lint and format**

Run: `ruff check examples/numerical_integration_demo.py && ruff format examples/numerical_integration_demo.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add examples/numerical_integration_demo.py docs/tutorials/numerical_integration_demo.md
git commit -m "docs(examples): add numerical_integration_demo + tutorial"
```

---

### Task 5: probability_demo.py + tutorial

**Files:**
- Create: `examples/probability_demo.py`
- Create: `docs/tutorials/probability_demo.md`

- [ ] **Step 1: Write `examples/probability_demo.py`**

```python
"""Probability distributions and sampling demo."""

from cds.probability import (
    binomial_pmf,
    exponential_pdf,
    gaussian_pdf,
    poisson_pmf,
    uniform_pdf,
    uniform_sample,
)


def main() -> None:
    print("=== Continuous PDFs ===")
    print(f"gaussian_pdf(0; mu=0, sigma=1) = {gaussian_pdf(0.0, mu=0.0, sigma=1.0):.6f}")
    print(f"  (1/sqrt(2π) ≈ {1.0 / (2 * 3.141592653589793) ** 0.5:.6f})")
    print(f"uniform_pdf(0.5; a=0, b=1) = {uniform_pdf(0.5, a=0.0, b=1.0):.6f}  (expect 1)")
    print(f"exponential_pdf(1; lambda=2) = {exponential_pdf(1.0, lambda_=2.0):.6f}")

    print("\n=== Discrete PMFs ===")
    print("Binomial PMF (n=10, p=0.5):")
    for k in range(0, 11):
        print(f"  P(X={k}) = {binomial_pmf(k, n=10, p=0.5):.4f}")
    print(f"Poisson PMF (lambda=3):")
    for k in range(0, 6):
        print(f"  P(X={k}) = {poisson_pmf(k, lambda_=3.0):.4f}")

    print("\n=== Sampling ===")
    samples = uniform_sample(low=0.0, high=1.0, n=5, seed=42)
    print(f"5 samples from Uniform(0,1), seed=42: {samples}")
    mean = sum(samples) / len(samples)
    print(f"sample mean = {mean:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python examples/probability_demo.py`
Expected: gaussian peak ≈ 0.398955; binomial symmetric around k=5.

- [ ] **Step 3: Write `docs/tutorials/probability_demo.md`**

```markdown
# 🎲 Probability Distributions Tutorial

`cds.probability` covers common continuous PDFs and discrete PMFs, plus reproducible sampling.

## 1. Continuous PDFs

```python
from cds.probability import gaussian_pdf, uniform_pdf, exponential_pdf

print(gaussian_pdf(0.0, mu=0.0, sigma=1.0))   # peak ≈ 0.399
print(uniform_pdf(0.5, a=0.0, b=1.0))          # 1.0 on support
print(exponential_pdf(1.0, lambda_=2.0))
```

## 2. Discrete PMFs

```python
from cds.probability import binomial_pmf, poisson_pmf

for k in range(11):
    print(k, binomial_pmf(k, n=10, p=0.5))   # symmetric around 5

for k in range(6):
    print(k, poisson_pmf(k, lambda_=3.0))
```

## 3. Reproducible Sampling

```python
from cds.probability import uniform_sample

print(uniform_sample(low=0.0, high=1.0, n=5, seed=42))  # deterministic
```
```

- [ ] **Step 4: Lint and format**

Run: `ruff check examples/probability_demo.py && ruff format examples/probability_demo.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add examples/probability_demo.py docs/tutorials/probability_demo.md
git commit -m "docs(examples): add probability_demo + tutorial"
```

---

### Task 6: montecarlo_demo.py + tutorial

**Files:**
- Create: `examples/montecarlo_demo.py`
- Create: `docs/tutorials/montecarlo_demo.md`

- [ ] **Step 1: Write `examples/montecarlo_demo.py`**

```python
"""Monte Carlo methods demo — pi estimation, integration, random walks, Buffon."""

import math

from cds.montecarlo import (
    buffon_needle,
    estimate_pi,
    mc_integrate,
    random_walk_1d,
    random_walk_2d,
)


def main() -> None:
    print("=== π Estimation ===")
    result = estimate_pi(samples=100_000, seed=42)
    print(f"estimate = {result.estimate:.6f}")
    print(f"true π  = {math.pi:.6f}")
    print(f"error   = {abs(result.estimate - math.pi):.5f}")

    print("\n=== Monte Carlo Integration: ∫_0^1 x^2 dx = 1/3 ===")
    integral = mc_integrate(lambda x: x**2, a=0.0, b=1.0, samples=50_000, seed=7)
    print(f"estimate = {integral.value:.6f}  (true = 1/3 ≈ 0.333333)")

    print("\n=== 1D Random Walk ===")
    w1 = random_walk_1d(steps=100, seed=1)
    print(f"final position after 100 steps (seed=1): {w1.position:.4f}")

    print("\n=== 2D Random Walk ===")
    w2 = random_walk_2d(steps=100, seed=2)
    print(f"final (x, y) after 100 steps (seed=2): ({w2.x:.4f}, {w2.y:.4f})")

    print("\n=== Buffon's Needle ===")
    bn = buffon_needle(drops=10_000, needle_length=0.5, line_spacing=1.0, seed=3)
    print(f"π via Buffon = {bn.pi_estimate:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python examples/montecarlo_demo.py`
Expected: π estimate within ~0.02 of true; integral within ~0.01 of 1/3.

- [ ] **Step 3: Write `docs/tutorials/montecarlo_demo.md`**

```markdown
# 🎰 Monte Carlo Methods Tutorial

`cds.montecarlo` estimates integrals, π, and random walks by sampling.

## 1. Estimating π

```python
import math
from cds.montecarlo import estimate_pi

r = estimate_pi(samples=100_000, seed=42)
print(r.estimate, abs(r.estimate - math.pi))
```

## 2. Integrating by Sampling

```python
from cds.montecarlo import mc_integrate

# ∫_0^1 x^2 dx = 1/3
print(mc_integrate(lambda x: x**2, a=0.0, b=1.0, samples=50_000, seed=7).value)
```

## 3. Random Walks

```python
from cds.montecarlo import random_walk_1d, random_walk_2d

print(random_walk_1d(steps=100, seed=1).position)
print(random_walk_2d(steps=100, seed=2).x, random_walk_2d(steps=100, seed=2).y)
```

## 4. Buffon's Needle

```python
from cds.montecarlo import buffon_needle

print(buffon_needle(drops=10_000, needle_length=0.5, line_spacing=1.0, seed=3).pi_estimate)
```
```

- [ ] **Step 4: Lint and format**

Run: `ruff check examples/montecarlo_demo.py && ruff format examples/montecarlo_demo.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add examples/montecarlo_demo.py docs/tutorials/montecarlo_demo.md
git commit -m "docs(examples): add montecarlo_demo + tutorial"
```

---

### Task 7: graph_demo.py + tutorial

**Files:**
- Create: `examples/graph_demo.py`
- Create: `docs/tutorials/graph_demo.md`

- [ ] **Step 1: Write `examples/graph_demo.py`**

```python
"""Graph algorithms demo — BFS/DFS, Dijkstra, Kruskal MST, topo sort, cycle detection."""

from cds.graph import (
    Graph,
    bfs,
    dfs,
    dijkstra,
    has_cycle,
    kruskal_mst,
    topological_sort,
)


def main() -> None:
    g = Graph(directed=False)
    edges = [
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)

    print("=== Traversal ===")
    print(f"BFS from A: {bfs(g, 'A')}")
    print(f"DFS from A: {dfs(g, 'A')}")

    print("\n=== Shortest Paths (Dijkstra) ===")
    dists = dijkstra(g, "A")
    for node, d in sorted(dists.items()):
        print(f"  dist(A -> {node}) = {d}")

    print("\n=== Minimum Spanning Tree (Kruskal) ===")
    mst = kruskal_mst(g)
    total = 0
    for u, v, w in mst:
        print(f"  {u} -- {v}  (w={w})")
        total += w
    print(f"  MST total weight = {total}")

    print("\n=== Cycle Detection (undirected) ===")
    print(f"has_cycle: {has_cycle(g)}")

    print("\n=== Topological Sort (DAG) ===")
    dag = Graph(directed=True)
    for u, v in [("course1", "course2"), ("course2", "course3"), ("course1", "course3")]:
        dag.add_edge(u, v)
    print(f"order: {topological_sort(dag)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python examples/graph_demo.py`
Expected: BFS/DFS lists, Dijkstra distances, MST edges with total weight.

- [ ] **Step 3: Write `docs/tutorials/graph_demo.md`**

```markdown
# 🕸️ Graph Algorithms Tutorial

`cds.graph` implements traversal, shortest paths, MST, topo sort, and cycle detection on a small `Graph` class.

## 1. Build a Graph

```python
from cds.graph import Graph

g = Graph(directed=False)
for u, v, w in [("A","B",4), ("A","C",2), ("B","C",1), ("B","D",5), ("C","D",8)]:
    g.add_edge(u, v, weight=w)
```

## 2. Traversal

```python
from cds.graph import bfs, dfs

print(bfs(g, "A"))
print(dfs(g, "A"))
```

## 3. Shortest Paths & MST

```python
from cds.graph import dijkstra, kruskal_mst

print(dijkstra(g, "A"))             # {node: distance}
print(kruskal_mst(g))               # list of (u, v, weight)
```

## 4. Topological Sort & Cycles

```python
from cds.graph import Graph, topological_sort, has_cycle

dag = Graph(directed=True)
dag.add_edge("a", "b")
dag.add_edge("b", "c")
print(topological_sort(dag))  # valid ordering
print(has_cycle(dag))         # False
```
```

- [ ] **Step 4: Lint and format**

Run: `ruff check examples/graph_demo.py && ruff format examples/graph_demo.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add examples/graph_demo.py docs/tutorials/graph_demo.md
git commit -m "docs(examples): add graph_demo + tutorial"
```

---

### Task 8: data_analysis_demo.py + tutorial

**Files:**
- Create: `examples/data_analysis_demo.py`
- Create: `docs/tutorials/data_analysis_demo.md`

- [ ] **Step 1: Write `examples/data_analysis_demo.py`**

```python
"""Data analysis demo — loading, transforming, and plotting tabular data."""

import os
import tempfile

from cds.data_analysis import (
    DataSet,
    DataTable,
    load_csv,
    moving_average,
    normalize,
    plot_bar,
    plot_line,
    z_score,
)


def main() -> None:
    # Build a tiny CSV in a temp file so the demo needs no fixtures.
    csv_text = "day,temperature,sales\n1,18.0,100\n2,19.5,110\n3,22.0,140\n4,21.5,135\n5,23.0,160\n"
    path = os.path.join(tempfile.gettempdir(), "_cds_demo.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv_text)

    print("=== Loading CSV ===")
    table = load_csv(path)
    print(f"columns: {table.columns}")
    print(f"rows: {len(table.rows)}")

    temps = [row[1] for row in table.rows]
    sales = [row[2] for row in table.rows]

    print("\n=== Normalisation ===")
    norm = normalize(temps)
    z = z_score(temps)
    print(f"normalized temps: {[round(v, 3) for v in norm]}")
    print(f"z-score temps   : {[round(v, 3) for v in z]}")

    print("\n=== Smoothing ===")
    ma = moving_average(sales, window=2)
    print(f"2-day moving avg of sales: {[round(v, 3) for v in ma]}")

    print("\n=== ASCII Visualisation ===")
    plot_line(temps, title="temperature")
    plot_bar(sales, title="sales")

    print("\n=== DataSet helper ===")
    ds = DataSet(name="week", observations=temps)
    print(f"{ds.name}: {len(ds.observations)} observations")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python examples/data_analysis_demo.py`
Expected: columns/rows printed, normalized values in [0,1], ASCII chart rendered.

- [ ] **Step 3: Write `docs/tutorials/data_analysis_demo.md`**

```markdown
# 📊 Data Analysis Tutorial

`cds.data_analysis` loads small CSVs, normalises and smooths series, and renders ASCII plots.

## 1. Load a CSV

```python
from cds.data_analysis import load_csv

table = load_csv("data.csv")
print(table.columns, len(table.rows))
```

## 2. Normalise & Smooth

```python
from cds.data_analysis import normalize, z_score, moving_average

temps = [18.0, 19.5, 22.0, 21.5, 23.0]
print(normalize(temps))           # min-max to [0,1]
print(z_score(temps))             # zero mean, unit variance
print(moving_average(temps, window=2))
```

## 3. ASCII Plots

```python
from cds.data_analysis import plot_line, plot_bar

plot_line(temps, title="temperature")
plot_bar([100, 110, 140, 135, 160], title="sales")
```
```

- [ ] **Step 4: Lint and format**

Run: `ruff check examples/data_analysis_demo.py && ruff format examples/data_analysis_demo.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add examples/data_analysis_demo.py docs/tutorials/data_analysis_demo.md
git commit -m "docs(examples): add data_analysis_demo + tutorial"
```

---

### Task 9: core_demo.py + tutorial

**Files:**
- Create: `examples/core_demo.py`
- Create: `docs/tutorials/core_demo.md`

- [ ] **Step 1: Write `examples/core_demo.py`**

```python
"""Core data-models demo — Domain, Hypothesis, HypothesisStatus."""

from cds.core import Domain, Hypothesis, HypothesisStatus


def main() -> None:
    print("=== Domains ===")
    for d in Domain:
        print(f"  {d.name} = {d.value}")

    print("\n=== Constructing a Hypothesis ===")
    h = Hypothesis(
        id="H-001",
        domain=Domain.COSMOLOGY,
        research_question="Why is the Hubble expansion accelerating?",
        statement="A cosmological constant (Λ) drives late-time acceleration.",
        status=HypothesisStatus.PROPOSED,
        confidence=0.62,
        rationale="Type Ia supernova data favours accelerated expansion.",
        assumptions=["GR holds on cosmological scales", "Λ > 0"],
        predictions=["(m-M) vs z curve bends downward vs matter-only"],
        tags=["dark-energy", "lambda"],
    )
    print(repr(h))
    print("\n--- Markdown render ---")
    print(h.to_markdown())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the public API matches**

Run: `python -c "from cds.core import Domain, Hypothesis, HypothesisStatus; h=Hypothesis(id='x', domain=Domain.PHYSICS, research_question='q', statement='s', status=HypothesisStatus.PROPOSED, confidence=0.5); print(hasattr(h, 'to_markdown'))"`

If `to_markdown` is missing or any kwarg name differs, **stop** and read `src/cds/core/models.py` to confirm the real constructor signature, then adapt Step 1 accordingly.

Expected: `True`.

- [ ] **Step 3: Run the example**

Run: `python examples/core_demo.py`
Expected: lists domains, prints repr + markdown.

- [ ] **Step 4: Write `docs/tutorials/core_demo.md`**

```markdown
# 🧱 Core Data Models Tutorial

`cds.core` defines the `Domain`, `Hypothesis`, and `HypothesisStatus` types used across CDS — particularly by the hypothesis engine.

## 1. Domains

```python
from cds.core import Domain

for d in Domain:
    print(d.name, d.value)
```

## 2. Building a Hypothesis

```python
from cds.core import Domain, Hypothesis, HypothesisStatus

h = Hypothesis(
    id="H-001",
    domain=Domain.COSMOLOGY,
    research_question="Why is the Hubble expansion accelerating?",
    statement="A cosmological constant (Λ) drives late-time acceleration.",
    status=HypothesisStatus.PROPOSED,
    confidence=0.62,
    assumptions=["GR holds on cosmological scales"],
    predictions=["(m-M) vs z bends downward"],
    tags=["dark-energy"],
)
print(h.to_markdown())
```

The same record type is what `cds.hypothesis.generate_hypotheses` returns — so you can inspect, persist, or feed it to the evaluator.
```

- [ ] **Step 5: Lint and format**

Run: `ruff check examples/core_demo.py && ruff format examples/core_demo.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add examples/core_demo.py docs/tutorials/core_demo.md
git commit -m "docs(examples): add core_demo + tutorial"
```

---

### Task 10: Smoke-test every example end-to-end

**Files:** none new.

- [ ] **Step 1: Run every example in a loop**

Run (bash, in repo root):
```bash
for f in examples/*_demo.py; do
  echo "=== $f ==="
  python "$f" > /dev/null && echo "OK" || echo "FAIL"
done
```

Expected: every line prints `OK`. If any fails, fix that example before proceeding.

- [ ] **Step 2: Verify lint across the examples directory**

Run: `ruff check examples/ && ruff format --check examples/`
Expected: clean.

- [ ] **Step 3: Verify coverage did not regress**

Run: `python -m pytest tests/ --cov=cds --cov-branch -q --no-header 2>&1 | tail -3`
Expected: `Total coverage: 98.59%` or higher (examples don't affect coverage; this is a safety net).

- [ ] **Step 4: Commit if anything was adjusted during smoke-test fixes**

If no fixes were needed, skip. Otherwise:
```bash
git add examples/ docs/tutorials/
git commit -m "docs(examples): smoke-test fixes"
```
