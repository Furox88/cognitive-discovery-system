# Mathematical Modeling Examples

This document provides detailed examples of using the `cds.modeling` module for scientific and engineering calculations.

## Expression Building

The modeling module uses an expression tree. Every expression is composed of:

- `Constant(value)` — a numeric literal
- `Variable(name)` — a named variable
- `BinaryOp(op, left, right)` — binary operations: `+`, `-`, `*`, `/`, `^`
- `UnaryFunc(func, arg)` — functions: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`

## Example 1: Projectile Motion

```python
from cds.modeling import Variable, Constant, BinaryOp, UnaryFunc
from cds.modeling import differentiate_numerically, integrate_numerically
import math

t = Variable("t")
v0 = Constant(20.0)     # initial velocity (m/s)
theta = Constant(math.pi / 4)  # 45 degrees
g = Constant(9.81)

# Horizontal position: x(t) = v₀ · cos(θ) · t
x_pos = BinaryOp("*", BinaryOp("*", v0, UnaryFunc("cos", theta)), t)

# Vertical position: y(t) = v₀ · sin(θ) · t - ½gt²
y_pos = BinaryOp("-",
    BinaryOp("*", BinaryOp("*", v0, UnaryFunc("sin", theta)), t),
    BinaryOp("*", Constant(0.5), BinaryOp("*", g, BinaryOp("^", t, Constant(2)))),
)

# Position at t = 1 second
print(f"x(1) = {x_pos.evaluate({'t': 1.0}):.2f} m")   # 14.14 m
print(f"y(1) = {y_pos.evaluate({'t': 1.0}):.2f} m")    # 9.24 m

# Velocity components (derivatives)
vx = differentiate_numerically(x_pos, "t", {"t": 1.0})
vy = differentiate_numerically(y_pos, "t", {"t": 1.0})
print(f"vx(1) = {vx:.2f} m/s")  # 14.14 m/s (constant)
print(f"vy(1) = {vy:.2f} m/s")  # 4.33 m/s (decreasing)
```

## Example 2: Simple Harmonic Motion

```python
# x(t) = A · cos(ωt + φ)
A = Constant(5.0)       # amplitude
omega = Constant(2.0)   # angular frequency
phi = Constant(0.0)     # phase

x_shm = BinaryOp("*", A,
    UnaryFunc("cos",
        BinaryOp("+",
            BinaryOp("*", omega, t),
            phi
        )
    )
)

# Position at t = π/4
pos = x_shm.evaluate({"t": math.pi / 4})
print(f"x(π/4) = {pos:.4f}")  # ~0.0 (cos(π/2) ≈ 0)

# Velocity = dx/dt
vel = differentiate_numerically(x_shm, "t", {"t": 0.0})
print(f"v(0) = {vel:.4f}")  # 0.0 (at max displacement, velocity is zero)
```

## Example 3: Numerical Integration — Work Done

```python
# Work = ∫ F(x) dx
# Force: F(x) = 3x² + 2x (position-dependent force)
x = Variable("x")
force = BinaryOp("+",
    BinaryOp("*", Constant(3), BinaryOp("^", x, Constant(2))),
    BinaryOp("*", Constant(2), x),
)

# Work done from x=0 to x=4
work = integrate_numerically(force, "x", 0, 4, {})
print(f"Work = {work:.2f} J")  # Analytical: 3(64/3) + 2(8) = 64 + 16 = 80 J
```

## Example 4: Radioactive Decay

```python
# N(t) = N₀ · e^(-λt)
t = Variable("t")
N0 = Constant(10000)     # initial atoms
lam = Constant(0.693)    # decay constant (half-life = ln2/λ = 1 year)

decay = BinaryOp("*", N0,
    UnaryFunc("exp",
        BinaryOp("*", Constant(-1), BinaryOp("*", Constant(0.693), t))
    )
)

# After 1 half-life
print(f"N(1) = {decay.evaluate({'t': 1.0}):.0f}")   # ~5000
print(f"N(2) = {decay.evaluate({'t': 2.0}):.0f}")   # ~2500
print(f"N(3) = {decay.evaluate({'t': 3.0}):.0f}")   # ~1250

# Decay rate = -dN/dt
rate = -differentiate_numerically(decay, "t", {"t": 1.0})
print(f"Decay rate at t=1: {rate:.0f} atoms/year")
```

## Example 5: Electric Circuit — RC Charging

```python
# Voltage across capacitor: V(t) = V₀(1 - e^(-t/RC))
t = Variable("t")
V0 = Constant(12.0)    # source voltage
RC = Constant(2.0)     # time constant (R·C)

v_cap = BinaryOp("*", V0,
    BinaryOp("-", Constant(1),
        UnaryFunc("exp",
            BinaryOp("/", BinaryOp("*", Constant(-1), t), RC)
        )
    )
)

# Voltage at different times
for time in [0.5, 1.0, 2.0, 5.0]:
    v = v_cap.evaluate({"t": time})
    print(f"V({time}s) = {v:.2f} V")

# Charging current ∝ dV/dt
dv_dt = differentiate_numerically(v_cap, "t", {"t": 1.0})
print(f"dV/dt at t=1s = {dv_dt:.4f} V/s")
```

## Supported Operations

| Type | Symbols/Functions |
|------|-------------------|
| Binary operators | `+`, `-`, `*`, `/`, `^` |
| Unary functions | `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs` |
| Differentiation | `differentiate_numerically(expr, var, env, h=1e-8)` |
| Integration | `integrate_numerically(expr, var, a, b, env, n=1000)` |
