"""Mathematical modeling demo — symbolic expressions, models, solvers, and fitting.

A runnable companion to ``docs/tutorials/modeling_demo.md``. Exercises every
public entry point of ``cds.modeling``: building and evaluating expressions,
symbolic differentiation, simplification, LaTeX export, grouping equations into
a :class:`~cds.modeling.MathModel`, root finding, and least-squares fitting.
"""

from cds.modeling import (
    Constant,
    Exp,
    Log,
    MathModel,
    Sin,
    Variable,
    fit_parameters,
    solve_equation,
)


def main() -> None:
    # --- 1. Build & evaluate an expression ---
    print("=== Build & Evaluate ===")
    x = Variable("x")
    expr = x**2 + 3 * x  # x^2 + 3x
    print(f"expr.evaluate({{x: 2}}) = {expr.evaluate({'x': 2})}")  # 10.0
    print(f"expr.to_str() = {expr.to_str()}")  # ((x ** 2) + (3 * x))

    # --- 2. Symbolic differentiation ---
    print("\n=== Symbolic Differentiation ===")
    f = Sin(x) * Exp(x)  # sin(x) * e^x
    df = f.diff("x")
    print(f"f.to_str()  = {f.to_str()}")
    print(f"df.to_str() = {df.to_str()}")
    # Analytically e^0 * (sin 0 + cos 0) = 1.0
    print(f"df.evaluate({{x: 0.0}}) = {df.evaluate({'x': 0.0})}")  # 1.0

    # --- 3. Simplify & export to LaTeX ---
    print("\n=== Simplify & LaTeX ===")
    print(f"(x + 0).simplify() = {(x + 0).simplify().to_str()}")  # x
    print(f"(x * 1).simplify() = {(x * 1).simplify().to_str()}")  # x
    print(f"(x ** 0).simplify() = {(x ** 0).simplify().to_str()}")  # 1.0
    expr = (Variable("x") ** Constant(2.0)) / Variable("y")
    print(f"expr.to_latex() = {expr.to_latex()}")  # \frac{x^{2}}{y}

    # --- 4. Group equations into a MathModel ---
    # Standard constant-acceleration kinematics:
    #   velocity: v(t) = v0 + a*t
    #   position: x(t) = x0 + v0*t + 0.5*a*t^2
    print("\n=== MathModel ===")
    t, a = Variable("t"), Variable("a")
    model = MathModel(
        name="Kinematics",
        description="Constant-acceleration motion",
        parameters={"a": 9.81},
        variables=["t", "v0", "x0"],
    )
    v0 = Variable("v0")
    x0 = Variable("x0")
    model.add_equation("velocity", v0 + a * t)
    model.add_equation("position", x0 + v0 * t + Constant(0.5) * a * t * t)
    env = {"t": 2.0, "v0": 0.0, "x0": 0.0}
    print(f"model.evaluate({env}) = {model.evaluate(env)}")
    # {'velocity': 19.62, 'position': 19.62}
    print("Jacobian column w.r.t. t:")
    for label, dexpr in model.jacobian("t").items():
        print(f"  d({label})/dt = {dexpr.to_str()}")

    # --- 5. Solve x^2 - 2 = 0  =>  x = sqrt(2) ---
    print("\n=== Solve Equation ===")
    result = solve_equation(x**2 - Constant(2.0), variable="x", x0=1.0)
    print(f"root of x^2 - 2 = {result.x:.6f} (sqrt(2) = {2**0.5:.6f})")
    print(f"converged={result.converged}, residual={result.residual:.2e}")

    # --- 6. Fit parameters to observed data ---
    # Model: position(t) = v * t (linear with zero intercept). We fix v as a
    # parameter and recover it from synthetic noisy-ish observations.
    print("\n=== Fit Parameters (least squares) ===")
    fit_model = MathModel(
        name="LinearMotion",
        description="position = v * t",
        parameters={"v": 0.0},
        variables=["t"],
    )
    fit_model.add_equation("position", Variable("v") * Variable("t"))
    observed = [
        ({"t": 0.0}, 0.0),
        ({"t": 1.0}, 3.0),
        ({"t": 2.0}, 6.0),
        ({"t": 3.0}, 9.0),
        ({"t": 4.0}, 12.0),
    ]
    fit = fit_parameters(
        fit_model,
        observed=observed,
        parameter_names=["v"],
        x0=[0.1],
        target_label="position",
    )
    print(f"fitted v = {fit.parameters['v']:.6f} (expected 3.0)")
    print(f"residual sum of squares = {fit.residual:.3e}, converged={fit.converged}")

    # Show Log is wired in too (kept light — just evaluate at a sanity point).
    print("\n=== Misc (Log) ===")
    y = Variable("y")
    print(f"Log(y).evaluate({{y: 1.0}}) = {Log(y).evaluate({'y': 1.0})}")  # 0.0


if __name__ == "__main__":
    main()
