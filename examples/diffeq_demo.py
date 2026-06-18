"""Ordinary differential equation solver demo."""

from cds.diffeq import euler_method, midpoint_method, rk4, rk45, solve_system


def main() -> None:
    # dy/dt = -y  ->  exact solution y(t) = e^-t, so y(1) = e^-1 ≈ 0.3679
    f = lambda t, y: -y  # noqa: E731
    t0, y0, t_end = 0.0, 1.0, 1.0

    print("=== Solving dy/dt = -y, y(0)=1, evaluate at t=1 (exact = e^-1 ≈ 0.3679) ===")
    sol_euler = euler_method(f, t0=t0, y0=y0, t_end=t_end, dt=0.01)
    sol_mid = midpoint_method(f, t0=t0, y0=y0, t_end=t_end, dt=0.01)
    sol_rk4 = rk4(f, t0=t0, y0=y0, t_end=t_end, dt=0.1)
    sol_rk45 = rk45(f, t0=t0, y0=y0, t_end=t_end, rtol=1e-6)
    print(f"Euler   (dt=0.01): {sol_euler.y[-1]:.6f}")
    print(f"Midpoint(dt=0.01): {sol_mid.y[-1]:.6f}")
    print(f"RK4     (dt=0.1) : {sol_rk4.y[-1]:.6f}")
    print(f"RK45 adaptive    : {sol_rk45.y[-1]:.6f}")

    print("\n=== System: predator-prey (Lotka-Volterra) ===")
    alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1

    def lotka(t: float, state: list[float]) -> list[float]:
        x, y = state
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]

    ts, ys = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=15.0, dt=0.05)
    print(f"Final prey/predator state: {ys[-1]}")
    print(f"Trajectory length: {len(ts)} points")


if __name__ == "__main__":
    main()
