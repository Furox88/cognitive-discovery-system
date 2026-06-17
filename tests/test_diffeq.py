"""Tests for ODE solvers."""

import math

from cds.diffeq import euler_method, midpoint_method, rk4, rk45, solve_system


class TestEulerMethod:
    def test_exponential_decay(self) -> None:
        # dy/dt = -y, y(0) = 1  =>  y(t) = e^(-t)
        sol = euler_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.001)
        assert abs(sol.y[-1] - math.exp(-1)) < 0.01

    def test_linear_growth(self) -> None:
        # dy/dt = 1, y(0) = 0  =>  y(t) = t
        sol = euler_method(lambda t, y: 1.0, 0, 0.0, 2.0, dt=0.01)
        assert abs(sol.y[-1] - 2.0) < 0.02

    def test_method_name(self) -> None:
        sol = euler_method(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "euler"

    def test_initial_condition(self) -> None:
        sol = euler_method(lambda t, y: 0, 0, 5.0, 1.0, dt=0.1)
        assert sol.y[0] == 5.0
        assert sol.t[0] == 0


class TestRK4:
    def test_exponential_decay_high_accuracy(self) -> None:
        # dy/dt = -y, y(0) = 1  =>  y(t) = e^(-t)
        sol = rk4(lambda t, y: -y, 0, 1.0, 1.0, dt=0.01)
        assert abs(sol.y[-1] - math.exp(-1)) < 1e-8

    def test_sine_derivative(self) -> None:
        # dy/dt = cos(t), y(0) = 0  =>  y(t) = sin(t)
        sol = rk4(lambda t, y: math.cos(t), 0, 0.0, math.pi, dt=0.01)
        assert abs(sol.y[-1] - math.sin(math.pi)) < 1e-6

    def test_quadratic(self) -> None:
        # dy/dt = 2t, y(0) = 0  =>  y(t) = t²
        sol = rk4(lambda t, y: 2 * t, 0, 0.0, 3.0, dt=0.01)
        assert abs(sol.y[-1] - 9.0) < 1e-6

    def test_method_name(self) -> None:
        sol = rk4(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "rk4"

    def test_better_than_euler(self) -> None:
        def f(t: float, y: float) -> float:
            return -y

        sol_euler = euler_method(f, 0, 1.0, 1.0, dt=0.1)
        sol_rk4 = rk4(f, 0, 1.0, 1.0, dt=0.1)
        exact = math.exp(-1)
        assert abs(sol_rk4.y[-1] - exact) < abs(sol_euler.y[-1] - exact)


class TestMidpointMethod:
    def test_exponential_decay(self) -> None:
        sol = midpoint_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.01)
        assert abs(sol.y[-1] - math.exp(-1)) < 1e-4

    def test_method_name(self) -> None:
        sol = midpoint_method(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "midpoint"


class TestRK45:
    def test_exponential_decay_adaptive(self) -> None:
        # dy/dt = -y, y(0) = 1  =>  y(t) = e^(-t)
        sol = rk45(lambda t, y: -y, 0, 1.0, 2.0, dt=0.1, atol=1e-8, rtol=1e-8)
        assert abs(sol.y[-1] - math.exp(-2.0)) < 1e-6
        # adaptive steps should be fewer than fixed-step with same accuracy
        assert sol.steps < 200

    def test_method_name(self) -> None:
        sol = rk45(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "rk45"


class TestSolveSystem:
    def test_harmonic_oscillator(self) -> None:
        # x'' = -x  =>  [x, v]' = [v, -x]
        # x(0)=1, v(0)=0  =>  x(t) = cos(t)
        def f(t: float, y: list[float]) -> list[float]:
            return [y[1], -y[0]]

        t_vals, y_vals = solve_system(f, 0, [1.0, 0.0], math.pi, dt=0.001)
        # x(π) = cos(π) = -1
        assert abs(y_vals[-1][0] - (-1.0)) < 1e-4

    def test_coupled_decay(self) -> None:
        # y1' = -y1, y2' = -2*y2
        def f(t: float, y: list[float]) -> list[float]:
            return [-y[0], -2 * y[1]]

        t_vals, y_vals = solve_system(f, 0, [1.0, 1.0], 1.0, dt=0.01)
        assert abs(y_vals[-1][0] - math.exp(-1)) < 1e-6
        assert abs(y_vals[-1][1] - math.exp(-2)) < 1e-5

    def test_initial_conditions_preserved(self) -> None:
        t_vals, y_vals = solve_system(
            lambda t, y: [0, 0],
            0,
            [3.0, 7.0],
            1.0,
            dt=0.1,
        )
        assert y_vals[0] == [3.0, 7.0]
