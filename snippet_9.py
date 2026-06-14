from cds.diffeq import rk4, solve_system
import math

# dy/dt = -y, y(0)=1  =>  y(t) = e^(-t)
sol = rk4(lambda t, y: -y, t0=0, y0=1.0, t_end=2.0)
print(f"y(2) = {sol.y[-1]:.6f}")  # ~0.135335 (e^-2)

# Harmonic oscillator: x'' = -x
def harmonic(t, y):
    return [y[1], -y[0]]
t_vals, y_vals = solve_system(harmonic, 0, [1.0, 0.0], math.pi)
print(f"x(π) = {y_vals[-1][0]:.4f}")  # ~-1.0
