from cds.montecarlo import estimate_pi, mc_integrate
import math

result = estimate_pi(n_samples=100_000, seed=42)
print(f"π ≈ {result.estimate:.4f}")  # ~3.1416

area = mc_integrate(math.sin, 0, math.pi, n_samples=100_000)
print(f"∫sin(x)dx = {area.estimate:.4f}")  # ~2.0
