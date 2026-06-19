# Monte Carlo Methods Tutorial

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
w2 = random_walk_2d(steps=100, seed=2)
print(w2.x, w2.y)
```

## 4. Buffon's Needle

```python
from cds.montecarlo import buffon_needle

print(buffon_needle(drops=10_000, needle_length=0.5, line_spacing=1.0, seed=3).pi_estimate)
```

Run the full demo with `python examples/montecarlo_demo.py`.
