# Cognitive Discovery System (CDS)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)

**Open-source research assistant for scientific discovery, structured reasoning, and computational science.**

CDS provides tools for hypothesis generation, quantum circuit simulation, statistical analysis, numerical math, and scientific computing — all in one Python package.

> Currently in **alpha (v0.1.0)**. Contributions welcome!

## Modules

| Module | Description |
|--------|-------------|
| `cds.quantum` | Single & multi-qubit simulation — Hadamard, Pauli gates, CNOT, SWAP, Toffoli, Bell states, GHZ, entanglement detection |
| `cds.stats` | Descriptive stats, correlation, linear regression |
| `cds.math_utils` | Numerical derivatives, integrals (Simpson's rule), matrix ops, determinants |
| `cds.data_analysis` | CSV loader, normalization, z-score, moving average |
| `cds.scientific` | Physical constants, formulas (KE, gravity, gas law, Schwarzschild radius, de Broglie) |
| `cds.hypothesis` | AI-assisted hypothesis generation (LLM-ready prompt templates) |

## Quick Start

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run CLI
cds --help
cds constants
cds calc ke

# Run tests
pytest
```

## Examples

### Quantum Circuit (single qubit)
```python
from cds.quantum import QuantumCircuit, hadamard, pauli_x, simulate

circuit = QuantumCircuit().add(hadamard()).add(pauli_x())
result = circuit.run()
print(result.probabilities())

counts = simulate(circuit, shots=1000)
print(counts)  # {0: ~500, 1: ~500}
```

### Multi-Qubit & Entanglement
```python
from cds.quantum import (
    QuantumRegister, h_gate, cnot, bell_state,
    ghz_state, is_entangled,
)

# Create a Bell state (|00> + |11>) / sqrt(2)
reg = bell_state(0)
print(is_entangled(reg))  # True
print(reg.measure_shots(shots=1000))  # {'00': ~500, '11': ~500}

# 3-qubit GHZ state
ghz = ghz_state(3)
print(ghz.probabilities())  # [0.5, 0, 0, 0, 0, 0, 0, 0.5]
```

### Statistics
```python
from cds.stats import mean, stdev, correlation, linear_regression

data = [12.5, 14.3, 11.8, 15.1, 13.7]
print(f"mean={mean(data):.2f}, std={stdev(data):.2f}")

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]
reg = linear_regression(x, y)
print(f"slope={reg.slope:.2f}, R²={reg.r_squared:.3f}")
```

### Numerical Calculus
```python
from cds.math_utils import derivative, integral
import math

# d/dx(sin(x)) at x=0
print(derivative(math.sin, 0))  # ~1.0

# integral of x^2 from 0 to 3
print(integral(lambda x: x**2, 0, 3))  # ~9.0
```

### Scientific Computing
```python
from cds.scientific import kinetic_energy, gravitational_force, get_constant

print(get_constant("c"))  # speed of light
print(kinetic_energy(mass=10, velocity=5))  # 125 J
print(gravitational_force(5.972e24, 70, 6.371e6))  # ~686 N (your weight)
```

## Architecture

```
src/cds/
├── quantum/        # Quantum circuit simulation
├── stats/          # Statistical analysis
├── math_utils/     # Numerical calculus & linear algebra
├── data_analysis/  # CSV loading & data transforms
├── scientific/     # Physical constants & formulas
├── hypothesis/     # Hypothesis generation
├── core/           # Shared models, config
├── agents/         # LLM agent abstractions (planned)
├── knowledge/      # Notes, concepts (planned)
└── cli.py          # Command-line interface
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

Looking for:
- Researchers with domain expertise
- LLM/agent engineers
- People building similar tools

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Maintainer: [@Furox88](https://github.com/Furox88)
- Issues & Discussions: [GitHub](https://github.com/Furox88/cognitive-discovery-system/issues)
