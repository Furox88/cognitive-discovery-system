# Getting Started

## Installation

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick Usage

### CLI

```bash
# Show available commands
cds --help

# List physical constants
cds constants

# Interactive physics calculator
cds calc ke    # kinetic energy
cds calc gravity
cds calc wave
cds calc gas

# Generate hypotheses
cds hypothesize "what causes the Hubble tension?" --domain cosmology
```

### Python API

```python
# Quantum simulation
from cds.quantum import bell_state, is_entangled, ghz_state
reg = bell_state(0)
print(is_entangled(reg))  # True

# Optimization
from cds.optimization import gradient_descent
result = gradient_descent(lambda x: (x - 3) ** 2, x0=10.0, lr=0.1)
print(result.x)  # ~3.0

# Signal processing
from cds.signals import fft_radix2
spectrum = fft_radix2([complex(i) for i in range(8)])

# Probability
from cds.probability import gaussian_pdf, binomial_pmf
print(gaussian_pdf(0.0))  # 0.3989...
print(binomial_pmf(3, 5, 0.5))  # 0.3125

# Statistics
from cds.stats import mean, linear_regression
print(mean([1, 2, 3, 4, 5]))  # 3.0

# Scientific computing
from cds.scientific import kinetic_energy, get_constant
print(get_constant("c"))  # 299792458.0
print(kinetic_energy(10, 5))  # 125.0
```

## Running Tests

```bash
pytest           # run all 162 tests
pytest -v        # verbose output
pytest -x        # stop on first failure
```

## Running Examples

```bash
python examples/quantum_demo.py
python examples/optimization_demo.py
python examples/signals_demo.py
python examples/stats_demo.py
```

## Project Structure

```
src/cds/
├── quantum/        # Quantum circuit simulation (single & multi-qubit)
├── optimization/   # Gradient descent, Newton, Adam, line search
├── signals/        # DFT, FFT, convolution, filtering
├── probability/    # Probability distributions & sampling
├── stats/          # Statistical analysis & regression
├── math_utils/     # Numerical calculus & linear algebra
├── data_analysis/  # CSV loading & data transforms
├── scientific/     # Physical constants & formulas
├── hypothesis/     # Hypothesis generation
├── core/           # Shared models, config
└── cli.py          # Command-line interface

examples/           # Runnable demo scripts
tests/              # 162 tests (pytest)
```
