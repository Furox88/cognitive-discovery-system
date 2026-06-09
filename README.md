# Cognitive Discovery System (CDS)

[![CI](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Open-source research assistant for scientific discovery, mathematical modeling, and structured reasoning.**

CDS provides a Python toolkit for researchers who want to organize their scientific workflow programmatically — from generating and tracking hypotheses, through mathematical modeling and numerical analysis, to managing research notes and mapping concept relationships.

---

## Features

| Module | Description |
|--------|-------------|
| `cds.hypothesis` | Create, evaluate, and track scientific hypotheses through their lifecycle |
| `cds.modeling` | Symbolic math expressions with numerical differentiation and integration |
| `cds.notes` | Research note management with full-text search and cross-referencing |
| `cds.concept_map` | Directed graph for concept relationships with BFS pathfinding |
| `cds.cli` | Command-line interface for quick interactions |

## Architecture

```
cognitive-discovery-system/
├── src/cds/
│   ├── __init__.py          # Package metadata
│   ├── hypothesis.py        # Hypothesis generation & lifecycle
│   ├── modeling.py          # Mathematical expression engine
│   ├── notes.py             # Research note management
│   ├── concept_map.py       # Concept relationship graphs
│   └── cli.py               # Command-line interface
├── tests/                   # Full test suite (pytest + coverage)
├── docs/                    # Documentation
├── .github/workflows/       # CI pipeline
├── pyproject.toml           # Project config (PyPI-ready)
└── CONTRIBUTING.md          # Contributor guidelines
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[test]"
```

## Quick Start

### Hypothesis Management

```python
from cds.hypothesis import Hypothesis, HypothesisStore

# Create a hypothesis
h = Hypothesis(
    statement="The reaction rate doubles for every 10°C increase in temperature",
    rationale="Based on the Arrhenius equation and collision theory",
    variables=["temperature", "reaction_rate"],
    tags=["chemistry", "kinetics"],
    confidence=0.7,
)

# Move through the scientific lifecycle
h.propose()           # DRAFT → PROPOSED
h.start_testing()     # PROPOSED → TESTING
h.support(confidence=0.92)  # TESTING → SUPPORTED

print(h.summary())
# [supported] The reaction rate doubles for every 10°C increase... (confidence=0.92)

# Organize hypotheses in a store
store = HypothesisStore()
store.add(h)
kinetics = store.filter_by_tag("kinetics")
```

### Mathematical Modeling

Build symbolic expressions and perform numerical analysis — useful for physics and engineering workflows:

```python
from cds.modeling import Variable, Constant, BinaryOp, UnaryFunc
from cds.modeling import differentiate_numerically, integrate_numerically
import math

# Define variables
x = Variable("x")
t = Variable("t")

# Example 1: Quadratic expression  f(x) = x² + 2x + 1
expr = BinaryOp("+",
    BinaryOp("+",
        BinaryOp("^", x, Constant(2)),    # x²
        BinaryOp("*", Constant(2), x),    # 2x
    ),
    Constant(1),                            # + 1
)

# Evaluate at x = 3  →  9 + 6 + 1 = 16
result = expr.evaluate({"x": 3.0})
print(f"f(3) = {result}")  # f(3) = 16.0

# Example 2: Numerical differentiation
# d/dx(x²) = 2x  →  at x=5, derivative = 10
x_squared = BinaryOp("^", x, Constant(2))
derivative = differentiate_numerically(x_squared, "x", {"x": 5.0})
print(f"d/dx(x²) at x=5 = {derivative:.4f}")  # 10.0000

# Example 3: Trigonometric functions
# sin(x) + cos(x)
trig_expr = BinaryOp("+",
    UnaryFunc("sin", x),
    UnaryFunc("cos", x),
)
print(f"sin(π/4) + cos(π/4) = {trig_expr.evaluate({'x': math.pi/4}):.4f}")  # 1.4142

# Example 4: Numerical integration
# ∫₀^π sin(x) dx = 2.0
sin_x = UnaryFunc("sin", x)
area = integrate_numerically(sin_x, "x", 0, math.pi, {})
print(f"∫₀^π sin(x) dx = {area:.4f}")  # 2.0000

# Example 5: Physics — kinetic energy  E = ½mv²
m = Variable("m")
v = Variable("v")
kinetic_energy = BinaryOp("*",
    Constant(0.5),
    BinaryOp("*", m, BinaryOp("^", v, Constant(2))),
)
# E at m=10 kg, v=3 m/s  →  0.5 * 10 * 9 = 45 J
energy = kinetic_energy.evaluate({"m": 10.0, "v": 3.0})
print(f"Kinetic energy = {energy} J")  # 45.0 J

# Example 6: Exponential decay  N(t) = N₀ · e^(-λt)
N0 = Constant(1000)        # initial count
lam = Constant(0.1)        # decay constant
decay = BinaryOp("*", N0,
    UnaryFunc("exp",
        BinaryOp("*", Constant(-1), BinaryOp("*", lam, t))
    ),
)
# After t=5:  1000 * e^(-0.5) ≈ 606.53
print(f"N(5) = {decay.evaluate({'t': 5.0}):.2f}")  # 606.53
```

### Research Notes

```python
from cds.notes import Note, Notebook

notebook = Notebook()

# Create and organize notes
note1 = Note(
    title="Arrhenius Equation",
    body="k = A·exp(-Ea/RT) where k is the rate constant, A is the "
         "pre-exponential factor, Ea is activation energy, R is the gas "
         "constant, and T is temperature in Kelvin.",
    tags=["chemistry", "kinetics"],
)
note2 = Note(
    title="Collision Theory",
    body="Reaction rate depends on the frequency and energy of molecular collisions.",
    tags=["chemistry", "kinetics"],
)
note2.add_reference(note1.id)  # cross-reference

notebook.add(note1)
notebook.add(note2)

# Search and filter
results = notebook.search("activation energy")
kinetics_notes = notebook.filter_by_tag("kinetics")
refs = notebook.referenced_by(note1.id)  # notes that cite note1
```

### Concept Mapping

```python
from cds.concept_map import Concept, ConceptMap, Relation

cm = ConceptMap()

# Build a concept graph
cm.add_concept(Concept("Force", "A push or pull on an object"))
cm.add_concept(Concept("Mass", "Amount of matter in an object"))
cm.add_concept(Concept("Acceleration", "Rate of change of velocity"))
cm.add_concept(Concept("Velocity", "Speed with direction"))

cm.add_relation(Relation("Force", "Mass", label="depends_on"))
cm.add_relation(Relation("Force", "Acceleration", label="equals_m_times"))
cm.add_relation(Relation("Acceleration", "Velocity", label="derivative_of"))

# Query the graph
neighbors = cm.neighbors("Force")         # ["Mass", "Acceleration"]
path = cm.find_path("Force", "Velocity")   # ["Force", "Acceleration", "Velocity"]
results = cm.search("velocity")            # [Concept("Velocity", ...)]
```

### CLI Usage

```bash
# Show system info
cds info

# Create a hypothesis
cds hypothesis new "Water boils at 100°C at sea level" "Standard atmospheric pressure"

# Create a research note
cds note new "Lab Observation" "The solution turned blue after adding CuSO4"

# Add a concept
cds concept new "Thermodynamics" --description "Study of heat and energy transfer"
```

## Running Tests

```bash
# Run full test suite with coverage report
pytest

# Run specific test module
pytest tests/test_modeling.py -v

# Generate HTML coverage report
pytest --cov-report=html
open htmlcov/index.html
```

## Roadmap

- [ ] **v0.2.0** — Persistent storage (SQLite backend for hypotheses, notes, and concept maps)
- [ ] **v0.3.0** — Export/import (JSON, Markdown, LaTeX export for research papers)
- [ ] **v0.4.0** — AI-assisted hypothesis generation (LLM integration for brainstorming)
- [ ] **v0.5.0** — Web UI (interactive dashboard for managing research workflows)
- [ ] **v1.0.0** — Stable API with plugin system and full documentation

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

To report security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
