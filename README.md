# Cognitive Discovery System (CDS)

[![CI](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Open-source research assistant for scientific discovery, mathematical modeling, and structured reasoning.

## What is this?

CDS is a Python toolkit for researchers. It helps you:

- **Track hypotheses** through their lifecycle (draft → testing → supported/refuted)
- **Model math** with symbolic expressions, differentiation, and integration
- **Manage research notes** with tagging, search, and cross-referencing
- **Map concept relationships** as directed graphs with pathfinding

## Install

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Usage

### Hypotheses

```python
from cds.hypothesis import Hypothesis, HypothesisStore

h = Hypothesis(
    statement="Reaction rate doubles per 10°C increase",
    rationale="Arrhenius equation",
    tags=["chemistry", "kinetics"],
    confidence=0.7,
)

h.propose()
h.start_testing()
h.support(confidence=0.92)
print(h.summary())

store = HypothesisStore()
store.add(h)
store.filter_by_tag("kinetics")
```

### Math Modeling

```python
from cds.modeling import Variable, Constant, BinaryOp, UnaryFunc
from cds.modeling import differentiate_numerically, integrate_numerically
import math

x = Variable("x")

# f(x) = x^2 + 2x + 1
f = BinaryOp("+",
    BinaryOp("+", BinaryOp("^", x, Constant(2)), BinaryOp("*", Constant(2), x)),
    Constant(1),
)
print(f.evaluate({"x": 3.0}))  # 16.0

# derivative of x^2 at x=5 -> 10
deriv = differentiate_numerically(BinaryOp("^", x, Constant(2)), "x", {"x": 5.0})

# integral of sin(x) from 0 to pi -> 2.0
area = integrate_numerically(UnaryFunc("sin", x), "x", 0, math.pi, {})

# kinetic energy: E = 0.5 * m * v^2
m, v = Variable("m"), Variable("v")
ke = BinaryOp("*", Constant(0.5), BinaryOp("*", m, BinaryOp("^", v, Constant(2))))
print(ke.evaluate({"m": 10.0, "v": 3.0}))  # 45.0 J

# exponential decay: N(t) = 1000 * e^(-0.1t)
t = Variable("t")
decay = BinaryOp("*", Constant(1000),
    UnaryFunc("exp", BinaryOp("*", Constant(-0.1), t))
)
print(f"N(5) = {decay.evaluate({'t': 5.0}):.1f}")  # 606.5
```

### Notes

```python
from cds.notes import Note, Notebook

nb = Notebook()
n1 = Note(title="Arrhenius Equation", body="k = A*exp(-Ea/RT)")
n1.add_tag("kinetics")
nb.add(n1)

results = nb.search("arrhenius")
```

### Concept Maps

```python
from cds.concept_map import Concept, ConceptMap, Relation

cm = ConceptMap()
cm.add_concept(Concept("Force", "A push or pull"))
cm.add_concept(Concept("Mass"))
cm.add_concept(Concept("Acceleration"))

cm.add_relation(Relation("Force", "Mass", label="depends_on"))
cm.add_relation(Relation("Force", "Acceleration", label="equals_m_times"))

path = cm.find_path("Force", "Acceleration")  # ["Force", "Acceleration"]
```

### CLI

```bash
cds info
cds hypothesis new "Water boils at 100C at sea level" "Standard atmospheric pressure"
cds note new "Lab Observation" "Solution turned blue after adding CuSO4"
cds concept new "Thermodynamics" --description "Study of heat and energy"
```

## Tests

```bash
pytest           # run tests with coverage
pytest -v        # verbose
pytest --cov-report=html  # html report
```

## Project Structure

```
src/cds/
  hypothesis.py    # hypothesis tracking
  modeling.py       # math expressions
  notes.py          # note management
  concept_map.py    # concept graphs
  cli.py            # command line
tests/              # test suite
```

## Roadmap

- [ ] SQLite persistence
- [ ] JSON/LaTeX export
- [ ] AI-assisted hypothesis generation
- [ ] Web dashboard
- [ ] Plugin system

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
