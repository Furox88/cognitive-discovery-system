# Getting Started

## Install

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Quick example

```python
from cds.hypothesis import Hypothesis

h = Hypothesis(
    statement="Objects fall at constant acceleration regardless of mass",
    rationale="Galileo's experiments",
    variables=["mass", "acceleration"],
    tags=["physics"],
)
h.propose()
print(h.summary())
```

```python
from cds.modeling import Variable, Constant, BinaryOp, UnaryFunc
from cds.modeling import differentiate_numerically, integrate_numerically

# s(t) = 0.5 * g * t^2
g = Constant(9.81)
t = Variable("t")
position = BinaryOp("*", Constant(0.5), BinaryOp("*", g, BinaryOp("^", t, Constant(2))))

print(f"Position at t=3s: {position.evaluate({'t': 3.0}):.2f} m")
velocity = differentiate_numerically(position, "t", {"t": 3.0})
print(f"Velocity at t=3s: {velocity:.2f} m/s")
```

```python
from cds.notes import Note, Notebook

nb = Notebook()
note = Note(title="Free Fall Results", body="Acceleration ~9.81 m/s^2", tags=["physics"])
nb.add(note)
print(nb.search("gravity"))
```

## Running tests

```bash
pytest
```
