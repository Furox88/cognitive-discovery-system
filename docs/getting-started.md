# Getting Started with CDS

This guide walks you through installing the Cognitive Discovery System and running your first research workflow.

## Prerequisites

- Python 3.9 or higher
- pip (comes with Python)

## Installation

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Your First Workflow

### Step 1: Formulate a Hypothesis

```python
from cds.hypothesis import Hypothesis

h = Hypothesis(
    statement="Objects in free fall experience constant acceleration regardless of mass",
    rationale="Galileo's experiments at the Tower of Pisa (thought experiment)",
    variables=["mass", "acceleration", "gravity"],
    tags=["physics", "mechanics"],
)
h.propose()
print(h.summary())
```

### Step 2: Model the Physics

```python
from cds.modeling import Variable, Constant, BinaryOp, UnaryFunc
from cds.modeling import differentiate_numerically, integrate_numerically

# Position of a falling object: s(t) = ½gt²
g = Constant(9.81)  # m/s²
t = Variable("t")
position = BinaryOp("*", Constant(0.5), BinaryOp("*", g, BinaryOp("^", t, Constant(2))))

# Position after 3 seconds
s = position.evaluate({"t": 3.0})
print(f"Position at t=3s: {s:.2f} m")  # 44.15 m

# Velocity = ds/dt (numerical derivative)
velocity = differentiate_numerically(position, "t", {"t": 3.0})
print(f"Velocity at t=3s: {velocity:.2f} m/s")  # 29.43 m/s

# Distance traveled from t=0 to t=5
distance = integrate_numerically(position, "t", 0, 5, {})
print(f"Distance from t=0 to t=5: {distance:.2f} m")  # 204.38 m
```

### Step 3: Document Your Findings

```python
from cds.notes import Note, Notebook

notebook = Notebook()

note = Note(
    title="Free Fall Experiment — Results",
    body="Confirmed that acceleration due to gravity is approximately 9.81 m/s². "
         "Objects of different masses fell at the same rate within experimental error.",
    tags=["physics", "experiment", "gravity"],
)
notebook.add(note)

# Search your notes later
results = notebook.search("gravity")
print(f"Found {len(results)} note(s) about gravity")
```

### Step 4: Map the Concepts

```python
from cds.concept_map import Concept, ConceptMap, Relation

cm = ConceptMap()
cm.add_concept(Concept("Gravity", "Attractive force between masses"))
cm.add_concept(Concept("Free Fall", "Motion under gravity only"))
cm.add_concept(Concept("Acceleration", "Rate of velocity change"))
cm.add_concept(Concept("Terminal Velocity", "Max speed in fluid"))

cm.add_relation(Relation("Gravity", "Free Fall", label="causes"))
cm.add_relation(Relation("Free Fall", "Acceleration", label="produces"))
cm.add_relation(Relation("Acceleration", "Terminal Velocity", label="limited_by"))

# Find the chain from Gravity to Terminal Velocity
path = cm.find_path("Gravity", "Terminal Velocity")
print(f"Concept path: {' → '.join(path)}")
# Gravity → Free Fall → Acceleration → Terminal Velocity
```

## Next Steps

- Read the [full API examples](../README.md#quick-start) in the README
- Check out the [mathematical modeling section](../README.md#mathematical-modeling) for more math examples
- See [CONTRIBUTING.md](../CONTRIBUTING.md) if you'd like to contribute
