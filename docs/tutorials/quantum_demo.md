# Quantum Circuit Simulation

Explore quantum mechanics using the `cds.quantum` module. This guide shows how to create Bell states and detect entanglement using our pure-Python simulator.

## 1. Creating a Bell State

A Bell state is a maximally entangled quantum state. We can prepare it using a Hadamard gate followed by a CNOT:

```python
from cds.quantum import QuantumCircuit, hadamard, cnot, is_entangled

# Create a Bell State (|00> + |11>) / sqrt(2)
circuit = QuantumCircuit()
circuit.add(hadamard())
circuit.add(cnot(control=0, target=1))

result_reg = circuit.run()
print(f"Is entangled? {is_entangled(result_reg)}")
print(f"Probabilities: {result_reg.probabilities()}")
```

## 2. Optimized Shot-Based Sampling

For large-scale measurements, we use an optimized O(1) sampling algorithm:

```python
from cds.quantum import simulate

counts = simulate(circuit, shots=10000)
print(f"Measurement counts: {counts}")
```
