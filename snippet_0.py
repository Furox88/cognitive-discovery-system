from cds.quantum import QuantumCircuit, hadamard, pauli_x, simulate

circuit = QuantumCircuit().add(hadamard()).add(pauli_x())
result = circuit.run()
print(result.probabilities())

counts = simulate(circuit, shots=1000)
print(counts)  # {0: ~500, 1: ~500}
