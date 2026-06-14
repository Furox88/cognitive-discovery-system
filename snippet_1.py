from cds.quantum import (
    QuantumRegister, h_gate, cnot, bell_state,
    ghz_state, is_entangled,
)

# Bell state (|00⟩ + |11⟩) / √2
reg = bell_state(0)
print(is_entangled(reg))  # True
print(reg.measure_shots(shots=1000))  # {'00': ~500, '11': ~500}

# 4-qubit GHZ state
ghz = ghz_state(4)
counts = ghz.measure_shots(shots=1000)
print(counts)  # {'0000': ~500, '1111': ~500}
