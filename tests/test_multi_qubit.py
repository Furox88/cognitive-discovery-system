"""Tests for multi-qubit quantum operations and entanglement."""

import math

from cds.quantum.multi_qubit import (
    QuantumRegister,
    bell_state,
    cnot,
    cz,
    ghz_state,
    h_gate,
    is_entangled,
    rz_gate,
    swap,
    toffoli,
    x_gate,
    y_gate,
    z_gate,
)

# --- QuantumRegister basics ---


def test_zeros_register() -> None:
    reg = QuantumRegister.zeros(2)
    assert reg.n_qubits == 2
    assert reg.size == 4
    assert abs(reg.amplitudes[0] - 1) < 1e-9
    assert all(abs(a) < 1e-9 for a in reg.amplitudes[1:])


def test_from_bits() -> None:
    reg = QuantumRegister.from_bits(3, 5)  # |101>
    assert abs(reg.amplitudes[5] - 1) < 1e-9
    assert sum(abs(a) ** 2 for a in reg.amplitudes) - 1.0 < 1e-9


def test_probabilities() -> None:
    reg = QuantumRegister.zeros(1)
    probs = reg.probabilities()
    assert abs(probs[0] - 1.0) < 1e-9
    assert abs(probs[1]) < 1e-9


def test_normalize() -> None:
    reg = QuantumRegister(n_qubits=1, amplitudes=[3 + 0j, 4 + 0j])
    reg.normalize()
    total = sum(abs(a) ** 2 for a in reg.amplitudes)
    assert abs(total - 1.0) < 1e-9


def test_measure_deterministic() -> None:
    reg = QuantumRegister.zeros(2)
    result = reg.measure(seed=42)
    assert result == 0


def test_measure_shots() -> None:
    reg = QuantumRegister.zeros(2)
    counts = reg.measure_shots(shots=100, seed=42)
    assert counts.get("00", 0) == 100


def test_expectation_zero_state() -> None:
    reg = QuantumRegister.zeros(2)
    assert reg.expectation() == 0.0


# --- Single-qubit gates on register ---


def test_x_gate_flips_qubit_0() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)
    # |00> -> |01>
    assert abs(reg.amplitudes[1] - 1) < 1e-9


def test_x_gate_flips_qubit_1() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 1)
    # |00> -> |10>
    assert abs(reg.amplitudes[2] - 1) < 1e-9


def test_h_gate_superposition() -> None:
    reg = QuantumRegister.zeros(1)
    reg = h_gate(reg, 0)
    probs = reg.probabilities()
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[1] - 0.5) < 1e-9


def test_z_gate_on_zero() -> None:
    reg = QuantumRegister.zeros(1)
    reg = z_gate(reg, 0)
    # Z|0> = |0>
    assert abs(reg.amplitudes[0] - 1) < 1e-9


def test_z_gate_on_one() -> None:
    reg = QuantumRegister.from_bits(1, 1)
    reg = z_gate(reg, 0)
    # Z|1> = -|1>
    assert abs(reg.amplitudes[1] - (-1)) < 1e-9


def test_y_gate() -> None:
    reg = QuantumRegister.zeros(1)
    reg = y_gate(reg, 0)
    # Y|0> = i|1>
    assert abs(reg.amplitudes[0]) < 1e-9
    assert abs(reg.amplitudes[1] - 1j) < 1e-9


def test_rz_gate() -> None:
    reg = QuantumRegister.zeros(1)
    reg = h_gate(reg, 0)
    reg = rz_gate(reg, 0, math.pi)
    probs = reg.probabilities()
    # still 50/50 after Rz
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[1] - 0.5) < 1e-9


def test_double_x_is_identity() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)
    reg = x_gate(reg, 0)
    assert abs(reg.amplitudes[0] - 1) < 1e-9


def test_double_h_is_identity() -> None:
    reg = QuantumRegister.zeros(1)
    reg = h_gate(reg, 0)
    reg = h_gate(reg, 0)
    assert abs(reg.amplitudes[0] - 1) < 1e-9


# --- CNOT ---


def test_cnot_no_flip_when_control_zero() -> None:
    reg = QuantumRegister.zeros(2)
    reg = cnot(reg, 0, 1)
    # control is 0, no change
    assert abs(reg.amplitudes[0] - 1) < 1e-9


def test_cnot_flips_when_control_one() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)  # |01>
    reg = cnot(reg, 0, 1)  # should flip target -> |11>
    assert abs(reg.amplitudes[3] - 1) < 1e-9


def test_cnot_creates_entanglement() -> None:
    reg = QuantumRegister.zeros(2)
    reg = h_gate(reg, 0)
    reg = cnot(reg, 0, 1)
    # Bell state: (|00> + |11>) / sqrt(2)
    assert is_entangled(reg)


# --- CZ ---


def test_cz_on_11() -> None:
    reg = QuantumRegister.from_bits(2, 3)  # |11>
    reg = cz(reg, 0, 1)
    # CZ|11> = -|11>
    assert abs(reg.amplitudes[3] - (-1)) < 1e-9


def test_cz_on_00() -> None:
    reg = QuantumRegister.zeros(2)
    reg = cz(reg, 0, 1)
    # no change
    assert abs(reg.amplitudes[0] - 1) < 1e-9


# --- SWAP ---


def test_swap_01_to_10() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)  # |01>
    reg = swap(reg, 0, 1)  # -> |10>
    assert abs(reg.amplitudes[2] - 1) < 1e-9


# --- Toffoli ---


def test_toffoli_flips_when_both_controls_set() -> None:
    reg = QuantumRegister.zeros(3)
    reg = x_gate(reg, 0)  # |001>
    reg = x_gate(reg, 1)  # |011>
    reg = toffoli(reg, 0, 1, 2)  # -> |111>
    assert abs(reg.amplitudes[7] - 1) < 1e-9


def test_toffoli_no_flip_when_one_control() -> None:
    reg = QuantumRegister.zeros(3)
    reg = x_gate(reg, 0)  # |001>
    reg = toffoli(reg, 0, 1, 2)  # control1 is 0, no change
    assert abs(reg.amplitudes[1] - 1) < 1e-9


# --- Bell states ---


def test_bell_phi_plus() -> None:
    reg = bell_state(0)
    s = 1 / math.sqrt(2)
    assert abs(reg.amplitudes[0] - s) < 1e-9  # |00>
    assert abs(reg.amplitudes[3] - s) < 1e-9  # |11>
    assert is_entangled(reg)


def test_bell_phi_minus() -> None:
    reg = bell_state(1)
    s = 1 / math.sqrt(2)
    assert abs(abs(reg.amplitudes[0]) - s) < 1e-9
    assert abs(abs(reg.amplitudes[3]) - s) < 1e-9
    assert is_entangled(reg)


def test_bell_psi_plus() -> None:
    reg = bell_state(2)
    s = 1 / math.sqrt(2)
    assert abs(abs(reg.amplitudes[1]) - s) < 1e-9
    assert abs(abs(reg.amplitudes[2]) - s) < 1e-9
    assert is_entangled(reg)


def test_bell_psi_minus() -> None:
    reg = bell_state(3)
    assert is_entangled(reg)


def test_bell_measurements() -> None:
    reg = bell_state(0)
    counts = reg.measure_shots(shots=10000, seed=7)
    assert "00" in counts
    assert "11" in counts
    # should be roughly 50/50
    total = sum(counts.values())
    r00 = counts.get("00", 0) / total
    assert 0.45 < r00 < 0.55


# --- GHZ ---


def test_ghz_3_qubit() -> None:
    reg = ghz_state(3)
    probs = reg.probabilities()
    # only |000> and |111> should have probability
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[7] - 0.5) < 1e-9
    for i in range(1, 7):
        assert abs(probs[i]) < 1e-9


def test_ghz_4_qubit() -> None:
    reg = ghz_state(4)
    probs = reg.probabilities()
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[15] - 0.5) < 1e-9


# --- Entanglement checks ---


def test_separable_state() -> None:
    reg = QuantumRegister.zeros(2)
    assert not is_entangled(reg)


def test_product_superposition_not_entangled() -> None:
    # H|0> ⊗ |0> is separable
    reg = QuantumRegister.zeros(2)
    reg = h_gate(reg, 0)
    assert not is_entangled(reg)


def test_entangled_after_cnot() -> None:
    reg = QuantumRegister.zeros(2)
    reg = h_gate(reg, 0)
    reg = cnot(reg, 0, 1)
    assert is_entangled(reg)


# --- 3-qubit circuits ---


def test_3_qubit_circuit() -> None:
    reg = QuantumRegister.zeros(3)
    reg = h_gate(reg, 0)
    reg = h_gate(reg, 1)
    reg = h_gate(reg, 2)
    # all 8 basis states equal
    probs = reg.probabilities()
    for p in probs:
        assert abs(p - 0.125) < 1e-9


def test_quantum_teleportation_circuit() -> None:
    """Simplified teleportation-like circuit."""
    reg = QuantumRegister.zeros(3)
    # prepare qubit 0 in some state
    reg = x_gate(reg, 0)  # |1>
    # create Bell pair on qubits 1,2
    reg = h_gate(reg, 1)
    reg = cnot(reg, 1, 2)
    # CNOT from 0 to 1
    reg = cnot(reg, 0, 1)
    reg = h_gate(reg, 0)
    # circuit ran without error
    total_prob = sum(abs(a) ** 2 for a in reg.amplitudes)
    assert abs(total_prob - 1.0) < 1e-9
