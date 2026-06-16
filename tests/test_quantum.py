"""Tests for quantum module."""

import math

from cds.quantum.circuit import (
    QuantumCircuit,
    Qubit,
    hadamard,
    pauli_x,
    pauli_z,
    phase_gate,
)
from cds.quantum.simulator import simulate


def test_qubit_default_is_zero_state() -> None:
    q = Qubit()
    p0, p1 = q.probabilities()
    assert p0 == 1.0
    assert p1 == 0.0


def test_pauli_x_flips() -> None:
    q = pauli_x().apply(Qubit())
    p0, p1 = q.probabilities()
    assert abs(p0) < 1e-9
    assert abs(p1 - 1.0) < 1e-9


def test_hadamard_creates_superposition() -> None:
    q = hadamard().apply(Qubit())
    p0, p1 = q.probabilities()
    assert abs(p0 - 0.5) < 1e-9
    assert abs(p1 - 0.5) < 1e-9


def test_double_hadamard_is_identity() -> None:
    circuit = QuantumCircuit().add(hadamard()).add(hadamard())
    q = circuit.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-9
    assert abs(p1) < 1e-9


def test_pauli_z_on_zero() -> None:
    q = pauli_z().apply(Qubit())
    # Z|0> = |0>
    assert abs(q.alpha - 1) < 1e-9


def test_phase_gate() -> None:
    g = phase_gate(math.pi)
    q = g.apply(Qubit(alpha=0, beta=1))
    # should flip sign of beta
    assert abs(q.beta.real - (-1)) < 1e-6


def test_circuit_len() -> None:
    c = QuantumCircuit().add(hadamard()).add(pauli_x())
    assert len(c) == 2


def test_simulate_pure_zero() -> None:
    c = QuantumCircuit()  # no gates, stays |0>
    counts = simulate(c, shots=100, seed=42)
    assert counts.get(0, 0) == 100
    assert counts.get(1, 0) == 0


def test_simulate_hadamard_roughly_even() -> None:
    c = QuantumCircuit().add(hadamard())
    counts = simulate(c, shots=10000, seed=7)
    ratio = counts.get(0, 0) / 10000
    assert 0.45 < ratio < 0.55


def test_qubit_normalize() -> None:
    q = Qubit(alpha=3 + 0j, beta=4 + 0j)
    q.normalize()
    p0, p1 = q.probabilities()
    assert abs(p0 + p1 - 1.0) < 1e-9


def test_double_pauli_x_is_identity() -> None:
    circuit = QuantumCircuit().add(pauli_x()).add(pauli_x())
    q = circuit.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-9


def test_pauli_z_on_one() -> None:
    q = pauli_z().apply(pauli_x().apply(Qubit()))
    # Z|1> = -|1>
    assert abs(abs(q.beta) - 1) < 1e-9


def test_circuit_empty() -> None:
    c = QuantumCircuit()
    q = c.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-9


def test_simulate_pauli_x_always_one() -> None:
    c = QuantumCircuit().add(pauli_x())
    counts = simulate(c, shots=100, seed=42)
    assert counts.get(1, 0) == 100


def test_phase_gate_zero() -> None:
    g = phase_gate(0)
    q = g.apply(Qubit(alpha=0, beta=1))
    assert abs(q.beta - 1) < 1e-9


def test_circuit_three_gates() -> None:
    c = QuantumCircuit().add(hadamard()).add(pauli_x()).add(hadamard())
    assert len(c) == 3
    q = c.run()
    # H X H = Z, so |0> -> Z|0> = |0>
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-6
