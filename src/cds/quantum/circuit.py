"""Basic quantum circuit representation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Qubit:
    """Single qubit state as (alpha, beta) amplitudes."""

    alpha: complex = 1 + 0j
    beta: complex = 0 + 0j

    def probabilities(self) -> tuple[float, float]:
        p0 = abs(self.alpha) ** 2
        p1 = abs(self.beta) ** 2
        return (p0, p1)

    def normalize(self) -> None:
        norm = math.sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm


@dataclass
class QuantumGate:
    """A 2x2 unitary gate stored as flat list [a, b, c, d]."""

    name: str
    matrix: list[complex]

    def apply(self, q: Qubit) -> Qubit:
        a, b, c, d = self.matrix
        new_alpha = a * q.alpha + b * q.beta
        new_beta = c * q.alpha + d * q.beta
        return Qubit(alpha=new_alpha, beta=new_beta)


# common gates
def hadamard() -> QuantumGate:
    s = 1 / math.sqrt(2)
    return QuantumGate("H", [s, s, s, -s])


def pauli_x() -> QuantumGate:
    return QuantumGate("X", [0, 1, 1, 0])


def pauli_z() -> QuantumGate:
    return QuantumGate("Z", [1, 0, 0, -1])


def phase_gate(theta: float) -> QuantumGate:
    return QuantumGate(f"P({theta:.2f})", [1, 0, 0, complex(math.cos(theta), math.sin(theta))])


@dataclass
class QuantumCircuit:
    """Simple circuit that applies gates sequentially to a single qubit."""

    gates: list[QuantumGate] = field(default_factory=list)

    def add(self, gate: QuantumGate) -> QuantumCircuit:
        self.gates.append(gate)
        return self

    def run(self, initial: Qubit | None = None) -> Qubit:
        q = initial or Qubit()
        for g in self.gates:
            q = g.apply(q)
        return q

    def __len__(self) -> int:
        return len(self.gates)
