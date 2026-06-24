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
        """Return (P(|0>), P(|1>)) measurement probabilities for this qubit."""
        p0 = abs(self.alpha) ** 2
        p1 = abs(self.beta) ** 2
        return (p0, p1)

    def normalize(self) -> None:
        """Renormalize the state amplitudes in-place to unit length."""
        # Complex amplitudes: the norm is sqrt(|alpha|^2 + |beta|^2). We sum
        # the squared magnitudes first and take a single sqrt — numerically
        # equivalent to math.hypot for reals, but math.hypot rejects complex
        # inputs, so this is the correct hypotenuse for the complex plane.
        mag = (abs(self.alpha) ** 2) + (abs(self.beta) ** 2)
        norm = math.sqrt(mag)
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm


@dataclass
class QuantumGate:
    """A 2x2 unitary gate stored as flat list [a, b, c, d]."""

    name: str
    matrix: list[complex]

    def apply(self, q: Qubit) -> Qubit:
        """Apply this gate to `q` and return a new Qubit (state is not mutated)."""
        a, b, c, d = self.matrix
        new_alpha = a * q.alpha + b * q.beta
        new_beta = c * q.alpha + d * q.beta
        return Qubit(alpha=new_alpha, beta=new_beta)


# common gates
def hadamard() -> QuantumGate:
    """Hadamard gate H = (1/sqrt(2)) * [[1, 1], [1, -1]]."""
    s = 1 / math.sqrt(2)
    return QuantumGate("H", [s, s, s, -s])


def pauli_x() -> QuantumGate:
    """Pauli-X (NOT) gate X = [[0, 1], [1, 0]]."""
    return QuantumGate("X", [0, 1, 1, 0])


def pauli_z() -> QuantumGate:
    """Pauli-Z gate Z = [[1, 0], [0, -1]]."""
    return QuantumGate("Z", [1, 0, 0, -1])


def phase_gate(theta: float) -> QuantumGate:
    """Phase rotation gate P(theta) = diag(1, e^{i*theta})."""
    return QuantumGate(f"P({theta:.2f})", [1, 0, 0, complex(math.cos(theta), math.sin(theta))])


@dataclass
class QuantumCircuit:
    """Simple circuit that applies gates sequentially to a single qubit."""

    gates: list[QuantumGate] = field(default_factory=list)

    def add(self, gate: QuantumGate) -> QuantumCircuit:
        """Append a gate to the circuit (returns self for fluent chaining)."""
        self.gates.append(gate)
        return self

    def run(self, initial: Qubit | None = None) -> Qubit:
        """Apply all gates sequentially; starts from `initial` or |0> if None."""
        q = initial or Qubit()
        for g in self.gates:
            q = g.apply(q)
        return q

    def __len__(self) -> int:
        """Return the number of gates in the circuit."""
        return len(self.gates)
