"""Multi-qubit quantum register with entanglement support."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class QuantumRegister:
    """N-qubit state vector. Amplitudes stored as list of 2^n complex numbers."""

    n_qubits: int
    amplitudes: list[complex]

    @classmethod
    def zeros(cls, n: int) -> QuantumRegister:
        """All qubits in |0> state."""
        amps: list[complex] = [0 + 0j] * (2 ** n)
        amps[0] = 1 + 0j
        return cls(n_qubits=n, amplitudes=amps)

    @classmethod
    def from_bits(cls, n: int, value: int) -> QuantumRegister:
        """Computational basis state |value>."""
        amps: list[complex] = [0 + 0j] * (2 ** n)
        amps[value] = 1 + 0j
        return cls(n_qubits=n, amplitudes=amps)

    @property
    def size(self) -> int:
        return len(self.amplitudes)

    def probabilities(self) -> list[float]:
        return [abs(a) ** 2 for a in self.amplitudes]

    def normalize(self) -> None:
        norm = math.sqrt(sum(abs(a) ** 2 for a in self.amplitudes))
        if norm > 0:
            self.amplitudes = [a / norm for a in self.amplitudes]

    def measure(self, seed: int | None = None) -> int:
        """Measure the register and collapse its state vector."""
        if seed is not None:
            random.seed(seed)
        probs = self.probabilities()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                # State Collapse logic
                # All other amplitudes become 0, measured state becomes 1.0
                new_amps = [0.0 + 0j] * len(self.amplitudes)
                new_amps[i] = 1.0 + 0j
                self.amplitudes = new_amps
                return i
        
        # Fallback for floating point edge cases
        final_idx = len(probs) - 1
        new_amps = [0.0 + 0j] * len(self.amplitudes)
        new_amps[final_idx] = 1.0 + 0j
        self.amplitudes = new_amps
        return final_idx


    def measure_shots(
        self, shots: int = 1000, seed: int | None = None,
    ) -> dict[str, int]:
        """Run multiple measurements, return counts as binary strings."""
        if seed is not None:
            random.seed(seed)
        counts: dict[str, int] = {}
        probs = self.probabilities()
        for _ in range(shots):
            r = random.random()
            cumulative = 0.0
            result = len(probs) - 1
            for i, p in enumerate(probs):
                cumulative += p
                if r < cumulative:
                    result = i
                    break
            label = format(result, f"0{self.n_qubits}b")
            counts[label] = counts.get(label, 0) + 1
        return counts

    def expectation(self) -> float:
        """Expected value treating basis index as eigenvalue."""
        return sum(
            i * abs(a) ** 2 for i, a in enumerate(self.amplitudes)
        )


def _gate_2x2(
    reg: QuantumRegister, target: int, matrix: list[complex],
) -> QuantumRegister:
    """Apply a 2x2 gate to a specific qubit in the register."""
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    step = 1 << target
    a, b, c, d = matrix

    for i in range(0, 1 << n, step << 1):
        for j in range(step):
            idx0 = i + j
            idx1 = idx0 + step
            v0 = reg.amplitudes[idx0]
            v1 = reg.amplitudes[idx1]
            new_amps[idx0] = a * v0 + b * v1
            new_amps[idx1] = c * v0 + d * v1

    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


def h_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Hadamard on qubit `target`."""
    s = 1 / math.sqrt(2)
    return _gate_2x2(reg, target, [s, s, s, -s])


def x_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Pauli-X (NOT) on qubit `target`."""
    return _gate_2x2(reg, target, [0, 1, 1, 0])


def z_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Pauli-Z on qubit `target`."""
    return _gate_2x2(reg, target, [1, 0, 0, -1])


def y_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Pauli-Y on qubit `target`."""
    return _gate_2x2(reg, target, [0, -1j, 1j, 0])


def rz_gate(
    reg: QuantumRegister, target: int, theta: float,
) -> QuantumRegister:
    """Rotation around Z axis."""
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2), math.sin(theta / 2))
    return _gate_2x2(reg, target, [e_neg, 0, 0, e_pos])


def cnot(
    reg: QuantumRegister, control: int, target: int,
) -> QuantumRegister:
    """Controlled-NOT gate."""
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    for i in range(1 << n):
        if i & (1 << control):
            j = i ^ (1 << target)
            if j > i:
                new_amps[i], new_amps[j] = reg.amplitudes[j], reg.amplitudes[i]
    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


def cz(
    reg: QuantumRegister, control: int, target: int,
) -> QuantumRegister:
    """Controlled-Z gate."""
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    for i in range(1 << n):
        if (i & (1 << control)) and (i & (1 << target)):
            new_amps[i] = -reg.amplitudes[i]
    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


def swap(
    reg: QuantumRegister, q1: int, q2: int,
) -> QuantumRegister:
    """SWAP gate — exchange two qubits."""
    reg = cnot(reg, q1, q2)
    reg = cnot(reg, q2, q1)
    reg = cnot(reg, q1, q2)
    return reg


def toffoli(
    reg: QuantumRegister, c1: int, c2: int, target: int,
) -> QuantumRegister:
    """Toffoli (CCNOT) gate — 3-qubit controlled-controlled-NOT."""
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    for i in range(1 << n):
        if (i & (1 << c1)) and (i & (1 << c2)):
            j = i ^ (1 << target)
            if j > i:
                new_amps[i], new_amps[j] = (
                    reg.amplitudes[j], reg.amplitudes[i],
                )
    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


# ----- common state preparation -----

def bell_state(which: int = 0) -> QuantumRegister:
    """Create one of the 4 Bell states (0-3).

    0: |Φ+> = (|00> + |11>) / √2
    1: |Φ-> = (|00> - |11>) / √2
    2: |Ψ+> = (|01> + |10>) / √2
    3: |Ψ-> = (|01> - |10>) / √2
    """
    reg = QuantumRegister.zeros(2)
    if which in (2, 3):
        reg = x_gate(reg, 1)
    reg = h_gate(reg, 0)
    reg = cnot(reg, 0, 1)
    if which in (1, 3):
        reg = z_gate(reg, 0)
    return reg


def ghz_state(n: int) -> QuantumRegister:
    """GHZ state: (|00...0> + |11...1>) / √2"""
    reg = QuantumRegister.zeros(n)
    reg = h_gate(reg, 0)
    for i in range(1, n):
        reg = cnot(reg, 0, i)
    return reg


def is_entangled(reg: QuantumRegister) -> bool:
    """Check if a 2-qubit state is entangled (not separable).

    Uses concurrence: for |ψ> = a|00> + b|01> + c|10> + d|11>,
    concurrence = 2|ad - bc|. If > 0, it's entangled.
    """
    if reg.n_qubits != 2:
        raise ValueError("entanglement check only for 2-qubit states")
    a, b, c, d = reg.amplitudes
    concurrence = 2 * abs(a * d - b * c)
    return concurrence > 1e-9
