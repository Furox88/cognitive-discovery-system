"""Run a circuit many times and collect measurement statistics."""
from __future__ import annotations

import random
from collections import Counter

from cds.quantum.circuit import QuantumCircuit, Qubit


def measure(q: Qubit) -> int:
    p0, _ = q.probabilities()
    return 0 if random.random() < p0 else 1


def simulate(circuit: QuantumCircuit, shots: int = 1000, seed: int | None = None) -> dict[int, int]:
    if seed is not None:
        random.seed(seed)
    results: list[int] = []
    for _ in range(shots):
        q = circuit.run()
        results.append(measure(q))
    return dict(Counter(results))
