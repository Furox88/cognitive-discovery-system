"""Run a circuit many times and collect measurement statistics."""
from __future__ import annotations

import random
from collections import Counter

from cds.quantum.circuit import QuantumCircuit, Qubit


def measure(q: Qubit) -> int:
    """Measure a qubit and collapse its state vector."""
    p0, _ = q.probabilities()
    outcome = 0 if random.random() < p0 else 1
    
    # Quantum State Collapse
    if outcome == 0:
        q.alpha, q.beta = 1.0 + 0j, 0.0 + 0j
    else:
        q.alpha, q.beta = 0.0 + 0j, 1.0 + 0j
        
    return outcome


def simulate(circuit: QuantumCircuit, shots: int = 1000, seed: int | None = None) -> dict[int, int]:
    """Run a circuit many times and collect measurement statistics.

    Optimized to compute the state vector only once, then probabilistically sample.
    """
    rng = random.Random(seed)

    # Compute the final quantum state exactly once (Massive performance boost)
    q = circuit.run()
    p0, _ = q.probabilities()

    # Probabilistically sample the distribution 'shots' times
    results = [0 if rng.random() < p0 else 1 for _ in range(shots)]
    return dict(Counter(results))
