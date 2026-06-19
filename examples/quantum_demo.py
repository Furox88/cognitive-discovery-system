"""Quantum computing demo — Bell states and entanglement."""

from cds.quantum import (
    QuantumCircuit,
    bell_state,
    ghz_state,
    hadamard,
    is_entangled,
    simulate,
)

# --- Single-qubit circuit ---


def main() -> None:
    print("=== Single-Qubit Circuit ===")
    circuit = QuantumCircuit().add(hadamard())
    result = circuit.run()
    print(f"After Hadamard: P(0)={result.probabilities()[0]:.3f}")

    counts = simulate(circuit, shots=10000)
    print(f"Measurement (10k shots): {counts}\n")

    # --- Bell state ---
    print("=== Bell State |Φ+⟩ ===")
    phi_plus = bell_state(0)
    print(f"Entangled? {is_entangled(phi_plus)}")
    bell_counts = phi_plus.measure_shots(shots=10000, seed=42)
    print(f"Measurements: {bell_counts}\n")

    # --- GHZ state ---
    print("=== 4-Qubit GHZ State ===")
    ghz = ghz_state(4)
    probs = ghz.probabilities()
    print(f"P(|0000⟩) = {probs[0]:.3f}")
    print(f"P(|1111⟩) = {probs[15]:.3f}")
    ghz_counts = ghz.measure_shots(shots=10000, seed=7)
    print(f"Measurements: {ghz_counts}")


if __name__ == "__main__":
    main()
