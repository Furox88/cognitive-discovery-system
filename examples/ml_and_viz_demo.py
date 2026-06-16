"""
Demo showing the new Machine Learning (Neural Network) and Visualization modules.
Everything is 100% Pure Python.
"""

from cds.data_analysis import plot_bar, plot_line
from cds.ml import MLP, Layer


def run_demo():
    print("--- 🧠 Pure Python Neural Network Demo ---")

    # Simple XOR-like logic training data
    # Input: [x1, x2], Output: [x1 OR x2]
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [1]]

    # Create a simple MLP: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
    net = MLP([Layer(2, 4, activation="relu"), Layer(4, 1, activation="sigmoid")])

    print("Initial predictions (Untrained):")
    for xi in X:
        print(f"  In: {xi} -> Out: {net.predict(xi)[0]:.4f}")

    print("\nTraining for 200 epochs using Adam optimizer...")
    history = net.train(X, y, epochs=200, lr=0.1)
    print(f"Training Result: {history}")

    print("\nFinal predictions (Trained):")
    for xi in X:
        print(f"  In: {xi} -> Out: {net.predict(xi)[0]:.4f}")

    print("\n--- 📈 ASCII Visualization Demo ---")

    # 1. Bar Chart
    stats = {"Quantum": 95.5, "Signals": 88.2, "Math": 92.0, "ML": 99.1}
    print(plot_bar(stats, title="Module Readiness Score"))

    # 2. Line Plot (Sine Wave)
    import math

    wave = [math.sin(x * 0.2) for x in range(50)]
    print(plot_line(wave, title="Generated Sine Wave (ASCII)"))


if __name__ == "__main__":
    run_demo()
