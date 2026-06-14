"""Pure Python Neural Network module."""
from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from cds.optimization.minimize import adam


class Layer:
    """A basic dense (fully-connected) neural network layer."""

    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        # Xavier/Glorot initialization for weights
        limit = (6.0 / (input_size + output_size)) ** 0.5
        self.weights = [[random.uniform(-limit, limit) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [0.0] * output_size
        self.activation = activation

    def forward(self, x: list[float]) -> list[float]:
        """Compute layer output for a single input vector."""
        out = []
        for i in range(len(self.weights)):
            z = sum(w * xi for w, xi in zip(self.weights[i], x)) + self.biases[i]
            out.append(self._activate(z))
        return out

    def _activate(self, z: float) -> float:
        if self.activation == "relu":
            return max(0.0, z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + (2.718281828459 ** -z))
        return z  # identity


class MLP:
    """Multi-Layer Perceptron (Pure Python)."""

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def predict(self, x: list[float]) -> list[float]:
        """Compute the network output."""
        curr = x
        for layer in self.layers:
            curr = layer.forward(curr)
        return curr

    def get_parameters(self) -> list[float]:
        """Flatten all weights and biases into a single list."""
        params = []
        for layer in self.layers:
            for row in layer.weights:
                params.extend(row)
            params.extend(layer.biases)
        return params

    def set_parameters(self, params: list[float]) -> None:
        """Unflatten parameters back into weights and biases."""
        idx = 0
        for layer in self.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] = params[idx]
                    idx += 1
            for i in range(len(layer.biases)):
                layer.biases[i] = params[idx]
                idx += 1

    def train(
        self, 
        X: list[list[float]], 
        y: list[list[float]], 
        epochs: int = 100, 
        lr: float = 0.01
    ) -> dict[str, Any]:
        """Train the network using the Adam optimizer."""
        
        def loss_fn(params: list[float]) -> float:
            self.set_parameters(params)
            total_loss = 0.0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                total_loss += sum((p - target) ** 2 for p, target in zip(pred, yi))
            return total_loss / len(X)

        p0 = self.get_parameters()
        res = adam(loss_fn, p0, lr=lr, max_iter=epochs)
        self.set_parameters(res.x)
        
        return {
            "final_loss": res.value,
            "iterations": res.iterations,
            "converged": res.converged
        }
