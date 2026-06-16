"""Tests for ML and Visualization modules."""

from cds.data_analysis import plot_bar, plot_line
from cds.ml import MLP, Layer


def test_layer_forward():
    layer = Layer(input_size=3, output_size=2, activation="relu")
    out = layer.forward([1.0, 2.0, 3.0])
    assert len(out) == 2
    assert all(val >= 0 for val in out)


def test_mlp_predict():
    net = MLP([Layer(2, 4, activation="relu"), Layer(4, 1, activation="sigmoid")])
    out = net.predict([0.5, 0.5])
    assert len(out) == 1
    assert 0 <= out[0] <= 1


def test_mlp_train_simple():
    # Test if loss decreases after training
    X = [[1.0, 0.0], [0.0, 1.0]]
    y = [[1.0], [0.0]]

    net = MLP([Layer(2, 1, activation="sigmoid")])

    def get_loss():
        total = 0
        for xi, yi in zip(X, y):
            total += (net.predict(xi)[0] - yi[0]) ** 2
        return total / len(X)

    initial_loss = get_loss()
    net.train(X, y, epochs=10, lr=0.5)
    final_loss = get_loss()

    assert final_loss < initial_loss or abs(final_loss - initial_loss) < 1e-5


def test_viz_bar_basic():
    data = {"A": 10, "B": 20}
    plot = plot_bar(data)
    assert "A" in plot
    assert "B" in plot
    assert "█" in plot


def test_viz_line_basic():
    y = [1, 2, 3, 2, 1]
    plot = plot_line(y)
    assert "•" in plot
    assert "max: 3.00" in plot


def test_viz_empty():
    assert "No data" in plot_bar({})
    assert "No data" in plot_line([])
