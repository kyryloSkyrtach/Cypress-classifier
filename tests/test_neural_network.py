"""Tests for the NeuralNetwork class."""

import math
import random

import pytest

from app.neural_network import NeuralNetwork


@pytest.fixture()
def tiny_net():
    random.seed(0)
    return NeuralNetwork(input_size=3, hidden_size=4, output_size=2, learning_rate=0.1)


class TestNeuralNetworkStructure:
    def test_weight_shapes(self, tiny_net):
        net = tiny_net
        assert len(net.W1) == 4
        assert len(net.W1[0]) == 3
        assert len(net.W2) == 2
        assert len(net.W2[0]) == 4
        assert len(net.b1) == 4
        assert len(net.b2) == 2

    def test_predict_proba_sums_to_one(self, tiny_net):
        x = [1.0, -0.5, 0.3]
        probs = tiny_net.predict_proba(x)
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_predict_returns_valid_class(self, tiny_net):
        x = [0.5, 0.5, 0.5]
        pred = tiny_net.predict(x)
        assert pred in (0, 1)

    def test_all_probabilities_positive(self, tiny_net):
        x = [2.0, -1.0, 0.0]
        probs = tiny_net.predict_proba(x)
        assert all(p > 0 for p in probs)


class TestActivations:
    def test_relu_positive(self):
        assert NeuralNetwork._relu(3.5) == 3.5

    def test_relu_negative(self):
        assert NeuralNetwork._relu(-1.0) == 0.0

    def test_relu_zero(self):
        assert NeuralNetwork._relu(0.0) == 0.0

    def test_softmax_sum(self):
        values = [1.0, 2.0, 3.0]
        result = NeuralNetwork._softmax(values)
        assert abs(sum(result) - 1.0) < 1e-9

    def test_softmax_monotone(self):
        values = [1.0, 2.0, 3.0]
        result = NeuralNetwork._softmax(values)
        assert result[0] < result[1] < result[2]

    def test_softmax_numerical_stability(self):
        # Large values should not produce NaN/Inf
        values = [1000.0, 1001.0, 1002.0]
        result = NeuralNetwork._softmax(values)
        assert all(math.isfinite(p) for p in result)
        assert abs(sum(result) - 1.0) < 1e-9


class TestTraining:
    def test_loss_decreases_on_xor(self):
        """Network should learn XOR-like pattern with hidden layer."""
        random.seed(1)
        net = NeuralNetwork(input_size=2, hidden_size=8, output_size=2, learning_rate=0.05)
        x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 50
        y = [0, 1, 1, 0] * 50
        history = net.train(x, y, epochs=200, batch_size=16, verbose=False)
        # Loss should decrease overall
        assert history[-1] < history[0]

    def test_train_returns_loss_history(self):
        random.seed(2)
        net = NeuralNetwork(input_size=2, hidden_size=4, output_size=2, learning_rate=0.1)
        x = [[1.0, 0.0], [0.0, 1.0]] * 20
        y = [0, 1] * 20
        history = net.train(x, y, epochs=5, batch_size=4, verbose=False)
        assert len(history) == 5
        assert all(isinstance(v, float) for v in history)

    def test_evaluate_perfect(self):
        """After heavy training on linearly separable data, acc should be high."""
        random.seed(3)
        net = NeuralNetwork(input_size=2, hidden_size=8, output_size=2, learning_rate=0.1)
        x_class0 = [[v, 0.0] for v in [1.0, 2.0, 3.0] * 20]
        x_class1 = [[v, 0.0] for v in [-1.0, -2.0, -3.0] * 20]
        x = x_class0 + x_class1
        y = [0] * 60 + [1] * 60
        net.train(x, y, epochs=100, batch_size=16, verbose=False)
        acc = net.evaluate(x, y)
        assert acc > 0.9
