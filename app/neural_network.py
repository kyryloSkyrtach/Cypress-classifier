#Neural Network with one hidden layer for multi-class classification.

import math
import random


class NeuralNetwork:
    """Feed-forward neural network: input → hidden (ReLU) → output (Softmax).

    Parameters
    ----------
    input_size:   number of input features
    hidden_size:  number of neurons in the hidden layer
    output_size:  number of output classes
    learning_rate: step size for gradient descent
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.01,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # He initialisation for weights (good default for ReLU)
        scale1 = math.sqrt(2.0 / input_size)
        scale2 = math.sqrt(2.0 / hidden_size)

        self.W1 = [[random.gauss(0, scale1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size

        self.W2 = [[random.gauss(0, scale2) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [0.0] * output_size

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def _relu_deriv(x: float) -> float:
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def _softmax(values: list[float]) -> list[float]:
        max_v = max(values)
        exps = [math.exp(v - max_v) for v in values]
        total = sum(exps)
        return [e / total for e in exps]

    # ------------------------------------------------------------------
    # Linear algebra helpers (pure Python, no numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def _dot(row: list[float], vec: list[float]) -> float:
        return sum(r * v for r, v in zip(row, vec, strict=False))

    @staticmethod
    def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
        return [NeuralNetwork._dot(row, vec) for row in mat]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, x: list[float]) -> tuple[list[float], list[float], list[float], list[float]]:
        """Return (z1, h, z2, y_hat) where h is hidden activations."""
        z1 = [self._dot(self.W1[j], x) + self.b1[j] for j in range(self.hidden_size)]
        h = [self._relu(v) for v in z1]
        z2 = [self._dot(self.W2[k], h) + self.b2[k] for k in range(self.output_size)]
        y_hat = self._softmax(z2)
        return z1, h, z2, y_hat

    def predict_proba(self, x: list[float]) -> list[float]:
        """Return class probabilities for a single sample."""
        _, _, _, y_hat = self._forward(x)
        return y_hat

    def predict(self, x: list[float]) -> int:
        """Return predicted class index."""
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    # ------------------------------------------------------------------
    # Backpropagation (cross-entropy loss)
    # ------------------------------------------------------------------

    def _backward(
        self,
        x: list[float],
        y_true: int,
        z1: list[float],
        h: list[float],
        y_hat: list[float],
    ) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
        """Compute gradients via backprop."""
        # Output layer gradient (softmax + cross-entropy derivative: y_hat - one_hot)
        delta2 = [y_hat[k] - (1.0 if k == y_true else 0.0) for k in range(self.output_size)]

        grad_w2 = [[delta2[k] * h[j] for j in range(self.hidden_size)] for k in range(self.output_size)]
        db2 = delta2[:]

        # Hidden layer gradient
        delta1 = [
            sum(self.W2[k][j] * delta2[k] for k in range(self.output_size)) * self._relu_deriv(z1[j])
            for j in range(self.hidden_size)
        ]

        grad_w1 = [[delta1[j] * x[i] for i in range(self.input_size)] for j in range(self.hidden_size)]
        db1 = delta1[:]

        return grad_w1, db1, grad_w2, db2

    def _apply_gradients(
        self,
        grad_w1: list[list[float]],
        db1: list[float],
        grad_w2: list[list[float]],
        db2: list[float],
        n: int,
    ) -> None:
        """Update weights with averaged gradients (mini-batch)."""
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                self.W1[j][i] -= self.lr * grad_w1[j][i] / n
            self.b1[j] -= self.lr * db1[j] / n

        for k in range(self.output_size):
            for j in range(self.hidden_size):
                self.W2[k][j] -= self.lr * grad_w2[k][j] / n
            self.b2[k] -= self.lr * db2[k] / n

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        x_train: list[list[float]],
        y_train: list[int],
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train the network; return list of per-epoch loss values."""
        loss_history: list[float] = []
        data = list(zip(x_train, y_train, strict=False))

        for epoch in range(1, epochs + 1):
            random.shuffle(data)
            epoch_loss = 0.0

            for start in range(0, len(data), batch_size):
                batch = data[start : start + batch_size]
                # Accumulate gradients
                acc_grad_w1 = [[0.0] * self.input_size for _ in range(self.hidden_size)]
                acc_db1 = [0.0] * self.hidden_size
                acc_grad_w2 = [[0.0] * self.hidden_size for _ in range(self.output_size)]
                acc_db2 = [0.0] * self.output_size

                for x, y in batch:
                    z1, h, _z2, y_hat = self._forward(x)
                    # Cross-entropy loss
                    p = max(y_hat[y], 1e-12)
                    epoch_loss -= math.log(p)

                    grad_w1, db1, grad_w2, db2 = self._backward(x, y, z1, h, y_hat)

                    for j in range(self.hidden_size):
                        for i in range(self.input_size):
                            acc_grad_w1[j][i] += grad_w1[j][i]
                        acc_db1[j] += db1[j]

                    for k in range(self.output_size):
                        for j in range(self.hidden_size):
                            acc_grad_w2[k][j] += grad_w2[k][j]
                        acc_db2[k] += db2[k]

                self._apply_gradients(acc_grad_w1, acc_db1, acc_grad_w2, acc_db2, len(batch))

            avg_loss = epoch_loss / len(data)
            loss_history.append(avg_loss)

            if verbose and epoch % 10 == 0:
                acc = self.evaluate(x_train, y_train)
                print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  train_acc={acc:.3f}")

        return loss_history

    def evaluate(self, x: list[list[float]], y: list[int]) -> float:
        """Return accuracy on given dataset."""
        correct = sum(1 for xi, yi in zip(x, y, strict=False) if self.predict(xi) == yi)
        return correct / len(y)
