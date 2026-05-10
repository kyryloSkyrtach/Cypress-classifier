"""Tests for evaluation metrics."""

from app.metrics import accuracy, classification_report, confusion_matrix


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0

    def test_zero(self):
        assert accuracy([0, 1, 2], [1, 2, 0]) == 0.0

    def test_partial(self):
        assert abs(accuracy([0, 1, 2, 0], [0, 1, 0, 0]) - 0.75) < 1e-9

    def test_empty(self):
        assert accuracy([], []) == 0.0


class TestConfusionMatrix:
    def test_shape(self):
        cm = confusion_matrix([0, 1, 2], [0, 1, 2], n_classes=3)
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_diagonal_perfect(self):
        y = [0, 1, 2, 0, 1, 2]
        cm = confusion_matrix(y, y, n_classes=3)
        for i in range(3):
            assert cm[i][i] == 2
        off = sum(cm[i][j] for i in range(3) for j in range(3) if i != j)
        assert off == 0

    def test_off_diagonal(self):
        y_true = [0, 0, 1]
        y_pred = [0, 1, 1]
        cm = confusion_matrix(y_true, y_pred, n_classes=2)
        assert cm[0][0] == 1
        assert cm[0][1] == 1
        assert cm[1][1] == 1


class TestClassificationReport:
    def test_returns_string(self):
        y = [0, 1, 2, 0, 1, 2]
        report = classification_report(y, y)
        assert isinstance(report, str)

    def test_contains_class_names(self):
        y = [0, 1, 2]
        report = classification_report(y, y)
        assert "timeout" in report
        assert "network_error" in report
        assert "ui_bug" in report

    def test_perfect_accuracy_shown(self):
        y = [0, 1, 2, 0, 1, 2]
        report = classification_report(y, y)
        assert "1.000" in report
