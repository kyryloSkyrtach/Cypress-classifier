"""Cypress Test Failure Classifier -- main entry point."""

from __future__ import annotations

import argparse
import json
import random

from app.data_generator import generate_dataset
from app.features import (
    IDX_TO_LABEL,
    extract_features,
    load_cypress_report,
    normalize,
)
from app.metrics import classification_report, confusion_matrix, print_confusion_matrix
from app.neural_network import NeuralNetwork

# Hyper-parameters
HIDDEN_SIZE = 16
LEARNING_RATE = 0.05
EPOCHS = 150
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
SEED = 42


def _train_test_split(
    x: list[list[float]],
    y: list[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    data = list(zip(x, y, strict=False))
    random.Random(seed).shuffle(data)  # noqa: S311
    split = int(len(data) * train_ratio)
    train, test = data[:split], data[split:]
    x_tr, y_tr = zip(*train, strict=False)
    x_te, y_te = zip(*test, strict=False)
    return list(x_tr), list(y_tr), list(x_te), list(y_te)


def _build_trained_model() -> tuple[NeuralNetwork, list[float], list[float]]:
    """Train model on synthetic data; return (model, means, stds)."""
    records = generate_dataset(n=600, seed=SEED)
    raw_x = [extract_features(r) for r in records]
    raw_y = [{"timeout": 0, "network_error": 1, "ui_bug": 2}[r["label"]] for r in records]
    x_train, y_train, x_test, _y_test = _train_test_split(raw_x, raw_y, TRAIN_RATIO, SEED)
    x_train_n, _x_test_n, means, stds = normalize(x_train, x_test)
    net = NeuralNetwork(6, HIDDEN_SIZE, 3, LEARNING_RATE)
    net.train(x_train_n, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=False)
    return net, means, stds


def run_experiment(verbose: bool = True) -> dict:
    """Train the model and return evaluation metrics dict."""
    records = generate_dataset(n=600, seed=SEED)
    raw_x = [extract_features(r) for r in records]
    raw_y = [{"timeout": 0, "network_error": 1, "ui_bug": 2}[r["label"]] for r in records]

    x_train, y_train, x_test, y_test = _train_test_split(raw_x, raw_y, TRAIN_RATIO, SEED)
    x_train_n, x_test_n, _means, _stds = normalize(x_train, x_test)

    net = NeuralNetwork(
        input_size=6,
        hidden_size=HIDDEN_SIZE,
        output_size=3,
        learning_rate=LEARNING_RATE,
    )
    if verbose:
        print("=" * 60)
        print("Cypress Test Failure Classifier -- Neural Network Training")
        print("=" * 60)
        print(f"Architecture : 6 -> {HIDDEN_SIZE} (ReLU) -> 3 (Softmax)")
        print(f"Samples      : {len(x_train)} train / {len(x_test)} test")
        print(f"Epochs       : {EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
        print("-" * 60)

    net.train(x_train_n, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=verbose)

    y_pred = [net.predict(xi) for xi in x_test_n]
    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS ON TEST SET")
        print("=" * 60)
        print(classification_report(y_test, y_pred))
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print_confusion_matrix(cm)

    correct = sum(t == p for t, p in zip(y_test, y_pred, strict=False))
    return {
        "accuracy": correct / len(y_test),
        "confusion_matrix": cm,
        "n_test": len(y_test),
        "correct": correct,
    }


def predict_single(
    log_json: str,
    model: NeuralNetwork,
    means: list[float],
    stds: list[float],
) -> str:
    """Classify one JSON log string; return predicted label name."""
    entry = json.loads(log_json)
    raw = extract_features(entry)
    normed = [(v - m) / s for v, m, s in zip(raw, means, stds, strict=False)]
    idx = model.predict(normed)
    probs = model.predict_proba(normed)
    label = IDX_TO_LABEL[idx]
    print(f"\nPrediction : {label}")
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {IDX_TO_LABEL[i]:<15} {p:.4f}")
    return label


def classify_report(report_path: str) -> None:
    """
    Load a Cypress JSON report, classify every failed test, print summary.

    This is the main user-facing feature: the developer points the tool at
    a real Cypress JSON report and instantly sees what kind of failure each
    test has.
    """
    print("Training model on synthetic data...", end=" ", flush=True)
    model, means, stds = _build_trained_model()
    print("done.\n")

    failed_tests = load_cypress_report(report_path)

    if not failed_tests:
        print("No failed tests found in the report.")
        return

    # Classify each failed test
    results: list[tuple[dict, str, list[float]]] = []
    for test in failed_tests:
        raw = extract_features(test)
        normed = [(v - m) / s for v, m, s in zip(raw, means, stds, strict=False)]
        label = IDX_TO_LABEL[model.predict(normed)]
        probs = model.predict_proba(normed)
        results.append((test, label, probs))

    # Per-test output
    print("=" * 65)
    print(f"CYPRESS REPORT ANALYSIS  —  {report_path}")
    print(f"Failed tests: {len(results)}")
    print("=" * 65)

    for test, label, probs in results:
        suite = test.get("suite", "")
        title = test.get("title", "")
        msg = test.get("error_message", "")[:80]
        conf = max(probs)
        print(f"\n  Suite : {suite}")
        print(f"  Test  : {title}")
        if msg:
            print(f"  Error : {msg}")
        print(f"  --> Predicted cause : [{label.upper()}]  (confidence: {conf:.1%})")

    # Summary table
    counts: dict[str, int] = {}
    for _, label, _ in results:
        counts[label] = counts.get(label, 0) + 1

    total = len(results)
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Total failed tests : {total}")
    for label in ("timeout", "network_error", "ui_bug"):
        n = counts.get(label, 0)
        bar = "#" * n
        print(f"  {label:<15} : {n:>3}  ({n/total:>5.1%})  {bar}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cypress failure classifier")
    parser.add_argument("--predict", type=str, default=None,
                        help="JSON string of a single log entry to classify")
    parser.add_argument("--report", type=str, default=None,
                        help="Path to a Cypress JSON report file to analyse")
    args = parser.parse_args()

    if args.report:
        classify_report(args.report)
    elif args.predict:
        net, means, stds = _build_trained_model()
        predict_single(args.predict, net, means, stds)
    else:
        run_experiment(verbose=True)


if __name__ == "__main__":
    main()
