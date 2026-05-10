# Cypress Test Failure Classifier

[![python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![ruff](https://github.com/kyryloSkyrtach/Cypress-classifier/workflows/Ruff/badge.svg)](https://github.com/kyryloSkyrtach/Cypress-classifier/actions?query=branch%3Amain)
[![pytest](https://github.com/kyryloSkyrtach/Cypress-classifier/workflows/Pytest/badge.svg)](https://github.com/kyryloSkyrtach/Cypress-classifier/actions?query=branch%3Amain)
[![markdown](https://github.com/kyryloSkyrtach/Cypress-classifier/workflows/Markdown%20Lint/badge.svg)](https://github.com/kyryloSkyrtach/Cypress-classifier/actions?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://license.md/licenses/mit-license/)

Automatic classification of Cypress E2E test failure causes using a Neural Network.

> Authors: Kyrylo Skyrtach, Mark Volobuiev

## Problem

When a Cypress test fails, the developer receives a notification and must manually analyse
logs to determine the cause. This project automatically classifies each failed test into
one of three categories:

| Class | Description |
|---|---|
| `timeout` | Test waited too long for a DOM element or server response |
| `network_error` | API returned 4xx/5xx or connection was not established |
| `ui_bug` | Element was found, but its value or state did not match the expectation |

## How It Works

1. A neural network is trained on **synthetic Cypress log data** (600 samples)
1. The trained model classifies failed tests from a **real Cypress JSON report**
1. The developer instantly sees what kind of failure each test has — no manual log reading

Synthetic data is used as a training set because real labelled Cypress logs are not
publicly available. This is standard practice when real data requires manual labelling.

## Algorithm

Feed-forward neural network with one hidden layer trained via backpropagation:

```text
Input (6 features) -> Hidden layer / 16 neurons / ReLU -> Output (3 classes) / Softmax
```

- He weight initialisation
- Mini-batch stochastic gradient descent (batch size 32)
- Cross-entropy loss
- Z-score normalisation
- **No ML libraries used** (scikit-learn, tensorflow etc.) — pure Python + stdlib

## Input Features (extracted from Cypress logs)

| # | Feature | Description |
|---|---|---|
| 0 | `execution_time_ms` | Total test duration (ms) |
| 1 | `failed_step_index` | Index of the step where the test failed |
| 2 | `retry_count` | Number of automatic Cypress retries |
| 3 | `error_code_category` | Encoded HTTP error category (0-5) |
| 4 | `dom_selector_depth` | CSS selector nesting depth |
| 5 | `network_call_count` | Number of network calls during the test |

## Requirements

- Python >= 3.14
- No ML libraries required

## Installation

```shell
git clone <repo-url>
cd cypress-classifier
uv sync --dev
```

If `uv` is not installed:

```shell
pip install uv
```

## Running

### Analyse a real Cypress JSON report (main use case)

```shell
uv run python -m app.main --report data/example_report.json
```

### Full training experiment (generates data, trains, evaluates)

```shell
uv run python -m app.main
```

### Classify a single JSON log entry

```shell
uv run python -m app.main --predict "{\"execution_time_ms\":18000,\"failed_step_index\":9,\"retry_count\":3,\"error_code\":null,\"dom_selector_depth\":8,\"network_call_count\":4}"
```

## Test

```shell
uv run pytest
```

56 tests covering neural network, feature extraction, report parser, metrics, integration.

## Project Structure

```text
cypress-classifier/
├── app/
│   ├── neural_network.py     # NN from scratch: forward pass, backprop, SGD
│   ├── features.py           # Feature extraction + Cypress JSON report parser
│   ├── data_generator.py     # Synthetic Cypress log generator (training data)
│   ├── metrics.py            # Accuracy, confusion matrix, F1 (no sklearn)
│   └── main.py               # Training pipeline + CLI (--report / --predict)
├── data/
│   └── example_report.json   # Example Cypress JSON report (3 failed tests)
├── tests/                    # 56 tests
├── CHANGELOG.md
├── CONTRIBUTING.md
├── README.md
└── REPORT.md
```

## Division of Work

### Kyrylo Skyrtach

- `app/neural_network.py` — full implementation of the neural network from scratch:
  forward pass (matrix-vector multiplication, ReLU, Softmax), backpropagation
  (chain rule, delta propagation), mini-batch SGD, He weight initialisation
- `app/main.py` — training pipeline and CLI (`--report` / `--predict` modes)
- `tests/test_neural_network.py` — unit tests for the neural network
- `tests/test_integration.py` — end-to-end pipeline integration tests
- `README.md`, `REPORT.md` — documentation
- `data/example_report.json` — example Cypress report with 3 failure types

### Mark Volobuiev

- `app/features.py` — feature extraction from Cypress logs, Z-score normalisation,
  Cypress JSON report parser (HTTP error detection, DOM depth inference, stack trace parsing)
- `app/data_generator.py` — synthetic Cypress log generator (3 classes, Gaussian distributions)
- `app/metrics.py` — accuracy, confusion matrix, precision/recall/F1 without sklearn
- `tests/test_features.py`, `tests/test_data_generator.py`, `tests/test_metrics.py`

## Security

If you discover any security-related issues, please email [ks30366@zpsb.pl](mailto:ks30366@zpsb.pl) instead of using the issue tracker.

---

Copyright (c) 2026 Kyrylo Skyrtach, Mark Volobuiev