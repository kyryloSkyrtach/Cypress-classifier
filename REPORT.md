# REPORT — Cypress Test Failure Classifier

**Authors:** Kyrylo Skyrtach, Mark Volobuiev

---

## 1. Problem Description

In projects using the Cypress E2E testing framework, test failures can occur for
fundamentally different reasons. The developer receives a failure notification and must
manually read JSON logs to determine the root cause — a time-consuming process that
slows down debugging.

**Goal:** Automatically classify each failed test into one of three categories:

- **timeout** — the test waited too long for a DOM element or server response
- **network_error** — the API returned an HTTP 4xx/5xx error or the connection failed
- **ui_bug** — the element was found, but its value or state did not match the expectation

**Practical result:** The developer points the tool at a Cypress JSON report and
immediately sees a breakdown like:

```text
timeout        5  (25%)
network_error  8  (40%)
ui_bug         7  (35%)
```

instead of reading each log manually.

---

## 2. Data Description

### Why synthetic data?

Real Cypress log datasets are not publicly available — test logs are internal
to specific projects and contain proprietary information. Synthetic data was
generated to simulate realistic failure scenarios for each of the three classes,
allowing the model to be trained and evaluated without access to real logs.

### Synthetic dataset (600 records, balanced)

Each record simulates a failed Cypress test log with 6 numeric features:

| Feature | timeout (typical) | network_error (typical) | ui_bug (typical) |
|---|---|---|---|
| `execution_time_ms` | ~15 000 ms | ~2 000 ms | ~4 000 ms |
| `failed_step_index` | ~8 | ~3 | ~5 |
| `retry_count` | ~3 | ~1 | ~0 |
| `error_code_category` | 0 (no HTTP error) | 2–4 (4xx/5xx) | 0 (no HTTP error) |
| `dom_selector_depth` | ~7 | ~3 | ~4 |
| `network_call_count` | ~5 | ~12 | ~4 |

Features follow Gaussian distributions with class-specific means, creating realistic
overlap between classes. Data is split 80/20 (480 train / 120 test). Z-score
normalisation is applied (computed on train set only).

### Real Cypress JSON report parser

In addition to synthetic training data, the project includes a parser for real Cypress
JSON reports (cypress --reporter json format). The parser:

1. Extracts all failed tests from the report
2. Infers numeric features from error messages and stack traces:
   - HTTP status codes via regex (`[45]\d{2}`)
   - Connection errors via keywords (`ECONNREFUSED`, `net::`)
   - DOM selector depth from CSS selectors in assertion messages
   - Step index from line numbers in stack traces
3. Passes the feature vectors to the trained model for classification

---

## 3. Algorithm — Neural Network with One Hidden Layer

### Why not a simple perceptron?

The original plan was three binary perceptrons (One-vs-Rest). Following the lecturer's
recommendation, we upgraded to a neural network with one hidden layer.

Advantages:

- Can learn **non-linear decision boundaries** (classes overlap in feature space)
- A single model handles all three classes simultaneously via Softmax output
- Backpropagation provides a principled gradient-based learning rule

### Architecture

```text
Input (6)  ->  Hidden (16, ReLU)  ->  Output (3, Softmax)
```

### Implementation (pure Python, no ML libraries)

| Component | Implementation |
|---|---|
| Forward pass | Matrix-vector dot products, ReLU, Softmax |
| Backward pass | Chain rule, delta propagation |
| Weight init | He initialisation (`σ = sqrt(2/fan_in)`) |
| Optimisation | Mini-batch SGD (batch=32, lr=0.05) |
| Loss | Cross-entropy `L = -log(p_correct_class)` |
| Normalisation | Z-score (fit on train set) |

---

## 4. Experiment Results

### Training curve

| Epoch | Loss | Train Accuracy |
|---|---|---|
| 10 | 0.0547 | 99.6% |
| 50 | 0.0076 | 100% |
| 100 | 0.0034 | 100% |
| 150 | 0.0021 | 100% |

### Test set results (120 samples)

```text
Class            Precision    Recall        F1   Support
---------------------------------------------------------
timeout              1.000     1.000     1.000        44
network_error        1.000     1.000     1.000        36
ui_bug               1.000     1.000     1.000        40
---------------------------------------------------------
macro avg            1.000     1.000     1.000       120

Accuracy: 1.0000  (120/120)
```

### Confusion matrix

```text
                  timeout  network_error  ui_bug
timeout      |        44              0       0
network_error|         0             36       0
ui_bug       |         0              0      40
```

All 120 test samples classified correctly — perfect separation of the three classes.

### Real report classification

The example report (`data/example_report.json`, 3 failed tests) was classified correctly:

| Test | Error | Predicted | Confidence |
|---|---|---|---|
| should show error with invalid credentials | AssertionError on div content | `ui_bug` | 99.9% |
| should load dashboard data from API | HTTP 503 response | `network_error` | 100.0% |
| should redirect to login on session timeout | Timed out after 16000ms | `timeout` | 100.0% |

---

## 5. Conclusions

1. The neural network achieves **100% accuracy** on the synthetic test set (120/120).
2. The Cypress JSON report parser correctly classifies all three failure types
   in a real-format report using heuristic feature extraction from error messages.
3. Perfect accuracy on synthetic data is expected — the features were designed to
   clearly separate the three classes. In a real setting with noisier data,
   accuracy would be lower.
4. In a production setting, the parser could be extended with actual Cypress
   plugin data (retry counts, real network call logs) for even higher accuracy.
5. Implementing backpropagation from scratch demonstrated deep understanding of
   gradient-based learning without relying on ML library black-boxes.

### Possible improvements

- Collect and manually label real Cypress logs for training
- Extend parser with Cypress plugin data (actual retry counts, HAR network logs)
- Add momentum or Adam optimiser for faster convergence
- Experiment with larger hidden layer or two hidden layers

---

## 6. Division of Work

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