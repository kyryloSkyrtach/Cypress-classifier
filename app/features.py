"""Feature extraction from Cypress test log entries."""

from __future__ import annotations

import json
import re
from pathlib import Path

LABEL_TO_IDX: dict[str, int] = {
    "timeout": 0,
    "network_error": 1,
    "ui_bug": 2,
}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}

NUM_FEATURES = 6

FEATURE_NAMES = [
    "execution_time_ms",
    "failed_step_index",
    "retry_count",
    "error_code_category",
    "dom_selector_depth",
    "network_call_count",
]


def _encode_error_code(error_code: int | None) -> float:
    """Map HTTP status code to a 0-5 category float."""
    if error_code is None:
        return 0.0
    if error_code == 0:
        return 1.0   # connection refused
    if 400 <= error_code < 500:
        return 2.0   # 4xx client error
    if 500 <= error_code < 600:
        return 3.0   # 5xx server error
    if error_code in (408, 504):
        return 4.0   # explicit timeout codes
    return 5.0


def extract_features(log_entry: dict) -> list[float]:
    """Convert a single log dict to a numeric feature vector."""
    return [
        float(log_entry.get("execution_time_ms", 0)),
        float(log_entry.get("failed_step_index", 0)),
        float(log_entry.get("retry_count", 0)),
        _encode_error_code(log_entry.get("error_code")),
        float(log_entry.get("dom_selector_depth", 0)),
        float(log_entry.get("network_call_count", 0)),
    ]


def extract_label(log_entry: dict) -> int:
    """Return integer class index from log entry's 'label' field."""
    label = log_entry.get("label", "").lower()
    if label not in LABEL_TO_IDX:
        msg = f"Unknown label '{label}'. Valid labels: {list(LABEL_TO_IDX)}"
        raise ValueError(msg)
    return LABEL_TO_IDX[label]


def normalize(
    x_train: list[list[float]],
    x_test: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    """Z-score normalisation fitted on train set, applied to both sets."""
    n_features = len(x_train[0])
    means = [0.0] * n_features
    stds = [1.0] * n_features

    for f in range(n_features):
        col = [row[f] for row in x_train]
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / len(col)
        std = var**0.5 if var > 0 else 1.0
        means[f] = mean
        stds[f] = std

    def _norm(dataset: list[list[float]]) -> list[list[float]]:
        return [[(row[f] - means[f]) / stds[f] for f in range(n_features)] for row in dataset]

    return _norm(x_train), _norm(x_test), means, stds


def load_jsonl(path: str | Path) -> tuple[list[list[float]], list[int]]:
    """Load a .jsonl file and return (features, labels)."""
    features: list[list[float]] = []
    labels: list[int] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            features.append(extract_features(entry))
            labels.append(extract_label(entry))
    return features, labels


# ---------------------------------------------------------------------------
# Cypress JSON report parser
# ---------------------------------------------------------------------------

def _infer_error_code(message: str) -> int | None:
    """Try to extract an HTTP status code from an error message string."""
    match = re.search(r"\b([45]\d{2})\b", message)
    if match:
        return int(match.group(1))
    if re.search(r"net::|ECONNREFUSED|ENOTFOUND|network", message, re.IGNORECASE):
        return 0   # connection-level error
    return None


def _infer_dom_depth(message: str) -> int:
    """Estimate DOM selector depth from error message."""
    # Count '>' and ' ' separators in CSS selectors found in the message
    selector_match = re.search(r"'([^']+)'", message)
    if selector_match:
        selector = selector_match.group(1)
        return max(1, selector.count(">") + selector.count(" ") + 1)
    return 1


def _infer_step_index(stack: str) -> int:
    """Estimate step index from line number in stack trace."""
    match = re.search(r":(\d+):\d+\)", stack)
    if match:
        return min(int(match.group(1)) // 5, 20)  # normalise line → step
    return 1


def parse_cypress_report(report: dict) -> list[dict]:
    """
    Parse a Cypress JSON report and extract feature dicts for failed tests.

    Parameters
    ----------
    report:
        Parsed Cypress JSON report (mochawesome or cypress --reporter json format).

    Returns
    -------
    List of feature dicts, one per failed test, ready for extract_features().
    Each dict also contains 'title' and 'suite' for display purposes.
    """
    failed_tests: list[dict] = []

    results = report.get("results", [])
    for suite_idx, suite in enumerate(results):
        suite_name = suite.get("suite", suite.get("title", f"Suite {suite_idx}"))
        tests = suite.get("tests", [])

        for step_idx, test in enumerate(tests):
            if test.get("status") != "failed":
                continue

            duration = float(test.get("duration", 0))
            err = test.get("err", {})
            message = err.get("message", "")
            stack = err.get("stack", "")

            error_code = _infer_error_code(message)
            dom_depth = _infer_dom_depth(message)
            step_index = _infer_step_index(stack) if stack else step_idx + 1

            # Heuristic: network errors usually have short duration
            # timeouts have long duration, ui_bugs are in between
            network_calls = 0
            if error_code is not None:
                network_calls = 8  # likely had network activity
            elif duration > 8000:
                network_calls = 3
            else:
                network_calls = 4

            retry_count = 0  # standard Cypress doesn't retry by default

            failed_tests.append({
                # display metadata
                "title": test.get("title", "unknown"),
                "suite": suite_name,
                "full_title": test.get("fullTitle", ""),
                "error_message": message,
                # numeric features
                "execution_time_ms": duration,
                "failed_step_index": step_index,
                "retry_count": retry_count,
                "error_code": error_code,
                "dom_selector_depth": dom_depth,
                "network_call_count": network_calls,
            })

    return failed_tests


def load_cypress_report(path: str | Path) -> list[dict]:
    """Load a Cypress JSON report file and return list of failed test feature dicts."""
    with Path(path).open(encoding="utf-8") as fh:
        report = json.load(fh)
    return parse_cypress_report(report)
