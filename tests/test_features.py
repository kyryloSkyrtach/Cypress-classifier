"""Tests for feature extraction, normalisation and Cypress report parser."""

import pytest

from app.features import (
    NUM_FEATURES,
    extract_features,
    extract_label,
    normalize,
    parse_cypress_report,
)

SYNTHETIC_ENTRY = {
    "execution_time_ms": 5000,
    "failed_step_index": 3,
    "retry_count": 1,
    "error_code": 404,
    "dom_selector_depth": 4,
    "network_call_count": 8,
    "label": "network_error",
}

CYPRESS_REPORT = {
    "stats": {"tests": 3, "failures": 2},
    "results": [
        {
            "suite": "Login Tests",
            "tests": [
                {"title": "passes", "status": "passed", "duration": 1000, "err": {}},
                {
                    "title": "ui fails",
                    "status": "failed",
                    "duration": 2200,
                    "err": {
                        "message": "AssertionError: expected '<div.msg>' to contain 'Error'",
                        "stack": "at Context.eval (login.cy.js:45:10)",
                    },
                },
                {
                    "title": "api fails",
                    "status": "failed",
                    "duration": 900,
                    "err": {
                        "message": "cy.request() failed with status code 503",
                        "stack": "at Context.eval (api.cy.js:12:5)",
                    },
                },
            ],
        }
    ],
}


class TestExtractFeatures:
    def test_output_length(self):
        assert len(extract_features(SYNTHETIC_ENTRY)) == NUM_FEATURES

    def test_all_floats(self):
        assert all(isinstance(f, float) for f in extract_features(SYNTHETIC_ENTRY))

    def test_missing_keys_default_zero(self):
        features = extract_features({})
        assert features[0] == 0.0
        assert features[1] == 0.0

    def test_error_code_404_encodes_2(self):
        assert extract_features({"error_code": 404})[3] == 2.0

    def test_error_code_500_encodes_3(self):
        assert extract_features({"error_code": 500})[3] == 3.0

    def test_error_code_none_encodes_zero(self):
        assert extract_features({"error_code": None})[3] == 0.0

    def test_error_code_0_encodes_1(self):
        assert extract_features({"error_code": 0})[3] == 1.0


class TestExtractLabel:
    def test_timeout(self):
        assert extract_label({"label": "timeout"}) == 0

    def test_network_error(self):
        assert extract_label({"label": "network_error"}) == 1

    def test_ui_bug(self):
        assert extract_label({"label": "ui_bug"}) == 2

    def test_case_insensitive(self):
        assert extract_label({"label": "TIMEOUT"}) == 0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown label"):
            extract_label({"label": "unknown"})


class TestNormalize:
    def _make(self):
        x_train = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        x_test = [[0.0, 0.0], [4.0, 40.0]]
        return x_train, x_test

    def test_train_mean_zero(self):
        x_tr_n, _, _, _ = normalize(*self._make())
        for f in range(2):
            assert abs(sum(r[f] for r in x_tr_n) / len(x_tr_n)) < 1e-9

    def test_shape_preserved(self):
        x_train, x_test = self._make()
        x_tr_n, x_te_n, _, _ = normalize(x_train, x_test)
        assert len(x_tr_n) == 3
        assert len(x_te_n) == 2

    def test_stds_positive(self):
        _, _, _, stds = normalize(*self._make())
        assert all(s > 0 for s in stds)


class TestParseCypressReport:
    def test_only_failed_returned(self):
        results = parse_cypress_report(CYPRESS_REPORT)
        assert len(results) == 2

    def test_titles_extracted(self):
        results = parse_cypress_report(CYPRESS_REPORT)
        titles = [r["title"] for r in results]
        assert "ui fails" in titles
        assert "api fails" in titles

    def test_passed_not_included(self):
        results = parse_cypress_report(CYPRESS_REPORT)
        assert all(r["title"] != "passes" for r in results)

    def test_each_has_feature_keys(self):
        results = parse_cypress_report(CYPRESS_REPORT)
        required = {"execution_time_ms", "failed_step_index", "retry_count",
                    "error_code", "dom_selector_depth", "network_call_count"}
        for r in results:
            assert required.issubset(r.keys())

    def test_duration_mapped(self):
        results = parse_cypress_report(CYPRESS_REPORT)
        durations = {r["title"]: r["execution_time_ms"] for r in results}
        assert durations["ui fails"] == 2200.0
        assert durations["api fails"] == 900.0

    def test_http_error_code_detected(self):
        results = parse_cypress_report(CYPRESS_REPORT)
        api_test = next(r for r in results if r["title"] == "api fails")
        assert api_test["error_code"] == 503

    def test_no_failures_returns_empty(self):
        report = {"results": [{"suite": "S", "tests": [
            {"title": "t", "status": "passed", "duration": 100, "err": {}}
        ]}]}
        assert parse_cypress_report(report) == []

    def test_feature_vector_length(self):
        results = parse_cypress_report(CYPRESS_REPORT)
        for r in results:
            assert len(extract_features(r)) == NUM_FEATURES
