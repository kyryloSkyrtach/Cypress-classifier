"""Tests for the synthetic data generator."""

from app.data_generator import generate_dataset


class TestGenerateDataset:
    def test_correct_count(self):
        records = generate_dataset(n=300, seed=0)
        assert len(records) == 300

    def test_three_classes_present(self):
        records = generate_dataset(n=300, seed=1)
        labels = {r["label"] for r in records}
        assert labels == {"timeout", "network_error", "ui_bug"}

    def test_required_keys_present(self):
        records = generate_dataset(n=10, seed=2)
        required = {
            "execution_time_ms",
            "failed_step_index",
            "retry_count",
            "error_code",
            "dom_selector_depth",
            "network_call_count",
            "label",
        }
        for rec in records:
            assert required.issubset(rec.keys())

    def test_numeric_fields_non_negative(self):
        records = generate_dataset(n=60, seed=3)
        for rec in records:
            assert rec["execution_time_ms"] >= 0
            assert rec["failed_step_index"] >= 0
            assert rec["retry_count"] >= 0
            assert rec["dom_selector_depth"] >= 0
            assert rec["network_call_count"] >= 0

    def test_deterministic(self):
        r1 = generate_dataset(n=100, seed=42)
        r2 = generate_dataset(n=100, seed=42)
        assert [rec["label"] for rec in r1] == [rec["label"] for rec in r2]

    def test_balanced(self):
        records = generate_dataset(n=300, seed=5)
        counts = {}
        for rec in records:
            counts[rec["label"]] = counts.get(rec["label"], 0) + 1
        assert counts["timeout"] == 100
        assert counts["network_error"] == 100
        assert counts["ui_bug"] == 100
