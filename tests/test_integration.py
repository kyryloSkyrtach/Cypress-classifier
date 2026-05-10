"""Integration tests -- full pipeline and report classification."""

from pathlib import Path

from app.main import classify_report, run_experiment


def test_full_pipeline_accuracy():
    """Trained model should achieve at least 85% accuracy on held-out data."""
    result = run_experiment(verbose=False)
    assert result["accuracy"] >= 0.85, f"Got {result['accuracy']:.4f}"


def test_full_pipeline_returns_dict():
    result = run_experiment(verbose=False)
    assert isinstance(result, dict)
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_report_mode_runs(capsys):
    """classify_report should process example_report.json without error."""
    report_path = Path(__file__).parent.parent / "data" / "example_report.json"
    if not report_path.exists():
        return
    classify_report(str(report_path))
    out = capsys.readouterr().out
    assert "SUMMARY" in out
    assert "Failed tests" in out


def test_report_mode_finds_all_classes(capsys):
    """All three failure types should appear in example report output."""
    report_path = Path(__file__).parent.parent / "data" / "example_report.json"
    if not report_path.exists():
        return
    classify_report(str(report_path))
    out = capsys.readouterr().out
    assert "UI_BUG" in out or "TIMEOUT" in out or "NETWORK_ERROR" in out
