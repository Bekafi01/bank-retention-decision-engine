import pytest
import numpy as np
import pandas as pd

from src.monitoring.drift_detector import calculate_psi, evaluate_feature_drift


def test_calculate_psi_no_drift():
    np.random.seed(42)
    expected = np.random.normal(loc=100.0, scale=15.0, size=2000)
    actual = np.random.normal(loc=100.0, scale=15.0, size=2000)

    psi = calculate_psi(expected, actual)
    assert isinstance(psi, float)
    # Identical distributions should have negligible PSI (< 0.05)
    assert psi < 0.05


def test_calculate_psi_significant_drift():
    np.random.seed(42)
    expected = np.random.normal(loc=100.0, scale=15.0, size=2000)
    actual = np.random.normal(loc=140.0, scale=30.0, size=2000)

    psi = calculate_psi(expected, actual)
    assert isinstance(psi, float)
    # Heavily shifted distributions should trigger significant drift (PSI >= 0.20)
    assert psi >= 0.20


def test_evaluate_feature_drift():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "CreditScore": np.random.randint(500, 800, size=1000),
        "Balance": np.random.normal(70000, 20000, size=1000),
        "Geography": np.random.choice(["France", "Germany", "Spain"], p=[0.5, 0.25, 0.25], size=1000),
        "NumOfProducts": np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1], size=1000),
    })

    # Shift Balance and Geography intentionally
    curr_df = pd.DataFrame({
        "CreditScore": np.random.randint(500, 800, size=1000),
        "Balance": np.random.normal(120000, 35000, size=1000),  # Heavy drift
        "Geography": np.random.choice(["France", "Germany", "Spain"], p=[0.1, 0.8, 0.1], size=1000),  # Heavy shift
        "NumOfProducts": np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1], size=1000),
    })

    report = evaluate_feature_drift(ref_df, curr_df)

    assert "is_overall_drift_detected" in report
    assert report["is_overall_drift_detected"] is True
    assert "Balance" in report["drifted_features"]
    assert "Geography" in report["drifted_features"]
    assert "CreditScore" not in report["drifted_features"]
