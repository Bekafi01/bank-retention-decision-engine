import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.train_pipeline import (
    get_candidate_models,
    compute_expected_calibration_error,
    create_full_pipeline,
)
from src.evaluate import compute_bootstrap_ci


def test_get_candidate_models():
    zoo = get_candidate_models()
    assert "Logistic_Regression" in zoo
    assert "Random_Forest" in zoo
    assert "LightGBM" in zoo
    assert "XGBoost" in zoo
    assert "CatBoost" in zoo


def test_compute_expected_calibration_error():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Perfectly calibrated probabilities
    y_proba_perfect = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    ece_perfect = compute_expected_calibration_error(y_true, y_proba_perfect, n_bins=5)
    assert ece_perfect < 0.2

    # Poorly calibrated probabilities
    y_proba_bad = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    ece_bad = compute_expected_calibration_error(y_true, y_proba_bad, n_bins=5)
    assert ece_bad > ece_perfect


def test_compute_bootstrap_ci():
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 10)
    y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.8, 0.85, 0.9] * 10)

    ci = compute_bootstrap_ci(y_true, y_proba, n_bootstraps=100)
    assert "ROC_AUC" in ci
    assert "PR_AUC" in ci
    assert "Brier_Score" in ci
    assert ci["ROC_AUC"]["ci_lower"] <= ci["ROC_AUC"]["mean"] <= ci["ROC_AUC"]["ci_upper"]
    assert ci["PR_AUC"]["ci_lower"] <= ci["PR_AUC"]["mean"] <= ci["PR_AUC"]["ci_upper"]


def test_calibrated_pipeline_inference():
    df = pd.DataFrame({
        "CreditScore": [600, 700, 500, 750],
        "Geography": ["France", "Germany", "Spain", "France"],
        "Gender": ["Female", "Male", "Female", "Male"],
        "Age": [35, 45, 25, 55],
        "Tenure": [3, 7, 1, 8],
        "Balance": [50000.0, 0.0, 120000.0, 80000.0],
        "NumOfProducts": [1, 2, 3, 1],
        "HasCrCard": [1, 0, 1, 1],
        "IsActiveMember": [1, 1, 0, 0],
        "EstimatedSalary": [80000.0, 95000.0, 45000.0, 110000.0],
        "Complain": [0, 0, 1, 1],
        "SatisfactionScore": [4, 5, 1, 2],
        "CardType": ["GOLD", "SILVER", "DIAMOND", "PLATINUM"],
        "PointEarned": [500, 600, 800, 400],
    })
    y = np.array([0, 0, 1, 1])

    base_pipe = create_full_pipeline(LogisticRegression(), df)
    cal_pipe = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=2)
    cal_pipe.fit(df, y)

    proba = cal_pipe.predict_proba(df)[:, 1]
    assert len(proba) == 4
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
