import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import ARTIFACTS_DIR, RAW_DATA_PATH
from src.preprocess import load_and_clean_data, split_features_and_target
from src.explainability import BankChurnExplainer


@pytest.fixture(scope="module")
def champion_pipeline():
    raw_pipe_path = ARTIFACTS_DIR / "raw_pipeline.pkl"
    if not raw_pipe_path.exists():
        pytest.skip("raw_pipeline.pkl not found. Run train_pipeline first.")
    return joblib.load(raw_pipe_path)


def test_explainer_initialization(champion_pipeline):
    explainer = BankChurnExplainer(champion_pipeline)
    feature_names = explainer.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 10


def test_single_customer_explanation(champion_pipeline):
    explainer = BankChurnExplainer(champion_pipeline)
    customer_df = pd.DataFrame([{
        "CreditScore": 620,
        "Geography": "France",
        "Gender": "Female",
        "Age": 48,
        "Tenure": 4,
        "Balance": 125000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 85000.0,
        "Complain": 1,
        "SatisfactionScore": 2,
        "CardType": "SILVER",
        "PointEarned": 450,
    }])

    attributions = explainer.explain_customer(customer_df, top_k=5)
    assert isinstance(attributions, dict)
    assert len(attributions) <= 5
    # Every attribution value should be a float
    for feat, val in attributions.items():
        assert isinstance(feat, str)
        assert isinstance(val, float)


def test_prescriptive_counterfactual_engine(champion_pipeline):
    explainer = BankChurnExplainer(champion_pipeline)
    # High-risk customer with complaint and inactivity
    customer_df = pd.DataFrame([{
        "CreditScore": 600,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 3,
        "Balance": 110000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 75000.0,
        "Complain": 1,
        "SatisfactionScore": 1,
        "CardType": "SILVER",
        "PointEarned": 300,
    }])

    rx = explainer.generate_prescriptive_counterfactuals(customer_df)
    assert "baseline_probability" in rx
    assert "simulated_probability" in rx
    assert "absolute_risk_drop" in rx
    assert "risk_reduction_pct" in rx
    assert "recommended_interventions" in rx
    assert len(rx["recommended_interventions"]) >= 1

    # Simulated risk should be strictly lower than baseline risk
    assert rx["simulated_probability"] <= rx["baseline_probability"]
    assert rx["risk_reduction_pct"] >= 0.0
    assert rx["potential_deposit_retained"] == 110000.0
