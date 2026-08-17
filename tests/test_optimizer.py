import numpy as np
import pandas as pd
from src.business_optimizer import (
    calculate_nim_value,
    evaluate_retention_economics,
    optimize_decision_threshold,
    segment_portfolio_customers,
    compute_portfolio_deposit_risk,
)


def test_calculate_nim_value():
    balance = 100000.0
    nim = calculate_nim_value(balance, nim_rate=0.028)
    assert nim == 2800.0


def test_evaluate_retention_economics_vip():
    res = evaluate_retention_economics(
        balance=80000.0,
        churn_proba=0.75,
        threshold=0.35,
    )
    assert res["is_targeted"] is True
    assert res["is_vip"] is True
    assert res["tier"] == "CRITICAL"
    assert "RM VIP Outreach" in res["quadrant"]
    assert res["intervention_cost"] == 150.0
    assert res["expected_net_savings"] > 0


def test_evaluate_retention_economics_digital():
    res = evaluate_retention_economics(
        balance=25000.0,
        churn_proba=0.60,
        threshold=0.35,
    )
    assert res["is_targeted"] is True
    assert res["is_vip"] is False
    assert res["tier"] == "MEDIUM"
    assert "Digital Offer" in res["quadrant"]
    assert res["intervention_cost"] == 25.0


def test_evaluate_retention_economics_low_risk_wealth():
    res = evaluate_retention_economics(
        balance=120000.0,
        churn_proba=0.15,
        threshold=0.35,
    )
    assert res["is_targeted"] is False
    assert res["is_vip"] is True
    assert res["tier"] == "LOW"
    assert "Wealth Cross-Sell" in res["quadrant"]
    assert res["intervention_cost"] == 0.0


def test_optimize_decision_threshold():
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    y_proba = np.random.uniform(0.0, 1.0, size=n)
    balances = np.random.uniform(1000.0, 150000.0, size=n)

    res = optimize_decision_threshold(y_true, y_proba, balances)
    assert "optimal_threshold" in res
    assert "max_net_profit_saved" in res
    assert 0.0 < res["optimal_threshold"] < 1.0
    assert "threshold_curve" in res


def test_segment_portfolio_customers():
    df = pd.DataFrame({
        "CustomerId": [1, 2, 3, 4],
        "Balance": [90000.0, 30000.0, 150000.0, 10000.0],
    })
    probs = np.array([0.80, 0.70, 0.10, 0.05])
    segmented = segment_portfolio_customers(df, probs, threshold=0.35)

    assert "RetentionTier" in segmented.columns
    assert "RecommendedAction" in segmented.columns
    assert "Tier 1" in segmented["RetentionTier"].iloc[0]
    assert "Tier 2" in segmented["RetentionTier"].iloc[1]
    assert "Tier 3" in segmented["RetentionTier"].iloc[2]
    assert "Tier 4" in segmented["RetentionTier"].iloc[3]


def test_compute_portfolio_deposit_risk():
    df = pd.DataFrame({
        "Balance": [100000.0, 20000.0, 80000.0, 15000.0],
    })
    probs = np.array([0.85, 0.65, 0.10, 0.05])
    risk_summary = compute_portfolio_deposit_risk(df, probs, threshold=0.35)

    assert risk_summary["total_portfolio_customers"] == 4
    assert risk_summary["total_portfolio_deposits"] == 215000.0
    assert risk_summary["deposits_at_risk"] == 120000.0
    assert risk_summary["total_targeted_customers"] == 2
    assert risk_summary["tier_breakdown"]["tier_1_rm_vip"] == 1
    assert risk_summary["tier_breakdown"]["tier_2_digital_incentive"] == 1
    assert risk_summary["tier_breakdown"]["tier_3_wealth_cross_sell"] == 1
    assert risk_summary["tier_breakdown"]["tier_4_standard_service"] == 1

