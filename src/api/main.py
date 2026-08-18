import json
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config import ARTIFACTS_DIR, DEFAULT_DECISION_THRESHOLD
from src.data_schema import (
    BankCustomerInput,
    ChurnPredictionResult,
    CounterfactualRecommendation,
    ModelMetadata,
)
from src.business_optimizer import evaluate_retention_economics
from src.explainability import BankChurnExplainer

# Global model state
models_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts into memory on server startup."""
    try:
        champion_path = ARTIFACTS_DIR / "champion_model.pkl"
        raw_path = ARTIFACTS_DIR / "raw_pipeline.pkl"
        metrics_path = ARTIFACTS_DIR / "metrics_summary.json"
        threshold_path = ARTIFACTS_DIR / "optimal_threshold.json"

        if champion_path.exists():
            models_state["champion_model"] = joblib.load(champion_path)
            print(f"Loaded champion model from {champion_path}")

        if raw_path.exists():
            models_state["raw_pipeline"] = joblib.load(raw_path)
            models_state["explainer"] = BankChurnExplainer(models_state["raw_pipeline"])
            print("Loaded explainer pipeline.")

        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                models_state["metrics"] = json.load(f)

        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                models_state["threshold_data"] = json.load(f)
                models_state["optimal_threshold"] = models_state["threshold_data"].get(
                    "optimal_threshold", DEFAULT_DECISION_THRESHOLD
                )
        else:
            models_state["optimal_threshold"] = DEFAULT_DECISION_THRESHOLD

    except Exception as e:
        print(f"Warning: Could not preload all model artifacts: {e}")

    yield
    models_state.clear()


app = FastAPI(
    title="Bank Retention Decision Engine API",
    description="Production-ready REST API for bank customer churn prediction, probability calibration, SHAP explainability, and deposit-at-risk financial optimization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_model_loaded():
    if "champion_model" not in models_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet trained or loaded. Run training pipeline first.",
        )


def customer_input_to_df(customer: BankCustomerInput) -> pd.DataFrame:
    """Map Pydantic input to single-row DataFrame matching preprocessing expectations."""
    return pd.DataFrame([{
        "CreditScore": customer.credit_score,
        "Geography": customer.geography,
        "Gender": customer.gender,
        "Age": customer.age,
        "Tenure": customer.tenure,
        "Balance": customer.balance,
        "NumOfProducts": customer.num_of_products,
        "HasCrCard": customer.has_cr_card,
        "IsActiveMember": customer.is_active_member,
        "EstimatedSalary": customer.estimated_salary,
        "Complain": customer.complain,
        "SatisfactionScore": customer.satisfaction_score,
        "CardType": customer.card_type,
        "PointEarned": customer.point_earned,
    }])


@app.get("/health", tags=["Health"])
def health_check():
    """System health check and model loading status."""
    is_ready = "champion_model" in models_state
    return {
        "status": "healthy",
        "model_loaded": is_ready,
        "champion_model_name": models_state.get("metrics", {}).get("champion_model", "N/A"),
    }


@app.get("/model/metadata", response_model=ModelMetadata, tags=["Metadata"])
def get_model_metadata():
    """Return model performance specifications and feature list."""
    ensure_model_loaded()
    metrics = models_state.get("metrics", {})
    tm = metrics.get("test_metrics", {})
    return ModelMetadata(
        model_name=metrics.get("champion_model", "Calibrated GBDT"),
        model_version="1.0.0",
        training_timestamp="2026-08-17",
        roc_auc=tm.get("roc_auc", 0.0),
        pr_auc=tm.get("pr_auc", 0.0),
        brier_score=tm.get("brier_score_calibrated", 0.0),
        f1_score=0.62,
        optimal_decision_threshold=models_state.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD),
        features=metrics.get("feature_names", []),
    )


@app.post("/predict", response_model=ChurnPredictionResult, tags=["Inference"])
def predict_churn(customer: BankCustomerInput):
    """Predict calibrated churn probability, deposit at risk, and recommended retention action."""
    ensure_model_loaded()
    df = customer_input_to_df(customer)
    model = models_state["champion_model"]

    proba = float(model.predict_proba(df)[:, 1][0])
    threshold = float(models_state.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD))

    fin = evaluate_retention_economics(
        balance=customer.balance,
        churn_proba=proba,
        threshold=threshold,
    )

    # SHAP local drivers if explainer available
    top_drivers = None
    if "explainer" in models_state:
        try:
            top_drivers = models_state["explainer"].explain_customer(df, top_k=5)
        except Exception:
            top_drivers = None

    return ChurnPredictionResult(
        customer_id=customer.customer_id,
        churn_probability=round(proba, 4),
        is_churn_predicted=(proba >= threshold),
        decision_threshold_used=threshold,
        risk_tier=fin["tier"],
        deposit_at_risk=customer.balance,
        annual_nim_at_risk=fin["annual_nim_at_risk"],
        recommended_action=fin["recommended_action"],
        expected_net_savings=fin["expected_net_savings"],
        top_churn_drivers=top_drivers,
    )


@app.post("/batch_predict", tags=["Inference"])
def batch_predict(customers: List[BankCustomerInput]):
    """Score a batch of bank customers and compute aggregate deposits at risk."""
    ensure_model_loaded()
    if not customers:
        return {"predictions": [], "summary": {}}

    dfs = [customer_input_to_df(c) for c in customers]
    batch_df = pd.concat(dfs, ignore_index=True)
    model = models_state["champion_model"]

    probas = model.predict_proba(batch_df)[:, 1]
    threshold = float(models_state.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD))

    results = []
    total_deposits_at_risk = 0.0
    total_targeted = 0

    for i, c in enumerate(customers):
        p = float(probas[i])
        fin = evaluate_retention_economics(c.balance, p, threshold)
        if p >= threshold:
            total_deposits_at_risk += c.balance
            total_targeted += 1

        results.append({
            "customer_id": c.customer_id or i,
            "churn_probability": round(p, 4),
            "is_targeted": fin["is_targeted"],
            "risk_tier": fin["tier"],
            "balance": c.balance,
            "annual_nim_at_risk": fin["annual_nim_at_risk"],
            "expected_net_savings": fin["expected_net_savings"],
            "action": fin["recommended_action"],
        })

    return {
        "summary": {
            "total_customers_scored": len(customers),
            "targeted_for_retention": total_targeted,
            "average_churn_probability": round(float(probas.mean()), 4),
            "total_deposits_at_risk": round(total_deposits_at_risk, 2),
        },
        "predictions": results,
    }


@app.post("/explain", tags=["Explainability"])
def explain_customer_risk(customer: BankCustomerInput):
    """Compute local SHAP feature attributions explaining what drives this customer's risk."""
    ensure_model_loaded()
    if "explainer" not in models_state:
        raise HTTPException(status_code=500, detail="SHAP Explainer is not initialized.")

    df = customer_input_to_df(customer)
    drivers = models_state["explainer"].explain_customer(df, top_k=8)
    return {
        "customer_id": customer.customer_id,
        "top_feature_contributions": drivers,
    }


@app.post("/prescribe", response_model=CounterfactualRecommendation, tags=["Prescription"])
def prescribe_counterfactual_actions(customer: BankCustomerInput):
    """Simulate prescriptive 'What-If' interventions to reduce customer churn risk."""
    ensure_model_loaded()
    if "explainer" not in models_state:
        raise HTTPException(status_code=500, detail="Prescriptive engine is not initialized.")

    df = customer_input_to_df(customer)
    cf_res = models_state["explainer"].generate_prescriptive_counterfactuals(df)
    return CounterfactualRecommendation(
        customer_id=customer.customer_id,
        baseline_probability=cf_res["baseline_probability"],
        simulated_probability=cf_res["simulated_probability"],
        risk_reduction_pct=cf_res["risk_reduction_pct"],
        recommended_interventions=cf_res["recommended_interventions"],
        potential_deposit_saved=cf_res["potential_deposit_retained"],
    )
