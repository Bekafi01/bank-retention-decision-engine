import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.config import ARTIFACTS_DIR


@pytest.fixture(scope="module")
def client():
    # Using context manager ensures FastAPI lifespan startup handler runs
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_model_metadata(client):
    champion_path = ARTIFACTS_DIR / "champion_model.pkl"
    if not champion_path.exists():
        pytest.skip("champion_model.pkl not found")

    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "roc_auc" in data
    assert "optimal_decision_threshold" in data
    assert "features" in data
    assert len(data["features"]) > 0


def test_predict_single_customer_vip(client):
    champion_path = ARTIFACTS_DIR / "champion_model.pkl"
    if not champion_path.exists():
        pytest.skip("champion_model.pkl not found")

    payload = {
        "CustomerId": 15634602,
        "Surname": "Hargrave",
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 125000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 101348.88,
        "Complain": 1,
        "Satisfaction Score": 2,
        "Card Type": "DIAMOND",
        "Point Earned": 464,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "is_churn_predicted" in data
    assert "risk_tier" in data
    assert data["deposit_at_risk"] == 125000.0
    assert "annual_nim_at_risk" in data
    assert data["annual_nim_at_risk"] == round(125000.0 * 0.028, 2)
    assert "recommended_action" in data
    assert "top_churn_drivers" in data


def test_predict_validation_error(client):
    # Invalid credit score (<300) and invalid country
    payload = {
        "CreditScore": 150,
        "Geography": "Atlantis",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 10000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict(client):
    champion_path = ARTIFACTS_DIR / "champion_model.pkl"
    if not champion_path.exists():
        pytest.skip("champion_model.pkl not found")

    batch_payload = [
        {
            "CustomerId": 1,
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 45,
            "Tenure": 5,
            "Balance": 120000.0,
            "NumOfProducts": 3,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 80000.0,
            "Complain": 1,
            "Satisfaction Score": 1,
            "Card Type": "SILVER",
            "Point Earned": 300,
        },
        {
            "CustomerId": 2,
            "CreditScore": 750,
            "Geography": "Spain",
            "Gender": "Male",
            "Age": 30,
            "Tenure": 3,
            "Balance": 15000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 60000.0,
            "Complain": 0,
            "Satisfaction Score": 5,
            "Card Type": "PLATINUM",
            "Point Earned": 800,
        },
    ]

    response = client.post("/batch_predict", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["summary"]["total_customers_scored"] == 2
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_explain_endpoint(client):
    champion_path = ARTIFACTS_DIR / "champion_model.pkl"
    if not champion_path.exists():
        pytest.skip("champion_model.pkl not found")

    payload = {
        "CustomerId": 101,
        "CreditScore": 600,
        "Geography": "Germany",
        "Gender": "Male",
        "Age": 50,
        "Tenure": 4,
        "Balance": 95000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 70000.0,
        "Complain": 1,
        "Satisfaction Score": 2,
        "Card Type": "GOLD",
        "Point Earned": 400,
    }

    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_feature_contributions" in data
    assert isinstance(data["top_feature_contributions"], dict)


def test_prescribe_endpoint(client):
    champion_path = ARTIFACTS_DIR / "champion_model.pkl"
    if not champion_path.exists():
        pytest.skip("champion_model.pkl not found")

    payload = {
        "CustomerId": 202,
        "CreditScore": 580,
        "Geography": "France",
        "Gender": "Female",
        "Age": 48,
        "Tenure": 2,
        "Balance": 115000.0,
        "NumOfProducts": 4,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 85000.0,
        "Complain": 1,
        "Satisfaction Score": 1,
        "Card Type": "SILVER",
        "Point Earned": 250,
    }

    response = client.post("/prescribe", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline_probability" in data
    assert "simulated_probability" in data
    assert "risk_reduction_pct" in data
    assert "recommended_interventions" in data
    assert data["simulated_probability"] <= data["baseline_probability"]
