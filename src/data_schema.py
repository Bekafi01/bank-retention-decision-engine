from typing import Optional, List, Dict, Any, Literal
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from pydantic import BaseModel, Field, ConfigDict


# Pandera DataFrame Schema for Raw Bank Dataset Validation
raw_bank_data_schema = DataFrameSchema(
    columns={
        "CreditScore": Column(int, Check.in_range(300, 900), nullable=False),
        "Geography": Column(str, Check.isin(["France", "Germany", "Spain"]), nullable=False),
        "Gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
        "Age": Column(int, Check.in_range(18, 105), nullable=False),
        "Tenure": Column(int, Check.in_range(0, 50), nullable=False),
        "Balance": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "NumOfProducts": Column(int, Check.in_range(1, 10), nullable=False),
        "HasCrCard": Column(int, Check.isin([0, 1]), nullable=False),
        "IsActiveMember": Column(int, Check.isin([0, 1]), nullable=False),
        "EstimatedSalary": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "Exited": Column(int, Check.isin([0, 1]), nullable=False, required=False),
        "Complain": Column(int, Check.isin([0, 1]), nullable=False, required=False),
        "SatisfactionScore": Column(int, Check.in_range(1, 5), nullable=False, required=False),
        "CardType": Column(str, Check.isin(["SILVER", "GOLD", "PLATINUM", "DIAMOND"]), nullable=False, required=False),
        "PointEarned": Column(int, Check.greater_than_or_equal_to(0), nullable=False, required=False),
    },
    coerce=True,
    strict=False,
)

# Pandera DataFrame Schema for Processed Engineered Dataset
engineered_bank_data_schema = DataFrameSchema(
    columns={
        "BalanceToSalaryRatio": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "IsZeroBalance": Column(int, Check.isin([0, 1]), nullable=False),
        "WealthTier": Column(str, Check.isin(["Zero_Balance", "Mass_Market", "Affluent", "High_Net_Worth"]), nullable=False),
        "TenureToAgeRatio": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "CreditScoreToAgeRatio": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "IsMultiProductRisk": Column(int, Check.isin([0, 1]), nullable=False),
        "ComplaintInactivityRisk": Column(int, Check.isin([0, 1]), nullable=False),
        "LoyaltyIndex": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
    },
    coerce=True,
    strict=False,
)


class BankCustomerInput(BaseModel):
    """Raw bank customer input schema for single inference."""
    model_config = ConfigDict(populate_by_name=True)

    customer_id: Optional[int] = Field(default=None, alias="CustomerId")
    surname: Optional[str] = Field(default=None, alias="Surname")
    credit_score: int = Field(..., ge=300, le=900, alias="CreditScore", description="Credit score between 300 and 900")
    geography: Literal["France", "Germany", "Spain"] = Field(..., alias="Geography", description="Country of residence")
    gender: Literal["Male", "Female"] = Field(..., alias="Gender", description="Gender of the customer")
    age: int = Field(..., ge=18, le=105, alias="Age", description="Customer age")
    tenure: int = Field(..., ge=0, le=50, alias="Tenure", description="Years customer has been with bank")
    balance: float = Field(..., ge=0.0, alias="Balance", description="Account deposit balance in €/$")
    num_of_products: int = Field(..., ge=1, le=10, alias="NumOfProducts", description="Number of bank products held")
    has_cr_card: int = Field(..., ge=0, le=1, alias="HasCrCard", description="1 if holds credit card, else 0")
    is_active_member: int = Field(..., ge=0, le=1, alias="IsActiveMember", description="1 if active member, else 0")
    estimated_salary: float = Field(..., ge=0.0, alias="EstimatedSalary", description="Estimated annual salary")
    complain: int = Field(default=0, ge=0, le=1, alias="Complain", description="1 if customer filed a complaint")
    satisfaction_score: int = Field(default=3, ge=1, le=5, alias="Satisfaction Score", description="Satisfaction rating 1 to 5")
    card_type: Literal["SILVER", "GOLD", "PLATINUM", "DIAMOND"] = Field(
        default="SILVER", alias="Card Type", description="Card tier held"
    )
    point_earned: int = Field(default=500, ge=0, alias="Point Earned", description="Loyalty points accumulated")


class ChurnPredictionResult(BaseModel):
    """Detailed response for a customer churn prediction."""
    customer_id: Optional[int] = None
    churn_probability: float = Field(..., description="Calibrated probability of churn [0.0 - 1.0]")
    is_churn_predicted: bool = Field(..., description="True if probability exceeds optimal decision threshold")
    decision_threshold_used: float = Field(..., description="The decision cutoff threshold applied")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(..., description="Assigned risk segment")
    deposit_at_risk: float = Field(..., description="Customer balance at risk of departure")
    annual_nim_at_risk: float = Field(..., description="Annual Net Interest Margin at risk")
    recommended_action: str = Field(..., description="Strategic retention recommendation")
    expected_net_savings: float = Field(..., description="Expected monetary benefit from targeted intervention")
    top_churn_drivers: Optional[Dict[str, float]] = Field(default=None, description="Local SHAP feature attributions")


class CounterfactualRecommendation(BaseModel):
    """Prescriptive 'what-if' recommendation for relationship managers."""
    customer_id: Optional[int] = None
    baseline_probability: float
    simulated_probability: float
    risk_reduction_pct: float
    recommended_interventions: List[str]
    potential_deposit_saved: float


class ModelMetadata(BaseModel):
    """Metadata and performance specs of the champion model."""
    model_name: str
    model_version: str
    training_timestamp: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    f1_score: float
    optimal_decision_threshold: float
    features: List[str]
