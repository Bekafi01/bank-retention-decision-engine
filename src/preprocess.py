import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    ID_COLS,
    TARGET_COL,
)
from src.data_schema import raw_bank_data_schema, engineered_bank_data_schema


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize messy column names to standard snake_case/clean format."""
    col_mapping = {
        "RowNumber": "RowNumber",
        "CustomerId": "CustomerId",
        "Surname": "Surname",
        "CreditScore": "CreditScore",
        "Geography": "Geography",
        "Gender": "Gender",
        "Age": "Age",
        "Tenure": "Tenure",
        "Balance": "Balance",
        "NumOfProducts": "NumOfProducts",
        "HasCrCard": "HasCrCard",
        "IsActiveMember": "IsActiveMember",
        "EstimatedSalary": "EstimatedSalary",
        "Exited": "Exited",
        "Complain": "Complain",
        "Satisfaction Score": "SatisfactionScore",
        "SatisfactionScore": "SatisfactionScore",
        "Card Type": "CardType",
        "CardType": "CardType",
        "Point Earned": "PointEarned",
        "PointEarned": "PointEarned",
    }
    df = df.rename(columns=lambda col: col_mapping.get(col, col))
    return df


class BankingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Banking & Wealth Domain Feature Engineering.
    Computes wealth ratios, customer stickiness, loyalty scores, and complaint interactions.
    """

    def __init__(self):
        self.card_weights = {
            "SILVER": 1.0,
            "GOLD": 2.0,
            "PLATINUM": 3.0,
            "DIAMOND": 4.0,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # 1. Wealth & Liquidity Ratios
        if "Balance" in X_out.columns and "EstimatedSalary" in X_out.columns:
            X_out["BalanceToSalaryRatio"] = X_out["Balance"] / (X_out["EstimatedSalary"] + 1.0)
            X_out["IsZeroBalance"] = (X_out["Balance"] == 0).astype(int)
            
            # Wealth tiers
            X_out["WealthTier"] = pd.cut(
                X_out["Balance"],
                bins=[-1.0, 1.0, 50000.0, 120000.0, float("inf")],
                labels=["Zero_Balance", "Mass_Market", "Affluent", "High_Net_Worth"],
            ).astype(str)

        # 2. Age & Tenure Dynamics
        if "Tenure" in X_out.columns and "Age" in X_out.columns:
            X_out["TenureToAgeRatio"] = X_out["Tenure"] / np.maximum(X_out["Age"], 18)

        if "CreditScore" in X_out.columns and "Age" in X_out.columns:
            X_out["CreditScoreToAgeRatio"] = X_out["CreditScore"] / np.maximum(X_out["Age"], 18)

        # 3. Product Bundling & Friction Dynamics
        if "NumOfProducts" in X_out.columns:
            # In retail banking, 1 product is standard, 2 is sweet spot, 3-4 has high churn risk
            X_out["IsMultiProductRisk"] = (X_out["NumOfProducts"] >= 3).astype(int)

        # 4. Service Complaint & Inactivity Multipliers
        if "Complain" in X_out.columns:
            if "IsActiveMember" in X_out.columns:
                # A customer who complained and is also inactive is an extreme churn flight risk
                X_out["ComplaintInactivityRisk"] = X_out["Complain"] * (1 - X_out["IsActiveMember"])
            if "SatisfactionScore" in X_out.columns:
                # Disgruntled complainer interaction (higher when satisfaction is lower)
                X_out["ComplaintRisk"] = X_out["Complain"] * (6 - X_out["SatisfactionScore"])

        # 5. Loyalty & Engagement Index
        if (
            "PointEarned" in X_out.columns
            and "SatisfactionScore" in X_out.columns
            and "CardType" in X_out.columns
        ):
            card_tier_num = X_out["CardType"].astype(str).str.upper().map(self.card_weights).fillna(1.0)
            X_out["LoyaltyIndex"] = (
                (X_out["PointEarned"] / 1000.0)
                * (X_out["SatisfactionScore"] / 5.0)
                * card_tier_num
            )

        return X_out


def load_and_clean_data(file_path: Optional[str] = None, validate: bool = True) -> pd.DataFrame:
    """Load raw bank churn CSV, standardize schema, validate, and drop ID columns."""
    target_path = file_path or str(RAW_DATA_PATH)
    df = pd.read_csv(target_path)
    df = standardize_column_names(df)

    # Fill defaults if optional columns were missing from standard 10k dataset
    if "Complain" not in df.columns:
        df["Complain"] = 0
    if "SatisfactionScore" not in df.columns:
        df["SatisfactionScore"] = 3
    if "CardType" not in df.columns:
        df["CardType"] = "SILVER"
    if "PointEarned" not in df.columns:
        df["PointEarned"] = 500

    # Ensure target is integer binary
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Validate raw schema with Pandera
    if validate:
        raw_bank_data_schema.validate(df)

    # Drop non-predictive identifiers if present
    drop_cols = [c for c in ID_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Separate features X and target y."""
    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df.copy()
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y


def build_preprocessor_pipeline(X_sample: pd.DataFrame) -> ColumnTransformer:
    """
    Construct leak-free Scikit-Learn ColumnTransformer for numerical scaling
    and categorical one-hot encoding after feature engineering.
    """
    fe = BankingFeatureEngineer()
    X_transformed = fe.transform(X_sample)

    numeric_cols = X_transformed.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X_transformed.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor


def create_full_pipeline(model: BaseEstimator, X_sample: pd.DataFrame) -> Pipeline:
    """Combine Feature Engineering, Preprocessor, and Estimator into an atomic pipeline."""
    return Pipeline(
        steps=[
            ("feature_engineer", BankingFeatureEngineer()),
            ("preprocessor", build_preprocessor_pipeline(X_sample)),
            ("classifier", model),
        ]
    )


def process_and_save_data(raw_path: Optional[str] = None) -> pd.DataFrame:
    """Execute Phase 1 data pipeline: load, clean, engineer, validate, and save."""
    print("=" * 70)
    print("BANK RETENTION DECISION ENGINE: PHASE 1 DATA PIPELINE")
    print("=" * 70)

    print(f"[1/4] Loading raw data from: {raw_path or RAW_DATA_PATH}")
    df_clean = load_and_clean_data(raw_path, validate=True)
    print(f" -> Cleaned records: {len(df_clean):,} rows, {df_clean.shape[1]} columns")
    print(f" -> Overall Churn Rate: {df_clean[TARGET_COL].mean():.2%}")

    print("[2/4] Executing Banking Domain Feature Engineering...")
    fe = BankingFeatureEngineer()
    df_engineered = fe.transform(df_clean)
    print(f" -> Engineered dataset shape: {df_engineered.shape}")

    print("[3/4] Validating engineered dataset against Pandera Schema...")
    engineered_bank_data_schema.validate(df_engineered)
    print(" -> Pandera schema validation passed successfully.")

    print(f"[4/4] Persisting engineered dataset...")
    df_engineered.to_parquet(PROCESSED_DATA_PATH, index=False)
    csv_fallback_path = PROCESSED_DATA_PATH.with_suffix(".csv")
    df_engineered.to_csv(csv_fallback_path, index=False)
    print(f" -> Saved Parquet to: {PROCESSED_DATA_PATH}")
    print(f" -> Saved CSV copy to: {csv_fallback_path}")
    print("=" * 70)
    print("PHASE 1 COMPLETE: DATA PIPELINE READY")
    print("=" * 70)
    return df_engineered


if __name__ == "__main__":
    process_and_save_data()

