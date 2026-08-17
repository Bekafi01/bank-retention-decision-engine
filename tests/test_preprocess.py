import pytest
import pandas as pd
import numpy as np
from pandera.errors import SchemaError
from sklearn.linear_model import LogisticRegression

from src.preprocess import (
    BankingFeatureEngineer,
    standardize_column_names,
    load_and_clean_data,
    build_preprocessor_pipeline,
    create_full_pipeline,
)
from src.data_schema import raw_bank_data_schema, engineered_bank_data_schema
from src.config import RAW_DATA_PATH


def test_standardize_column_names():
    raw_df = pd.DataFrame({
        "Satisfaction Score": [4],
        "Card Type": ["GOLD"],
        "Point Earned": [600],
        "CreditScore": [700],
    })
    cleaned_df = standardize_column_names(raw_df)
    assert "SatisfactionScore" in cleaned_df.columns
    assert "CardType" in cleaned_df.columns
    assert "PointEarned" in cleaned_df.columns


def test_banking_feature_engineer():
    df = pd.DataFrame({
        "CreditScore": [650, 720],
        "Age": [40, 50],
        "Tenure": [5, 10],
        "Balance": [100000.0, 0.0],
        "NumOfProducts": [3, 1],
        "IsActiveMember": [0, 1],
        "EstimatedSalary": [50000.0, 60000.0],
        "Complain": [1, 0],
        "SatisfactionScore": [2, 5],
        "CardType": ["SILVER", "PLATINUM"],
        "PointEarned": [400, 800],
    })

    fe = BankingFeatureEngineer()
    df_trans = fe.transform(df)

    assert "BalanceToSalaryRatio" in df_trans.columns
    assert "IsZeroBalance" in df_trans.columns
    assert "WealthTier" in df_trans.columns
    assert "TenureToAgeRatio" in df_trans.columns
    assert "CreditScoreToAgeRatio" in df_trans.columns
    assert "IsMultiProductRisk" in df_trans.columns
    assert "ComplaintInactivityRisk" in df_trans.columns
    assert "ComplaintRisk" in df_trans.columns
    assert "LoyaltyIndex" in df_trans.columns

    # Verify logic
    assert df_trans["IsZeroBalance"].iloc[1] == 1
    assert df_trans["IsZeroBalance"].iloc[0] == 0
    assert df_trans["WealthTier"].iloc[0] == "Affluent"
    assert df_trans["WealthTier"].iloc[1] == "Zero_Balance"
    assert df_trans["IsMultiProductRisk"].iloc[0] == 1
    assert df_trans["ComplaintInactivityRisk"].iloc[0] == 1
    assert df_trans["ComplaintInactivityRisk"].iloc[1] == 0
    assert df_trans["ComplaintRisk"].iloc[0] == 4  # 1 * (6 - 2)
    assert df_trans["ComplaintRisk"].iloc[1] == 0  # 0 * (6 - 5)


def test_pandera_raw_schema_validation():
    valid_df = pd.DataFrame({
        "CreditScore": [600, 700],
        "Geography": ["France", "Germany"],
        "Gender": ["Female", "Male"],
        "Age": [35, 45],
        "Tenure": [3, 7],
        "Balance": [50000.0, 0.0],
        "NumOfProducts": [1, 2],
        "HasCrCard": [1, 0],
        "IsActiveMember": [1, 1],
        "EstimatedSalary": [80000.0, 95000.0],
        "Exited": [0, 1],
        "Complain": [0, 1],
        "SatisfactionScore": [4, 2],
        "CardType": ["GOLD", "DIAMOND"],
        "PointEarned": [500, 900],
    })
    validated = raw_bank_data_schema.validate(valid_df)
    assert len(validated) == 2

    # Test invalid credit score trigger SchemaError
    invalid_df = valid_df.copy()
    invalid_df["CreditScore"] = [250, 700]  # below 300
    with pytest.raises(SchemaError):
        raw_bank_data_schema.validate(invalid_df)


def test_pandera_engineered_schema_validation():
    df = pd.DataFrame({
        "CreditScore": [600],
        "Geography": ["France"],
        "Gender": ["Female"],
        "Age": [35],
        "Tenure": [3],
        "Balance": [150000.0],
        "NumOfProducts": [4],
        "HasCrCard": [1],
        "IsActiveMember": [0],
        "EstimatedSalary": [75000.0],
        "Exited": [1],
        "Complain": [1],
        "SatisfactionScore": [1],
        "CardType": ["DIAMOND"],
        "PointEarned": [850],
    })
    fe = BankingFeatureEngineer()
    df_eng = fe.transform(df)
    validated = engineered_bank_data_schema.validate(df_eng)
    assert validated["WealthTier"].iloc[0] == "High_Net_Worth"
    assert validated["IsMultiProductRisk"].iloc[0] == 1


def test_load_and_clean_data_real_file():
    if RAW_DATA_PATH.exists():
        df = load_and_clean_data(str(RAW_DATA_PATH), validate=True)
        assert len(df) == 10000
        assert "RowNumber" not in df.columns
        assert "CustomerId" not in df.columns
        assert "Surname" not in df.columns
        assert "Exited" in df.columns


def test_preprocessor_pipeline_transformation():
    df = pd.DataFrame({
        "CreditScore": [650, 720, 580],
        "Geography": ["France", "Germany", "Spain"],
        "Gender": ["Female", "Male", "Female"],
        "Age": [40, 50, 35],
        "Tenure": [5, 10, 2],
        "Balance": [100000.0, 0.0, 45000.0],
        "NumOfProducts": [3, 1, 2],
        "HasCrCard": [1, 1, 0],
        "IsActiveMember": [0, 1, 1],
        "EstimatedSalary": [50000.0, 60000.0, 40000.0],
        "Complain": [1, 0, 0],
        "SatisfactionScore": [2, 5, 4],
        "CardType": ["SILVER", "PLATINUM", "GOLD"],
        "PointEarned": [400, 800, 650],
    })

    preprocessor = build_preprocessor_pipeline(df)
    fe = BankingFeatureEngineer()
    df_eng = fe.transform(df)
    transformed_matrix = preprocessor.fit_transform(df_eng)

    assert isinstance(transformed_matrix, np.ndarray)
    assert transformed_matrix.shape[0] == 3
    assert transformed_matrix.shape[1] > 10


def test_full_pipeline_fit_predict():
    df = pd.DataFrame({
        "CreditScore": [650, 720, 580, 800],
        "Geography": ["France", "Germany", "Spain", "France"],
        "Gender": ["Female", "Male", "Female", "Male"],
        "Age": [40, 50, 35, 60],
        "Tenure": [5, 10, 2, 8],
        "Balance": [100000.0, 0.0, 45000.0, 120000.0],
        "NumOfProducts": [3, 1, 2, 1],
        "HasCrCard": [1, 1, 0, 1],
        "IsActiveMember": [0, 1, 1, 0],
        "EstimatedSalary": [50000.0, 60000.0, 40000.0, 90000.0],
        "Complain": [1, 0, 0, 1],
        "SatisfactionScore": [2, 5, 4, 1],
        "CardType": ["SILVER", "PLATINUM", "GOLD", "DIAMOND"],
        "PointEarned": [400, 800, 650, 950],
    })
    y = np.array([1, 0, 0, 1])

    pipeline = create_full_pipeline(LogisticRegression(), df)
    pipeline.fit(df, y)
    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)

    assert len(preds) == 4
    assert probs.shape == (4, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

