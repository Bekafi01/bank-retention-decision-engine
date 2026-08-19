import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import ks_2samp, chisquare


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and production distributions.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate shift (monitor)
    PSI >= 0.2: Significant drift (retrain recommended)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantiles based on reference data
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
    actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division with epsilon
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.maximum(0.0, psi_val))


def evaluate_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    psi_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Evaluate feature distribution drift across numerical and categorical features.
    """
    if features is None:
        features = [
            c for c in reference_df.columns
            if c in current_df.columns and c not in ["CustomerId", "RowNumber", "Surname", "Exited"]
        ]

    drift_report = {}
    drifted_features = []

    for col in features:
        if col not in current_df.columns or col not in reference_df.columns:
            continue

        ref_col = reference_df[col]
        curr_col = current_df[col]

        if pd.api.types.is_numeric_dtype(ref_col):
            psi = calculate_psi(ref_col.values, curr_col.values)
            ks_stat, ks_pval = ks_2samp(ref_col.dropna(), curr_col.dropna())
            is_drift = bool(psi >= psi_threshold or ks_pval < 0.01)

            drift_report[col] = {
                "type": "numeric",
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(ks_pval), 6),
                "is_drift": is_drift,
            }
        else:
            # Categorical distribution comparison
            ref_dist = ref_col.value_counts(normalize=True)
            curr_dist = curr_col.value_counts(normalize=True)

            all_cats = list(set(ref_dist.index).union(set(curr_dist.index)))
            ref_vec = np.array([ref_dist.get(cat, 1e-4) for cat in all_cats])
            curr_vec = np.array([curr_dist.get(cat, 1e-4) for cat in all_cats])

            # Normalize to 1
            ref_vec /= ref_vec.sum()
            curr_vec /= curr_vec.sum()

            psi = float(np.sum((curr_vec - ref_vec) * np.log(curr_vec / ref_vec)))
            is_drift = bool(psi >= psi_threshold)

            drift_report[col] = {
                "type": "categorical",
                "psi": round(psi, 4),
                "is_drift": is_drift,
            }

        if is_drift:
            drifted_features.append(col)

    return {
        "is_overall_drift_detected": len(drifted_features) > 0,
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "feature_details": drift_report,
    }
