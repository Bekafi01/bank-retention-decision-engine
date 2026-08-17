import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

from src.config import (
    RAW_DATA_PATH,
    ARTIFACTS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
)
from src.preprocess import load_and_clean_data, split_features_and_target
from src.business_optimizer import optimize_decision_threshold


def compute_bootstrap_ci(
    y_true: np.ndarray, y_proba: np.ndarray, n_bootstraps: int = 1000
) -> Dict[str, Dict[str, float]]:
    """Compute 95% Bootstrap Confidence Intervals for evaluation metrics."""
    np.random.seed(RANDOM_STATE)
    boot_aucs, boot_praucs, boot_briers = [], [], []

    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[indices], y_proba[indices]))
        boot_praucs.append(average_precision_score(y_true[indices], y_proba[indices]))
        boot_briers.append(brier_score_loss(y_true[indices], y_proba[indices]))

    return {
        "ROC_AUC": {
            "mean": float(np.mean(boot_aucs)),
            "ci_lower": float(np.percentile(boot_aucs, 2.5)),
            "ci_upper": float(np.percentile(boot_aucs, 97.5)),
        },
        "PR_AUC": {
            "mean": float(np.mean(boot_praucs)),
            "ci_lower": float(np.percentile(boot_praucs, 2.5)),
            "ci_upper": float(np.percentile(boot_praucs, 97.5)),
        },
        "Brier_Score": {
            "mean": float(np.mean(boot_briers)),
            "ci_lower": float(np.percentile(boot_briers, 2.5)),
            "ci_upper": float(np.percentile(boot_briers, 97.5)),
        },
    }


def generate_evaluation_report(
    data_path: str = str(RAW_DATA_PATH),
) -> pd.DataFrame:
    """Evaluate all fitted models and generate comparative visualization artifacts."""
    print("\n--- Running Comparative Model Evaluation & Calibration Analysis ---")
    df = load_and_clean_data(data_path)
    X, y = split_features_and_target(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    fitted_pipelines = joblib.load(ARTIFACTS_DIR / "all_fitted_pipelines.pkl")
    calibrated_champion = joblib.load(ARTIFACTS_DIR / "champion_model.pkl")

    with open(ARTIFACTS_DIR / "metrics_summary.json", "r") as f:
        metrics_meta = json.load(f)
    champion_name = metrics_meta["champion_model"]

    results = []
    probas = {}

    for name, pipeline in fitted_pipelines.items():
        p = pipeline.predict_proba(X_test)[:, 1]
        probas[name] = p
        auc = roc_auc_score(y_test, p)
        pr_auc = average_precision_score(y_test, p)
        brier = brier_score_loss(y_test, p)
        y_pred = (p >= 0.50).astype(int)

        results.append({
            "Model": name,
            "ROC-AUC": round(auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Brier Score": round(brier, 4),
            "F1 (th=0.5)": round(f1_score(y_test, y_pred), 4),
            "Recall (th=0.5)": round(recall_score(y_test, y_pred), 4),
            "Precision (th=0.5)": round(precision_score(y_test, y_pred), 4),
        })

    # Add Calibrated Champion
    cal_p = calibrated_champion.predict_proba(X_test)[:, 1]
    probas[f"{champion_name} (Calibrated)"] = cal_p
    opt_th = metrics_meta["test_metrics"]["optimal_threshold"]
    cal_pred = (cal_p >= opt_th).astype(int)

    results.append({
        "Model": f"{champion_name} (Calibrated, th={opt_th:.2f})",
        "ROC-AUC": round(roc_auc_score(y_test, cal_p), 4),
        "PR-AUC": round(average_precision_score(y_test, cal_p), 4),
        "Brier Score": round(brier_score_loss(y_test, cal_p), 4),
        "F1 (th=0.5)": round(f1_score(y_test, cal_pred), 4),
        "Recall (th=0.5)": round(recall_score(y_test, cal_pred), 4),
        "Precision (th=0.5)": round(precision_score(y_test, cal_pred), 4),
    })

    results_df = pd.DataFrame(results)
    print("\n--- Comparative Metrics Leaderboard ---")
    print(results_df.to_string(index=False))

    # Bootstrap CIs for champion
    ci_stats = compute_bootstrap_ci(y_test.values, cal_p)
    print("\n--- Champion 95% Bootstrap Confidence Intervals ---")
    for metric, vals in ci_stats.items():
        print(f" {metric:12s}: {vals['mean']:.4f} (95% CI: [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}])")

    # 1. Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    plt.title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / "roc_curve_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Plot Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        pr_val = average_precision_score(y_test, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_val:.3f})")
    plt.title("Precision-Recall Curve Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / "pr_curve_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Calibration Reliability Curves
    plt.figure(figsize=(8, 6))
    for name in [champion_name, f"{champion_name} (Calibrated)"]:
        prob_true, prob_pred = calibration_curve(y_test, probas[name], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.title("Reliability Diagram (Probability Calibration)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Empirical Fraction of Positives")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / "calibration_reliability_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Financial Cost Curve & Net Profit vs Threshold
    test_balances = X_test["Balance"].values
    opt_curve = optimize_decision_threshold(y_test.values, cal_p, test_balances)
    tc = opt_curve["threshold_curve"]

    plt.figure(figsize=(9, 5))
    plt.plot(tc["thresholds"], tc["net_profits"], color="#1f77b4", lw=2.5, label="Net Retained Profit (€)")
    plt.axvline(opt_curve["optimal_threshold"], color="red", linestyle="--", label=f"Optimal p* = {opt_curve['optimal_threshold']:.2f}")
    plt.axvline(0.50, color="gray", linestyle=":", label="Default p = 0.50")
    plt.title("Financial Decision Curve: Net Profit vs Churn Threshold", fontsize=13, fontweight="bold")
    plt.xlabel("Decision Probability Threshold")
    plt.ylabel("Net Profit (€)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / "cost_curve_profit_optimization.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nAll comparative evaluation figures saved to: {FIGURES_DIR}")
    return results_df


if __name__ == "__main__":
    generate_evaluation_report()
