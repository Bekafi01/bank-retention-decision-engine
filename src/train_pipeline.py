import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    classification_report,
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna

# Suppress Optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    ARTIFACTS_DIR,
    RANDOM_STATE,
    CV_SPLITS,
    TEST_SIZE,
    TARGET_COL,
)
from src.preprocess import (
    load_and_clean_data,
    split_features_and_target,
    BankingFeatureEngineer,
    build_preprocessor_pipeline,
    create_full_pipeline,
)
from src.business_optimizer import optimize_decision_threshold


def compute_expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper if i < n_bins - 1 else y_proba <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_proba[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate model zoo spanning linear, bagging, boosting, and ensembles."""
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=2.5,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=2.5,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def optimize_champion_hyperparameters(
    champion_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 25,
) -> BaseEstimator:
    """Run Optuna Bayesian optimization on the winning GBDT architecture."""
    print(f"\n[Optuna] Tuning {champion_name} hyperparameters ({n_trials} trials, 5-Fold CV)...")
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        if "LightGBM" in champion_name:
            model = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 350, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 63),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.5, 4.0),
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=-1,
            )
        elif "CatBoost" in champion_name:
            model = CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 150, 400, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                depth=trial.suggest_int("depth", 4, 8),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            )
        else:  # XGBoost or default
            model = xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 350, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.5, 4.0),
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            )

        pipeline = create_full_pipeline(model, X_train)
        y_cv_proba = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        score = (average_precision_score(y_train, y_cv_proba) * 0.6) + (roc_auc_score(y_train, y_cv_proba) * 0.4)
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[Optuna] Best Trial Value: {study.best_value:.4f}")
    print(f"[Optuna] Best Parameters: {study.best_params}")

    # Reconstruct optimal model
    best_params = study.best_params
    if "LightGBM" in champion_name:
        best_model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    elif "CatBoost" in champion_name:
        best_model = CatBoostClassifier(**best_params, auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
    else:
        best_model = xgb.XGBClassifier(**best_params, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)

    return best_model


def train_and_benchmark(
    data_path: str = str(RAW_DATA_PATH),
    run_optuna: bool = True,
) -> Dict[str, Any]:
    """
    Complete end-to-end training, cross-validation benchmarking,
    calibration, threshold optimization, and artifact serialization.
    """
    print(f"\n[1/6] Loading and engineering data from: {data_path}")
    df = load_and_clean_data(data_path, validate=True)
    print(f"Dataset shape: {df.shape}, Churn rate: {df[TARGET_COL].mean():.2%}")

    X, y = split_features_and_target(df)

    # Train / Test Split (Stratified holdout)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set: {len(X_train):,} rows | Holdout test set: {len(X_test):,} rows")

    models_zoo = get_candidate_models()
    cv_results = {}
    fitted_pipelines = {}

    print(f"\n[2/6] Running {CV_SPLITS}-Fold Stratified Cross-Validation on Model Zoo...")
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models_zoo.items():
        pipeline = create_full_pipeline(model, X_train)
        
        # Cross-validation out-of-fold probability predictions
        y_cv_proba = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]

        auc = roc_auc_score(y_train, y_cv_proba)
        pr_auc = average_precision_score(y_train, y_cv_proba)
        brier = brier_score_loss(y_train, y_cv_proba)

        cv_results[name] = {
            "CV_ROC_AUC": float(auc),
            "CV_PR_AUC": float(pr_auc),
            "CV_Brier_Score": float(brier),
        }
        print(f" -> {name:20s} | CV ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    # Select champion based on highest composite CV PR-AUC & ROC-AUC
    champion_name = max(cv_results, key=lambda k: cv_results[k]["CV_PR_AUC"] + cv_results[k]["CV_ROC_AUC"])
    print(f"\n[3/6] Selected Champion Architecture: {champion_name}")

    if run_optuna and champion_name in ["LightGBM", "XGBoost", "CatBoost"]:
        tuned_model = optimize_champion_hyperparameters(champion_name, X_train, y_train, n_trials=25)
        champion_raw_pipeline = create_full_pipeline(tuned_model, X_train)
    else:
        champion_raw_pipeline = fitted_pipelines[champion_name]

    # Fit raw champion pipeline on training set
    champion_raw_pipeline.fit(X_train, y_train)
    fitted_pipelines[champion_name] = champion_raw_pipeline

    # Probability Calibration on Champion Model strictly using CV on X_train (No test leakage!)
    print(f"\n[4/6] Calibrating Champion Model probabilities with 5-Fold CV Platt Scaling on X_train...")
    calibrated_champion = CalibratedClassifierCV(
        estimator=champion_raw_pipeline, method="sigmoid", cv=CV_SPLITS
    )
    calibrated_champion.fit(X_train, y_train)

    # Final evaluation on holdout test set
    test_raw_proba = champion_raw_pipeline.predict_proba(X_test)[:, 1]
    test_cal_proba = calibrated_champion.predict_proba(X_test)[:, 1]

    raw_brier = brier_score_loss(y_test, test_raw_proba)
    cal_brier = brier_score_loss(y_test, test_cal_proba)
    raw_ece = compute_expected_calibration_error(y_test.values, test_raw_proba)
    cal_ece = compute_expected_calibration_error(y_test.values, test_cal_proba)
    test_auc = roc_auc_score(y_test, test_cal_proba)
    test_pr_auc = average_precision_score(y_test, test_cal_proba)

    print(f" -> Holdout Test ROC-AUC: {test_auc:.4f}")
    print(f" -> Holdout Test PR-AUC:  {test_pr_auc:.4f}")
    print(f" -> Brier Score: Uncalibrated={raw_brier:.4f} -> Calibrated={cal_brier:.4f} (Reduction: {(raw_brier - cal_brier)/raw_brier:.1%})")
    print(f" -> Expected Calibration Error (ECE): Uncalibrated={raw_ece:.4f} -> Calibrated={cal_ece:.4f}")

    # Threshold Optimization for Deposit-at-Risk
    print(f"\n[5/6] Optimizing Financial Decision Cutoff Threshold on Deposits...")
    test_balances = X_test["Balance"].values
    opt_results = optimize_decision_threshold(y_test.values, test_cal_proba, test_balances)
    optimal_th = opt_results["optimal_threshold"]
    print(f" -> Optimal Threshold: {optimal_th:.3f}")
    print(f" -> Max Net Retained NIM Saved: €{opt_results['max_net_profit_saved']:,.2f}")
    print(f" -> Profit Lift vs Default (0.50): €{opt_results['profit_lift_over_default']:,.2f}")

    # Extract feature names from fitted pipeline
    preprocessor = champion_raw_pipeline.named_steps["preprocessor"]
    fe_feature_names = preprocessor.get_feature_names_out().tolist()

    # Serialization of Artifacts
    print(f"\n[6/6] Serializing Production Artifacts to: {ARTIFACTS_DIR}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_champion, ARTIFACTS_DIR / "champion_model.pkl")
    joblib.dump(champion_raw_pipeline, ARTIFACTS_DIR / "raw_pipeline.pkl")
    joblib.dump(fitted_pipelines, ARTIFACTS_DIR / "all_fitted_pipelines.pkl")

    metrics_payload = {
        "champion_model": champion_name,
        "cv_benchmarks": cv_results,
        "test_metrics": {
            "roc_auc": float(test_auc),
            "pr_auc": float(test_pr_auc),
            "brier_score_uncalibrated": float(raw_brier),
            "brier_score_calibrated": float(cal_brier),
            "ece_uncalibrated": float(raw_ece),
            "ece_calibrated": float(cal_ece),
            "optimal_threshold": float(optimal_th),
            "max_net_profit_saved": float(opt_results["max_net_profit_saved"]),
            "profit_lift_vs_default": float(opt_results["profit_lift_over_default"]),
        },
        "feature_names": fe_feature_names,
    }

    with open(ARTIFACTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(ARTIFACTS_DIR / "optimal_threshold.json", "w") as f:
        json.dump(opt_results, f, indent=2)

    print("Pipeline execution and artifact serialization complete.")
    return metrics_payload


if __name__ == "__main__":
    train_and_benchmark()
