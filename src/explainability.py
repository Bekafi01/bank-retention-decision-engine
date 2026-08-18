import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RAW_DATA_PATH
from src.preprocess import BankingFeatureEngineer


class BankChurnExplainer:
    """
    Model Explainability & Prescriptive Counterfactual Engine for Banking Churn.
    Combines TreeSHAP attributions with actionable 'What-If' recommendations.
    """

    def __init__(self, raw_pipeline):
        self.pipeline = raw_pipeline
        self.fe = raw_pipeline.named_steps["feature_engineer"]
        self.preprocessor = raw_pipeline.named_steps["preprocessor"]
        self.classifier = raw_pipeline.named_steps["classifier"]
        
        # Determine appropriate explainer based on classifier family
        try:
            self.explainer = shap.TreeExplainer(self.classifier)
        except Exception:
            feature_count = len(self.get_feature_names())
            self.explainer = shap.LinearExplainer(
                self.classifier,
                masker=shap.maskers.Independent(np.zeros((1, feature_count))),
            )

    def get_feature_names(self) -> List[str]:
        """Retrieve processed feature names from column transformer."""
        raw_names = self.preprocessor.get_feature_names_out().tolist()
        # Clean up sklearn prefix prefixes like 'num__' and 'cat__'
        cleaned_names = [
            name.replace("num__", "").replace("cat__", "").replace("remainder__", "")
            for name in raw_names
        ]
        return cleaned_names

    def transform_single(self, df_single: pd.DataFrame) -> np.ndarray:
        """Apply feature engineering and column transformer to input."""
        df_eng = self.fe.transform(df_single)
        X_trans = self.preprocessor.transform(df_eng)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        return X_trans

    def explain_customer(self, customer_df: pd.DataFrame, top_k: int = 8) -> Dict[str, float]:
        """Compute local SHAP feature attributions for a single customer."""
        X_trans = self.transform_single(customer_df)
        feature_names = self.get_feature_names()

        shap_values = self.explainer.shap_values(X_trans)
        # Handle binary classification formats across LightGBM, XGBoost, CatBoost, and Scikit-Learn
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        contrib_series = pd.Series(sv, index=feature_names)
        top_factors = contrib_series.reindex(contrib_series.abs().sort_values(ascending=False).index).head(top_k)
        return {k: round(float(v), 4) for k, v in top_factors.to_dict().items()}

    def generate_prescriptive_counterfactuals(
        self, customer_df: pd.DataFrame, target_threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Simulate actionable business interventions to calculate realistic churn risk reduction.
        Tests:
          1. Complaint resolution (Complain: 1 -> 0, SatisfactionScore: +1)
          2. Digital activation (IsActiveMember: 0 -> 1)
          3. Product restructuring (NumOfProducts: >=3 -> 2)
          4. Loyalty Card tier bump (CardType: SILVER -> PLATINUM)
        """
        baseline_proba = float(self.pipeline.predict_proba(customer_df)[:, 1][0])
        sim_df = customer_df.copy()
        applied_actions = []
        intervention_details = []

        # Action 1: Resolve open complaint
        if "Complain" in sim_df.columns and sim_df["Complain"].iloc[0] == 1:
            sim_df["Complain"] = 0
            if "SatisfactionScore" in sim_df.columns:
                sim_df["SatisfactionScore"] = min(5, sim_df["SatisfactionScore"].iloc[0] + 1)
            applied_actions.append("Expedite resolution of outstanding customer complaint & offer apology goodwill perk.")
            intervention_details.append({
                "action": "Resolve Complaint & Restore Satisfaction",
                "lever": "Complain = 0, Satisfaction +1",
            })

        # Action 2: Digital engagement & mobile banking activation
        if "IsActiveMember" in sim_df.columns and sim_df["IsActiveMember"].iloc[0] == 0:
            sim_df["IsActiveMember"] = 1
            applied_actions.append("Enroll in automated wealth dashboard & mobile app active user onboarding.")
            intervention_details.append({
                "action": "Activate Mobile & Web Banking",
                "lever": "IsActiveMember = 1",
            })

        # Action 3: Product restructuring (if customer has excessive 3-4 products)
        if "NumOfProducts" in sim_df.columns and sim_df["NumOfProducts"].iloc[0] >= 3:
            sim_df["NumOfProducts"] = 2
            applied_actions.append("Consolidate overlapping accounts into optimized high-yield 2-product bundle.")
            intervention_details.append({
                "action": "Optimize Product Bundle",
                "lever": "NumOfProducts = 2",
            })

        # Action 4: Loyalty Tier Upgrade
        if "CardType" in sim_df.columns and str(sim_df["CardType"].iloc[0]).upper() in ["SILVER", "GOLD"]:
            sim_df["CardType"] = "PLATINUM"
            applied_actions.append("Complimentary 1-year upgrade to Platinum Card tier with annual fee waiver.")
            intervention_details.append({
                "action": "Upgrade Card Tier",
                "lever": "CardType = PLATINUM",
            })

        if not applied_actions:
            applied_actions.append("Personalized Relationship Manager touchpoint & deposit rate review.")
            intervention_details.append({
                "action": "Executive Relationship Manager Outreach",
                "lever": "Personalized Outreach",
            })

        simulated_proba = float(self.pipeline.predict_proba(sim_df)[:, 1][0])
        risk_reduction = max(0.0, baseline_proba - simulated_proba)
        risk_reduction_pct = (risk_reduction / (baseline_proba + 1e-6)) * 100.0

        customer_balance = float(customer_df["Balance"].iloc[0]) if "Balance" in customer_df.columns else 0.0
        potential_deposit_saved = round(customer_balance * (risk_reduction / max(baseline_proba, 1e-6)), 2)

        return {
            "baseline_probability": round(baseline_proba, 4),
            "simulated_probability": round(simulated_proba, 4),
            "absolute_risk_drop": round(risk_reduction, 4),
            "risk_reduction_pct": round(risk_reduction_pct, 1),
            "recommended_interventions": applied_actions,
            "intervention_details": intervention_details,
            "potential_deposit_retained": round(customer_balance, 2),
            "expected_deposit_saved": potential_deposit_saved,
        }

    def generate_global_shap_plots(
        self, sample_df: pd.DataFrame, output_dir: Optional[Path] = None, max_display: int = 12
    ) -> Dict[str, Path]:
        """
        Generate publication-quality global SHAP summary beeswarm and bar importance plots.
        """
        target_dir = output_dir or FIGURES_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        X_trans = self.transform_single(sample_df)
        feature_names = self.get_feature_names()

        shap_values = self.explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # 1. Global Bar Feature Importance Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_trans, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
        plt.title("TreeSHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        bar_path = target_dir / "shap_global_importance_bar.png"
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Global Summary Beeswarm Plot
        plt.figure(figsize=(11, 7))
        shap.summary_plot(sv, X_trans, feature_names=feature_names, max_display=max_display, show=False)
        plt.title("TreeSHAP Global Summary Beeswarm Plot", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        beeswarm_path = target_dir / "shap_summary_beeswarm.png"
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"SHAP global plots saved:\n -> {bar_path}\n -> {beeswarm_path}")
        return {"bar_plot": bar_path, "beeswarm_plot": beeswarm_path}


if __name__ == "__main__":
    import joblib
    raw_pipe = joblib.load(ARTIFACTS_DIR / "raw_pipeline.pkl")
    explainer = BankChurnExplainer(raw_pipe)
    
    # Run global SHAP plotting on sample of raw dataset
    from src.preprocess import load_and_clean_data, split_features_and_target
    df_raw = load_and_clean_data(validate=False)
    X, _ = split_features_and_target(df_raw)
    sample_X = X.sample(n=min(600, len(X)), random_state=42)
    explainer.generate_global_shap_plots(sample_X)

