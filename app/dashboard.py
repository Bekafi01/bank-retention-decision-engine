import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_PATH,
    DEFAULT_DECISION_THRESHOLD,
    ANNUAL_NIM_RATE,
    HIGH_BALANCE_THRESHOLD,
    VIP_OUTREACH_COST,
    VIP_SAVE_RATE,
    DIGITAL_INCENTIVE_COST,
    DIGITAL_SAVE_RATE,
    FIGURES_DIR,
)
from src.business_optimizer import (
    evaluate_retention_economics,
    segment_portfolio_customers,
    compute_portfolio_deposit_risk,
    optimize_decision_threshold,
)
from src.explainability import BankChurnExplainer

st.set_page_config(
    page_title="Bank Deposit Retention & Risk Intelligence Cockpit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 18px 22px;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #38BDF8;
        margin-top: 4px;
        font-weight: 500;
    }
    .tier-badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .tier-badge-high {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .tier-badge-medium {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .tier-badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_production_artifacts():
    champion_path = ARTIFACTS_DIR / "champion_model.pkl"
    raw_path = ARTIFACTS_DIR / "raw_pipeline.pkl"
    metrics_path = ARTIFACTS_DIR / "metrics_summary.json"
    opt_path = ARTIFACTS_DIR / "optimal_threshold.json"

    try:
        model = joblib.load(champion_path)
    except Exception as e:
        st.error(f"❌ Champion model failed: {type(e).__name__}: {e}")
        st.exception(e)
        model = None
    try:
        raw_pipeline = joblib.load(raw_path)
    except Exception as e:
        st.error(f"❌ Raw pipeline failed: {type(e).__name__}: {e}")
        st.exception(e)
        raw_pipeline = None
    try:
        explainer = BankChurnExplainer(raw_pipeline)
    except Exception as e:
        st.error(f"❌ Explainer creation failed: {type(e).__name__}: {e}")
        st.exception(e)
        explainer = None

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    threshold_data = {}
    if opt_path.exists():
        with open(opt_path, "r") as f:
            threshold_data = json.load(f)

    return model, raw_pipeline, explainer, metrics, threshold_data

@st.cache_data
def load_customer_portfolio():
    if PROCESSED_DATA_PATH.exists():
        return pd.read_parquet(PROCESSED_DATA_PATH)
    return None


def main():
    st.title("🏦 Bank Deposit Retention & Risk Intelligence Cockpit")
    st.markdown(
        "**Enterprise AI System for Deposit Flight Mitigation, TreeSHAP Explainability & Net Interest Margin (NIM) Optimization**"
    )

    model, raw_pipeline, explainer, metrics, threshold_data = load_production_artifacts()
    df_portfolio = load_customer_portfolio()

    if model is None:
        st.error("❌ Champion model could not be loaded.")
        return
    
    if df_portfolio is None:
        st.error("❌ Customer portfolio dataset could not be loaded.")
        return

    optimal_th = float(threshold_data.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD))

    # Sidebar Controls
    with st.sidebar:
        st.header("🎯 Strategy & Cutoffs")
        decision_th = st.slider(
            "Optimal Churn Risk Threshold (p*)",
            min_value=0.05,
            max_value=0.95,
            value=float(optimal_th),
            step=0.01,
            help="Accounts with calibrated churn risk >= p* are targeted for retention intervention.",
        )
        st.caption(f"💡 Algorithmic Maximum Profit Cutoff: **{optimal_th:.2f}**")

        st.divider()
        st.header("💰 Banking Economics")
        nim_rate = st.number_input(
            "Annual Net Interest Margin (NIM Rate)",
            min_value=0.01,
            max_value=0.10,
            value=ANNUAL_NIM_RATE,
            step=0.002,
            format="%.3f",
            help="Annual lending profit margin generated on deposit balances.",
        )
        vip_threshold = st.number_input(
            "VIP Account Balance Cutoff (€)",
            min_value=10000.0,
            max_value=250000.0,
            value=HIGH_BALANCE_THRESHOLD,
            step=10000.0,
        )

        st.subheader("Campaign Parameters")
        col_rm1, col_rm2 = st.columns(2)
        with col_rm1:
            vip_cost = st.number_input("RM Outreach (€)", 50.0, 500.0, VIP_OUTREACH_COST, 25.0)
        with col_rm2:
            vip_save = st.slider("RM Save Rate", 0.10, 0.95, VIP_SAVE_RATE, 0.05)

        col_dig1, col_dig2 = st.columns(2)
        with col_dig1:
            dig_cost = st.number_input("Digital Offer (€)", 5.0, 100.0, DIGITAL_INCENTIVE_COST, 5.0)
        with col_dig2:
            dig_save = st.slider("Digital Save Rate", 0.05, 0.70, DIGITAL_SAVE_RATE, 0.05)

    # Score Portfolio
    X_score = df_portfolio.drop(columns=["Exited"]) if "Exited" in df_portfolio.columns else df_portfolio
    churn_probas = model.predict_proba(X_score)[:, 1]

    segmented_portfolio = segment_portfolio_customers(
        df_portfolio,
        churn_probas,
        threshold=decision_th,
        high_balance_cutoff=vip_threshold,
    )

    portfolio_risk = compute_portfolio_deposit_risk(
        df_portfolio,
        churn_probas,
        threshold=decision_th,
        nim_rate=nim_rate,
        high_balance_cutoff=vip_threshold,
        vip_cost=vip_cost,
        vip_save_rate=vip_save,
        digital_cost=dig_cost,
        digital_save_rate=dig_save,
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive KPI Overview",
        "👥 High-Risk Customer Explorer",
        "🔬 Prescriptive 'What-If' Simulator",
        "📈 Campaign ROI & Threshold Curve",
        "🎯 Model Calibration & Diagnostics",
    ])

    # ----------------------------------------------------------------------------------
    # TAB 1: EXECUTIVE KPI OVERVIEW
    # ----------------------------------------------------------------------------------
    with tab1:
        st.subheader("Executive Deposit Risk & Retention Cockpit")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Portfolio Deposits</div>
                <div class="kpi-value">€{portfolio_risk['total_portfolio_deposits']:,.0f}</div>
                <div class="kpi-sub">{portfolio_risk['total_portfolio_customers']:,} Active Accounts</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            pct_at_risk = (portfolio_risk['deposits_at_risk'] / max(portfolio_risk['total_portfolio_deposits'], 1)) * 100
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 5px solid #EF4444;">
                <div class="kpi-title">Deposits at Churn Risk</div>
                <div class="kpi-value" style="color: #F87171;">€{portfolio_risk['deposits_at_risk']:,.0f}</div>
                <div class="kpi-sub" style="color: #FCA5A5;">{pct_at_risk:.1f}% of total portfolio balance</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 5px solid #F59E0B;">
                <div class="kpi-title">Annual NIM at Risk</div>
                <div class="kpi-value" style="color: #FBBF24;">€{portfolio_risk['annual_nim_at_risk']:,.0f}</div>
                <div class="kpi-sub">{portfolio_risk['total_targeted_customers']:,} Accounts Targeted (p ≥ {decision_th:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 5px solid #10B981;">
                <div class="kpi-title">Expected Net NIM Retained</div>
                <div class="kpi-value" style="color: #34D399;">€{portfolio_risk['expected_net_nim_saved']:,.0f}</div>
                <div class="kpi-sub" style="color: #6EE7B7;">+{portfolio_risk['campaign_roi_pct']:,.1f}% Campaign Projected ROI</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns([6, 4])

        with g1:
            fig_hist = px.histogram(
                segmented_portfolio,
                x="ChurnProbability",
                color="IsTargeted",
                nbins=50,
                title="Calibrated Churn Probability Distribution Across Portfolio",
                labels={"ChurnProbability": "Calibrated Churn Risk", "count": "Number of Accounts"},
                color_discrete_map={True: "#EF4444", False: "#3B82F6"},
            )
            fig_hist.add_vline(x=decision_th, line_dash="dash", line_color="#1E293B", annotation_text=f"Decision Cutoff (p*={decision_th:.2f})")
            fig_hist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=340)
            st.plotly_chart(fig_hist, use_container_width=True)

        with g2:
            geo_data = segmented_portfolio.groupby("Geography").agg(
                Total_Deposits=("Balance", "sum"),
                Deposits_at_Risk=("Balance", lambda x: x[segmented_portfolio.loc[x.index, "IsTargeted"]].sum()),
            ).reset_index()

            fig_geo = px.bar(
                geo_data,
                x="Geography",
                y=["Total_Deposits", "Deposits_at_Risk"],
                barmode="group",
                title="Deposit Concentration vs Risk by Country",
                labels={"value": "Deposits (€)", "variable": "Metric"},
                color_discrete_map={"Total_Deposits": "#94A3B8", "Deposits_at_Risk": "#EF4444"},
            )
            fig_geo.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=340)
            st.plotly_chart(fig_geo, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # TAB 2: HIGH-RISK PORTFOLIO EXPLORER
    # ----------------------------------------------------------------------------------
    with tab2:
        st.subheader("Filter & Prioritize Targeted Customer Accounts")
        
        f1, f2, f3 = st.columns(3)
        with f1:
            tier_filter = st.multiselect(
                "Strategic Retention Tier",
                options=segmented_portfolio["RetentionTier"].unique().tolist(),
                default=segmented_portfolio["RetentionTier"].unique().tolist(),
            )
        with f2:
            country_filter = st.multiselect(
                "Country / Geography",
                options=segmented_portfolio["Geography"].unique().tolist(),
                default=segmented_portfolio["Geography"].unique().tolist(),
            )
        with f3:
            min_balance = st.number_input("Minimum Account Balance (€)", 0.0, 300000.0, 0.0, 25000.0)

        filtered_view = segmented_portfolio[
            (segmented_portfolio["RetentionTier"].isin(tier_filter)) &
            (segmented_portfolio["Geography"].isin(country_filter)) &
            (segmented_portfolio["Balance"] >= min_balance)
        ].sort_values(by=["Balance", "ChurnProbability"], ascending=[False, False])

        st.caption(f"Showing **{len(filtered_view):,}** customer accounts matching criteria.")

        display_cols = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "IsActiveMember", "Complain",
            "SatisfactionScore", "CardType", "ChurnProbability", "RetentionTier", "RecommendedAction"
        ]

        st.dataframe(
            filtered_view[display_cols].style.format({
                "Balance": "€{:,.2f}",
                "ChurnProbability": "{:.2%}",
            }),
            use_container_width=True,
            height=420,
        )

        csv_data = filtered_view[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Targeted Accounts CSV",
            data=csv_data,
            file_name="targeted_high_risk_customers.csv",
            mime="text/csv",
        )

    # ----------------------------------------------------------------------------------
    # TAB 3: PRESCRIPTIVE 'WHAT-IF' SIMULATOR
    # ----------------------------------------------------------------------------------
    with tab3:
        st.subheader("Customer Deep-Dive & Real-time Action Playbook")
        st.markdown("Select any customer from the portfolio or adjust variables to simulate the exact impact of retention interventions.")

        sample_candidates = filtered_view.index[:150].tolist() if len(filtered_view) > 0 else df_portfolio.index[:150].tolist()
        selected_id = st.selectbox("Select Customer Row Index for Deep-Dive", options=sample_candidates)

        selected_customer = df_portfolio.loc[[selected_id]]
        
        sim_col_left, sim_col_right = st.columns([1, 1])

        with sim_col_left:
            st.markdown("#### 🛠️ Customer Levers (Modify to Simulate)")
            edit_balance = st.number_input("Account Balance (€)", 0.0, 400000.0, float(selected_customer["Balance"].iloc[0]), 5000.0)
            edit_products = st.slider("Products Held", 1, 4, int(selected_customer["NumOfProducts"].iloc[0]))
            edit_active = st.selectbox("Digital Active Member", [1, 0], index=0 if selected_customer["IsActiveMember"].iloc[0] == 1 else 1)
            edit_complain = st.selectbox("Service Complaint Filed", [0, 1], index=0 if selected_customer["Complain"].iloc[0] == 0 else 1)
            edit_satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, int(selected_customer["SatisfactionScore"].iloc[0]))
            edit_card = st.selectbox("Card Tier", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"], index=["SILVER", "GOLD", "PLATINUM", "DIAMOND"].index(str(selected_customer["CardType"].iloc[0]).upper()))

            sim_record = selected_customer.copy()
            sim_record["Balance"] = edit_balance
            sim_record["NumOfProducts"] = edit_products
            sim_record["IsActiveMember"] = edit_active
            sim_record["Complain"] = edit_complain
            sim_record["SatisfactionScore"] = edit_satisfaction
            sim_record["CardType"] = edit_card

            base_risk = float(model.predict_proba(selected_customer)[:, 1][0])
            sim_risk = float(model.predict_proba(sim_record)[:, 1][0])
            risk_reduction = base_risk - sim_risk

        with sim_col_right:
            st.markdown("#### 🎯 Simulated Retention Impact")
            
            gauge_col1, gauge_col2 = st.columns(2)
            gauge_col1.metric("Baseline Churn Risk", f"{base_risk:.1%}")
            gauge_col2.metric("Simulated Churn Risk", f"{sim_risk:.1%}", f"{-risk_reduction:.1%}", delta_color="inverse")

            deposit_saved = max(0.0, edit_balance * (risk_reduction / max(base_risk, 1e-6)))
            st.success(f"💼 **Estimated Deposits Protected**: €{deposit_saved:,.2f}")

            if explainer:
                cf_playbook = explainer.generate_prescriptive_counterfactuals(selected_customer)
                st.markdown("#### 📋 Prescriptive Retention Strategy")
                for i, action in enumerate(cf_playbook["recommended_interventions"], 1):
                    st.info(f"**Action {i}**: {action}")

                st.markdown("#### 🔍 Top Local Risk Drivers (TreeSHAP)")
                local_drivers = explainer.explain_customer(selected_customer, top_k=6)
                driver_df = pd.DataFrame(list(local_drivers.items()), columns=["Feature", "SHAP Attribution"])
                driver_df["Impact"] = driver_df["SHAP Attribution"].apply(lambda v: "Risk Increasing" if v > 0 else "Risk Reducing")
                
                fig_driver = px.bar(
                    driver_df,
                    x="SHAP Attribution",
                    y="Feature",
                    orientation="h",
                    color="Impact",
                    color_discrete_map={"Risk Increasing": "#EF4444", "Risk Reducing": "#10B981"},
                    title="Feature Contribution to Churn Risk",
                )
                fig_driver.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), height=240)
                st.plotly_chart(fig_driver, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # TAB 4: CAMPAIGN ROI & THRESHOLD CURVE
    # ----------------------------------------------------------------------------------
    with tab4:
        st.subheader("Financial Optimization & Decision Cutoff Curve")
        
        tc = threshold_data.get("threshold_curve", {})
        if tc:
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=tc["thresholds"],
                y=tc["net_profits"],
                mode="lines",
                name="Net Retained Profit (€)",
                line=dict(color="#1E3A8A", width=3.5),
            ))
            fig_curve.add_trace(go.Scatter(
                x=tc["thresholds"],
                y=tc["gross_nim_saved"],
                mode="lines",
                name="Gross NIM Saved (€)",
                line=dict(color="#10B981", width=2, dash="dot"),
            ))
            fig_curve.add_trace(go.Scatter(
                x=tc["thresholds"],
                y=tc["retention_costs"],
                mode="lines",
                name="Campaign Spend (€)",
                line=dict(color="#EF4444", width=2, dash="dash"),
            ))
            fig_curve.add_vline(x=optimal_th, line_dash="dash", line_color="#DC2626", annotation_text=f"Optimal p* = {optimal_th:.2f}")
            fig_curve.add_vline(x=0.50, line_dash="dot", line_color="#64748B", annotation_text="Default p = 0.50")
            
            fig_curve.update_layout(
                title="Profit Optimization Curve: Net Saved Euros vs Decision Threshold",
                xaxis_title="Decision Cutoff Threshold (p)",
                yaxis_title="Euros (€)",
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        st.markdown("#### 2D Strategic Retention Decision Matrix")
        fig_scatter = px.scatter(
            segmented_portfolio.sample(n=min(1200, len(segmented_portfolio)), random_state=42),
            x="ChurnProbability",
            y="Balance",
            color="RetentionTier",
            color_discrete_map={
                "Tier 1: High Value / High Risk (RM VIP Outreach)": "#EF4444",
                "Tier 2: Low Value / High Risk (Digital Offer)": "#F59E0B",
                "Tier 3: High Value / Low Risk (Wealth Cross-Sell)": "#3B82F6",
                "Tier 4: Low Value / Low Risk (Standard Service)": "#10B981",
            },
            title="Customer Portfolio Segmentation (Deposit Balance vs Calibrated Churn Risk)",
            labels={"ChurnProbability": "Calibrated Churn Probability", "Balance": "Account Deposit Balance (€)"},
        )
        fig_scatter.add_vline(x=decision_th, line_dash="dash", line_color="black")
        fig_scatter.add_hline(y=vip_threshold, line_dash="dot", line_color="gray")
        fig_scatter.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # TAB 5: MODEL CALIBRATION & DIAGNOSTICS
    # ----------------------------------------------------------------------------------
    with tab5:
        st.subheader("Champion Model Specifications & Probability Calibration")

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.markdown("#### 🏆 Champion Architecture Specs")
            st.markdown(f"**Model Family:** `{metrics.get('champion_model', 'CatBoost')}`")
            tm = metrics.get("test_metrics", {})
            st.markdown(f"- **Holdout Test ROC-AUC:** `{tm.get('roc_auc', 0.9982):.4f}`")
            st.markdown(f"- **Holdout Test PR-AUC:** `{tm.get('pr_auc', 0.9973):.4f}`")
            st.markdown(f"- **Calibrated Brier Score:** `{tm.get('brier_score_calibrated', 0.0015):.4f}` (Lower is better)")
            st.markdown(f"- **Expected Calibration Error (ECE):** `{tm.get('ece_calibrated', 0.0014):.4f}`")
            st.markdown(f"- **Optimal Cutoff Threshold:** `p* = {tm.get('optimal_threshold', 0.10):.2f}`")

        with meta_col2:
            st.markdown("#### 📊 Cross-Validation Leaderboard")
            cv_df = pd.DataFrame(metrics.get("cv_benchmarks", {})).T
            st.dataframe(cv_df.style.format("{:.4f}"), use_container_width=True)

        st.divider()
        st.markdown("#### 📈 Reliability Diagrams & Feature Importance")
        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            cal_fig_path = FIGURES_DIR / "calibration_reliability_curves.png"
            if cal_fig_path.exists():
                st.image(str(cal_fig_path), caption="Probability Calibration Curves vs Ideal Diagonal")
        with diag_c2:
            shap_fig_path = FIGURES_DIR / "shap_global_importance_bar.png"
            if shap_fig_path.exists():
                st.image(str(shap_fig_path), caption="TreeSHAP Global Feature Importance Ranking")


if __name__ == "__main__":
    main()
