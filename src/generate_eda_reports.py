"""
Exploratory Data Analysis and Feature Profiling Generator for Bank Churn & Retention Engine.
Produces publication-grade visualizations for the portfolio and executive reports.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import PROCESSED_DATA_PATH, FIGURES_DIR, TARGET_COL

# Apply modern aesthetic theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

COLORS = {
    "primary": "#1E3A8A",      # Navy
    "secondary": "#0284C7",    # Sky Blue
    "accent": "#EF4444",       # Crimson/Coral Churn
    "success": "#10B981",      # Emerald Retained
    "neutral": "#64748B",      # Slate
    "card_bg": "#F8FAFC",
}


def generate_eda_figures():
    """Generate high-resolution EDA figures saved to reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"Loaded processed dataset ({len(df):,} rows) for EDA generation...")

    # 1. Churn Rate by Geography, Gender, and Age Group
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Geography Churn Rate
    geo_churn = df.groupby("Geography")[TARGET_COL].mean().reset_index()
    sns.barplot(
        data=geo_churn, x="Geography", y=TARGET_COL, ax=axes[0],
        palette=[COLORS["secondary"], COLORS["accent"], COLORS["primary"]]
    )
    axes[0].set_title("Churn Rate by Geography")
    axes[0].set_ylabel("Churn Proportion")
    axes[0].set_ylim(0, 0.45)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Gender Churn Rate
    gender_churn = df.groupby("Gender")[TARGET_COL].mean().reset_index()
    sns.barplot(
        data=gender_churn, x="Gender", y=TARGET_COL, ax=axes[1],
        palette=[COLORS["accent"], COLORS["primary"]]
    )
    axes[1].set_title("Churn Rate by Gender")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 0.35)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Age Distribution by Churn
    sns.kdeplot(data=df[df[TARGET_COL] == 0]["Age"], ax=axes[2], label="Retained (0)", color=COLORS["success"], fill=True, alpha=0.3)
    sns.kdeplot(data=df[df[TARGET_COL] == 1]["Age"], ax=axes[2], label="Churned (1)", color=COLORS["accent"], fill=True, alpha=0.3)
    axes[2].set_title("Age Density: Retained vs Churned")
    axes[2].set_xlabel("Customer Age")
    axes[2].set_ylabel("Density")
    axes[2].legend()

    plt.tight_layout()
    fig_path1 = FIGURES_DIR / "01_demographics_churn_distribution.png"
    plt.savefig(fig_path1)
    plt.close()
    print(f"Saved: {fig_path1}")

    # 2. Wealth Dynamics: Balance to Salary Ratio & Wealth Tiers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wealth Tier Churn
    tier_order = ["Zero_Balance", "Mass_Market", "Affluent", "High_Net_Worth"]
    tier_churn = df.groupby("WealthTier")[TARGET_COL].mean().reindex(tier_order).reset_index()
    sns.barplot(data=tier_churn, x="WealthTier", y=TARGET_COL, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Churn Rate Across Wealth Tiers")
    axes[0].set_xlabel("Wealth Tier (Deposit Concentration)")
    axes[0].set_ylabel("Churn Proportion")
    axes[0].set_ylim(0, 0.35)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Balance vs Salary Ratio boxplot
    sns.boxplot(data=df, x=TARGET_COL, y="BalanceToSalaryRatio", ax=axes[1], palette=[COLORS["success"], COLORS["accent"]], showfliers=False)
    axes[1].set_title("Balance-to-Salary Ratio by Churn Status")
    axes[1].set_xticklabels(["Retained", "Churned"])
    axes[1].set_ylabel("Balance / Salary Ratio")

    plt.tight_layout()
    fig_path2 = FIGURES_DIR / "02_wealth_dynamics.png"
    plt.savefig(fig_path2)
    plt.close()
    print(f"Saved: {fig_path2}")

    # 3. Service Complaints & Multi-Product Flight Risk
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Product count inflection
    prod_churn = df.groupby("NumOfProducts")[TARGET_COL].mean().reset_index()
    sns.barplot(data=prod_churn, x="NumOfProducts", y=TARGET_COL, ax=axes[0], palette="Reds")
    axes[0].set_title("Churn Rate by Number of Products Held")
    axes[0].set_xlabel("Products Held")
    axes[0].set_ylabel("Churn Proportion")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Complaint Impact
    complain_churn = df.groupby("Complain")[TARGET_COL].mean().reset_index()
    sns.barplot(data=complain_churn, x="Complain", y=TARGET_COL, ax=axes[1], palette=[COLORS["success"], COLORS["accent"]])
    axes[1].set_title("Churn Rate: Non-Complaining vs Complaining Customers")
    axes[1].set_xticklabels(["No Complaint Filed", "Complaint Filed"])
    axes[1].set_ylabel("Churn Proportion")
    axes[1].set_ylim(0, 1.05)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path3 = FIGURES_DIR / "03_product_and_complaint_drivers.png"
    plt.savefig(fig_path3)
    plt.close()
    print(f"Saved: {fig_path3}")

    # 4. Correlation Matrix of Engineered Features
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", vmin=-0.5, vmax=0.5,
        annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8}
    )
    plt.title("Correlation Matrix: Banking & Engineered Features", pad=15)
    plt.tight_layout()
    fig_path4 = FIGURES_DIR / "04_engineered_feature_correlations.png"
    plt.savefig(fig_path4)
    plt.close()
    print(f"Saved: {fig_path4}")


if __name__ == "__main__":
    generate_eda_figures()
