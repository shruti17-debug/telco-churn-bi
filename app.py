"""
Telco Customer Churn Prediction & Retention Analytics
Complete BI project — single-file Streamlit application.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)
import os

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f8fafc; }
    [data-testid="stSidebar"] { background: #1e293b; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
        border-left: 4px solid #3b82f6;
        margin-bottom: 8px;
    }
    .metric-card.red  { border-left-color: #ef4444; }
    .metric-card.green{ border-left-color: #22c55e; }
    .metric-card.amber{ border-left-color: #f59e0b; }
    .metric-label { font-size:13px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
    .metric-value { font-size:32px; font-weight:700; color:#1e293b; line-height:1.2; }
    .metric-delta { font-size:12px; color:#64748b; margin-top:4px; }
    .section-header {
        font-size:20px; font-weight:700; color:#1e293b;
        border-bottom:2px solid #e2e8f0; padding-bottom:8px; margin:24px 0 16px;
    }
    .insight-box {
        background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
        padding:14px 18px; margin:8px 0; font-size:14px; color:#1e40af;
    }
    .risk-high   { background:#fef2f2; border-color:#fca5a5; color:#991b1b; }
    .risk-medium { background:#fffbeb; border-color:#fde68a; color:#92400e; }
    stDataFrame thead { background:#1e293b !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix TotalCharges (blank strings → NaN → median imputation)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # SeniorCitizen: already 0/1 but label it for display
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Tenure buckets
    bins   = [0, 12, 24, 48, 72]
    labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
    df["TenureBucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)

    # Monthly charge tier
    df["ChargeTier"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 95, 120],
        labels=["Low (<$35)", "Mid ($35–$65)", "High ($65–$95)", "Premium (>$95)"],
    )

    return df


# ─────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING & MODEL TRAINING
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_models(df: pd.DataFrame):
    model_df = df.copy()

    cat_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]

    le = LabelEncoder()
    for col in cat_cols:
        model_df[col] = le.fit_transform(model_df[col].astype(str))

    feature_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]

    X = model_df[feature_cols]
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    def metrics(y_true, y_pred, y_prob):
        return {
            "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
            "Precision": round(precision_score(y_true, y_pred) * 100, 2),
            "Recall":    round(recall_score(y_true, y_pred) * 100, 2),
            "F1":        round(f1_score(y_true, y_pred) * 100, 2),
            "AUC-ROC":   round(roc_auc_score(y_true, y_prob) * 100, 2),
            "CM":        confusion_matrix(y_true, y_pred),
            "FPR":       roc_curve(y_true, y_prob)[0],
            "TPR":       roc_curve(y_true, y_prob)[1],
        }

    lr_metrics = metrics(y_test, lr_pred, lr_prob)
    rf_metrics = metrics(y_test, rf_pred, rf_prob)

    # Pick winner by AUC-ROC
    if rf_metrics["AUC-ROC"] >= lr_metrics["AUC-ROC"]:
        best_name, best_model, best_metrics = "Random Forest", rf, rf_metrics
        best_pred, best_prob = rf_pred, rf_prob
    else:
        best_name, best_model, best_metrics = "Logistic Regression", lr, lr_metrics
        best_pred, best_prob = lr_pred, lr_prob

    # Score ALL customers with best model
    if best_name == "Random Forest":
        all_prob = best_model.predict_proba(X)[:, 1]
    else:
        all_prob = best_model.predict_proba(scaler.transform(X))[:, 1]

    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return {
        "lr": lr, "rf": rf,
        "lr_metrics": lr_metrics, "rf_metrics": rf_metrics,
        "best_name": best_name, "best_metrics": best_metrics,
        "X_test": X_test, "y_test": y_test,
        "best_pred": best_pred, "best_prob": best_prob,
        "all_prob": all_prob,
        "feature_importance": fi,
        "feature_cols": feature_cols,
    }


# ─────────────────────────────────────────────────────────
# HELPER — colour palette
# ─────────────────────────────────────────────────────────
CHURN_COLORS  = {"No": "#3b82f6", "Yes": "#ef4444"}
PALETTE_BLUE  = ["#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a"]
PALETTE_MIXED = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#06b6d4"]


def fmt_kpi(label: str, value: str, delta: str = "", colour: str = "") -> str:
    cls = f"metric-card {colour}"
    return f"""
    <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>"""


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.sidebar.markdown("## 📡 Churn Intelligence Hub")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Executive Overview", "📊 Trend & Driver Analysis", "🚨 Risk & Action Center", "🤖 Model Performance"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    # Global filters
    st.sidebar.markdown("### Filters")
    contracts = st.sidebar.multiselect(
        "Contract Type",
        options=df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist(),
    )
    internet = st.sidebar.multiselect(
        "Internet Service",
        options=df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist(),
    )
    tenure_range = st.sidebar.slider(
        "Tenure (months)", 0, int(df["tenure"].max()), (0, int(df["tenure"].max()))
    )

    filtered = df[
        df["Contract"].isin(contracts) &
        df["InternetService"].isin(internet) &
        df["tenure"].between(tenure_range[0], tenure_range[1])
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} customers")
    return page, filtered


# ─────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────
def page_executive(df: pd.DataFrame, result: dict):
    st.markdown('<div class="section-header">🏠 Executive Overview</div>', unsafe_allow_html=True)

    churn_rate   = df["Churn"].mean() * 100
    total_cust   = len(df)
    churned_cust = df["Churn"].sum()
    rev_at_risk  = df[df["Churn"] == 1]["MonthlyCharges"].sum()
    avg_tenure   = df["tenure"].mean()
    avg_monthly  = df["MonthlyCharges"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(fmt_kpi("Churn Rate", f"{churn_rate:.1f}%", f"{churned_cust:,} customers lost", "red"), unsafe_allow_html=True)
    with k2:
        st.markdown(fmt_kpi("Total Customers", f"{total_cust:,}", "Active subscriber base", ""), unsafe_allow_html=True)
    with k3:
        st.markdown(fmt_kpi("Monthly Revenue at Risk", f"${rev_at_risk:,.0f}", "From predicted churners", "amber"), unsafe_allow_html=True)
    with k4:
        st.markdown(fmt_kpi("Avg Customer Tenure", f"{avg_tenure:.1f} mo", f"Avg monthly bill: ${avg_monthly:.2f}", "green"), unsafe_allow_html=True)

    st.markdown("")

    col_a, col_b = st.columns([1, 1])

    # Churn distribution donut
    with col_a:
        st.markdown("##### Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes  = [100 - churn_rate, churn_rate]
        labels = ["Retained", "Churned"]
        colors = ["#3b82f6", "#ef4444"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
        ax.set_title("Customer Churn Share", fontweight="bold", pad=12)
        st.pyplot(fig); plt.close()

    # Monthly charges distribution by churn
    with col_b:
        st.markdown("##### Monthly Charges by Churn Status")
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, color in [("No", "#3b82f6"), ("Yes", "#ef4444")]:
            subset = df[df["Churn"] == (1 if label == "Yes" else 0)]["MonthlyCharges"]
            ax.hist(subset, bins=30, alpha=0.65, color=color, label=label, edgecolor="white")
        ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Customers")
        ax.legend(title="Churned", frameon=False)
        ax.set_title("Monthly Charge Distribution", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    col_c, col_d = st.columns([1, 1])

    # Churn by contract type
    with col_c:
        st.markdown("##### Churn Rate by Contract")
        grp = df.groupby("Contract")["Churn"].mean().reset_index()
        grp["ChurnPct"] = grp["Churn"] * 100
        grp = grp.sort_values("ChurnPct", ascending=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(grp["Contract"], grp["ChurnPct"],
                       color=["#22c55e" if v < 20 else "#f59e0b" if v < 45 else "#ef4444"
                              for v in grp["ChurnPct"]], edgecolor="white")
        for bar, val in zip(bars, grp["ChurnPct"]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Churn Rate (%)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title("Churn Rate by Contract Type", fontweight="bold")
        st.pyplot(fig); plt.close()

    # Senior vs non-senior churn
    with col_d:
        st.markdown("##### Churn Rate: Senior vs Non-Senior")
        grp2 = df.groupby("SeniorCitizenLabel")["Churn"].mean().reset_index()
        grp2["ChurnPct"] = grp2["Churn"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors_bar = ["#3b82f6", "#f59e0b"]
        ax.bar(grp2["SeniorCitizenLabel"], grp2["ChurnPct"], color=colors_bar, width=0.45, edgecolor="white")
        for i, (_, row) in enumerate(grp2.iterrows()):
            ax.text(i, row["ChurnPct"] + 0.5, f"{row['ChurnPct']:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Senior Citizen Churn Comparison", fontweight="bold")
        st.pyplot(fig); plt.close()

    # Key insights
    st.markdown("---")
    st.markdown("#### 💡 Key Executive Insights")
    monthly_churn = df[df["Churn"] == 1]["Contract"].value_counts(normalize=True).get("Month-to-month", 0) * 100
    fiber_churn   = df[df["InternetService"] == "Fiber optic"]["Churn"].mean() * 100
    senior_churn  = df[df["SeniorCitizen"] == 1]["Churn"].mean() * 100
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="insight-box risk-high">📌 <b>{monthly_churn:.0f}%</b> of churned customers were on Month-to-month contracts — the single biggest churn driver.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="insight-box risk-medium">📌 Fiber Optic subscribers churn at <b>{fiber_churn:.1f}%</b> — nearly 3× the rate of DSL customers.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="insight-box">📌 Senior citizens churn at <b>{senior_churn:.1f}%</b> — targeted support programs could reduce this significantly.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE 2 — TREND & DRIVER ANALYSIS
# ─────────────────────────────────────────────────────────
def page_trends(df: pd.DataFrame):
    st.markdown('<div class="section-header">📊 Trend & Driver Analysis</div>', unsafe_allow_html=True)

    # Internet service
    row1_a, row1_b = st.columns(2)
    with row1_a:
        st.markdown("##### Churn Rate by Internet Service")
        g = df.groupby("InternetService")["Churn"].mean().reset_index()
        g["pct"] = g["Churn"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        clr = ["#22c55e" if v < 20 else "#f59e0b" if v < 40 else "#ef4444" for v in g["pct"]]
        ax.bar(g["InternetService"], g["pct"], color=clr, edgecolor="white", width=0.5)
        for i, (_, row) in enumerate(g.iterrows()):
            ax.text(i, row["pct"] + 0.3, f"{row['pct']:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with row1_b:
        st.markdown("##### Churn Rate by Tenure Bucket")
        g2 = df.groupby("TenureBucket", observed=True)["Churn"].mean().reset_index()
        g2["pct"] = g2["Churn"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(g2["TenureBucket"].astype(str), g2["pct"], marker="o", color="#3b82f6", linewidth=2.5, markersize=8)
        ax.fill_between(range(len(g2)), g2["pct"], alpha=0.12, color="#3b82f6")
        ax.set_xticks(range(len(g2))); ax.set_xticklabels(g2["TenureBucket"].astype(str), rotation=15)
        ax.set_ylabel("Churn Rate (%)")
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(g2["pct"]):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        st.pyplot(fig); plt.close()

    row2_a, row2_b = st.columns(2)
    with row2_a:
        st.markdown("##### Churn Rate by Payment Method")
        g3 = df.groupby("PaymentMethod")["Churn"].mean().reset_index().sort_values("Churn")
        g3["pct"] = g3["Churn"] * 100
        # Shorten labels
        g3["shortLabel"] = g3["PaymentMethod"].str.replace(" (automatic)", "\n(auto)", regex=False)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        clr = [PALETTE_MIXED[i % len(PALETTE_MIXED)] for i in range(len(g3))]
        ax.barh(g3["shortLabel"], g3["pct"], color=clr, edgecolor="white")
        for bar, val in zip(ax.patches, g3["pct"]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax.set_xlabel("Churn Rate (%)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with row2_b:
        st.markdown("##### Churn Rate by Monthly Charge Tier")
        g4 = df.groupby("ChargeTier", observed=True)["Churn"].mean().reset_index()
        g4["pct"] = g4["Churn"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.8))
        clr4 = ["#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8"]
        ax.bar(g4["ChargeTier"].astype(str), g4["pct"], color=clr4, edgecolor="white", width=0.55)
        for i, (_, row) in enumerate(g4.iterrows()):
            ax.text(i, row["pct"] + 0.3, f"{row['pct']:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xticklabels(g4["ChargeTier"].astype(str), rotation=10)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    # Heatmap: contract × internet
    st.markdown("##### Churn Heatmap: Contract Type × Internet Service")
    pivot = df.pivot_table(values="Churn", index="Contract", columns="InternetService", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5,
                cbar_kws={"label": "Churn %"}, ax=ax)
    ax.set_title("Churn Rate (%) by Contract × Internet Service", fontweight="bold")
    st.pyplot(fig); plt.close()

    # Add-on service churn comparison
    st.markdown("##### Churn Rate: Add-on Services (subscribers vs non-subscribers)")
    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    rows = []
    for svc in services:
        for val in ["Yes", "No"]:
            sub = df[df[svc] == val]
            if len(sub):
                rows.append({"Service": svc.replace("Streaming", "Streaming "), "Subscriber": val, "ChurnRate": sub["Churn"].mean() * 100})
    svc_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4))
    x      = np.arange(len(services))
    width  = 0.35
    yes_rates = svc_df[svc_df["Subscriber"] == "Yes"]["ChurnRate"].values
    no_rates  = svc_df[svc_df["Subscriber"] == "No"]["ChurnRate"].values
    ax.bar(x - width / 2, yes_rates, width, label="Subscriber", color="#3b82f6", edgecolor="white")
    ax.bar(x + width / 2, no_rates,  width, label="Non-Subscriber", color="#ef4444", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("Streaming", "Streaming\n") for s in services], fontsize=9)
    ax.set_ylabel("Churn Rate (%)"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────
# PAGE 3 — RISK & ACTION CENTER
# ─────────────────────────────────────────────────────────
def page_risk(df: pd.DataFrame, result: dict):
    st.markdown('<div class="section-header">🚨 Risk & Action Center</div>', unsafe_allow_html=True)

    all_prob = result["all_prob"]
    risk_df  = df.copy()
    risk_df["ChurnProbability"] = (all_prob * 100).round(1)
    risk_df["RiskTier"] = pd.cut(
        risk_df["ChurnProbability"],
        bins=[0, 40, 70, 100],
        labels=["Low (<40%)", "Medium (40–70%)", "High (>70%)"],
    )

    # Summary KPIs
    high   = (risk_df["RiskTier"] == "High (>70%)").sum()
    medium = (risk_df["RiskTier"] == "Medium (40–70%)").sum()
    low    = (risk_df["RiskTier"] == "Low (<40%)").sum()
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(fmt_kpi("High Risk Customers", f"{high:,}", "> 70% churn probability", "red"), unsafe_allow_html=True)
    with k2:
        st.markdown(fmt_kpi("Medium Risk Customers", f"{medium:,}", "40–70% churn probability", "amber"), unsafe_allow_html=True)
    with k3:
        st.markdown(fmt_kpi("Low Risk Customers", f"{low:,}", "< 40% churn probability", "green"), unsafe_allow_html=True)

    st.markdown("")

    # Top high-value at-risk customers
    st.markdown("#### 🎯 High-Value Customers Predicted to Churn (Top 20 by Monthly Charges)")
    top_risk = (
        risk_df[risk_df["ChurnProbability"] >= 50]
        .sort_values("MonthlyCharges", ascending=False)
        .head(20)[["customerID", "Contract", "InternetService", "tenure",
                   "MonthlyCharges", "TotalCharges", "ChurnProbability", "PaymentMethod"]]
        .reset_index(drop=True)
    )
    top_risk.index = top_risk.index + 1

    def colour_prob(val):
        if val >= 70:
            return "background-color:#fef2f2; color:#991b1b; font-weight:bold"
        elif val >= 50:
            return "background-color:#fffbeb; color:#92400e; font-weight:bold"
        return ""

    styled = top_risk.style.applymap(colour_prob, subset=["ChurnProbability"])
    st.dataframe(styled, use_container_width=True)

    # Retention actions
    st.markdown("---")
    st.markdown("#### 💼 Recommended Retention Actions")

    action_map = {
        "Month-to-month": {
            "title": "🔄 Contract Upgrade Incentive",
            "action": "Offer 3–6 months free or 20% discount to upgrade to 1- or 2-year contract. Month-to-month customers churn at 42% vs 3% for 2-year contracts.",
            "tier": "risk-high",
        },
        "Fiber optic": {
            "title": "🌐 Fiber Optic Satisfaction Programme",
            "action": "Proactive outreach to Fiber Optic subscribers: service quality check, speed upgrade offer, or bill credit. Fiber churn rate is ~42%.",
            "tier": "risk-medium",
        },
        "Electronic check": {
            "title": "💳 Auto-Pay Migration Campaign",
            "action": "Incentivise switch to automatic payment (credit card / bank transfer) with $5/mo bill credit. Electronic check customers churn at 45%.",
            "tier": "risk-high",
        },
        "No_security": {
            "title": "🔒 Security Bundle Upsell",
            "action": "Bundle OnlineSecurity + TechSupport at a discounted rate. Customers without these services churn at ~2× the rate of subscribers.",
            "tier": "risk-medium",
        },
        "Senior": {
            "title": "👴 Senior Loyalty Programme",
            "action": "Dedicated support line, simplified billing, and senior discount plan. Senior citizens churn at ~42% vs 24% overall.",
            "tier": "risk-medium",
        },
        "Short_tenure": {
            "title": "🆕 New Customer Onboarding",
            "action": "0–12 month customers churn most. Introduce a 90-day welcome programme: check-in calls, tutorial resources, and first-year discount lock.",
            "tier": "risk-high",
        },
    }

    pairs = list(action_map.values())
    for i in range(0, len(pairs), 2):
        c1, c2 = st.columns(2)
        for col, item in zip([c1, c2], pairs[i:i+2]):
            with col:
                st.markdown(
                    f'<div class="insight-box {item["tier"]}"><b>{item["title"]}</b><br>{item["action"]}</div>',
                    unsafe_allow_html=True,
                )

    # Risk distribution chart
    st.markdown("---")
    st.markdown("#### 📈 Churn Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(risk_df["ChurnProbability"], bins=40, color="#3b82f6", edgecolor="white", alpha=0.8)
    ax.axvline(40, color="#f59e0b", linestyle="--", linewidth=2, label="Medium Risk Threshold (40%)")
    ax.axvline(70, color="#ef4444", linestyle="--", linewidth=2, label="High Risk Threshold (70%)")
    ax.set_xlabel("Predicted Churn Probability (%)")
    ax.set_ylabel("Number of Customers")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────
# PAGE 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────
def page_model(result: dict):
    st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)

    lr_m = result["lr_metrics"]
    rf_m = result["rf_metrics"]

    st.markdown(f"#### Best Model: **{result['best_name']}** ✅")
    st.caption("Both models trained; best selected by AUC-ROC score on 20% held-out test set.")

    # Side-by-side metrics
    col_lr, col_rf = st.columns(2)
    for col, name, m in [(col_lr, "Logistic Regression", lr_m), (col_rf, "Random Forest", rf_m)]:
        with col:
            best_tag = " ✅ (Selected)" if name == result["best_name"] else ""
            st.markdown(f"**{name}{best_tag}**")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            for mc, label, key in [
                (mc1, "Accuracy",  "Accuracy"),
                (mc2, "Precision", "Precision"),
                (mc3, "Recall",    "Recall"),
                (mc4, "F1",        "F1"),
                (mc5, "AUC-ROC",   "AUC-ROC"),
            ]:
                with mc:
                    st.metric(label, f"{m[key]:.1f}%")

    st.markdown("---")

    col_cm1, col_cm2 = st.columns(2)
    for col, name, m in [(col_cm1, "Logistic Regression", lr_m), (col_cm2, "Random Forest", rf_m)]:
        with col:
            st.markdown(f"##### Confusion Matrix — {name}")
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(m["CM"], annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Predicted No", "Predicted Yes"],
                        yticklabels=["Actual No", "Actual Yes"],
                        linewidths=0.5, cbar=False)
            ax.set_title(f"{name}", fontweight="bold")
            st.pyplot(fig); plt.close()

    # ROC curves
    st.markdown("##### ROC Curves — Both Models")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lr_m["FPR"], lr_m["TPR"], color="#3b82f6", lw=2, label=f"Logistic Regression (AUC = {lr_m['AUC-ROC']:.1f}%)")
    ax.plot(rf_m["FPR"], rf_m["TPR"], color="#ef4444", lw=2, label=f"Random Forest (AUC = {rf_m['AUC-ROC']:.1f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig); plt.close()

    # Feature importance
    st.markdown("##### Random Forest — Feature Importance (Top 15)")
    fi = result["feature_importance"].head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_fi = [PALETTE_BLUE[min(int(v * 25), 4)] for v in (fi.values / fi.values.max())]
    ax.barh(fi.index[::-1], fi.values[::-1], color=colors_fi[::-1], edgecolor="white")
    ax.set_xlabel("Feature Importance Score")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Feature Importance — Random Forest", fontweight="bold")
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    # ── locate dataset ──
    csv_candidates = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_churn.csv",
        "churn.csv",
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        st.error(
            "📂 Dataset not found. Please place **WA_Fn-UseC_-Telco-Customer-Churn.csv** "
            "in the same directory as this script and refresh the page."
        )
        st.info(
            "Download from Kaggle: "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )
        st.stop()

    with st.spinner("Loading & preparing data …"):
        df = load_and_clean(csv_path)

    with st.spinner("Training models (first run only — cached thereafter) …"):
        result = build_models(df)

    page, filtered_df = render_sidebar(df)

    if page == "🏠 Executive Overview":
        page_executive(filtered_df, result)
    elif page == "📊 Trend & Driver Analysis":
        page_trends(filtered_df)
    elif page == "🚨 Risk & Action Center":
        page_risk(filtered_df, result)
    elif page == "🤖 Model Performance":
        page_model(result)


if __name__ == "__main__":
    main()
