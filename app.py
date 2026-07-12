"""
DemandIQ — Sales Forecasting & Intelligence Dashboard
A professional, production-ready Streamlit dashboard for sales analysis,
forecasting, anomaly detection, and product segmentation.
Upgraded with a modern, high-end SaaS analytics design system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─── LOAD PRE-TRAINED ARTIFACTS FROM NOTEBOOK ────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load all pre-trained models and transformers exported from analysis.ipynb."""
    artifacts = {}
    try:
        artifacts["model"] = joblib.load("model.pkl")
    except Exception as e:
        st.error(f"❌ Could not load model.pkl: {e}")
        artifacts["model"] = None
    try:
        artifacts["scaler"] = joblib.load("scaler.pkl")
    except Exception as e:
        artifacts["scaler"] = None
    try:
        artifacts["pca"] = joblib.load("pca.pkl")
    except Exception as e:
        artifacts["pca"] = None
    try:
        artifacts["kmeans"] = joblib.load("kmeans.pkl")
    except Exception as e:
        artifacts["kmeans"] = None
    try:
        artifacts["columns"] = joblib.load("columns.pkl")
    except Exception as e:
        artifacts["columns"] = None
    return artifacts

# ─── PAGE CONFIGURATION ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DemandIQ — Sales Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN SYSTEM & PALETTE ─────────────────────────────────────────────────
COLORS = {
    "primary": "#3B82F6",   # Blue
    "secondary": "#8B5CF6", # Purple
    "accent": "#22D3EE",    # Cyan
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "background": "#0B1120",
    "card_bg": "#111827",
    "text": "#E5E7EB",
}

PLOTLY_PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["success"],
    COLORS["warning"],
    COLORS["danger"]
]


PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0B1120",
    plot_bgcolor="#0B1120",
    font=dict(family="Inter, sans-serif", color="#E5E7EB"),

    margin=dict(l=40, r=40, t=50, b=40),

    xaxis=dict(
        gridcolor="rgba(148,163,184,0.08)",
        zerolinecolor="rgba(148,163,184,0.08)",
        tickfont=dict(color="#94A3B8"),
        title=dict(font=dict(color="#E5E7EB")),
        showline=True,
        linecolor="rgba(255,255,255,0.08)"
    ),

    yaxis=dict(
        gridcolor="rgba(148,163,184,0.08)",
        zerolinecolor="rgba(148,163,184,0.08)",
        tickfont=dict(color="#94A3B8"),
        title=dict(font=dict(color="#E5E7EB")),
        showline=True,
        linecolor="rgba(255,255,255,0.08)"
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),

    hoverlabel=dict(
        bgcolor="#111827",
        font_size=13,
        font_family="Inter, sans-serif",
        font_color="#F9FAFB",
        bordercolor="rgba(255,255,255,0.08)"
    ),

    hovermode="x unified"
)
# ─── PREMIUM CUSTOM CSS INJECTION ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Main Background and fonts */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
    color: #ffffff;
}
.stApp {
    background-color: #0E1117 !important;
}

/* Sidebar premium background */
[data-testid="stSidebar"] {
    background-color: #121620 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Sidebar navigation styled items */
div[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 12px 16px !important;
    border-radius: 12px !important;
    margin-bottom: 6px !important;
    background-color: #161C2A !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: #A0AEC0 !important;
    width: 100% !important;
    transition: all 0.25s ease-in-out !important;
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #1C2436 !important;
    border-color: rgba(79, 139, 249, 0.4) !important;
    color: #ffffff !important;
}
div[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
    background: linear-gradient(135deg, rgba(79, 139, 249, 0.2) 0%, rgba(124, 77, 255, 0.2) 100%) !important;
    border: 1px solid rgba(79, 139, 249, 0.6) !important;
    color: #ffffff !important;
}

/* Styled Container Card */
.card {
    background: linear-gradient(135deg, #1C2333 0%, #251B40 100%);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 24px;
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.card:hover {
    transform: translateY(-2px);
    border-color: rgba(79, 139, 249, 0.35);
}

/* KPI Card layout styling */
.kpi-card {
    background: linear-gradient(135deg, #1C2333 0%, #251B40 100%);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
    height: 136px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(79, 139, 249, 0.4);
    box-shadow: 0 12px 40px rgba(79, 139, 249, 0.2);
}
.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.kpi-label {
    font-size: 13px;
    font-weight: 600;
    color: #A0AEC0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(90deg, #4F8BF9, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 4px;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 12px;
    color: #A0AEC0;
    font-weight: 500;
    margin-top: 4px;
}

/* Dropdowns and select widgets */
div[data-baseweb="select"] > div {
    background-color: #1C2333 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Premium styling for dataframes and tables */
div[data-testid="stDataFrame"] {
    background-color: #1C2333 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  UI LAYOUT MODULES (COMPONENTS)
# ═══════════════════════════════════════════════════════════════════════════

def page_header(title, description, icon="📊"):
    """Render a premium page header banner."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(79, 139, 249, 0.12) 0%, rgba(124, 77, 255, 0.12) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 28px;
        backdrop-filter: blur(12px);
    ">
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 32px;">{icon}</span>
            <div>
                <h1 style="
                    margin: 0;
                    font-size: 24px;
                    font-weight: 800;
                    background: linear-gradient(90deg, #4F8BF9, #00E5FF);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">{title}</h1>
                <p style="margin: 4px 0 0 0; color: #A0AEC0; font-size: 14px; font-weight: 400;">{description}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle):
    """Render a section header with standard divider line."""
    st.markdown(f"""
    <div style="margin-top: 28px; margin-bottom: 12px;">
        <h3 style="margin: 0; font-size: 18px; font-weight: 700; color: #ffffff;">{title}</h3>
        <p style="margin: 2px 0 10px 0; font-size: 13px; color: #A0AEC0;">{subtitle}</p>
        <hr style="margin: 0 0 18px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);">
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", tooltip=""):
    """Render same-height, perfectly-aligned KPI card with tooltip support."""
    tooltip_html = f'<span style="cursor: help; color: #A0AEC0; font-size: 12px; margin-left: 4px;" title="{tooltip}">ⓘ</span>' if tooltip else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">{label}</span>
                {tooltip_html}
            </div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  DATA PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    df["YearWeek"] = df["Order Date"].dt.to_period("W")
    return df


@st.cache_data
def build_monthly_series(df):
    monthly = df.groupby("YearMonth")["Sales"].sum().reset_index()
    monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()
    return monthly.sort_values("Date").reset_index(drop=True)


@st.cache_data
def build_weekly_series(df):
    weekly = df.groupby("YearWeek")["Sales"].sum().reset_index()
    weekly["Date"] = weekly["YearWeek"].dt.to_timestamp()
    return weekly.sort_values("Date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MODELING & PREDICTION PIPELINE — USES PRE-TRAINED ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════

# Feature names as exported from the notebook (model.pkl)
FEATURE_COLS = ["lag1", "lag2", "lag3", "rolling_mean", "month", "quarter"]


def create_features(series_df, target_col="Sales"):
    """Create lag/rolling features matching the notebook's exported model feature names."""
    df = series_df.copy()
    df["lag1"] = df[target_col].shift(1)
    df["lag2"] = df[target_col].shift(2)
    df["lag3"] = df[target_col].shift(3)
    df["rolling_mean"] = df[target_col].shift(1).rolling(window=3).mean()
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df.dropna(inplace=True)
    return df


@st.cache_data
def evaluate_model(series_df, _model):
    """Evaluate the pre-trained model on a test split (no re-training)."""
    df = create_features(series_df)
    feature_cols = FEATURE_COLS

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_test, y_test = test[feature_cols], test["Sales"]

    preds = _model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return train, test, preds, mae, rmse, feature_cols


def forecast_future(model, series_df, feature_cols, horizon=3):
    df = create_features(series_df)
    history = df["Sales"].tolist()
    last_date = df["Date"].iloc[-1]
    forecasts = []

    for i in range(horizon):
        lag1 = history[-1]
        lag2 = history[-2]
        lag3 = history[-3]
        rm3 = np.mean(history[-3:])
        next_date = last_date + pd.DateOffset(months=i + 1)
        month_num = next_date.month
        quarter_num = (next_date.month - 1) // 3 + 1
        row = pd.DataFrame(
            [[lag1, lag2, lag3, rm3, month_num, quarter_num]],
            columns=feature_cols,
        )
        pred = model.predict(row)[0]
        history.append(pred)
        forecasts.append({"Date": next_date, "Forecast": pred})

    return pd.DataFrame(forecasts)


@st.cache_data
def detect_anomalies(weekly_df):
    df = weekly_df.copy()
    iso = IsolationForest(contamination=0.08, random_state=42)
    df["ISO_Anomaly"] = iso.fit_predict(df[["Sales"]])
    df["ISO_Anomaly"] = df["ISO_Anomaly"].map({1: 0, -1: 1})

    rolling_mean = df["Sales"].rolling(window=8, min_periods=1).mean()
    rolling_std = df["Sales"].rolling(window=8, min_periods=1).std().fillna(0)
    rolling_std = rolling_std.replace(0, 1)
    df["Z_Score"] = (df["Sales"] - rolling_mean) / rolling_std
    df["Z_Anomaly"] = (df["Z_Score"].abs() > 2).astype(int)

    df["Anomaly"] = ((df["ISO_Anomaly"] == 1) | (df["Z_Anomaly"] == 1)).astype(int)
    df["Type"] = df.apply(
        lambda r: ("High ⬆" if r["Z_Score"] > 0 else "Low ⬇") if r["Anomaly"] == 1 else "Normal",
        axis=1,
    )
    return df


@st.cache_data
def run_clustering(df, _scaler, _kmeans, _pca):
    """Segment sub-categories using pre-trained scaler, KMeans and PCA from notebook."""
    sub_sales = df.groupby("Sub-Category").agg(
        TotalSales=("Sales", "sum"),
        AvgOrder=("Sales", "mean"),
        Order_Count=("Sales", "count"),
    ).reset_index()

    mid = df["Order Date"].min() + (df["Order Date"].max() - df["Order Date"].min()) / 2
    first_half = df[df["Order Date"] <= mid].groupby("Sub-Category")["Sales"].sum().rename("First_Half")
    second_half = df[df["Order Date"] > mid].groupby("Sub-Category")["Sales"].sum().rename("Second_Half")
    growth = pd.concat([first_half, second_half], axis=1).fillna(0)
    growth["Growth"] = ((growth["Second_Half"] - growth["First_Half"]) / growth["First_Half"].replace(0, 1)) * 100

    monthly_sub = df.groupby(["Sub-Category", "YearMonth"])["Sales"].sum().reset_index()
    volatility = monthly_sub.groupby("Sub-Category")["Sales"].std().rename("Volatility").fillna(0).reset_index()

    cluster_df = sub_sales.merge(growth[["Growth"]], left_on="Sub-Category", right_index=True, how="left")
    cluster_df = cluster_df.merge(volatility, on="Sub-Category", how="left")
    cluster_df.fillna(0, inplace=True)

    # Use the exact feature names from columns.pkl: ['TotalSales', 'Growth', 'Volatility', 'AvgOrder']
    feature_cols = ["TotalSales", "Growth", "Volatility", "AvgOrder"]

    # Transform with the pre-trained scaler (no re-fitting)
    X_scaled = _scaler.transform(cluster_df[feature_cols])

    # Predict clusters with the pre-trained KMeans (no re-fitting)
    cluster_df["Cluster"] = _kmeans.predict(X_scaled)

    # Project with pre-trained PCA (no re-fitting)
    pca_coords = _pca.transform(X_scaled)
    cluster_df["PC1"] = pca_coords[:, 0]
    cluster_df["PC2"] = pca_coords[:, 1]

    # Build label map dynamically based on actual number of clusters
    n_clusters = _kmeans.n_clusters
    base_labels = ["Stars ⭐", "Growth 🚀", "Stable 🛡️", "Niche 🎯"]
    label_map = {i: base_labels[i] for i in range(min(n_clusters, len(base_labels)))}
    cluster_df["Cluster_Label"] = cluster_df["Cluster"].map(label_map)

    explained = _pca.explained_variance_ratio_

    # Rename for display compatibility
    cluster_df = cluster_df.rename(columns={
        "TotalSales": "Total_Sales",
        "Growth": "Growth_Rate",
        "AvgOrder": "Avg_Order_Value",
    })

    return cluster_df, explained


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — SALES OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def page_sales_overview(df):
    page_header("Sales Overview", "Comprehensive insights into revenue metrics, performance trends, and product categories", "📊")

    # ── Sidebar Filter Setup ──
    st.sidebar.markdown("""
    <div style="margin-top: 10px; margin-bottom: 12px;">
        <h4 style="margin: 0; font-size: 14px; font-weight: 700; color: #ffffff;">🔍 Filter Settings</h4>
        <hr style="margin: 6px 0 16px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);">
    </div>
    """, unsafe_allow_html=True)
    
    regions = ["All"] + sorted(df["Region"].unique().tolist())
    selected_region = st.sidebar.selectbox("Region Scope", regions, key="ov_region")

    categories = ["All"] + sorted(df["Category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Category Scope", categories, key="ov_category")

    years = ["All"] + sorted(df["Year"].unique().tolist())
    selected_year = st.sidebar.selectbox("Year Range", years, key="ov_year")

    # Filters calculation
    fdf = df.copy()
    if selected_region != "All":
        fdf = fdf[fdf["Region"] == selected_region]
    if selected_category != "All":
        fdf = fdf[fdf["Category"] == selected_category]
    if selected_year != "All":
        fdf = fdf[fdf["Year"] == int(selected_year)]

    # Metrics prep
    total_sales = fdf["Sales"].sum()
    monthly = build_monthly_series(fdf)
    avg_monthly = monthly["Sales"].mean() if len(monthly) > 0 else 0
    best_month_row = monthly.loc[monthly["Sales"].idxmax()] if len(monthly) > 0 else None
    best_month_str = best_month_row["Date"].strftime("%b %Y") if best_month_row is not None else "N/A"

    yearly = fdf.groupby("Year")["Sales"].sum().sort_index()
    if len(yearly) >= 2:
        last_year, prev_year = yearly.iloc[-1], yearly.iloc[-2]
        yoy = ((last_year - prev_year) / prev_year) * 100
        yoy_str = f"{yoy:+.1f}%"
    else:
        yoy_str = "N/A"

    # KPI Layout in 1 Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Sales", f"${total_sales:,.0f}", f"{len(fdf):,} transactions", "Sum of all order transactions in the filtered range")
    with c2:
        kpi_card("Avg Monthly Sales", f"${avg_monthly:,.0f}", f"{len(monthly)} calendar months", "Average monthly sales aggregated over the entire period")
    with c3:
        kpi_card("Best Month", best_month_str, f"${best_month_row['Sales']:,.0f}" if best_month_row is not None else "N/A", "The calendar month with the highest recorded sales")
    with c4:
        kpi_card("YoY Growth", yoy_str, "vs previous year", "Year-over-Year revenue comparison of the last two years")

    # Row 1 Charts: Revenue Trends
    section_header("Revenue Trends", "Visualizing structural sales patterns across yearly and monthly horizons")
    col_a, col_b = st.columns(2)

    with col_a:
        yearly_df = fdf.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(
            yearly_df, x="Year", y="Sales",
            text_auto=",.0f",
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
            marker_cornerradius=8
        )
        fig.update_layout(title="Yearly Distribution Summary", **PLOTLY_LAYOUT)
        fig.update_layout(yaxis_title="Sales ($)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if len(monthly) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["Date"], y=monthly["Sales"],
                mode="lines+markers",
                line=dict(color=COLORS["secondary"], width=3, shape="spline"),
                marker=dict(size=6, color=COLORS["secondary"], line=dict(width=1, color="white")),
                fill="tozeroy",
                fillcolor="rgba(124, 77, 255, 0.06)",
                name="Monthly Sales",
            ))
            fig.update_layout(title="Timeline Historical Growth Path", **PLOTLY_LAYOUT)
            fig.update_layout(yaxis_title="Sales ($)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    # Row 2 Charts: Category Insights
    section_header("Category Insights", "Distribution of revenue metrics by regional territories and product categories")
    col_c, col_d = st.columns(2)

    with col_c:
        region_df = fdf.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=True)
        fig = px.bar(
            region_df, x="Sales", y="Region", orientation="h",
            text_auto=",.0f",
            color="Region",
            color_discrete_sequence=PLOTLY_PALETTE,
        )
        fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0, marker_cornerradius=8)
        fig.update_layout(title="Region Revenue Volume", showlegend=False, **PLOTLY_LAYOUT)
        fig.update_layout(xaxis_title="Sales ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        cat_df = fdf.groupby("Category")["Sales"].sum().reset_index()
        fig = px.pie(
            cat_df, names="Category", values="Sales",
            color_discrete_sequence=PLOTLY_PALETTE,
            hole=0.6,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(title="Sector Share Ratio", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECAST EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

def page_forecast_explorer(df):
    page_header("Forecast Explorer", "Predictive revenue analysis using an optimized XGBoost feature regression model", "📈")

    # Main page styled control card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        segment_type = st.selectbox("Segment Filter Type", ["Overall", "Category", "Region"], key="fc_seg")
    with col_c2:
        segment_value = None
        if segment_type == "Category":
            segment_value = st.selectbox("Select Category", sorted(df["Category"].unique()), key="fc_cat")
        elif segment_type == "Region":
            segment_value = st.selectbox("Select Region", sorted(df["Region"].unique()), key="fc_reg")
        else:
            st.selectbox("Select Segment", ["Overall Sales Data"], disabled=True, key="fc_seg_disabled")
    with col_c3:
        horizon = st.slider("Forecast Horizon (months)", 1, 3, 3, key="fc_hor")
    st.markdown('</div>', unsafe_allow_html=True)

    # Filter application
    fdf = df.copy()
    if segment_type == "Category" and segment_value:
        fdf = fdf[fdf["Category"] == segment_value]
    elif segment_type == "Region" and segment_value:
        fdf = fdf[fdf["Region"] == segment_value]

    monthly = build_monthly_series(fdf)

    if len(monthly) < 10:
        st.warning("Insufficient historical series points to generate reliable predictions. Please select a wider scope.")
        return

    with st.spinner("Loading pre-trained XGBoost model & generating forecast..."):
        artifacts = load_artifacts()
        model = artifacts["model"]
        if model is None:
            st.error("❌ model.pkl not found. Please run the notebook export cell first.")
            return
        train_df, test_df, preds, mae, rmse, feat_cols = evaluate_model(monthly, model)
        future_df = forecast_future(model, monthly, feat_cols, horizon)

    # Calculate shaded confidence interval bounds
    future_df["Lower"] = (future_df["Forecast"] - 1.96 * rmse).clip(lower=0)
    future_df["Upper"] = future_df["Forecast"] + 1.96 * rmse

    # KPI rows
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Mean Absolute Error (MAE)", f"${mae:,.0f}", "average error margin", "Average absolute difference between actual values and model projections")
    with c2:
        kpi_card("Root Mean Sq. Error (RMSE)", f"${rmse:,.0f}", "residual deviation", "Standard deviation of model prediction residuals")
    with c3:
        kpi_card("Total Cumulative Forecast", f"${future_df['Forecast'].sum():,.0f}", f"over next {horizon} mo", "Sum of predicted sales across the forecast horizon")

    # Forecast timeline
    section_header("Actual vs Forecast Projection", "Historical actual sales compared with model predictions and confidence bands")
    
    fig = go.Figure()
    
    # 1. Shaded Confidence Interval
    fig.add_trace(go.Scatter(
        x=pd.concat([future_df["Date"], future_df["Date"].iloc[::-1]]),
        y=pd.concat([future_df["Upper"], future_df["Lower"].iloc[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 229, 255, 0.07)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval (95%)"
    ))

    # 2. Historical actual path (Solid Line)
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Sales"],
        mode="lines+markers", name="Actual Sales",
        line=dict(color=COLORS["primary"], width=3, shape="spline"),
        marker=dict(size=5, color=COLORS["primary"], line=dict(width=1, color="white")),
    ))

    # 3. Future predicted path (Dashed Line)
    fig.add_trace(go.Scatter(
        x=future_df["Date"], y=future_df["Forecast"],
        mode="lines+markers", name="XGBoost Forecast",
        line=dict(color=COLORS["accent"], width=3, dash="dash", shape="spline"),
        marker=dict(size=7, symbol="diamond", color=COLORS["accent"], line=dict(width=1, color="white")),
    ))

    fig.update_layout(
        yaxis_title="Monthly Sales ($)",
        xaxis_title="",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dynamic Insight Box
    last_sales = monthly["Sales"].iloc[-horizon:].sum()
    forecast_sales = future_df["Forecast"].sum()
    pct_change = ((forecast_sales - last_sales) / last_sales) * 100
    direction = "increase" if pct_change >= 0 else "decrease"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.08) 0%, rgba(79, 139, 249, 0.08) 100%);
        border-left: 4px solid #00E5FF;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px;">💡</span>
            <span style="color: #ffffff; font-size: 14px; font-weight: 500;">
                Sales are expected to <b>{direction}</b> by <b>{abs(pct_change):.1f}%</b> over the next <b>{horizon} months</b> compared to the previous equivalent period.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Forecast breakdown table
    section_header("Forecast Details", "Detailed tabular breakdown of forecasted values")
    display_df = future_df[["Date", "Forecast", "Lower", "Upper"]].copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    display_df["Forecast"] = display_df["Forecast"].apply(lambda x: f"${x:,.0f}")
    display_df["Lower Limit"] = display_df["Lower"].apply(lambda x: f"${x:,.0f}")
    display_df["Upper Limit"] = display_df["Upper"].apply(lambda x: f"${x:,.0f}")
    display_df.drop(columns=["Lower", "Upper"], inplace=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY REPORT
# ═══════════════════════════════════════════════════════════════════════════

def page_anomaly_report(df):
    page_header("Anomaly Report", "Unusual sales pattern detection using Isolation Forest and Z-score modeling", "🚨")

    with st.spinner("Analyzing data for sales anomalies..."):
        weekly = build_weekly_series(df)
        anom_df = detect_anomalies(weekly)

    total_anomalies = anom_df["Anomaly"].sum()
    high_count = (anom_df["Type"] == "High ⬆").sum()
    low_count = (anom_df["Type"] == "Low ⬇").sum()

    # KPI Layout
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Weeks Analyzed", f"{len(anom_df)}", "entire period", "Total count of weekly sales aggregations")
    with c2:
        kpi_card("Anomalies Flagged", f"{total_anomalies}", f"{total_anomalies/len(anom_df)*100:.1f}% rate", "Number of outlier sales periods detected")
    with c3:
        kpi_card("Unusual Spikes ⬆", f"{high_count}", "higher than threshold", "Periods with sales significantly above the rolling average")
    with c4:
        kpi_card("Unusual Dips ⬇", f"{low_count}", "lower than threshold", "Periods with sales significantly below the rolling average")

    # Chart segment
    section_header("Outlier Sales Periods", "Weekly sales history featuring flagged high and low anomaly points")

    fig = go.Figure()
    
    # 1. Normal Weekly Line
    fig.add_trace(go.Scatter(
        x=anom_df["Date"], y=anom_df["Sales"],
        mode="lines+markers", name="Normal Sales",
        line=dict(color="rgba(255, 255, 255, 0.4)", width=2, shape="spline"),
        marker=dict(size=4, color="rgba(255, 255, 255, 0.5)"),
        fill="tozeroy",
        fillcolor="rgba(79, 139, 249, 0.03)",
    ))

    # 2. Anomalies (Red, larger size)
    anom_points = anom_df[anom_df["Anomaly"] == 1]
    if len(anom_points) > 0:
        fig.add_trace(go.Scatter(
            x=anom_points["Date"], y=anom_points["Sales"],
            mode="markers", name="Anomaly Flag",
            marker=dict(color=COLORS["danger"], size=10, symbol="circle", line=dict(width=1.5, color="white")),
            hovertext=anom_points["Type"],
        ))

    fig.update_layout(
        yaxis_title="Weekly Sales ($)",
        xaxis_title="",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabular anomaly details
    section_header("Detailed Outlier Audit Log", "List of all anomalies detected with Z-score metric ratings")
    
    if len(anom_points) > 0:
        table_df = anom_points[["Date", "Sales", "Type", "Z_Score"]].copy()
        table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
        table_df["Sales"] = table_df["Sales"].apply(lambda x: f"${x:,.0f}")
        table_df["Z-Score"] = table_df["Z_Score"].apply(lambda x: f"{x:.2f}")
        table_df.drop(columns=["Z_Score"], inplace=True)
        
        # Display styled dataframe
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.info("No anomalies detected in the current series dataset.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PRODUCT SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════

def page_product_segments(df):
    page_header("Product Segments", "Sub-category cohort performance analysis using K-Means clustering and PCA", "🧠")

    with st.spinner("Loading pre-trained KMeans & PCA from notebook artifacts..."):
        artifacts = load_artifacts()
        scaler = artifacts["scaler"]
        kmeans = artifacts["kmeans"]
        pca = artifacts["pca"]
        if any(x is None for x in [scaler, kmeans, pca]):
            st.error("❌ One or more clustering artifacts (scaler.pkl, kmeans.pkl, pca.pkl) not found. Please run the notebook export cell first.")
            return
        cluster_df, explained = run_clustering(df, scaler, kmeans, pca)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Cohorts Analyzed", f"{len(cluster_df)}", "product sub-categories", "Total count of unique product sub-categories")
    with c2:
        kpi_card("Target Clusters", f"{kmeans.n_clusters} Groups", f"K-Means (k={kmeans.n_clusters})", "Number of segment divisions from the pre-trained clustering model")
    with c3:
        total_var = (explained[0] + explained[1]) * 100
        kpi_card("Explained Variance", f"{total_var:.1f}%", "via PC1 & PC2", "Proportion of source variance captured in 2D layout space")

    # Segment Cards (Action Items)
    section_header("Segment Action Items & Insights", "Targeted business directives grouped by cohort performance characteristics")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(79, 139, 249, 0.12) 0%, rgba(0, 229, 255, 0.08) 100%);
            border: 1px solid rgba(79, 139, 249, 0.25);
            border-radius: 12px;
            padding: 18px;
            height: 140px;
        ">
            <h4 style="margin:0 0 6px 0; color:#4F8BF9; font-size:15px; font-weight:700;">⭐ Stars</h4>
            <p style="margin:0; font-size:12px; color:#A0AEC0; line-height:1.4;">High volume and consistent demand profile.</p>
            <div style="margin-top:12px; font-weight:700; color:{COLORS["accent"]}; font-size:12px;">➔ Stock aggressively</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(124, 77, 255, 0.12) 0%, rgba(0, 229, 255, 0.08) 100%);
            border: 1px solid rgba(124, 77, 255, 0.25);
            border-radius: 12px;
            padding: 18px;
            height: 140px;
        ">
            <h4 style="margin:0 0 6px 0; color:#7C4DFF; font-size:15px; font-weight:700;">🚀 Growth</h4>
            <p style="margin:0; font-size:12px; color:#A0AEC0; line-height:1.4;">Significant upward trend velocity metrics.</p>
            <div style="margin-top:12px; font-weight:700; color:{COLORS["accent"]}; font-size:12px;">➔ Expand market reach</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(0, 200, 151, 0.12) 0%, rgba(0, 229, 255, 0.08) 100%);
            border: 1px solid rgba(0, 200, 151, 0.25);
            border-radius: 12px;
            padding: 18px;
            height: 140px;
        ">
            <h4 style="margin:0 0 6px 0; color:#00C897; font-size:15px; font-weight:700;">🛡️ Stable</h4>
            <p style="margin:0; font-size:12px; color:#A0AEC0; line-height:1.4;">Consistent baseline volumes, low volatility.</p>
            <div style="margin-top:12px; font-weight:700; color:{COLORS["accent"]}; font-size:12px;">➔ Optimize supply cost</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255, 101, 132, 0.12) 0%, rgba(0, 229, 255, 0.08) 100%);
            border: 1px solid rgba(255, 101, 132, 0.25);
            border-radius: 12px;
            padding: 18px;
            height: 140px;
        ">
            <h4 style="margin:0 0 6px 0; color:#FF6584; font-size:15px; font-weight:700;">🎯 Niche</h4>
            <p style="margin:0; font-size:12px; color:#A0AEC0; line-height:1.4;">Low-to-medium demand with high volatility.</p>
            <div style="margin-top:12px; font-weight:700; color:{COLORS["danger"]}; font-size:12px;">➔ Monitor closely</div>
        </div>
        """, unsafe_allow_html=True)

    # 2D Cluster Space Plot
    section_header("Clustered Projection Space", "Dimensional mapping of sub-categories based on total sales, volatility, and growth rates")

    fig = px.scatter(
        cluster_df, x="PC1", y="PC2",
        color="Cluster_Label",
        text="Sub-Category",
        size="Total_Sales",
        size_max=40,
        color_discrete_map={
            "Stars ⭐": COLORS["primary"],
            "Growth 🚀": COLORS["secondary"],
            "Stable 🛡️": COLORS["success"],
            "Niche 🎯": COLORS["danger"]
        },
        hover_data={"Total_Sales": ":,.0f", "Growth_Rate": ":+.1f%", "Volatility": ":,.0f", "Avg_Order_Value": ":,.2f", "PC1": False, "PC2": False, "Cluster_Label": False},
        labels={
            "Total_Sales": "Total Sales",
            "Growth_Rate": "Growth Rate",
            "Avg_Order_Value": "Avg Order Value",
            "Cluster_Label": "Segment Cohort"
        }
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=10,
        marker=dict(line=dict(width=1.5, color="white")),
        hovertemplate="<b>%{text}</b><br><br>" +
                      "Total Sales: $%{customdata[0]:,.0f}<br>" +
                      "Growth Rate: %{customdata[1]:+.1f}%<br>" +
                      "Volatility: $%{customdata[2]:,.0f}<br>" +
                      "Avg Order Value: $%{customdata[3]:,.2f}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title=f"Principal Component 1 ({explained[0]*100:.1f}% var)",
        yaxis_title=f"Principal Component 2 ({explained[1]*100:.1f}% var)",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabular cluster values
    section_header("Cohort Performance Metrics", "Raw metric values for all categorized sub-product cohorts")
    table_df = cluster_df[["Sub-Category", "Cluster_Label", "Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"]].copy()
    table_df["Total_Sales"] = table_df["Total_Sales"].apply(lambda x: f"${x:,.0f}")
    table_df["Growth_Rate"] = table_df["Growth_Rate"].apply(lambda x: f"{x:+.1f}%")
    table_df["Volatility"] = table_df["Volatility"].apply(lambda x: f"${x:,.0f}")
    table_df["Avg_Order_Value"] = table_df["Avg_Order_Value"].apply(lambda x: f"${x:,.2f}")
    table_df = table_df.rename(columns={
        "Cluster_Label": "Assigned Segment",
        "Total_Sales": "Total Sales",
        "Growth_Rate": "Growth Rate",
        "Avg_Order_Value": "Avg Order Value",
    })
    st.dataframe(table_df.sort_values("Assigned Segment"), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION & MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Sidebar platform branding header
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 24px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 24px;">
            <h2 style="
                margin: 0;
                font-size: 24px;
                font-weight: 800;
                background: linear-gradient(90deg, #4F8BF9, #00E5FF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            ">DemandIQ</h2>
            <p style="margin: 4px 0 0 0; color: #A0AEC0; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;">Sales Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"],
        label_visibility="collapsed",
    )

    # Initialize data load
    df = load_data()

    if page == "Sales Overview":
        page_sales_overview(df)
    elif page == "Forecast Explorer":
        page_forecast_explorer(df)
    elif page == "Anomaly Report":
        page_anomaly_report(df)
    elif page == "Product Segments":
        page_product_segments(df)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="text-align:center;color:#4A5568;font-size:0.7rem;">'
        "DemandIQ v1.1 • Premium SaaS UI<br>"
        "© 2026 Sales Intelligence</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
