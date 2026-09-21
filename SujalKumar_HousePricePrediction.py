"""
House Price Prediction — End-to-End Data Analytics & ML Dashboard
=================================================================
Run with:  streamlit run app.py
Dataset  : archive/House Price Prediction Dataset.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH = os.path.join("archive", "House Price Prediction Dataset.csv")

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
C_BG        = "#0f1117"   # page background
C_SURFACE   = "#1a1d27"   # card surface
C_BORDER    = "#2a2d3e"   # subtle border
C_ACCENT    = "#6366f1"   # indigo accent
C_ACCENT2   = "#22d3ee"   # cyan accent
C_ACCENT3   = "#f59e0b"   # amber accent
C_TEXT      = "#f1f5f9"   # primary text
C_MUTED     = "#94a3b8"   # muted text
C_GREEN     = "#10b981"   # positive delta
C_RED       = "#f43f5e"   # negative delta

CHART_PALETTE = [C_ACCENT, C_ACCENT2, C_ACCENT3, "#a78bfa", "#34d399", "#fb923c"]
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=C_TEXT, size=13),
    title_font=dict(size=16, color=C_TEXT, family="Inter, Segoe UI, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
    margin=dict(l=16, r=16, t=48, b=16),
    coloraxis_colorbar=dict(tickfont=dict(color=C_MUTED)),
    xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_MUTED)),
    yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_MUTED)),
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ──────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }}
    .stApp {{
        background-color: {C_BG};
        color: {C_TEXT};
    }}

    /* ── Sidebar ───────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {C_SURFACE};
        border-right: 1px solid {C_BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {C_TEXT}; }}
    [data-testid="stSidebarNav"] {{ display: none; }}

    /* ── Top-bar / header strip ────────────── */
    header[data-testid="stHeader"] {{
        background: {C_BG};
        border-bottom: 1px solid {C_BORDER};
    }}

    /* ── Metric cards ───────────────────────── */
    .kpi-card {{
        background: linear-gradient(135deg, {C_SURFACE} 0%, #222540 100%);
        border: 1px solid {C_BORDER};
        border-radius: 16px;
        padding: 22px 18px 18px 18px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.18);
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {C_ACCENT}, {C_ACCENT2});
        border-radius: 16px 16px 0 0;
    }}
    .kpi-icon  {{ font-size: 28px; margin-bottom: 6px; }}
    .kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                  color: {C_MUTED}; font-weight: 500; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 28px; font-weight: 700; color: {C_TEXT};
                  letter-spacing: -0.5px; line-height: 1.1; }}
    .kpi-delta {{ font-size: 11px; color: {C_ACCENT2}; margin-top: 6px;
                  font-weight: 500; }}

    /* ── Section / hero headers ─────────────── */
    .hero-header {{
        background: linear-gradient(135deg, #1e2035 0%, #252845 100%);
        border: 1px solid {C_BORDER};
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}
    .hero-header::after {{
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        pointer-events: none;
    }}
    .hero-title {{
        font-size: 32px; font-weight: 700;
        background: linear-gradient(90deg, {C_TEXT}, {C_ACCENT2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }}
    .hero-sub {{
        font-size: 15px; color: {C_MUTED};
        margin: 0; font-weight: 400; line-height: 1.6;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.35);
        color: {C_ACCENT2};
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }}

    /* ── Section label ──────────────────────── */
    .section-label {{
        font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
        color: {C_ACCENT}; font-weight: 600; margin-bottom: 8px;
    }}
    .section-heading {{
        font-size: 20px; font-weight: 600; color: {C_TEXT};
        margin: 0 0 20px 0;
    }}

    /* ── Pipeline steps ─────────────────────── */
    .pipeline-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 8px 0 24px 0;
    }}
    .pipeline-step {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        position: relative;
    }}
    .pipeline-step-num {{
        position: absolute; top: -10px; left: 50%; transform: translateX(-50%);
        background: {C_ACCENT}; color: #fff;
        border-radius: 50%; width: 22px; height: 22px;
        font-size: 11px; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
    }}
    .pipeline-step-icon {{ font-size: 22px; margin-bottom: 8px; }}
    .pipeline-step-title {{ font-size: 12px; font-weight: 600; color: {C_TEXT}; margin-bottom: 4px; }}
    .pipeline-step-desc  {{ font-size: 11px; color: {C_MUTED}; line-height: 1.4; }}

    /* ── Info / result cards ─────────────────── */
    .result-card {{
        background: linear-gradient(135deg, {C_SURFACE} 0%, #1e2540 100%);
        border: 1px solid {C_BORDER};
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        height: 100%;
    }}
    .result-icon {{ font-size: 32px; margin-bottom: 10px; }}
    .result-label {{ font-size: 12px; color: {C_MUTED}; text-transform: uppercase;
                     letter-spacing: 0.8px; font-weight: 500; margin-bottom: 8px; }}
    .result-value {{ font-size: 34px; font-weight: 700; color: {C_ACCENT2};
                     letter-spacing: -1px; }}
    .result-sub   {{ font-size: 12px; color: {C_MUTED}; margin-top: 4px; }}

    /* ── Streamlit native overrides ─────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {C_SURFACE};
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {C_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: {C_MUTED};
        font-weight: 500;
        font-size: 13px;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C_ACCENT} !important;
        color: #fff !important;
    }}
    .stDataFrame, .stDataFrame table {{
        background: {C_SURFACE} !important;
        border-radius: 12px;
    }}
    [data-testid="stMetricValue"] {{
        color: {C_ACCENT2};
        font-size: 22px;
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{ color: {C_MUTED}; }}
    .stButton > button {{
        background: linear-gradient(135deg, {C_ACCENT}, #818cf8);
        color: #fff;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 15px;
        padding: 14px 28px;
        transition: all 0.2s ease;
        width: 100%;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99,102,241,0.35);
    }}
    [data-testid="stForm"] {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 16px;
        padding: 24px;
    }}
    .stSelectbox [data-baseweb="select"] > div,
    .stNumberInput input,
    .stSlider [data-testid="stThumb"] {{
        background: #1e2135 !important;
        border-color: {C_BORDER} !important;
        color: {C_TEXT} !important;
    }}
    hr {{ border-color: {C_BORDER}; }}
    .stRadio [data-testid="stMarkdownContainer"] p {{
        color: {C_MUTED};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# CHART HELPER
# ─────────────────────────────────────────────
def apply_theme(fig, height=420):
    """Apply the global dark theme to any Plotly figure."""
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=C_BORDER, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=C_BORDER, zeroline=False)
    return fig


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load CSV and apply data-cleaning steps."""
    df = pd.read_csv(path)

    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)
    df.drop_duplicates(inplace=True)

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    df["HouseAge"]     = 2024 - df["YearBuilt"]
    df["PricePerSqFt"] = (df["Price"] / df["Area"]).round(2)
    df["RoomsTotal"]   = df["Bedrooms"] + df["Bathrooms"]

    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(df: pd.DataFrame):
    """Encode categoricals, split data, train LR + RF, return artefacts."""
    feature_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "HouseAge",
                    "Location", "Condition", "Garage", "RoomsTotal"]
    target_col = "Price"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    le_location  = LabelEncoder()
    le_condition = LabelEncoder()
    le_garage    = LabelEncoder()

    X["Location"]  = le_location.fit_transform(X["Location"])
    X["Condition"] = le_condition.fit_transform(X["Condition"])
    X["Garage"]    = le_garage.fit_transform(X["Garage"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    metrics = {
        "Linear Regression": {
            "R²":   round(r2_score(y_test, y_pred_lr), 4),
            "MAE":  round(mean_absolute_error(y_test, y_pred_lr), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 2),
        },
        "Random Forest": {
            "R²":   round(r2_score(y_test, y_pred_rf), 4),
            "MAE":  round(mean_absolute_error(y_test, y_pred_rf), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 2),
        },
    }

    fi_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    encoders = {
        "Location":  le_location,
        "Condition": le_condition,
        "Garage":    le_garage,
    }

    return (
        rf, lr,
        X_train, X_test, y_train, y_test,
        y_pred_rf, y_pred_lr,
        metrics, fi_df,
        feature_cols, encoders,
    )


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    st.error(
        f"❌ Dataset not found at `{DATASET_PATH}`.  "
        "Please place the CSV in the `archive/` folder."
    )
    st.stop()

df = load_and_clean_data(DATASET_PATH)
(
    rf_model, lr_model,
    X_train, X_test, y_train, y_test,
    y_pred_rf, y_pred_lr,
    metrics, fi_df,
    feature_cols, encoders,
) = train_models(df)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 20px 0 8px 0;">
            <div style="font-size:48px;">🏠</div>
            <div style="font-size:18px; font-weight:700; color:{C_TEXT}; margin-top:6px;">
                House Price ML
            </div>
            <div style="font-size:11px; color:{C_MUTED}; margin-top:2px;">
                Prediction Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{C_BORDER}; margin: 12px 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📋 Data Overview", "📊 EDA", "🌐 3D Market Analysis", "🤖 Model Performance", "🔮 Predict Price"],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{C_BORDER}; margin: 16px 0 10px 0;'>", unsafe_allow_html=True)

    # Sidebar mini-stats
    st.markdown(
        f"""
        <div style="background:{C_BG}; border:1px solid {C_BORDER}; border-radius:12px; padding:14px; margin-bottom:12px;">
            <div style="font-size:10px; text-transform:uppercase; letter-spacing:1px; color:{C_MUTED}; margin-bottom:10px;">
                Quick Stats
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="font-size:12px; color:{C_MUTED};">Records</span>
                <span style="font-size:12px; font-weight:600; color:{C_TEXT};">{len(df):,}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="font-size:12px; color:{C_MUTED};">Avg Price</span>
                <span style="font-size:12px; font-weight:600; color:{C_ACCENT2};">${df['Price'].mean():,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:{C_MUTED};">RF R²</span>
                <span style="font-size:12px; font-weight:600; color:{C_GREEN};">{metrics['Random Forest']['R²']:.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style="font-size:10px; color:{C_MUTED}; text-align:center; line-height:1.6;">
            📦 <a href="https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset"
            style="color:{C_ACCENT2}; text-decoration:none;">Kaggle Dataset</a>
        </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════
if page == "🏠 Home":

    # Hero header
    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-badge">🏠 End-to-End ML Project</div>
            <div class="hero-title">House Price Prediction Dashboard</div>
            <div class="hero-sub">
                A complete data science pipeline — ingestion, cleaning, EDA, machine learning
                and live prediction — powered by Random Forest &amp; Linear Regression.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards
    kpis = [
        ("🗂️", "Total Records",   f"{len(df):,}",                         "after cleaning"),
        ("💰", "Average Price",   f"${df['Price'].mean():,.0f}",           "across all houses"),
        ("📐", "Average Area",    f"{df['Area'].mean():,.0f} sq ft",       "living space"),
        ("🏚️", "Avg House Age",   f"{df['HouseAge'].mean():.0f} yrs",      "years old"),
        ("🎯", "RF Model R²",     f"{metrics['Random Forest']['R²']:.4f}", "best model score"),
    ]

    cols = st.columns(5)
    for col, (icon, label, value, delta) in zip(cols, kpis):
        col.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline steps
    st.markdown(
        f"""
        <div class="section-label">Project Pipeline</div>
        <div class="pipeline-grid">
            <div class="pipeline-step">
                <div class="pipeline-step-num">1</div>
                <div class="pipeline-step-icon">📥</div>
                <div class="pipeline-step-title">Data Ingestion</div>
                <div class="pipeline-step-desc">Load CSV from archive/ folder</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">2</div>
                <div class="pipeline-step-icon">🧹</div>
                <div class="pipeline-step-title">Data Cleaning</div>
                <div class="pipeline-step-desc">Dedup, impute nulls, engineer features</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">3</div>
                <div class="pipeline-step-icon">📊</div>
                <div class="pipeline-step-title">EDA</div>
                <div class="pipeline-step-desc">Correlations, distributions, price drivers</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">4</div>
                <div class="pipeline-step-icon">🤖</div>
                <div class="pipeline-step-title">Modelling</div>
                <div class="pipeline-step-desc">LR + RF — R², MAE, RMSE</div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">5</div>
                <div class="pipeline-step-icon">🔮</div>
                <div class="pipeline-step-title">Prediction</div>
                <div class="pipeline-step-desc">Interactive live price estimator</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Price distribution
    st.markdown(f"<div class='section-label'>Price Distribution</div>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        df, x="Price", nbins=50,
        labels={"Price": "Sale Price (USD)"},
        color_discrete_sequence=[C_ACCENT],
    )
    fig_hist.update_traces(marker_line_width=0, opacity=0.85)
    fig_hist = apply_theme(fig_hist, height=380)
    fig_hist.update_layout(title="Overall House Price Distribution", bargap=0.06)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Bottom row: location pie + condition bar
    c1, c2 = st.columns(2)
    with c1:
        loc_counts = df["Location"].value_counts().reset_index()
        loc_counts.columns = ["Location", "Count"]
        fig_pie = px.pie(
            loc_counts, names="Location", values="Count",
            title="Properties by Location",
            color_discrete_sequence=CHART_PALETTE,
            hole=0.45,
        )
        fig_pie = apply_theme(fig_pie, height=340)
        fig_pie.update_traces(textfont_color=C_TEXT)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        cond_avg = df.groupby("Condition")["Price"].mean().reset_index().sort_values("Price", ascending=False)
        fig_cbar = px.bar(
            cond_avg, x="Condition", y="Price",
            title="Avg Price by Condition",
            color="Price",
            color_continuous_scale=[[0, C_SURFACE], [1, C_ACCENT]],
        )
        fig_cbar.update_traces(marker_line_width=0)
        fig_cbar = apply_theme(fig_cbar, height=340)
        st.plotly_chart(fig_cbar, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE: DATA OVERVIEW
# ═══════════════════════════════════════════════
elif page == "📋 Data Overview":
    st.markdown(
        f"""<div class="hero-header">
            <div class="hero-badge">📋 Dataset</div>
            <div class="hero-title">Data Overview</div>
            <div class="hero-sub">
                Explore raw records, descriptive statistics, and data-quality metrics for
                the <strong>{len(df):,}</strong>-record house price dataset.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["  📄 Sample Data  ", "  📈 Statistics  ", "  🔍 Data Quality  "])

    with tab1:
        st.markdown(f"<div class='section-label'>First 100 Rows</div>", unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True, height=440)

    with tab2:
        st.markdown(f"<div class='section-label'>Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(
            df.describe().T.style.background_gradient(cmap="Blues"),
            use_container_width=True,
        )

    with tab3:
        st.markdown(f"<div class='section-label'>Data Quality Report</div>", unsafe_allow_html=True)
        quality = pd.DataFrame({
            "Column":        df.columns.tolist(),
            "Dtype":         [str(df[c].dtype) for c in df.columns],
            "Non-Null":      df.notnull().sum().values,
            "Null %":        (df.isnull().mean() * 100).round(2).values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(quality, use_container_width=True)

        st.markdown(f"<br><div class='section-label'>Categorical Distributions</div>", unsafe_allow_html=True)
        cat_cols = st.columns(3)
        for i, col_name in enumerate(["Location", "Condition", "Garage"]):
            vc = df[col_name].value_counts().reset_index()
            vc.columns = [col_name, "Count"]
            fig = px.bar(
                vc, x=col_name, y="Count",
                title=f"Distribution of {col_name}",
                color="Count",
                color_continuous_scale=[[0, C_SURFACE], [1, C_ACCENT]],
            )
            fig.update_traces(marker_line_width=0)
            fig = apply_theme(fig, height=300)
            cat_cols[i].plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown(
        f"""<div class="hero-header">
            <div class="hero-badge">📊 Analysis</div>
            <div class="hero-title">Exploratory Data Analysis</div>
            <div class="hero-sub">
                Uncovering patterns, correlations, and price drivers through
                interactive visualisations.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Correlation heatmap
    st.markdown(f"<div class='section-label'>Correlation Matrix</div>", unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[[0, "#1e1b4b"], [0.5, C_SURFACE], [1, C_ACCENT]],
            zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=11, color=C_TEXT),
            hoverongaps=False,
        )
    )
    fig_heat = apply_theme(fig_heat, height=500)
    fig_heat.update_layout(title="Feature Correlation Matrix")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(f"<hr style='border-color:{C_BORDER}; margin:8px 0 20px 0;'>", unsafe_allow_html=True)

    # Row 1
    st.markdown(f"<div class='section-label'>Price Relationships</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.scatter(
            df, x="Area", y="Price",
            color="Condition",
            title="Price vs Living Area",
            labels={"Area": "Area (sq ft)", "Price": "Price (USD)"},
            opacity=0.7,
            color_discrete_sequence=CHART_PALETTE,
        )
        fig = apply_theme(fig, height=380)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.box(
            df, x="Location", y="Price",
            color="Location",
            title="Price Distribution by Location",
            color_discrete_sequence=CHART_PALETTE,
        )
        fig = apply_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        fig = px.violin(
            df, x="Condition", y="Price",
            color="Condition", box=True,
            title="Price Distribution by House Condition",
            color_discrete_sequence=CHART_PALETTE,
        )
        fig = apply_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        fig = px.scatter(
            df, x="HouseAge", y="Price",
            color="Location",
            trendline="ols",
            title="House Age vs Price",
            opacity=0.65,
            color_discrete_sequence=CHART_PALETTE,
        )
        fig = apply_theme(fig, height=380)
        fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<hr style='border-color:{C_BORDER}; margin:8px 0 20px 0;'>", unsafe_allow_html=True)

    # Row 3
    st.markdown(f"<div class='section-label'>Feature Averages</div>", unsafe_allow_html=True)
    col_l3, col_r3 = st.columns(2)

    with col_l3:
        avg_bed = df.groupby("Bedrooms")["Price"].mean().reset_index()
        fig = px.bar(
            avg_bed, x="Bedrooms", y="Price",
            title="Avg Price by Bedrooms",
            color="Price",
            color_continuous_scale=[[0, C_SURFACE], [1, C_ACCENT2]],
        )
        fig.update_traces(marker_line_width=0)
        fig = apply_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_r3:
        avg_garage = df.groupby("Garage")["Price"].mean().reset_index()
        fig = px.bar(
            avg_garage, x="Garage", y="Price",
            title="Avg Price: Garage vs No Garage",
            color="Garage",
            color_discrete_map={"Yes": C_ACCENT, "No": C_ACCENT3},
        )
        fig.update_traces(marker_line_width=0)
        fig = apply_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Price per sq ft by location
    st.markdown(f"<div class='section-label'>Price Efficiency</div>", unsafe_allow_html=True)
    avg_ppsf = (
        df.groupby("Location")["PricePerSqFt"]
        .mean().reset_index()
        .sort_values("PricePerSqFt", ascending=False)
    )
    fig = px.bar(
        avg_ppsf, x="Location", y="PricePerSqFt",
        title="Average Price per Sq Ft by Location",
        color="PricePerSqFt",
        color_continuous_scale=[[0, C_SURFACE], [1, C_ACCENT3]],
        text=avg_ppsf["PricePerSqFt"].round(0).astype(str).add(" $/sqft"),
    )
    fig.update_traces(marker_line_width=0, textposition="outside",
                      textfont=dict(color=C_TEXT, size=12))
    fig = apply_theme(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE: 3D MARKET ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🌐 3D Market Analysis":
    st.markdown(
        f"""<div class="hero-header">
            <div class="hero-badge">🌐 3D Visualisation</div>
            <div class="hero-title">3D Market Analysis</div>
            <div class="hero-sub">
                Explore the full housing market in three dimensions — rotate, zoom,
                and hover over individual properties to reveal their details.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────
    st.markdown(f"<div class='section-label'>View Controls</div>", unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

    with ctrl1:
        color_by = st.selectbox(
            "Colour points by",
            ["Bedrooms", "Bathrooms", "Floors", "Condition", "Location", "Garage"],
            index=0,
        )
    with ctrl2:
        marker_size = st.slider("Marker size", 2, 10, 5)
    with ctrl3:
        opacity_val = st.slider("Opacity", 0.2, 1.0, 0.75, step=0.05)
    with ctrl4:
        filter_condition = st.multiselect(
            "Filter by Condition",
            options=sorted(df["Condition"].unique()),
            default=sorted(df["Condition"].unique()),
        )

    # Apply condition filter
    df_3d = df[df["Condition"].isin(filter_condition)] if filter_condition else df.copy()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build custom hover template ───────────────
    hover_data_cols = {
        "Bedrooms":  True,
        "Bathrooms": True,
        "Floors":    True,
        "Condition": True,
        "Location":  True,
        "Garage":    True,
        "HouseAge":  True,
        "PricePerSqFt": True,
    }

    # ── Main 3D Scatter — Area × YearBuilt × Price ─
    st.markdown(f"<div class='section-label'>Area × Year Built × Price</div>", unsafe_allow_html=True)

    is_categorical = color_by in ["Condition", "Location", "Garage"]

    if is_categorical:
        fig_3d = px.scatter_3d(
            df_3d,
            x="Area",
            y="YearBuilt",
            z="Price",
            color=color_by,
            hover_name=None,
            hover_data=hover_data_cols,
            labels={
                "Area":      "Area (sq ft)",
                "YearBuilt": "Year Built",
                "Price":     "Price (USD)",
            },
            title=f"3D Property Market — coloured by {color_by}",
            color_discrete_sequence=CHART_PALETTE,
            opacity=opacity_val,
        )
    else:
        fig_3d = px.scatter_3d(
            df_3d,
            x="Area",
            y="YearBuilt",
            z="Price",
            color=color_by,
            hover_data=hover_data_cols,
            labels={
                "Area":      "Area (sq ft)",
                "YearBuilt": "Year Built",
                "Price":     "Price (USD)",
            },
            title=f"3D Property Market — coloured by {color_by}",
            color_continuous_scale=[
                [0.0,  "#1e1b4b"],
                [0.25, C_ACCENT],
                [0.5,  C_ACCENT2],
                [0.75, C_ACCENT3],
                [1.0,  "#fef3c7"],
            ],
            opacity=opacity_val,
        )

    # Tooltip template
    fig_3d.update_traces(
        marker=dict(
            size=marker_size,
            line=dict(width=0),
        ),
        hovertemplate=(
            "<b>🏠 Property Details</b><br>"
            "━━━━━━━━━━━━━━━━<br>"
            "<b>Area:</b> %{x:,.0f} sq ft<br>"
            "<b>Year Built:</b> %{y}<br>"
            "<b>Price:</b> $%{z:,.0f}<br>"
            "<extra></extra>"
        ),
    )

    # Scene / axes styling
    # axis_style holds everything EXCEPT the per-axis title text (avoids duplicate-key error)
    _axis_title_font = dict(font=dict(color=C_TEXT, size=12))
    axis_style = dict(
        backgroundcolor=C_BG,
        gridcolor=C_BORDER,
        showbackground=True,
        zerolinecolor=C_BORDER,
        tickfont=dict(color=C_MUTED, size=10),
    )

    def _ax(text):
        """Build a single axis dict with a styled title and shared style."""
        return dict(title=dict(text=text, **_axis_title_font), **axis_style)

    fig_3d.update_layout(
        height=680,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=C_TEXT, size=12),
        title_font=dict(size=16, color=C_TEXT),
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            xaxis=_ax("Area (sq ft)"),
            yaxis=_ax("Year Built"),
            zaxis=_ax("Price (USD)"),
            bgcolor=C_BG,
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.9),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="auto",
        ),
        legend=dict(
            bgcolor="rgba(26,29,39,0.85)",
            bordercolor=C_BORDER,
            font=dict(color=C_TEXT, size=11),
        ),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown(f"<hr style='border-color:{C_BORDER}; margin:8px 0 20px 0;'>", unsafe_allow_html=True)

    # ── Second row: two complementary 3D plots ────
    st.markdown(f"<div class='section-label'>Alternative Perspectives</div>", unsafe_allow_html=True)
    col3d_l, col3d_r = st.columns(2)

    # Left — Area × Bedrooms × Price (coloured by Bathrooms)
    with col3d_l:
        fig_3d_b = px.scatter_3d(
            df_3d,
            x="Area",
            y="Bedrooms",
            z="Price",
            color="Bathrooms",
            hover_data=hover_data_cols,
            labels={
                "Area":     "Area (sq ft)",
                "Bedrooms": "Bedrooms",
                "Price":    "Price (USD)",
            },
            title="Area × Bedrooms × Price",
            color_continuous_scale=[
                [0.0, "#1e1b4b"],
                [0.5, C_ACCENT],
                [1.0, C_ACCENT2],
            ],
            opacity=opacity_val,
        )
        fig_3d_b.update_traces(
            marker=dict(size=max(marker_size - 1, 2), line=dict(width=0)),
            hovertemplate=(
                "<b>Area:</b> %{x:,.0f} sq ft<br>"
                "<b>Bedrooms:</b> %{y}<br>"
                "<b>Price:</b> $%{z:,.0f}<br>"
                "<extra></extra>"
            ),
        )
        fig_3d_b.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=C_TEXT, size=11),
            title_font=dict(size=14, color=C_TEXT),
            margin=dict(l=0, r=0, t=44, b=0),
            scene=dict(
                xaxis=_ax("Area (sq ft)"),
                yaxis=_ax("Bedrooms"),
                zaxis=_ax("Price (USD)"),
                bgcolor=C_BG,
                camera=dict(eye=dict(x=1.8, y=1.4, z=0.8)),
            ),
        )
        st.plotly_chart(fig_3d_b, use_container_width=True)

    # Right — HouseAge × RoomsTotal × Price (coloured by PricePerSqFt)
    with col3d_r:
        fig_3d_c = px.scatter_3d(
            df_3d,
            x="HouseAge",
            y="RoomsTotal",
            z="Price",
            color="PricePerSqFt",
            hover_data=hover_data_cols,
            labels={
                "HouseAge":    "House Age (yrs)",
                "RoomsTotal":  "Total Rooms",
                "Price":       "Price (USD)",
                "PricePerSqFt":"$/sq ft",
            },
            title="House Age × Total Rooms × Price",
            color_continuous_scale=[
                [0.0, "#1e1b4b"],
                [0.5, C_ACCENT3],
                [1.0, "#fef3c7"],
            ],
            opacity=opacity_val,
        )
        fig_3d_c.update_traces(
            marker=dict(size=max(marker_size - 1, 2), line=dict(width=0)),
            hovertemplate=(
                "<b>House Age:</b> %{x} yrs<br>"
                "<b>Total Rooms:</b> %{y}<br>"
                "<b>Price:</b> $%{z:,.0f}<br>"
                "<extra></extra>"
            ),
        )
        fig_3d_c.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=C_TEXT, size=11),
            title_font=dict(size=14, color=C_TEXT),
            margin=dict(l=0, r=0, t=44, b=0),
            scene=dict(
                xaxis=_ax("House Age (yrs)"),
                yaxis=_ax("Total Rooms"),
                zaxis=_ax("Price (USD)"),
                bgcolor=C_BG,
                camera=dict(eye=dict(x=1.8, y=1.4, z=0.8)),
            ),
        )
        st.plotly_chart(fig_3d_c, use_container_width=True)

    # ── Insight cards ─────────────────────────────
    st.markdown(f"<hr style='border-color:{C_BORDER}; margin:8px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>How to Navigate the 3D Chart</div>", unsafe_allow_html=True)

    tip_cols = st.columns(4)
    tips = [
        ("🖱️", "Rotate",   "Click and drag to orbit the scene freely in any direction."),
        ("🔍", "Zoom",     "Scroll the mouse wheel or pinch to zoom in/out."),
        ("✋", "Pan",      "Right-click and drag (or two-finger drag) to pan the view."),
        ("💡", "Hover",    "Hover over any point to see full property details in the tooltip."),
    ]
    for col, (icon, title, desc) in zip(tip_cols, tips):
        col.markdown(
            f"""<div class="kpi-card" style="padding:16px 14px;">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{title}</div>
                <div style="font-size:11px; color:{C_MUTED}; line-height:1.5; margin-top:6px;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown(
        f"""<div class="hero-header">
            <div class="hero-badge">🤖 Machine Learning</div>
            <div class="hero-title">Model Performance</div>
            <div class="hero-sub">
                Benchmarking Linear Regression vs Random Forest on R², MAE and RMSE
                across an 80/20 train-test split.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Metrics comparison table + bar charts
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

    # Model metric KPI cards
    m_cols = st.columns(4)
    kpi_metrics = [
        ("🌲", "RF — R²",   f"{metrics['Random Forest']['R²']:.4f}",       C_GREEN),
        ("🌲", "RF — MAE",  f"${metrics['Random Forest']['MAE']:,.0f}",     C_ACCENT2),
        ("🌲", "RF — RMSE", f"${metrics['Random Forest']['RMSE']:,.0f}",    C_ACCENT3),
        ("📈", "LR — R²",   f"{metrics['Linear Regression']['R²']:.4f}",   C_MUTED),
    ]
    for col, (icon, label, val, color) in zip(m_cols, kpi_metrics):
        col.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}; font-size:24px;">{val}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.bar(
            metrics_df, x="Model", y="R²",
            title="R² Score — higher is better",
            color="Model",
            color_discrete_sequence=[C_ACCENT, C_ACCENT2],
            text=metrics_df["R²"].astype(str),
        )
        fig.update_traces(marker_line_width=0, textposition="outside",
                          textfont=dict(color=C_TEXT))
        fig = apply_theme(fig, height=340)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0, max(metrics_df["R²"]) * 1.4])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.bar(
            metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"]),
            x="variable", y="value", color="Model",
            barmode="group",
            title="MAE vs RMSE — lower is better",
            labels={"variable": "Metric", "value": "USD"},
            color_discrete_sequence=[C_ACCENT, C_ACCENT2],
        )
        fig.update_traces(marker_line_width=0)
        fig = apply_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<hr style='border-color:{C_BORDER}; margin:8px 0 20px 0;'>", unsafe_allow_html=True)

    # Actual vs Predicted scatter
    st.markdown(f"<div class='section-label'>Prediction Accuracy</div>", unsafe_allow_html=True)
    avp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_rf})
    fig = px.scatter(
        avp_df, x="Actual", y="Predicted",
        title="Actual vs Predicted — Random Forest",
        labels={"Actual": "Actual Price (USD)", "Predicted": "Predicted Price (USD)"},
        opacity=0.6,
        color_discrete_sequence=[C_ACCENT],
    )
    max_val = max(avp_df["Actual"].max(), avp_df["Predicted"].max())
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color=C_ACCENT3, dash="dash", width=2),
    )
    fig.add_annotation(
        x=max_val * 0.85, y=max_val * 0.92,
        text="Perfect prediction line",
        font=dict(color=C_ACCENT3, size=11),
        showarrow=False,
    )
    fig = apply_theme(fig, height=440)
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)

    col_res, col_fi = st.columns(2)

    with col_res:
        residuals = y_test.values - y_pred_rf
        fig_res = px.histogram(
            x=residuals, nbins=50,
            title="Residual Distribution (Actual − Predicted)",
            labels={"x": "Residual (USD)"},
            color_discrete_sequence=[C_ACCENT2],
        )
        fig_res.update_traces(marker_line_width=0, opacity=0.85)
        fig_res = apply_theme(fig_res, height=360)
        fig_res.update_layout(bargap=0.04)
        st.plotly_chart(fig_res, use_container_width=True)

    with col_fi:
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h",
            title="Feature Importance — Random Forest",
            color="Importance",
            color_continuous_scale=[[0, C_SURFACE], [1, C_ACCENT]],
        )
        fig_fi.update_traces(marker_line_width=0)
        fig_fi = apply_theme(fig_fi, height=360)
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE: PREDICT PRICE
# ═══════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.markdown(
        f"""<div class="hero-header">
            <div class="hero-badge">🔮 Live Prediction</div>
            <div class="hero-title">Predict House Price</div>
            <div class="hero-sub">
                Configure the property details below. The trained
                <strong>Random Forest</strong> model will estimate the market price
                instantly — compared against the Linear Regression baseline.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown(
            f"<div class='section-label' style='margin-bottom:16px;'>Property Configuration</div>",
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div style='font-size:12px;color:{C_MUTED};margin-bottom:4px;'>SIZE & ROOMS</div>", unsafe_allow_html=True)
            area      = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=2000, step=50)
            bedrooms  = st.slider("Bedrooms", 1, 10, 3)
            bathrooms = st.slider("Bathrooms", 1, 8, 2)

        with col2:
            st.markdown(f"<div style='font-size:12px;color:{C_MUTED};margin-bottom:4px;'>STRUCTURE & LOCATION</div>", unsafe_allow_html=True)
            floors     = st.slider("Floors", 1, 5, 1)
            year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000, step=1)
            location   = st.selectbox("Location", sorted(df["Location"].unique()))

        with col3:
            st.markdown(f"<div style='font-size:12px;color:{C_MUTED};margin-bottom:4px;'>QUALITY & EXTRAS</div>", unsafe_allow_html=True)
            condition   = st.selectbox("Condition", sorted(df["Condition"].unique()))
            garage      = st.selectbox("Garage", ["Yes", "No"])
            rooms_total = bedrooms + bathrooms

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Estimate Price Now", use_container_width=True)

    if submitted:
        house_age = 2024 - year_built

        input_data = pd.DataFrame([{
            "Area":       area,
            "Bedrooms":   bedrooms,
            "Bathrooms":  bathrooms,
            "Floors":     floors,
            "HouseAge":   house_age,
            "Location":   encoders["Location"].transform([location])[0],
            "Condition":  encoders["Condition"].transform([condition])[0],
            "Garage":     encoders["Garage"].transform([garage])[0],
            "RoomsTotal": rooms_total,
        }])

        rf_price = rf_model.predict(input_data)[0]
        lr_price = lr_model.predict(input_data)[0]
        avg_price = df["Price"].mean()
        ppsf = rf_price / area

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-label'>Prediction Results</div>",
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4 = st.columns(4)
        result_cards = [
            ("🌲", "Random Forest", f"${rf_price:,.0f}", "Primary model estimate"),
            ("📈", "Linear Regression", f"${lr_price:,.0f}", "Baseline model estimate"),
            ("📐", "Price per Sq Ft", f"${ppsf:,.2f}", "Normalised value"),
            ("📊", "vs Dataset Avg", f"{'+' if rf_price >= avg_price else ''}{((rf_price - avg_price)/avg_price*100):,.1f}%",
             f"Avg: ${avg_price:,.0f}"),
        ]
        for col, (icon, label, val, sub) in zip([r1, r2, r3, r4], result_cards):
            col.markdown(
                f"""<div class="result-card">
                    <div class="result-icon">{icon}</div>
                    <div class="result-label">{label}</div>
                    <div class="result-value">{val}</div>
                    <div class="result-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rf_price,
            delta={
                "reference": avg_price,
                "relative": False,
                "valueformat": ",.0f",
                "prefix": "$",
                "increasing": {"color": C_ACCENT2},
                "decreasing": {"color": C_ACCENT3},
            },
            number={"prefix": "$", "valueformat": ",.0f", "font": {"color": C_TEXT, "size": 36}},
            title={"text": "Predicted Price vs Dataset Average", "font": {"color": C_MUTED, "size": 14}},
            gauge={
                "axis": {
                    "range": [0, df["Price"].max() * 1.1],
                    "tickprefix": "$", "tickformat": ",.0f",
                    "tickfont": {"color": C_MUTED, "size": 10},
                },
                "bar":  {"color": C_ACCENT, "thickness": 0.25},
                "bgcolor": C_SURFACE,
                "borderwidth": 1,
                "bordercolor": C_BORDER,
                "steps": [
                    {"range": [0, df["Price"].quantile(0.33)],                             "color": "rgba(16,185,129,0.15)"},
                    {"range": [df["Price"].quantile(0.33), df["Price"].quantile(0.66)],     "color": "rgba(245,158,11,0.15)"},
                    {"range": [df["Price"].quantile(0.66), df["Price"].max() * 1.1],       "color": "rgba(244,63,94,0.15)"},
                ],
                "threshold": {
                    "line": {"color": C_ACCENT3, "width": 3},
                    "thickness": 0.75,
                    "value": avg_price,
                },
            },
        ))
        fig_gauge.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=C_TEXT),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Input summary
        st.markdown(
            f"<div class='section-label'>Input Summary</div>",
            unsafe_allow_html=True,
        )
        summary = pd.DataFrame([{
            "Area (sq ft)": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "Year Built": year_built,
            "House Age": house_age,
            "Location": location,
            "Condition": condition,
            "Garage": garage,
            "Total Rooms": rooms_total,
        }])
        st.dataframe(summary, use_container_width=True)
