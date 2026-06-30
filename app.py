import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk AI Workbench",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark-theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* global background */
  .stApp { background-color: #0d1117; color: #c9d1d9; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* tab bar */
  .stTabs [data-baseweb="tab-list"] {
      background-color: #161b22;
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: transparent;
      color: #8b949e;
      border-radius: 6px;
      font-weight: 600;
      font-size: 14px;
      padding: 8px 18px;
  }
  .stTabs [aria-selected="true"] {
      background-color: #1f6feb !important;
      color: #ffffff !important;
  }

  /* KPI cards */
  .kpi-card {
      background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 20px 24px;
      text-align: center;
  }
  .kpi-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .kpi-value { font-size: 28px; font-weight: 700; color: #f0f6fc; }
  .kpi-sub   { font-size: 12px; color: #3fb950; margin-top: 4px; }

  /* data table */
  .stDataFrame { border-radius: 8px; }
  div[data-testid="stDataFrame"] > div { border-radius: 8px; }

  /* section headers */
  .section-header {
      font-size: 18px; font-weight: 700; color: #f0f6fc;
      border-left: 4px solid #1f6feb;
      padding-left: 12px; margin: 24px 0 16px;
  }

  /* slider labels */
  .stSlider label { color: #c9d1d9 !important; }

  /* metric delta colours */
  [data-testid="stMetricDelta"] svg { display: none; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #161b22; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "#0d1117"
PLOTLY_PAPER = "#161b22"
PLOTLY_GRID = "#21262d"

# ── Colour helpers ────────────────────────────────────────────────────────────
GRADE_COLOURS = {
    "AAA": "#3fb950", "AA": "#3fb950", "A": "#3fb950",
    "BBB": "#3fb950", "BB": "#e3b341", "B": "#e3b341",
    "CCC": "#f85149",
}

def grade_colour(g):
    return GRADE_COLOURS.get(g, "#8b949e")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION  (cached so it's stable across reruns)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_portfolio(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 50
    industries = ["Technology", "Healthcare", "Real Estate", "Manufacturing", "Retail"]
    grades = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    # grade distributions per industry
    grade_weights = {
        "Technology":     [0.08, 0.12, 0.20, 0.25, 0.20, 0.10, 0.05],
        "Healthcare":     [0.10, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03],
        "Real Estate":    [0.05, 0.10, 0.20, 0.25, 0.20, 0.13, 0.07],
        "Manufacturing":  [0.06, 0.10, 0.18, 0.26, 0.22, 0.12, 0.06],
        "Retail":         [0.04, 0.08, 0.15, 0.22, 0.25, 0.16, 0.10],
    }
    pd_map = {"AAA": 0.001, "AA": 0.003, "A": 0.008, "BBB": 0.015,
              "BB": 0.040, "B": 0.090, "CCC": 0.200}
    lgd_map = {"AAA": 0.20, "AA": 0.25, "A": 0.30, "BBB": 0.35,
               "BB": 0.45, "B": 0.55, "CCC": 0.65}
    rate_map = {"AAA": 3.5, "AA": 4.0, "A": 4.8, "BBB": 5.5,
                "BB": 7.0, "B": 9.5, "CCC": 13.0}

    rows = []
    for i in range(n):
        ind = industries[i % len(industries)]
        grade = rng.choice(grades, p=grade_weights[ind])
        amount = round(rng.uniform(5, 500) * 1e6, -5)
        base_pd = pd_map[grade]
        pd_val = float(np.clip(rng.normal(base_pd, base_pd * 0.15), 0.0005, 0.40))
        lgd_val = float(np.clip(rng.normal(lgd_map[grade], 0.04), 0.05, 0.80))
        rate = float(np.clip(rng.normal(rate_map[grade], 0.30), 2.5, 18.0))
        el = pd_val * lgd_val * amount
        rows.append({
            "Loan ID": f"CML-{2024_0001 + i:04d}",
            "Industry": ind,
            "Credit Grade": grade,
            "Loan Amount ($)": amount,
            "Interest Rate (%)": round(rate, 2),
            "PD (%)": round(pd_val * 100, 3),
            "LGD (%)": round(lgd_val * 100, 1),
            "Expected Loss ($)": round(el, 0),
            # extra borrower features for ML tab
            "Debt-to-Income": round(float(rng.uniform(0.10, 0.80)), 3),
            "Credit Score": int(rng.integers(500, 820)),
            "Revenue Growth (%)": round(float(rng.normal(0.05, 0.12)), 3),
            "Leverage Ratio": round(float(rng.uniform(1.0, 6.0)), 2),
            "Interest Coverage": round(float(rng.uniform(0.8, 8.0)), 2),
        })

    df = pd.DataFrame(rows)

    # binary default label for ML (1 = default, 0 = performing)
    default_prob = df["PD (%)"] / 100
    df["Default"] = (np.random.default_rng(seed + 1).random(n) < default_prob).astype(int)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    features = ["Debt-to-Income", "Credit Score", "Revenue Growth (%)",
                 "Leverage Ratio", "Interest Coverage"]
    X = df[features].values
    y = df["Default"].values

    # augment with synthetic data so the model has enough samples
    rng = np.random.default_rng(99)
    n_aug = 2000
    X_aug = np.column_stack([
        rng.uniform(0.10, 0.80, n_aug),
        rng.integers(500, 820, n_aug),
        rng.normal(0.05, 0.12, n_aug),
        rng.uniform(1.0, 6.0, n_aug),
        rng.uniform(0.8, 8.0, n_aug),
    ])
    # label via a simple scoring rule
    score = (
        -0.5 * (X_aug[:, 0] - 0.5)
        - 0.004 * (X_aug[:, 1] - 660)
        + 0.8 * (0.05 - X_aug[:, 2])
        + 0.12 * (X_aug[:, 3] - 3.5)
        - 0.10 * (X_aug[:, 4] - 4.0)
    )
    prob_d = 1 / (1 + np.exp(-score * 3))
    y_aug = (rng.random(n_aug) < prob_d).astype(int)

    X_all = np.vstack([X, X_aug])
    y_all = np.concatenate([y, y_aug])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )
    model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                   min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_vals_test = explainer.shap_values(X_test)

    return model, explainer, X_train, X_test, y_train, y_test, features, shap_vals_test


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(90deg,#161b22,#1c2128);
            border:1px solid #30363d;border-radius:12px;
            padding:24px 32px;margin-bottom:24px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:36px;">🏦</div>
    <div>
      <div style="font-size:24px;font-weight:800;color:#f0f6fc;letter-spacing:-0.5px;">
        Credit Risk AI Workbench
      </div>
      <div style="font-size:13px;color:#8b949e;margin-top:2px;">
        Commercial &amp; Specialized Industries | Portfolio Analytics &amp; ML Default Intelligence
      </div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div style="font-size:11px;color:#8b949e;">RISK RATING</div>
      <div style="font-size:20px;font-weight:700;color:#3fb950;">INVESTMENT GRADE</div>
      <div style="font-size:11px;color:#8b949e;">As of 2024-Q4</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

df = generate_portfolio()
model, explainer, X_train, X_test, y_train, y_test, features, shap_vals_test = train_model(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Portfolio Overview",
    "🤖  ML Default Prediction",
    "⚡  Stress Testing",
    "🔍  Individual Loan Analysis",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    total_value   = df["Loan Amount ($)"].sum()
    total_el      = df["Expected Loss ($)"].sum()
    wtd_pd        = (df["PD (%)"] * df["Loan Amount ($)"]).sum() / total_value
    el_pct        = total_el / total_value * 100

    # Portfolio credit rating (weighted)
    grade_score = {"AAA":7,"AA":6,"A":5,"BBB":4,"BB":3,"B":2,"CCC":1}
    avg_score = (df["Credit Grade"].map(grade_score) * df["Loan Amount ($)"]).sum() / total_value
    if avg_score >= 6.5: port_rating = "AAA"
    elif avg_score >= 5.5: port_rating = "AA"
    elif avg_score >= 4.5: port_rating = "A"
    elif avg_score >= 3.5: port_rating = "BBB"
    elif avg_score >= 2.5: port_rating = "BB"
    elif avg_score >= 1.5: port_rating = "B"
    else: port_rating = "CCC"

    c1, c2, c3, c4 = st.columns(4)
    kpi_data = [
        (c1, "Total Portfolio Value", f"${total_value/1e9:.2f}B", "50 commercial loans"),
        (c2, "Weighted Avg PD",       f"{wtd_pd:.3f}%",           f"EL Rate: {el_pct:.2f}%"),
        (c3, "Total Expected Loss",   f"${total_el/1e6:.1f}M",    "Credit VaR (99%): est."),
        (c4, "Portfolio Rating",      port_rating,                 "Weighted composite"),
    ]
    for col, label, value, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Industry Exposure</div>', unsafe_allow_html=True)
        ind_exp = df.groupby("Industry")["Loan Amount ($)"].sum().reset_index()
        fig_pie = px.pie(
            ind_exp, values="Loan Amount ($)", names="Industry",
            color_discrete_sequence=["#1f6feb","#3fb950","#e3b341","#f85149","#a371f7"],
            hole=0.55,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                              textfont_size=12)
        fig_pie.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
            plot_bgcolor=PLOTLY_BG, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10), height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Credit Grade Distribution</div>', unsafe_allow_html=True)
        grade_dist = df.groupby("Credit Grade")["Loan Amount ($)"].sum().reset_index()
        ordered = ["AAA","AA","A","BBB","BB","B","CCC"]
        grade_dist["Credit Grade"] = pd.Categorical(grade_dist["Credit Grade"], categories=ordered, ordered=True)
        grade_dist = grade_dist.sort_values("Credit Grade")
        bar_colours = [grade_colour(g) for g in grade_dist["Credit Grade"]]

        fig_bar = go.Figure(go.Bar(
            x=grade_dist["Credit Grade"],
            y=grade_dist["Loan Amount ($)"] / 1e6,
            marker_color=bar_colours,
            text=(grade_dist["Loan Amount ($)"] / 1e6).map(lambda v: f"${v:.0f}M"),
            textposition="outside",
        ))
        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
            plot_bgcolor=PLOTLY_BG,
            yaxis_title="Exposure ($M)", xaxis_title="",
            margin=dict(t=10, b=10, l=10, r=10), height=300,
            yaxis_gridcolor=PLOTLY_GRID,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── PD vs LGD scatter ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Risk Map — PD vs LGD by Industry</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        df, x="PD (%)", y="LGD (%)",
        color="Industry", size="Loan Amount ($)",
        symbol="Credit Grade",
        hover_data=["Loan ID", "Credit Grade", "Loan Amount ($)", "Expected Loss ($)"],
        size_max=30,
        color_discrete_sequence=["#1f6feb","#3fb950","#e3b341","#f85149","#a371f7"],
    )
    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG, height=380,
        xaxis_gridcolor=PLOTLY_GRID, yaxis_gridcolor=PLOTLY_GRID,
        margin=dict(t=10, b=40, l=40, r=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Loan table ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Full Loan Portfolio</div>', unsafe_allow_html=True)

    display_cols = ["Loan ID","Industry","Credit Grade","Loan Amount ($)",
                    "Interest Rate (%)","PD (%)","LGD (%)","Expected Loss ($)"]
    display_df = df[display_cols].copy()
    display_df["Loan Amount ($)"] = display_df["Loan Amount ($)"].map(lambda v: f"${v:,.0f}")
    display_df["Expected Loss ($)"] = display_df["Expected Loss ($)"].map(lambda v: f"${v:,.0f}")

    def colour_grade(val):
        c = grade_colour(val)
        return f"background-color:{c}22; color:{c}; font-weight:700"

    def colour_pd(val):
        v = float(val)
        if v < 1: return "color:#3fb950"
        if v < 5: return "color:#e3b341"
        return "color:#f85149"

    styled = (display_df.style
              .map(colour_grade, subset=["Credit Grade"])
              .map(colour_pd, subset=["PD (%)"]))
    st.dataframe(styled, use_container_width=True, height=480)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ML DEFAULT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-header">Live Borrower Scoring</div>', unsafe_allow_html=True)
        dti   = st.slider("Debt-to-Income Ratio",     0.05, 0.95, 0.35, 0.01)
        score = st.slider("Credit Score",             450,  850,  680)
        rev_g = st.slider("Revenue Growth (%)",       -0.30, 0.40, 0.05, 0.01)
        lev   = st.slider("Leverage Ratio (Debt/EBITDA)", 0.5, 8.0, 3.0, 0.1)
        icr   = st.slider("Interest Coverage Ratio",  0.5, 10.0, 3.5, 0.1)

        X_live = np.array([[dti, score, rev_g, lev, icr]])
        pd_live = model.predict_proba(X_live)[0][1]

        if pd_live < 0.05:
            pd_colour, pd_label = "#3fb950", "LOW RISK"
        elif pd_live < 0.15:
            pd_colour, pd_label = "#e3b341", "MODERATE RISK"
        else:
            pd_colour, pd_label = "#f85149", "HIGH RISK"

        st.markdown(f"""
        <div style="margin-top:24px;background:#161b22;border:1px solid {pd_colour}44;
                    border-radius:12px;padding:24px;text-align:center;">
          <div style="font-size:12px;color:#8b949e;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;">
            Predicted Default Probability
          </div>
          <div style="font-size:48px;font-weight:800;color:{pd_colour};">
            {pd_live*100:.1f}%
          </div>
          <div style="font-size:14px;font-weight:700;color:{pd_colour};margin-top:4px;">
            {pd_label}
          </div>
        </div>""", unsafe_allow_html=True)

        # SHAP waterfall for live input
        st.markdown('<div class="section-header">SHAP Explanation</div>', unsafe_allow_html=True)
        shap_live = explainer.shap_values(X_live)
        # For binary classifier shap_values returns list [class0, class1]
        if isinstance(shap_live, list):
            sv = shap_live[1][0]
            base = explainer.expected_value[1]
        else:
            sv = shap_live[0]
            base = explainer.expected_value

        feat_labels = ["DTI","Credit Score","Rev Growth","Leverage","Int. Coverage"]
        colours = ["#f85149" if v > 0 else "#3fb950" for v in sv]
        fig_wf = go.Figure(go.Bar(
            x=feat_labels, y=sv,
            marker_color=colours,
            text=[f"{v:+.3f}" for v in sv],
            textposition="outside",
        ))
        fig_wf.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
            plot_bgcolor=PLOTLY_BG, height=260,
            yaxis_title="SHAP value (default risk ↑)",
            xaxis_title="",
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis_gridcolor=PLOTLY_GRID,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)

        # mean |SHAP| across test set
        if isinstance(shap_vals_test, list):
            sv_test = shap_vals_test[1]
        else:
            sv_test = shap_vals_test

        mean_shap = np.abs(sv_test).mean(axis=0)
        fi_df = pd.DataFrame({"Feature": features, "Mean |SHAP|": mean_shap})
        fi_df = fi_df.sort_values("Mean |SHAP|")

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Mean |SHAP|"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Mean |SHAP|"],
                colorscale=[[0,"#1f6feb"],[0.5,"#a371f7"],[1,"#f85149"]],
                showscale=False,
            ),
            text=fi_df["Mean |SHAP|"].map(lambda v: f"{v:.4f}"),
            textposition="outside",
        ))
        fig_fi.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
            plot_bgcolor=PLOTLY_BG, height=300,
            xaxis_title="Mean |SHAP Value| (avg impact on default probability)",
            margin=dict(t=10, b=10, l=10, r=80),
            xaxis_gridcolor=PLOTLY_GRID,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Confusion matrix + ROC side by side ──────────────────────────────
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)

        with cc1:
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Performing","Default"], y=["Performing","Default"],
                color_continuous_scale=[[0,"#161b22"],[0.5,"#1f6feb"],[1,"#a371f7"]],
            )
            fig_cm.update_layout(
                template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
                plot_bgcolor=PLOTLY_BG, height=300,
                title="Confusion Matrix",
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with cc2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#1f6feb", width=2),
                name=f"ROC (AUC = {roc_auc:.3f})",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#8b949e", dash="dash"), name="Random",
            ))
            fig_roc.update_layout(
                template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
                plot_bgcolor=PLOTLY_BG, height=300,
                title=f"ROC Curve (AUC = {roc_auc:.3f})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(t=40, b=40, l=40, r=10),
                xaxis_gridcolor=PLOTLY_GRID, yaxis_gridcolor=PLOTLY_GRID,
                legend=dict(x=0.5, y=0.05),
            )
            st.plotly_chart(fig_roc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Economic Scenario Parameters</div>', unsafe_allow_html=True)

    severity = st.slider(
        "Recession Severity Multiplier  (1.0 = Base, 3.0 = GFC-equivalent)",
        1.0, 3.5, 1.0, 0.1,
    )

    # grade-level PD multipliers per scenario
    base_mult   = {"AAA":1.0,"AA":1.0,"A":1.0,"BBB":1.0,"BB":1.0,"B":1.0,"CCC":1.0}
    recess_mult = {"AAA":1.5,"AA":1.8,"A":2.0,"BBB":2.5,"BB":3.0,"B":3.5,"CCC":4.0}
    severe_mult = {"AAA":2.5,"AA":3.0,"A":3.5,"BBB":4.5,"BB":5.5,"B":6.5,"CCC":8.0}

    def apply_severity(mult_dict, sev):
        return {k: 1 + (v-1)*sev for k, v in mult_dict.items()}

    scenarios = {
        "Base Case":        apply_severity(base_mult, 1.0),
        "Recession":        apply_severity(recess_mult, severity),
        "Severe Recession": apply_severity(severe_mult, min(severity * 1.5, 3.5)),
    }

    results = []
    for scen_name, mult in scenarios.items():
        stressed_pd = df["Credit Grade"].map(lambda g, m=mult: min(df.loc[df["Credit Grade"]==g,"PD (%)"].iloc[0]/100 * m.get(g, 1.0), 0.99))
        stressed_pd = df.apply(lambda row, m=mult: min(row["PD (%)"]/100 * m.get(row["Credit Grade"], 1.0), 0.99), axis=1)
        stressed_el = stressed_pd * (df["LGD (%)"]/100) * df["Loan Amount ($)"]
        total_loss = stressed_el.sum()
        loss_rate  = total_loss / df["Loan Amount ($)"].sum() * 100
        capital_req = total_loss * 1.08  # 8 % capital buffer on top
        results.append({
            "Scenario": scen_name,
            "Total Expected Loss ($M)": round(total_loss/1e6, 1),
            "Loss Rate (%)": round(loss_rate, 3),
            "Capital Required ($M)": round(capital_req/1e6, 1),
            "_losses": stressed_el.values,
        })

    res_df = pd.DataFrame(results)

    # ── Scenario table ────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)
    colours_scen = ["#3fb950","#e3b341","#f85149"]
    for col, (_, row), clr in zip([sc1,sc2,sc3], res_df.iterrows(), colours_scen):
        with col:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid {clr}44;
                        border-radius:12px;padding:20px;text-align:center;">
              <div style="font-size:13px;font-weight:700;color:{clr};
                          text-transform:uppercase;margin-bottom:12px;">
                {row['Scenario']}
              </div>
              <div style="font-size:30px;font-weight:800;color:#f0f6fc;">
                ${row['Total Expected Loss ($M)']:,.1f}M
              </div>
              <div style="font-size:12px;color:#8b949e;margin-top:4px;">Expected Loss</div>
              <hr style="border-color:#30363d;margin:12px 0">
              <div style="font-size:16px;color:{clr};font-weight:600;">
                {row['Loss Rate (%)']:.2f}% Loss Rate
              </div>
              <div style="font-size:13px;color:#8b949e;margin-top:6px;">
                Capital Req: ${row['Capital Required ($M)']:,.1f}M
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Loss distribution chart ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Credit Loss Distribution (Monte Carlo)</div>', unsafe_allow_html=True)

    rng2 = np.random.default_rng(7)
    n_sims = 10_000
    fig_dist = go.Figure()

    scen_styles = [
        ("Base Case",        "#3fb950", "solid"),
        ("Recession",        "#e3b341", "solid"),
        ("Severe Recession", "#f85149", "solid"),
    ]

    for (sname, sclr, sdash), (_, srow) in zip(scen_styles, res_df.iterrows()):
        loan_losses = srow["_losses"]
        # simulate correlated losses via a factor model
        macro_factor = rng2.normal(0, 1, n_sims)
        sim_total = np.zeros(n_sims)
        for loss_i in loan_losses:
            idio = rng2.normal(0, 1, n_sims)
            corr = 0.25
            combined = np.sqrt(corr)*macro_factor + np.sqrt(1-corr)*idio
            default_draw = (combined < -1.5).astype(float)
            sim_total += default_draw * loss_i

        p95 = np.percentile(sim_total, 95)
        p99 = np.percentile(sim_total, 99)
        el  = np.mean(sim_total)

        counts, bins = np.histogram(sim_total/1e6, bins=60)
        fig_dist.add_trace(go.Bar(
            x=(bins[:-1]+bins[1:])/2, y=counts,
            name=sname, marker_color=sclr, opacity=0.55,
        ))
        for pct, val, ls in [(95, p95,"dash"),(99, p99,"dot")]:
            fig_dist.add_vline(
                x=val/1e6, line_dash=ls, line_color=sclr,
                annotation_text=f"{sname} VaR {pct}%: ${val/1e6:.1f}M",
                annotation_font_size=10, annotation_font_color=sclr,
            )

    fig_dist.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG, height=420,
        barmode="overlay",
        xaxis_title="Portfolio Loss ($M)",
        yaxis_title="Simulation Frequency",
        margin=dict(t=10, b=50, l=50, r=10),
        xaxis_gridcolor=PLOTLY_GRID, yaxis_gridcolor=PLOTLY_GRID,
        legend=dict(x=0.75, y=0.95),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Industry stress heatmap ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Industry Loss Rate by Scenario</div>', unsafe_allow_html=True)

    heat_rows = []
    for scen_name, mult in scenarios.items():
        for ind in df["Industry"].unique():
            sub = df[df["Industry"]==ind]
            s_pd = sub.apply(
                lambda row, m=mult: min(row["PD (%)"]/100 * m.get(row["Credit Grade"],1.0), 0.99), axis=1
            )
            s_el = (s_pd * sub["LGD (%)"]/100 * sub["Loan Amount ($)"]).sum()
            s_exp = sub["Loan Amount ($)"].sum()
            heat_rows.append({"Scenario": scen_name, "Industry": ind,
                               "Loss Rate (%)": round(s_el/s_exp*100, 2)})

    heat_df = pd.DataFrame(heat_rows)
    pivot = heat_df.pivot(index="Industry", columns="Scenario", values="Loss Rate (%)")
    pivot = pivot[["Base Case","Recession","Severe Recession"]]

    fig_heat = px.imshow(
        pivot,
        labels=dict(x="Scenario", y="Industry", color="Loss Rate (%)"),
        color_continuous_scale=[[0,"#161b22"],[0.4,"#e3b341"],[1,"#f85149"]],
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG, height=320,
        margin=dict(t=10, b=10, l=120, r=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — INDIVIDUAL LOAN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    loan_options = df["Loan ID"].tolist()
    selected_id = st.selectbox("Select Loan", loan_options, index=0)
    loan = df[df["Loan ID"] == selected_id].iloc[0]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Loan detail cards ─────────────────────────────────────────────────────
    d1, d2, d3, d4, d5 = st.columns(5)
    details = [
        (d1, "Industry",        loan["Industry"]),
        (d2, "Credit Grade",    loan["Credit Grade"]),
        (d3, "Loan Amount",     f"${loan['Loan Amount ($)']/1e6:.1f}M"),
        (d4, "PD",              f"{loan['PD (%)']:.3f}%"),
        (d5, "Expected Loss",   f"${loan['Expected Loss ($)']/1e3:.0f}K"),
    ]
    for col, lbl, val in details:
        with col:
            grade_clr = grade_colour(loan["Credit Grade"]) if lbl == "Credit Grade" else "#f0f6fc"
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-value" style="font-size:22px;color:{grade_clr};">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">SHAP Waterfall — Why This Default Probability?</div>',
                unsafe_allow_html=True)

    # get shap for this loan
    X_loan = loan[features].values.reshape(1, -1)
    shap_loan = explainer.shap_values(X_loan)
    if isinstance(shap_loan, list):
        sv_loan = shap_loan[1][0]
        base_val = float(explainer.expected_value[1])
    else:
        sv_loan = shap_loan[0]
        base_val = float(explainer.expected_value)

    pd_pred = model.predict_proba(X_loan)[0][1]
    feat_vals = loan[features].values

    # Build waterfall
    feat_display = [
        f"DTI={feat_vals[0]:.2f}",
        f"Score={feat_vals[1]:.0f}",
        f"RevGrowth={feat_vals[2]:.1%}",
        f"Leverage={feat_vals[3]:.1f}",
        f"ICR={feat_vals[4]:.1f}",
    ]
    running = [base_val + sv_loan[:i+1].sum() for i in range(len(sv_loan))]
    clrs_wf = ["#f85149" if v > 0 else "#3fb950" for v in sv_loan]

    fig_wfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"]*len(sv_loan) + ["total"],
        x=feat_display + ["Final PD"],
        y=list(sv_loan) + [0],
        base=base_val,
        connector={"line": {"color": "#30363d"}},
        decreasing={"marker": {"color": "#3fb950"}},
        increasing={"marker": {"color": "#f85149"}},
        totals={"marker": {"color": "#1f6feb"}},
        text=[f"{v:+.3f}" for v in sv_loan] + [f"{pd_pred:.3f}"],
        textposition="outside",
    ))
    fig_wfall.add_hline(y=base_val, line_dash="dash", line_color="#8b949e",
                        annotation_text=f"Base rate: {base_val:.3f}")
    fig_wfall.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG, height=380,
        yaxis_title="Default Probability",
        margin=dict(t=10, b=60, l=60, r=10),
        yaxis_gridcolor=PLOTLY_GRID,
    )
    st.plotly_chart(fig_wfall, use_container_width=True)

    # ── Peer comparison ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Peer Comparison — Same Industry & Grade</div>',
                unsafe_allow_html=True)

    peers = df[
        (df["Industry"] == loan["Industry"]) &
        (df["Credit Grade"] == loan["Credit Grade"])
    ].copy()
    peers["_is_selected"] = peers["Loan ID"] == selected_id

    peer_fig = go.Figure()

    # all peers
    others = peers[~peers["_is_selected"]]
    peer_fig.add_trace(go.Scatter(
        x=others["PD (%)"], y=others["LGD (%)"],
        mode="markers",
        marker=dict(size=12, color="#1f6feb", opacity=0.7),
        name="Peers",
        hovertext=others["Loan ID"],
    ))
    # selected loan
    sel = peers[peers["_is_selected"]]
    peer_fig.add_trace(go.Scatter(
        x=sel["PD (%)"], y=sel["LGD (%)"],
        mode="markers",
        marker=dict(size=18, color="#f85149", symbol="star"),
        name=f"{selected_id} (selected)",
    ))

    # percentile annotation
    if len(peers) > 1:
        pct_rank = (peers["PD (%)"] <= loan["PD (%)"]).mean() * 100
        peer_fig.add_annotation(
            x=loan["PD (%)"], y=loan["LGD (%)"],
            text=f"  {pct_rank:.0f}th percentile PD",
            showarrow=False, font=dict(color="#f85149", size=12), xanchor="left",
        )

    peer_fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG, height=350,
        xaxis_title="PD (%)", yaxis_title="LGD (%)",
        title=f"{loan['Industry']} | {loan['Credit Grade']} — {len(peers)} loans",
        margin=dict(t=40, b=40, l=50, r=10),
        xaxis_gridcolor=PLOTLY_GRID, yaxis_gridcolor=PLOTLY_GRID,
    )
    st.plotly_chart(peer_fig, use_container_width=True)

    # ── Borrower feature radar ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Borrower Profile vs. Industry Median</div>',
                unsafe_allow_html=True)

    industry_peers = df[df["Industry"] == loan["Industry"]]
    feat_norm_labels = ["DTI", "Credit Score", "Rev Growth", "Leverage", "ICR"]

    def normalise(col, invert=False):
        mn, mx = df[col].min(), df[col].max()
        val_self = (loan[col] - mn) / (mx - mn)
        val_peer = (industry_peers[col].median() - mn) / (mx - mn)
        if invert:
            return 1 - val_self, 1 - val_peer
        return val_self, val_peer

    self_vals, peer_vals = [], []
    for col, inv in zip(features, [True, False, False, True, False]):
        s, p = normalise(col, inv)
        self_vals.append(s)
        peer_vals.append(p)

    theta = feat_norm_labels + [feat_norm_labels[0]]
    self_r = self_vals + [self_vals[0]]
    peer_r = peer_vals + [peer_vals[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=self_r, theta=theta, fill="toself",
        name=selected_id, line=dict(color="#f85149"),
        fillcolor="rgba(248,81,73,0.15)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=peer_r, theta=theta, fill="toself",
        name="Industry Median", line=dict(color="#1f6feb"),
        fillcolor="rgba(31,111,235,0.15)",
    ))
    fig_radar.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=PLOTLY_PAPER,
        polar=dict(
            bgcolor=PLOTLY_BG,
            radialaxis=dict(visible=True, range=[0,1], color="#8b949e"),
            angularaxis=dict(color="#8b949e"),
        ),
        height=380,
        legend=dict(x=0.8, y=1.1),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
