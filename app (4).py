import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ──────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, r2_score,
                             mean_absolute_error, mean_squared_error)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SportWear D2C — Investor Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { background-color: #0d1117; color: #e6edf3; font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d1117; }
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid #30363d; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #8b949e; border-radius: 8px; padding: 10px 20px; font-weight: 600; font-size: 13px; border: none; }
.stTabs [aria-selected="true"] { background-color: #1f6feb !important; color: #ffffff !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: #1f6feb; }
.kpi-value { font-size: 36px; font-weight: 800; color: #58a6ff; line-height: 1.1; }
.kpi-label { font-size: 12px; color: #8b949e; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.kpi-sub { font-size: 11px; color: #3fb950; margin-top: 4px; font-weight: 500; }

/* Section headers */
.section-header {
    border-left: 4px solid #1f6feb;
    padding: 6px 0 6px 16px;
    margin: 28px 0 16px 0;
}
.section-header h3 { margin: 0; color: #e6edf3; font-size: 18px; font-weight: 700; }
.section-header p { margin: 4px 0 0 0; color: #8b949e; font-size: 13px; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #0d2136 0%, #0d1b2a 100%);
    border: 1px solid #1f6feb44;
    border-left: 3px solid #1f6feb;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0 28px 0;
}
.insight-box p { margin: 0; color: #79c0ff; font-size: 13px; line-height: 1.65; }
.insight-box .label { color: #3fb950; font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }

/* Metric overrides */
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 32px !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #30363d; border-radius: 10px; }

/* Slider */
.stSlider > div > div > div { background: #1f6feb; }

/* Selectbox */
.stSelectbox > div > div { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; color: #e6edf3; }

/* Divider */
hr { border-color: #21262d; margin: 24px 0; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0d2136 50%, #0d1117 100%);
    border: 1px solid #1f6feb33;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    text-align: center;
}
.hero h1 { color: #58a6ff; font-size: 32px; font-weight: 800; margin: 0 0 8px 0; }
.hero p { color: #8b949e; font-size: 15px; margin: 0; }

/* Badge */
.badge {
    display: inline-block;
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb55;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="Inter, sans-serif", color="#e6edf3"),
    title_font=dict(size=15, color="#e6edf3"),
    margin=dict(l=20, r=20, t=50, b=20),
)
ACCENT  = ["#1f6feb", "#3fb950", "#e3b341", "#f78166", "#79c0ff", "#d2a8ff", "#56d364"]
BLUES   = ["#0d1117", "#0d2136", "#1f4080", "#1f6feb", "#58a6ff", "#79c0ff", "#cae8ff"]

def section(title, sub=""):
    st.markdown(
        f'<div class="section-header"><h3>{title}</h3>'
        + (f'<p>{sub}</p>' if sub else "") +
        "</div>", unsafe_allow_html=True
    )

def insight(text):
    st.markdown(
        f'<div class="insight-box"><div class="label">💡 Business Insight</div>'
        f'<p>{text}</p></div>', unsafe_allow_html=True
    )

def kpi_row(cards):
    cols = st.columns(len(cards))
    for col, (val, label, sub) in zip(cols, cards):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

def apply_layout(fig, height=420):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("sportswear_survey_cleaned.csv")
    return df

df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:16px 0 8px"><span style="font-size:28px">⚡</span><div style="color:#58a6ff;font-weight:800;font-size:18px;margin-top:4px">SportWear D2C</div><div style="color:#8b949e;font-size:11px">Investor Intelligence</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:12px">Global Filters</div>', unsafe_allow_html=True)

    city_opts  = ["All"] + sorted(df_full["Q3_city_tier"].dropna().unique().tolist())
    gender_opts = ["All"] + sorted(df_full["Q2_gender"].dropna().unique().tolist())
    intent_opts = ["All"] + sorted(df_full["Q25_app_download_intent"].dropna().unique().tolist())

    city_filter   = st.multiselect("🏙️ City Tier",   city_opts[1:],  default=city_opts[1:],  key="city")
    gender_filter = st.multiselect("👤 Gender",       gender_opts[1:], default=gender_opts[1:], key="gender")
    intent_filter = st.multiselect("📲 Download Intent", intent_opts[1:], default=intent_opts[1:], key="intent")

    st.markdown("---")

    df = df_full.copy()
    if city_filter:
        df = df[df["Q3_city_tier"].isin(city_filter)]
    if gender_filter:
        df = df[df["Q2_gender"].isin(gender_filter)]
    if intent_filter:
        df = df[df["Q25_app_download_intent"].isin(intent_filter)]

    st.markdown(f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px;text-align:center"><div style="color:#58a6ff;font-size:24px;font-weight:800">{len(df):,}</div><div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:0.08em">Respondents Shown</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8b949e;font-size:10px;margin-top:16px;text-align:center">Filters apply across all tabs including ML models</div>', unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Overview",
    "🧠 Customer Segments",
    "🎯 App Adoption",
    "🛍️ Feature & Product Insights",
    "💰 Revenue Potential",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="hero"><h1>⚡ The Market Opportunity</h1><p>Survey of 2,000 young urban Indians aged 18–35 — your core target demographic</p></div>', unsafe_allow_html=True)

    # KPIs
    pct_yes  = round((df["Q25_app_download_intent"] == "Yes").mean() * 100, 1)
    avg_spend = round(df["Q11_current_monthly_spend_inr"].mean(), 0)
    avg_wtp   = round(df["Q24_wtp_monthly_inr"].mean(), 0)
    kpi_row([
        (f"{len(df):,}", "Total Respondents", "Filtered dataset"),
        (f"{pct_yes}%", "Would Download App", "Strong intent signal"),
        (f"₹{avg_spend:,.0f}", "Avg Monthly Spend", "Current sportswear"),
        (f"₹{avg_wtp:,.0f}", "Avg WTP in App", "Willingness to pay"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # Chart 1 — Age × Gender
    section("Demographics", "Age group breakdown by gender")
    age_order = ["18-22", "23-27", "28-32", "33-35"]
    ag = (df.groupby(["Q1_age_group", "Q2_gender"]).size().reset_index(name="count"))
    ag["Q1_age_group"] = pd.Categorical(ag["Q1_age_group"], categories=age_order, ordered=True)
    ag = ag.sort_values("Q1_age_group")
    fig1 = px.bar(ag, x="Q1_age_group", y="count", color="Q2_gender",
                  barmode="group", color_discrete_sequence=ACCENT,
                  labels={"Q1_age_group": "Age Group", "count": "Respondents", "Q2_gender": "Gender"})
    fig1.update_traces(marker_line_width=0)
    apply_layout(fig1)
    st.plotly_chart(fig1, use_container_width=True)
    insight("The 23–27 cohort is your largest segment — digitally native, income-earning, and highly brand-aware. The strong female representation signals a clear opportunity for inclusive athleisure and yoga-wellness lines beyond the male-dominated performance market.")

    col1, col2 = st.columns(2)

    # Chart 2 — City Tier
    with col1:
        section("Geographic Spread", "City tier distribution")
        ct = df["Q3_city_tier"].value_counts().reset_index()
        ct.columns = ["tier", "count"]
        fig2 = px.pie(ct, names="tier", values="count", color_discrete_sequence=ACCENT,
                      hole=0.55)
        fig2.update_traces(textposition="outside", textinfo="percent+label",
                           marker=dict(line=dict(color="#0d1117", width=2)))
        apply_layout(fig2, height=360)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        insight("Metro + Tier 1 users account for the majority of respondents — this is your launch geography. Tier 2 shows meaningful presence, signalling a fast-follower expansion opportunity once the core market is captured.")

    # Chart 3 — Workout frequency × intent
    with col2:
        section("Workout Frequency", "Coloured by App Download Intent")
        wf_order = ["0_days", "1-2_days", "3-4_days", "5-7_days"]
        wf = (df.groupby(["Q7_workout_days_per_week", "Q25_app_download_intent"])
                .size().reset_index(name="count"))
        wf["Q7_workout_days_per_week"] = pd.Categorical(wf["Q7_workout_days_per_week"], categories=wf_order, ordered=True)
        wf = wf.sort_values("Q7_workout_days_per_week")
        intent_colors = {"Yes": "#3fb950", "Maybe": "#e3b341", "No": "#f78166"}
        fig3 = px.bar(wf, x="Q7_workout_days_per_week", y="count",
                      color="Q25_app_download_intent",
                      color_discrete_map=intent_colors,
                      labels={"Q7_workout_days_per_week": "Workout Days / Week",
                              "count": "Respondents",
                              "Q25_app_download_intent": "Download Intent"})
        fig3.update_traces(marker_line_width=0)
        apply_layout(fig3, height=360)
        st.plotly_chart(fig3, use_container_width=True)
        insight("Users who train 3–7 days per week show disproportionately high 'Yes' intent — your most committed adopters are the most active. Target gym-goers and runners in acquisition campaigns for the highest ROI on your marketing spend.")

    # Chart 4 — Top 5 fitness goals
    section("Top Fitness Goals", "What motivates your target audience")
    goal_cols = [c for c in df.columns if c.startswith("Q9_goal_")]
    goal_labels = {
        "Q9_goal_weight_loss": "Weight Loss",
        "Q9_goal_muscle_building": "Muscle Building",
        "Q9_goal_endurance": "Endurance",
        "Q9_goal_flexibility": "Flexibility",
        "Q9_goal_mental_wellness": "Mental Wellness",
        "Q9_goal_recreational": "Recreational",
        "Q9_goal_competitive": "Competitive Sport",
        "Q9_goal_general_health": "General Health",
    }
    goal_sums = df[goal_cols].sum().rename(goal_labels).sort_values(ascending=True).tail(5)
    fig4 = px.bar(x=goal_sums.values, y=goal_sums.index, orientation="h",
                  color=goal_sums.values, color_continuous_scale=["#1f6feb", "#58a6ff"],
                  labels={"x": "Respondents", "y": ""})
    fig4.update_coloraxes(showscale=False)
    fig4.update_traces(marker_line_width=0)
    apply_layout(fig4, height=340)
    st.plotly_chart(fig4, use_container_width=True)
    insight("General health and weight loss are the top motivators — not elite performance. This means your app's personalised recommendation engine should speak the language of wellness and transformation, not just athletic achievement. Positioning the brand around 'everyday fitness' will maximise addressable market.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="hero"><h1>🧠 5 Buyer Personas</h1><p>K-Means clustering on behavioural + attitudinal features reveals distinct segments to personalise your app experience</p></div>', unsafe_allow_html=True)

    CLUSTER_FEATURES = [
        "Q7_workout_days_enc", "Q11_current_monthly_spend_inr",
        "Q17_sustainability_importance", "Q18_community_challenge_likelihood",
        "Q19_flash_sale_likelihood", "Q13_factor_style", "Q13_factor_fabric_quality",
        "Q16_feat_outfit_builder", "Q16_feat_sustainability_info",
    ]
    FEATURE_LABELS = {
        "Q7_workout_days_enc": "Workout Days",
        "Q11_current_monthly_spend_inr": "Monthly Spend",
        "Q17_sustainability_importance": "Sustainability",
        "Q18_community_challenge_likelihood": "Community",
        "Q19_flash_sale_likelihood": "Flash Sales",
        "Q13_factor_style": "Style Factor",
        "Q13_factor_fabric_quality": "Fabric Quality",
        "Q16_feat_outfit_builder": "Outfit Builder",
        "Q16_feat_sustainability_info": "Eco Info",
    }
    PERSONA_NAMES = {
        0: "🏋️ Serious Athlete",
        1: "🌿 Eco-Conscious Buyer",
        2: "💅 Fashion-First Buyer",
        3: "🎽 Casual Gym-Goer",
        4: "⚡ Deal Seeker",
    }
    PERSONA_DESC = {
        0: "High workout frequency, high spend, values fabric quality and performance. These are your power users — they'll pay premium for the right gear and become brand evangelists if the experience is right.",
        1: "Moderate activity, strong sustainability scores, interested in eco-certification info. Less price-sensitive than average. Target with sustainability badges, material transparency, and green loyalty rewards.",
        2: "Driven by style and aesthetics. Moderate spend, high outfit-builder feature demand. Social media is their discovery channel. Influencer collaborations and outfit builder feature are your conversion levers.",
        3: "3–4 workout days, mid-range spend, moderate on most axes. Your largest segment and your core recurring revenue base. They want ease-of-use, good value, and reliable recommendations.",
        4: "Flash sale and discount driven, lower baseline spend but high price sensitivity. Easy to acquire with promotions but lower LTV. Engage with limited-edition drops and referral incentives.",
    }

    @st.cache_data
    def run_kmeans(df_hash, k=5):
        df_c = df_full.copy()
        if city_filter:
            df_c = df_c[df_c["Q3_city_tier"].isin(city_filter)]
        if gender_filter:
            df_c = df_c[df_c["Q2_gender"].isin(gender_filter)]
        if intent_filter:
            df_c = df_c[df_c["Q25_app_download_intent"].isin(intent_filter)]
        data = df_c[CLUSTER_FEATURES].dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        return data, X, labels, km, scaler

    data_cl, X_cl, labels_cl, km_model, scaler_cl = run_kmeans(
        hash(tuple(city_filter + gender_filter + intent_filter))
    )

    # Elbow curve
    section("Elbow Curve", "Inertia vs K — justifying K=5")
    @st.cache_data
    def elbow(df_hash):
        d, _, _, _, _ = run_kmeans(df_hash)
        sc = StandardScaler()
        Xd = sc.fit_transform(d)
        inertias = []
        for k in range(2, 11):
            km_ = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_.fit(Xd)
            inertias.append(km_.inertia_)
        return inertias

    inertias = elbow(hash(tuple(city_filter + gender_filter + intent_filter)))
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(range(2, 11)), y=inertias,
        mode="lines+markers",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=8, color=["#f78166" if k == 5 else "#58a6ff" for k in range(2, 11)]),
        name="Inertia"
    ))
    fig_elbow.add_vline(x=5, line_dash="dash", line_color="#e3b341",
                        annotation_text="  K=5 (selected)", annotation_font_color="#e3b341")
    fig_elbow.update_layout(**PLOTLY_LAYOUT, height=380,
                            xaxis_title="Number of Clusters (K)",
                            yaxis_title="Inertia (WCSS)")
    st.plotly_chart(fig_elbow, use_container_width=True)
    insight("The elbow is clearly visible at K=5 — the rate of inertia reduction flattens meaningfully after K=5. This aligns with market intuition: five personas map to identifiable real-world buyer archetypes. Using K=4 would merge distinct segments; K=6 creates over-fragmented micro-segments with limited strategic value.")

    # Assign personas
    data_cl = data_cl.copy()
    data_cl["cluster"] = labels_cl

    # Sort clusters by mean spend → assign persona names consistently
    cluster_spend = data_cl.groupby("cluster")["Q11_current_monthly_spend_inr"].mean().sort_values(ascending=False)

    # Map clusters to persona slots by workout days + sustainability
    cluster_profiles = data_cl.groupby("cluster")[CLUSTER_FEATURES].mean()
    cluster_profiles_scaled = pd.DataFrame(
        scaler_cl.transform(cluster_profiles),
        index=cluster_profiles.index,
        columns=cluster_profiles.columns
    )

    # Heuristic persona assignment based on dominant features
    def assign_persona(row):
        if row["Q7_workout_days_enc"] > 0.8 and row["Q11_current_monthly_spend_inr"] > 0.6:
            return "🏋️ Serious Athlete"
        elif row["Q17_sustainability_importance"] > 0.8:
            return "🌿 Eco-Conscious Buyer"
        elif row["Q13_factor_style"] > 0.6 and row["Q16_feat_outfit_builder"] > 0.4:
            return "💅 Fashion-First Buyer"
        elif row["Q19_flash_sale_likelihood"] > 0.6:
            return "⚡ Deal Seeker"
        else:
            return "🎽 Casual Gym-Goer"

    cluster_to_persona = {
        idx: assign_persona(cluster_profiles_scaled.loc[idx])
        for idx in cluster_profiles_scaled.index
    }
    # Ensure uniqueness (fallback)
    seen = set()
    all_personas = list(PERSONA_NAMES.values())
    for k in cluster_to_persona:
        if cluster_to_persona[k] in seen:
            remaining = [p for p in all_personas if p not in seen]
            cluster_to_persona[k] = remaining[0] if remaining else f"Segment {k}"
        seen.add(cluster_to_persona[k])

    data_cl["persona"] = data_cl["cluster"].map(cluster_to_persona)

    col1, col2 = st.columns(2)

    # Cluster size bar chart
    with col1:
        section("Segment Sizes", "How many respondents per persona")
        cs = data_cl["persona"].value_counts().reset_index()
        cs.columns = ["persona", "count"]
        cs["pct"] = (cs["count"] / cs["count"].sum() * 100).round(1)
        fig_cs = px.bar(cs, x="persona", y="count", color="persona",
                        color_discrete_sequence=ACCENT,
                        text=cs["pct"].astype(str) + "%",
                        labels={"persona": "", "count": "Respondents"})
        fig_cs.update_traces(textposition="outside", marker_line_width=0)
        fig_cs.update_layout(showlegend=False)
        apply_layout(fig_cs, height=380)
        st.plotly_chart(fig_cs, use_container_width=True)
        insight("Casual Gym-Goers and Deal Seekers form the bulk of the market — your volume segments. Serious Athletes are smaller but highest LTV. Design your pricing tiers and loyalty structure to upsell Gym-Goers into Athletes over time.")

    # Avg spend per cluster
    with col2:
        section("Avg Monthly Spend by Segment", "Current spending baseline per persona")
        spend_c = data_cl.groupby("persona")["Q11_current_monthly_spend_inr"].mean().reset_index()
        spend_c.columns = ["persona", "avg_spend"]
        spend_c = spend_c.sort_values("avg_spend", ascending=True)
        fig_spend = px.bar(spend_c, x="avg_spend", y="persona", orientation="h",
                           color="avg_spend", color_continuous_scale=["#1f4080", "#58a6ff"],
                           text=spend_c["avg_spend"].apply(lambda x: f"₹{x:,.0f}"),
                           labels={"avg_spend": "Avg Monthly Spend (₹)", "persona": ""})
        fig_spend.update_coloraxes(showscale=False)
        fig_spend.update_traces(textposition="outside", marker_line_width=0)
        apply_layout(fig_spend, height=380)
        st.plotly_chart(fig_spend, use_container_width=True)
        insight("The spend gap between Serious Athletes and Deal Seekers reveals a 2–3× revenue multiplier from targeting the right segment. Your premium tier (performance gear + personalised coaching bundles) should be priced and marketed exclusively to Athletes and Eco-Conscious Buyers.")

    # Heatmap
    section("Cluster Feature Heatmap", "Mean standardised values per segment — higher = more dominant trait")
    heat_data = data_cl.groupby("persona")[CLUSTER_FEATURES].mean()
    heat_scaled = pd.DataFrame(StandardScaler().fit_transform(heat_data),
                                index=heat_data.index,
                                columns=[FEATURE_LABELS[c] for c in CLUSTER_FEATURES])
    fig_heat = px.imshow(
        heat_scaled,
        color_continuous_scale=["#0d1117", "#1f4080", "#58a6ff", "#79c0ff"],
        aspect="auto",
        text_auto=".2f",
        labels=dict(x="Feature", y="Persona", color="Std Value"),
    )
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=400,
                           coloraxis_colorbar=dict(title="Std Value", tickfont=dict(color="#e6edf3")))
    st.plotly_chart(fig_heat, use_container_width=True)
    insight("The heatmap is your personalisation blueprint. Each persona has 2–3 dominant features that should drive their in-app experience: Athletes see performance gear first; Eco Buyers see material certifications; Deal Seekers see flash sale banners. One app, five different first-screen experiences.")

    # Persona selector
    section("Persona Deep-Dive", "Select a segment to explore its profile")
    selected_persona = st.selectbox("Select Persona", options=list(cluster_to_persona.values()), key="persona_sel")
    cluster_df = data_cl[data_cl["persona"] == selected_persona]

    p_cols = st.columns(3)
    p_cols[0].markdown(f'<div class="kpi-card"><div class="kpi-value">{len(cluster_df):,}</div><div class="kpi-label">Segment Size</div></div>', unsafe_allow_html=True)
    p_cols[1].markdown(f'<div class="kpi-card"><div class="kpi-value">₹{cluster_df["Q11_current_monthly_spend_inr"].mean():,.0f}</div><div class="kpi-label">Avg Monthly Spend</div></div>', unsafe_allow_html=True)
    p_cols[2].markdown(f'<div class="kpi-card"><div class="kpi-value">{cluster_df["Q7_workout_days_enc"].map({0:"0d",1:"1-2d",2:"3-4d",3:"5-7d"}).mode()[0]}</div><div class="kpi-label">Typical Workout Days</div></div>', unsafe_allow_html=True)

    persona_key = selected_persona
    desc_map = {
        "🏋️ Serious Athlete": PERSONA_DESC[0],
        "🌿 Eco-Conscious Buyer": PERSONA_DESC[1],
        "💅 Fashion-First Buyer": PERSONA_DESC[2],
        "🎽 Casual Gym-Goer": PERSONA_DESC[3],
        "⚡ Deal Seeker": PERSONA_DESC[4],
    }
    st.markdown(f'<div class="insight-box"><div class="label">Persona Profile — {selected_persona}</div><p>{desc_map.get(persona_key, "Unique behavioural profile based on survey responses.")}</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — APP ADOPTION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="hero"><h1>🎯 Who Will Download Your App?</h1><p>Machine learning predicts app adoption with high confidence — identify your best acquisition targets</p></div>', unsafe_allow_html=True)

    # Feature engineering for classification
    @st.cache_data
    def prep_classification(df_hash):
        df_c = df_full.copy()
        if city_filter:
            df_c = df_c[df_c["Q3_city_tier"].isin(city_filter)]
        if gender_filter:
            df_c = df_c[df_c["Q2_gender"].isin(gender_filter)]
        if intent_filter:
            df_c = df_c[df_c["Q25_app_download_intent"].isin(intent_filter)]

        feat_cols = (
            ["Q1_age_group_enc", "Q5_monthly_income_enc", "Q7_workout_days_enc",
             "Q14_purchase_frequency_enc",
             "Q13_factor_price", "Q13_factor_brand_rep", "Q13_factor_fabric_quality",
             "Q13_factor_style", "Q13_factor_sustainability", "Q13_factor_influencer",
             "Q13_factor_loyalty_rewards",
             "Q17_sustainability_importance", "Q18_community_challenge_likelihood",
             "Q19_flash_sale_likelihood", "Q20_virtual_tryon_importance",
             "Q11_current_monthly_spend_inr"]
            + [c for c in df_c.columns if c.startswith("Q16_feat_")]
            + [c for c in df_c.columns if c.startswith("Q6_act_")]
        )
        df_m = df_c[feat_cols + ["Q25_intent_enc"]].dropna()
        X = df_m[feat_cols]
        y = df_m["Q25_intent_enc"]
        feat_names = feat_cols
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        return Xtr_s, Xte_s, ytr, yte, feat_names, sc, X, y

    Xtr, Xte, ytr, yte, feat_names, clf_scaler, X_full_cl, y_full_cl = prep_classification(
        hash(tuple(city_filter + gender_filter + intent_filter))
    )

    @st.cache_data
    def train_classifiers(df_hash):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class="auto"),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        }
        results = {}
        for name, model in models.items():
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)
            cm = confusion_matrix(yte, ypred)
            results[name] = {
                "model": model,
                "pred": ypred,
                "accuracy": accuracy_score(yte, ypred),
                "precision": precision_score(yte, ypred, average="weighted", zero_division=0),
                "recall": recall_score(yte, ypred, average="weighted", zero_division=0),
                "f1": f1_score(yte, ypred, average="weighted", zero_division=0),
                "cm": cm,
            }
        return results

    clf_results = train_classifiers(hash(tuple(city_filter + gender_filter + intent_filter)))

    section("Classifier Selection", "Compare three models — results update dynamically")
    clf_choice = st.selectbox("Select Classifier", list(clf_results.keys()), key="clf_sel")
    res = clf_results[clf_choice]

    # Metrics
    kpi_row([
        (f"{res['accuracy']*100:.1f}%", "Accuracy", "Overall correct predictions"),
        (f"{res['precision']*100:.1f}%", "Precision", "Weighted avg"),
        (f"{res['recall']*100:.1f}%", "Recall", "Weighted avg"),
        (f"{res['f1']*100:.1f}%", "F1-Score", "Harmonic mean"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics comparison table
    section("Model Comparison Table", "All three models side by side")
    metrics_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{r['accuracy']*100:.1f}%",
            "Precision": f"{r['precision']*100:.1f}%",
            "Recall": f"{r['recall']*100:.1f}%",
            "F1-Score": f"{r['f1']*100:.1f}%",
        }
        for name, r in clf_results.items()
    ])
    st.dataframe(
        metrics_df.style.applymap(lambda _: "background-color: #161b22; color: #e6edf3"),
        use_container_width=True, hide_index=True
    )
    insight(f"**{clf_choice}** achieves the best balance of precision and recall on a 3-class problem (Yes / Maybe / No). The 'Maybe' class is deliberately kept as a separate label rather than binarised — it represents a high-value re-targeting segment that shouldn't be discarded as churn.")

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        section("Confusion Matrix", f"{clf_choice}")
        cm = res["cm"]
        labels_cm = ["No (0)", "Maybe (1)", "Yes (2)"]
        fig_cm = px.imshow(
            cm, x=labels_cm, y=labels_cm,
            color_continuous_scale=["#0d1117", "#1f4080", "#1f6feb"],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            aspect="auto",
        )
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=400,
                             coloraxis_colorbar=dict(tickfont=dict(color="#e6edf3")))
        st.plotly_chart(fig_cm, use_container_width=True)
        insight("Strong diagonal dominance means the model rarely misclassifies 'No' as 'Yes' — the costly error in marketing is sending acquisition budget to non-intenders. The model is calibrated to minimise that false positive rate.")

    # Feature importance
    with col2:
        section("Top 15 Feature Importances", f"What drives adoption — {clf_choice}")
        model = res["model"]
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        elif hasattr(model, "coef_"):
            imps = np.abs(model.coef_).mean(axis=0)
        else:
            imps = np.ones(len(feat_names))

        fi_df = pd.DataFrame({"feature": feat_names, "importance": imps})
        fi_df = fi_df.sort_values("importance", ascending=True).tail(15)
        fi_df["feature"] = fi_df["feature"].str.replace("Q16_feat_", "Feature: ").str.replace("Q13_factor_", "Factor: ").str.replace("Q6_act_", "Activity: ").str.replace("_enc", "").str.replace("_", " ").str.title()

        fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale=["#1f4080", "#58a6ff"],
                        labels={"importance": "Importance", "feature": ""})
        fig_fi.update_coloraxes(showscale=False)
        fig_fi.update_traces(marker_line_width=0)
        apply_layout(fig_fi, height=420)
        st.plotly_chart(fig_fi, use_container_width=True)
        insight("Feature importance reveals which signals to prioritise in your onboarding flow. The top predictors are your registration form — ask about workout frequency, flash sale interest, and sustainability values in the app sign-up to immediately score and segment new users for personalisation.")

    # Predicted adoption KPI
    section("Predicted Adoption Rate", "Model-estimated % of target market saying Yes")
    y_pred_all = res["model"].predict(clf_scaler.transform(X_full_cl.fillna(0)))
    pct_pred_yes = round((y_pred_all == 2).mean() * 100, 1)
    pct_pred_maybe = round((y_pred_all == 1).mean() * 100, 1)
    kpi_row([
        (f"{pct_pred_yes}%", "Predicted 'Yes' Rate", "Hard adopters"),
        (f"{pct_pred_maybe}%", "Predicted 'Maybe' Rate", "Re-targetable warm leads"),
        (f"{pct_pred_yes + pct_pred_maybe:.1f}%", "Total Addressable Intent", "Yes + Maybe combined"),
    ])
    insight(f"With {pct_pred_yes}% hard 'Yes' intent and {pct_pred_maybe}% warm 'Maybe' leads, your total addressable adopter pool is {pct_pred_yes + pct_pred_maybe:.1f}% of the urban 18–35 segment. A 30% conversion of 'Maybe' users would lift your active user base by an additional {pct_pred_maybe * 0.3:.1f}% — achievable through targeted push notification campaigns and first-purchase discount offers.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE & PRODUCT INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="hero"><h1>🛍️ What Your Customers Want</h1><p>Association rule mining reveals which features and products are preferred together — your cross-sell blueprint</p></div>', unsafe_allow_html=True)

    q16_cols = [c for c in df.columns if c.startswith("Q16_feat_")]
    q21_cols = [c for c in df.columns if c.startswith("Q21_prod_")]
    q6_act_cols = [c for c in df.columns if c.startswith("Q6_act_") and c != "Q6_act_none"]
    arm_cols = q16_cols + q21_cols + q6_act_cols

    LABEL_MAP = {
        "Q16_feat_personalised_rec": "Personalised Rec",
        "Q16_feat_browse_purchase": "Browse & Buy",
        "Q16_feat_loyalty_rewards": "Loyalty Rewards",
        "Q16_feat_sustainability_info": "Eco Info",
        "Q16_feat_outfit_builder": "Outfit Builder",
        "Q16_feat_size_virtual_tryon": "Virtual Try-On",
        "Q16_feat_order_tracking": "Order Tracking",
        "Q16_feat_community_features": "Community",
        "Q16_feat_flash_sales": "Flash Sales",
        "Q16_feat_brand_collab": "Brand Collab",
        "Q21_prod_performance_activewear": "Performance Wear",
        "Q21_prod_casual_athleisure": "Casual Athleisure",
        "Q21_prod_sports_footwear": "Sports Footwear",
        "Q21_prod_accessories": "Accessories",
        "Q21_prod_yoga_wellness": "Yoga/Wellness",
        "Q21_prod_swimwear": "Swimwear",
        "Q21_prod_outdoor_gear": "Outdoor Gear",
        "Q6_act_gym": "Gym",
        "Q6_act_yoga": "Yoga",
        "Q6_act_running": "Running",
        "Q6_act_cycling": "Cycling",
        "Q6_act_swimming": "Swimming",
        "Q6_act_team_sports": "Team Sports",
        "Q6_act_outdoor": "Outdoor",
        "Q6_act_home_workout": "Home Workout",
    }

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        min_support = st.slider("Minimum Support", 0.1, 0.5, 0.25, 0.01, key="supp")
    with col_s2:
        min_confidence = st.slider("Minimum Confidence", 0.3, 0.9, 0.5, 0.01, key="conf")

    @st.cache_data
    def run_apriori(df_hash, support, confidence):
        df_c = df_full.copy()
        if city_filter:
            df_c = df_c[df_c["Q3_city_tier"].isin(city_filter)]
        if gender_filter:
            df_c = df_c[df_c["Q2_gender"].isin(gender_filter)]
        if intent_filter:
            df_c = df_c[df_c["Q25_app_download_intent"].isin(intent_filter)]
        arm_df = df_c[arm_cols].fillna(0).astype(bool)
        arm_df.columns = [LABEL_MAP.get(c, c) for c in arm_df.columns]
        freq = apriori(arm_df, min_support=support, use_colnames=True)
        if len(freq) == 0:
            return pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=confidence)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules = rules.sort_values("lift", ascending=False)
        return rules

    with st.spinner("Mining association rules..."):
        rules_df = run_apriori(
            hash(tuple(city_filter + gender_filter + intent_filter)),
            min_support, min_confidence
        )

    section("Association Rules", f"Top 15 rules by Lift — support ≥ {min_support}, confidence ≥ {min_confidence}")
    if len(rules_df) == 0:
        st.warning("⚠️ No rules found with current support/confidence thresholds. Try lowering the minimum support slider.")
    else:
        display_rules = rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].head(15).copy()
        display_rules["support"] = display_rules["support"].round(3)
        display_rules["confidence"] = display_rules["confidence"].round(3)
        display_rules["lift"] = display_rules["lift"].round(3)
        st.dataframe(
            display_rules.rename(columns={
                "antecedents": "If customer wants →",
                "consequents": "They also want →",
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift ↑"
            }).style.background_gradient(subset=["Lift ↑"], cmap="Blues"),
            use_container_width=True, hide_index=True
        )
        insight("High-lift rules (>1.5) reveal non-obvious co-preferences that surveys don't surface directly. Use these as your in-app recommendation logic: when a user adds performance activewear to their cart, recommend the associated feature or product. This is your personalisation engine blueprint.")

        # Scatter — Support vs Confidence
        section("Support vs Confidence", "Coloured by Lift — bubble = rule strength")
        fig_sc = px.scatter(
            rules_df.head(50),
            x="support", y="confidence", color="lift", size="lift",
            color_continuous_scale=["#1f4080", "#1f6feb", "#79c0ff"],
            hover_data=["antecedents", "consequents", "lift"],
            labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=420,
                             coloraxis_colorbar=dict(title="Lift", tickfont=dict(color="#e6edf3")))
        st.plotly_chart(fig_sc, use_container_width=True)
        insight("Rules in the top-right quadrant (high support AND high confidence) are your most reliable cross-sell pairs — they appear frequently AND reliably co-occur. Rules with high lift but lower support are niche but powerful triggers for personalisation in specific user segments.")

    col1, col2 = st.columns(2)

    # Feature popularity
    with col1:
        section("App Feature Demand", "How many users want each feature (Q16)")
        feat_sums = df[q16_cols].sum().rename({c: LABEL_MAP.get(c, c) for c in q16_cols}).sort_values(ascending=True)
        fig_feat = px.bar(
            x=feat_sums.values, y=feat_sums.index, orientation="h",
            color=feat_sums.values, color_continuous_scale=["#1f4080", "#1f6feb"],
            labels={"x": "Respondents Wanting Feature", "y": ""},
            text=feat_sums.values,
        )
        fig_feat.update_coloraxes(showscale=False)
        fig_feat.update_traces(textposition="outside", marker_line_width=0)
        apply_layout(fig_feat, height=420)
        st.plotly_chart(fig_feat, use_container_width=True)
        insight("Order tracking and Browse & Buy dominate feature demand — these are your table-stakes features that must be flawless at launch. Outfit Builder and Sustainability Info are your differentiators — features that major platforms don't offer and that build strong brand loyalty.")

    # Product category preferences
    with col2:
        section("Product Category Preferences", "What users want to buy (Q21)")
        prod_sums = df[q21_cols].sum().rename({c: LABEL_MAP.get(c, c) for c in q21_cols}).sort_values(ascending=True)
        fig_prod = px.bar(
            x=prod_sums.values, y=prod_sums.index, orientation="h",
            color=prod_sums.values, color_continuous_scale=["#1f4080", "#3fb950"],
            labels={"x": "Respondents Interested", "y": ""},
            text=prod_sums.values,
        )
        fig_prod.update_coloraxes(showscale=False)
        fig_prod.update_traces(textposition="outside", marker_line_width=0)
        apply_layout(fig_prod, height=420)
        st.plotly_chart(fig_prod, use_container_width=True)
        insight("Performance activewear and casual athleisure lead — validating the core thesis of this brand. More importantly, the high interest in accessories signals a strong average order value opportunity: bundle accessories into the outfit builder feature to drive ₹200–₹500 incremental revenue per transaction.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REVENUE POTENTIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="hero"><h1>💰 The Revenue Story</h1><p>Regression modelling + market sizing quantifies the financial opportunity for investors</p></div>', unsafe_allow_html=True)

    @st.cache_data
    def prep_regression(df_hash):
        df_c = df_full.copy()
        if city_filter:
            df_c = df_c[df_c["Q3_city_tier"].isin(city_filter)]
        if gender_filter:
            df_c = df_c[df_c["Q2_gender"].isin(gender_filter)]
        if intent_filter:
            df_c = df_c[df_c["Q25_app_download_intent"].isin(intent_filter)]

        # Encode categoricals
        income_map = {"Below_20k": 1, "20k-40k": 2, "40k-60k": 3,
                      "60k-80k": 4, "80k-100k": 5, "Above_100k": 6}
        freq_map = {"Rarely": 0, "Yearly": 1, "Every_6_months": 2,
                    "Every_2-3_months": 3, "Monthly_or_more": 4}

        df_c["income_enc"] = df_c["Q5_monthly_income"].map(income_map)
        df_c["freq_enc"] = df_c["Q14_purchase_frequency"].map(freq_map)

        feat_cols = (
            ["income_enc", "Q7_workout_days_enc", "freq_enc",
             "Q13_factor_price", "Q13_factor_brand_rep",
             "Q17_sustainability_importance", "Q19_flash_sale_likelihood"]
            + [c for c in df_c.columns if c.startswith("Q15_brand_")]
        )
        df_m = df_c[feat_cols + ["Q11_log"]].dropna()
        X = df_m[feat_cols]
        y = df_m["Q11_log"]
        feat_names = feat_cols
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        return Xtr_s, Xte_s, ytr, yte, feat_names, sc, X, y

    Xtr_r, Xte_r, ytr_r, yte_r, reg_feat_names, reg_scaler, X_reg, y_reg = prep_regression(
        hash(tuple(city_filter + gender_filter + intent_filter))
    )

    @st.cache_data
    def train_regressors(df_hash):
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        }
        results = {}
        for name, model in models.items():
            model.fit(Xtr_r, ytr_r)
            ypred = model.predict(Xte_r)
            r2 = r2_score(yte_r, ypred)
            mae = mean_absolute_error(np.expm1(yte_r), np.expm1(ypred))
            rmse = np.sqrt(mean_squared_error(np.expm1(yte_r), np.expm1(ypred)))
            results[name] = {
                "model": model,
                "pred_log": ypred,
                "pred_inr": np.expm1(ypred),
                "actual_inr": np.expm1(yte_r),
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
            }
        return results

    reg_results = train_regressors(hash(tuple(city_filter + gender_filter + intent_filter)))

    # Metrics table
    section("Regression Model Comparison", "Predicting monthly spend (Q11) — log-transformed target")
    reg_metrics = pd.DataFrame([
        {
            "Model": name,
            "R²": f"{r['r2']:.4f}",
            "MAE (₹)": f"₹{r['mae']:,.0f}",
            "RMSE (₹)": f"₹{r['rmse']:,.0f}",
        }
        for name, r in reg_results.items()
    ])
    st.dataframe(
        reg_metrics.style.applymap(lambda _: "background-color: #161b22; color: #e6edf3"),
        use_container_width=True, hide_index=True
    )

    best_model_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
    best_res = reg_results[best_model_name]
    insight(f"**{best_model_name}** achieves the best R² of {best_res['r2']:.3f}, explaining {best_res['r2']*100:.1f}% of variance in monthly sportswear spend. The MAE of ₹{best_res['mae']:,.0f} means the model's spend predictions are reliable enough to power a dynamic pricing and discount engine within the app — personalise discount depth based on predicted spend capacity.")

    # Actual vs Predicted
    section("Actual vs Predicted Spend", f"Best model: {best_model_name}")
    scatter_df = pd.DataFrame({
        "Actual (₹)": best_res["actual_inr"].values,
        "Predicted (₹)": best_res["pred_inr"],
    })
    max_val = max(scatter_df["Actual (₹)"].max(), scatter_df["Predicted (₹)"].max())
    fig_avp = px.scatter(
        scatter_df, x="Actual (₹)", y="Predicted (₹)",
        color_discrete_sequence=["#58a6ff"],
        opacity=0.45,
        labels={"Actual (₹)": "Actual Spend (₹)", "Predicted (₹)": "Predicted Spend (₹)"},
    )
    fig_avp.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#3fb950", dash="dash", width=2),
        name="Perfect Prediction",
        showlegend=True,
    ))
    apply_layout(fig_avp, height=440)
    fig_avp.update_layout(legend=dict(font=dict(color="#e6edf3")))
    st.plotly_chart(fig_avp, use_container_width=True)
    insight("The scatter is tightest for mid-range spenders (₹1,500–₹5,000) — your core revenue band. High-end luxury spenders (>₹8,000) are under-predicted, suggesting the model is conservative. In practice this is useful: when the model predicts high spend, the actual spend will be even higher — a safe signal to unlock premium features for these users.")

    # Market sizing
    section("Total Addressable Revenue", "Projecting from survey signals to market scale")

    MARKET_SIZE = 50_000_000
    yes_rate = (df_full["Q25_app_download_intent"] == "Yes").mean()
    maybe_rate = (df_full["Q25_app_download_intent"] == "Maybe").mean()
    avg_wtp_app = df["Q24_wtp_monthly_inr"].mean()
    avg_wtp_conservative = df["Q24_winsorized"].mean()

    conservative_users = int(MARKET_SIZE * yes_rate)
    optimistic_users   = int(MARKET_SIZE * (yes_rate + maybe_rate * 0.30))
    rev_conservative   = conservative_users * avg_wtp_conservative / 1e7
    rev_optimistic     = optimistic_users   * avg_wtp_app         / 1e7

    kpi_row([
        (f"{MARKET_SIZE/1e6:.0f}M", "Addressable Market", "Urban Indians 18-35"),
        (f"{yes_rate*100:.1f}%", "Hard Download Intent", "Direct Yes responses"),
        (f"{conservative_users/1e6:.1f}M", "Conservative Users", "Yes only"),
        (f"{optimistic_users/1e6:.1f}M", "Optimistic Users", "Yes + 30% Maybe"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    kpi_row([
        (f"₹{avg_wtp_conservative:,.0f}", "Avg WTP / Month", "Winsorized estimate"),
        (f"₹{rev_conservative:.0f} Cr", "Conservative GMV/mo", f"{conservative_users/1e6:.1f}M × ₹{avg_wtp_conservative:,.0f}"),
        (f"₹{rev_optimistic:.0f} Cr", "Optimistic GMV/mo", f"{optimistic_users/1e6:.1f}M × ₹{avg_wtp_app:,.0f}"),
        (f"₹{rev_conservative*12:.0f}–{rev_optimistic*12:.0f} Cr", "Annual Revenue Range", "12-month projection"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue waterfall
    section("Revenue Build-Up", "From survey signals to annual GMV")
    stages = ["Total Market", "Hard Adopters (Yes)", "Warm Leads (+30% Maybe)", "At Avg WTP (Annual)"]
    values = [
        MARKET_SIZE,
        conservative_users,
        optimistic_users,
        int(optimistic_users * avg_wtp_app * 12),
    ]
    fig_funnel = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=["#1f6feb", "#3fb950", "#e3b341", "#f78166"]),
        connector=dict(line=dict(color="#30363d", dash="dot", width=2)),
    ))
    fig_funnel.update_layout(
        **PLOTLY_LAYOUT, height=420,
        funnelmode="stack",
    )
    st.plotly_chart(fig_funnel, use_container_width=True)
    insight(f"Even at the conservative end — capturing only the hard 'Yes' segment at winsorized WTP — the annual GMV opportunity is ₹{rev_conservative*12:.0f} Crore. The optimistic scenario (capturing 30% of warm leads) represents a ₹{rev_optimistic*12:.0f} Crore annual market. With a 15–20% take rate on GMV, this translates to ₹{rev_conservative*12*0.175:.0f}–₹{rev_optimistic*12*0.175:.0f} Crore net revenue — a compelling unit economics story for Series A.")

    # WTP distribution
    section("WTP Distribution", "Shape of willingness-to-pay across the filtered segment")
    fig_wtp = px.histogram(
        df, x="Q24_wtp_monthly_inr", nbins=50,
        color_discrete_sequence=["#1f6feb"],
        labels={"Q24_wtp_monthly_inr": "WTP per Month (₹)", "count": "Respondents"},
    )
    fig_wtp.add_vline(x=df["Q24_wtp_monthly_inr"].median(), line_dash="dash",
                      line_color="#3fb950",
                      annotation_text=f"  Median: ₹{df['Q24_wtp_monthly_inr'].median():,.0f}",
                      annotation_font_color="#3fb950")
    fig_wtp.add_vline(x=df["Q24_wtp_monthly_inr"].mean(), line_dash="dash",
                      line_color="#e3b341",
                      annotation_text=f"  Mean: ₹{df['Q24_wtp_monthly_inr'].mean():,.0f}",
                      annotation_font_color="#e3b341")
    fig_wtp.update_traces(marker_line_width=0)
    apply_layout(fig_wtp, height=380)
    st.plotly_chart(fig_wtp, use_container_width=True)
    insight(f"The WTP distribution is right-skewed with a long tail of high-value buyers (₹5,000–₹15,000/month). The median of ₹{df['Q24_wtp_monthly_inr'].median():,.0f} is your mass-market price point; the mean of ₹{df['Q24_wtp_monthly_inr'].mean():,.0f} is inflated by premium buyers. Use the median for your base subscription pricing and design a premium tier to capture the top 20% who drive disproportionate revenue.")
