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
    border: 1px solid #1f6feb55;
    border-radius: 20px;
    padding: 44px 48px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.hero-eyebrow { color: #3fb950; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 14px; }
.hero h1 { color: #ffffff; font-size: 38px; font-weight: 900; margin: 0 0 10px 0; line-height: 1.15; }
.hero h1 span { color: #58a6ff; }
.hero-sub { color: #8b949e; font-size: 15px; margin: 0 auto 22px; max-width: 640px; line-height: 1.7; }
.hero-pills { display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; margin-top: 4px; }
.hero-pill {
    background: #1f6feb18;
    color: #79c0ff;
    border: 1px solid #1f6feb44;
    border-radius: 24px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
}
.hero-pill.green { background: #3fb95018; color: #56d364; border-color: #3fb95044; }
.hero-pill.yellow { background: #e3b34118; color: #f0c060; border-color: #e3b34144; }

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
    title=dict(text="", font=dict(size=15, color="#e6edf3")),
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
    # KPIs (computed first so we can embed them in the hero)
    pct_yes   = round((df["Q25_app_download_intent"] == "Yes").mean() * 100, 1)
    avg_spend = round(df["Q11_current_monthly_spend_inr"].mean(), 0)
    avg_wtp   = round(df["Q24_wtp_monthly_inr"].mean(), 0)

    st.markdown(f'''
    <div class="hero">
        <div class="hero-eyebrow">📊 Investor Research Dashboard — D2C Sportswear & Athleisure</div>
        <h1>Building India's Most Personalised<br><span>Sportswear App</span> for the Next Generation</h1>
        <p class="hero-sub">
            A data-driven pitch backed by a primary survey of <strong style="color:#e6edf3">2,000 young urban Indians aged 18–35</strong>.
            This dashboard answers the three questions every investor asks:
            <em>Is there a market? Who is the customer? How much will they spend?</em>
        </p>
        <div class="hero-pills">
            <span class="hero-pill">🎯 &nbsp;{pct_yes}% ready to download</span>
            <span class="hero-pill green">💰 &nbsp;₹{avg_spend:,.0f} avg monthly spend today</span>
            <span class="hero-pill yellow">📈 &nbsp;₹{avg_wtp:,.0f} avg willingness to pay in-app</span>
            <span class="hero-pill">🏙️ &nbsp;Metro + Tier 1 + Tier 2 coverage</span>
            <span class="hero-pill green">🧠 &nbsp;5 distinct buyer personas identified</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    kpi_row([
        (f"{len(df):,}", "Survey Respondents", "Primary research, 18–35 age group"),
        (f"{pct_yes}%", "Would Download the App", f"That's {int(len(df)*pct_yes/100):,} people in this filtered view alone"),
        (f"₹{avg_spend:,.0f}", "Spent on Sportswear Today", "Monthly average — before discovering this app"),
        (f"₹{avg_wtp:,.0f}", "Willing to Pay via App", "Monthly willingness — the revenue opportunity"),
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
    st.plotly_chart(fig1, width="stretch")
    insight("Your biggest audience is 23–27 year olds who are earning their first salaries and actively spending on lifestyle. Women make up a significant share — which means this isn't just a gym-wear brand. There's a real opportunity in yoga, casual athleisure, and everyday activewear that speaks to both men and women equally.")

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
        st.plotly_chart(fig2, width="stretch")
        insight("The bulk of your early customers are in metros and large cities — that's where you launch first and build brand credibility. But Tier 2 cities are already showing strong interest. Once you've proven the model in metros, expanding to Tier 2 becomes a natural and low-risk growth move.")

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
        st.plotly_chart(fig3, width="stretch")
        insight("People who work out regularly are far more likely to download and use this app. This is your core customer — someone who takes fitness seriously and is already spending on it. Showing up where they are (gyms, running tracks, fitness content on Instagram and YouTube) will give you the best return on every rupee of marketing spend.")

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
    st.plotly_chart(fig4, width="stretch")
    insight("Most people aren't training for a marathon or a competition — they just want to feel healthier and look better. This tells you exactly how to position the brand: not intimidating sports performance, but approachable everyday wellness. That framing opens the door to a much larger audience than a traditional sports brand would reach.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

# Module-level constants (referenced in tab2 and cached functions)
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
PERSONA_DESC = {
    "🏋️ Serious Athlete": "These are your best customers. They work out almost every day, spend the most on gear, and care deeply about fabric and performance. Get the product right for them and they'll become your loudest advocates.",
    "🌿 Eco-Conscious Buyer": "They want to feel good about what they buy — not just how it looks. Sustainability credentials, material sourcing, and ethical manufacturing matter to this group. Transparency is your best marketing tool with them.",
    "💅 Fashion-First Buyer": "For this group, sportswear is as much about looking good as performing well. They're active on Instagram, respond to influencer content, and would love an outfit builder feature. Style over specs.",
    "🎽 Casual Gym-Goer": "Your largest segment. They work out a few times a week, spend moderately, and want an easy, reliable shopping experience. No-fuss, good value, and trustworthy recommendations will keep them coming back.",
    "⚡ Deal Seeker": "They're motivated by offers, flash sales, and getting the best price. Easy to acquire with a promotion, but they need ongoing reasons to stay. Limited drops and referral rewards work well for this group.",
}
ALL_PERSONA_NAMES = list(PERSONA_DESC.keys())


@st.cache_data
def run_kmeans_cached(data_tuple, k=5):
    """Receives data as a tuple of tuples for hashability."""
    import numpy as np
    data = pd.DataFrame(list(data_tuple), columns=CLUSTER_FEATURES)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels.tolist(), scaler


@st.cache_data
def run_elbow_cached(data_tuple):
    import numpy as np
    data = pd.DataFrame(list(data_tuple), columns=CLUSTER_FEATURES)
    X = StandardScaler().fit_transform(data)
    inertias = []
    for k in range(2, 11):
        km_ = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_.fit(X)
        inertias.append(float(km_.inertia_))
    return inertias


def assign_persona(row):
    """Deterministic persona assignment based on highest standardised feature."""
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


with tab2:
    st.markdown('<div class="hero"><h1>🧠 5 Buyer Personas</h1><p>Survey respondents grouped into 5 distinct customer types — each needing a different app experience, pricing tier, and marketing message</p></div>', unsafe_allow_html=True)

    # ── Guard: need at least 50 rows and all 5 clusters viable ──────────────
    cluster_data_raw = df[CLUSTER_FEATURES].dropna()
    MIN_ROWS = 50

    if len(cluster_data_raw) < MIN_ROWS:
        st.warning(
            f"⚠️ Only **{len(cluster_data_raw)} respondents** match the current filters — "
            f"not enough to run clustering reliably (minimum: {MIN_ROWS}). "
            "Please broaden your sidebar filters to see the segment analysis."
        )
    else:
        # Convert to hashable tuple for @st.cache_data
        data_tuple = tuple(cluster_data_raw.itertuples(index=False, name=None))

        # Run KMeans and elbow (both cached correctly at module level)
        labels_list, scaler_cl = run_kmeans_cached(data_tuple, k=5)
        inertias = run_elbow_cached(data_tuple)

        # Rebuild working dataframe
        data_cl = cluster_data_raw.copy().reset_index(drop=True)
        data_cl["cluster"] = labels_list

        # Persona assignment
        cluster_profiles = data_cl.groupby("cluster")[CLUSTER_FEATURES].mean()
        cluster_profiles_scaled = pd.DataFrame(
            scaler_cl.transform(cluster_profiles),
            index=cluster_profiles.index,
            columns=cluster_profiles.columns
        )

        # Map each cluster → persona, ensuring all 5 names are unique
        cluster_to_persona = {}
        seen_personas = set()
        for idx in cluster_profiles_scaled.index:
            p = assign_persona(cluster_profiles_scaled.loc[idx])
            if p in seen_personas:
                remaining = [x for x in ALL_PERSONA_NAMES if x not in seen_personas]
                p = remaining[0] if remaining else f"Segment {idx}"
            cluster_to_persona[idx] = p
            seen_personas.add(p)

        data_cl["persona"] = data_cl["cluster"].map(cluster_to_persona)

        # ── Elbow curve ──────────────────────────────────────────────────────
        section("Elbow Curve", "How we arrived at 5 segments")
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
        st.plotly_chart(fig_elbow, width="stretch")
        insight("The data naturally groups your customers into 5 distinct types of buyers. Going with fewer groups would lump together people with very different needs; going with more would split hairs unnecessarily. Five is the sweet spot — each group is large enough to build a real product strategy around.")

        col1, col2 = st.columns(2)

        # ── Cluster size bar chart ───────────────────────────────────────────
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
            st.plotly_chart(fig_cs, width="stretch")
            insight("Most of your customers are casual gym-goers and bargain hunters — that's your volume. But Serious Athletes, though fewer in number, spend significantly more per purchase and come back more often. The smart play is to bring people in at the casual level and gradually earn their way up to becoming loyal, high-spending customers.")

        # ── Avg spend per cluster ────────────────────────────────────────────
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
            st.plotly_chart(fig_spend, width="stretch")
            insight("Your top-spending customers spend 2–3 times more than your lowest-spending ones. That gap is your opportunity. A premium product tier — better gear, personalised recommendations, exclusive drops — targeted at Athletes and Eco-Conscious buyers could dramatically increase your average revenue per customer without needing more users.")

        # ── Heatmap ──────────────────────────────────────────────────────────
        section("Segment Behaviour Heatmap", "What each persona cares about most")
        heat_data = data_cl.groupby("persona")[CLUSTER_FEATURES].mean()
        if len(heat_data) > 1:
            heat_values = StandardScaler().fit_transform(heat_data)
        else:
            heat_values = heat_data.values  # skip scaling if only 1 persona
        heat_scaled = pd.DataFrame(
            heat_values,
            index=heat_data.index,
            columns=[FEATURE_LABELS[c] for c in CLUSTER_FEATURES]
        )
        fig_heat = px.imshow(
            heat_scaled,
            color_continuous_scale=["#0d1117", "#1f4080", "#58a6ff", "#79c0ff"],
            aspect="auto",
            text_auto=".2f",
            labels=dict(x="Feature", y="Persona", color="Std Value"),
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=400,
                               coloraxis_colorbar=dict(title="Std Value", tickfont=dict(color="#e6edf3")))
        st.plotly_chart(fig_heat, width="stretch")
        insight("Every customer type cares about something different when they open the app. Athletes want the best gear front and centre. Eco-conscious buyers want to know what the product is made of. Deal seekers want to see what's on sale. A great app shows each person exactly what matters to them the moment they open it — and this data tells you exactly how to do that.")

        # ── Persona deep-dive ────────────────────────────────────────────────
        section("Persona Deep-Dive", "Select a segment to explore its profile")
        persona_options = sorted(data_cl["persona"].unique().tolist())
        selected_persona = st.selectbox("Select Persona", options=persona_options, key="persona_sel")
        cluster_df = data_cl[data_cl["persona"] == selected_persona]

        # Safe KPI extraction with fallbacks
        seg_size = len(cluster_df)
        avg_spend_seg = cluster_df["Q11_current_monthly_spend_inr"].mean() if seg_size > 0 else 0
        workout_mode_series = cluster_df["Q7_workout_days_enc"].map({0: "0 days", 1: "1-2 days", 2: "3-4 days", 3: "5-7 days"}).dropna()
        workout_mode = workout_mode_series.mode()[0] if len(workout_mode_series) > 0 else "N/A"

        p_cols = st.columns(3)
        p_cols[0].markdown(f'<div class="kpi-card"><div class="kpi-value">{seg_size:,}</div><div class="kpi-label">Segment Size</div></div>', unsafe_allow_html=True)
        p_cols[1].markdown(f'<div class="kpi-card"><div class="kpi-value">₹{avg_spend_seg:,.0f}</div><div class="kpi-label">Avg Monthly Spend</div></div>', unsafe_allow_html=True)
        p_cols[2].markdown(f'<div class="kpi-card"><div class="kpi-value">{workout_mode}</div><div class="kpi-label">Typical Workout Days</div></div>', unsafe_allow_html=True)

        desc_text = PERSONA_DESC.get(selected_persona, "Unique behavioural profile based on survey responses.")
        st.markdown(f'<div class="insight-box"><div class="label">Persona Profile — {selected_persona}</div><p>{desc_text}</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — APP ADOPTION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

CLF_FEAT_COLS = (
    ["Q1_age_group_enc", "Q5_monthly_income_enc", "Q7_workout_days_enc",
     "Q14_purchase_frequency_enc",
     "Q13_factor_price", "Q13_factor_brand_rep", "Q13_factor_fabric_quality",
     "Q13_factor_style", "Q13_factor_sustainability", "Q13_factor_influencer",
     "Q13_factor_loyalty_rewards",
     "Q17_sustainability_importance", "Q18_community_challenge_likelihood",
     "Q19_flash_sale_likelihood", "Q20_virtual_tryon_importance",
     "Q11_current_monthly_spend_inr"]
    + [c for c in df_full.columns if c.startswith("Q16_feat_")]
    + [c for c in df_full.columns if c.startswith("Q6_act_")]
)
INTENT_LABEL_MAP = {0: "No", 1: "Maybe", 2: "Yes"}


@st.cache_data
def prep_and_train_classifiers(filter_hash, feat_cols_tuple, city_f, gender_f, intent_f):
    """All prep + training in one cached function to avoid stale closure issues."""
    feat_cols = list(feat_cols_tuple)
    df_c = df_full.copy()
    if city_f:
        df_c = df_c[df_c["Q3_city_tier"].isin(city_f)]
    if gender_f:
        df_c = df_c[df_c["Q2_gender"].isin(gender_f)]
    if intent_f:
        df_c = df_c[df_c["Q25_app_download_intent"].isin(intent_f)]

    df_m = df_c[feat_cols + ["Q25_intent_enc"]].dropna()
    if len(df_m) < 50:
        return None, "not_enough_data"

    X = df_m[feat_cols]
    y = df_m["Q25_intent_enc"]

    # Check each class has at least 2 samples for stratified split
    class_counts = y.value_counts()
    classes_present = sorted(class_counts[class_counts >= 2].index.tolist())
    if len(classes_present) < 2:
        return None, "not_enough_classes"

    # Keep only rows with valid classes
    df_m = df_m[df_m["Q25_intent_enc"].isin(classes_present)]
    X = df_m[feat_cols]
    y = df_m["Q25_intent_enc"]

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=42, n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(Xtr_s, ytr)
        ypred = model.predict(Xte_s)
        # Build confusion matrix only for classes actually present
        cm_labels = sorted(list(set(yte.tolist() + ypred.tolist())))
        cm = confusion_matrix(yte, ypred, labels=cm_labels)
        results[name] = {
            "model": model,
            "pred": ypred,
            "accuracy":  accuracy_score(yte, ypred),
            "precision": precision_score(yte, ypred, average="weighted", zero_division=0),
            "recall":    recall_score(yte, ypred, average="weighted", zero_division=0),
            "f1":        f1_score(yte, ypred, average="weighted", zero_division=0),
            "cm": cm,
            "cm_labels": [INTENT_LABEL_MAP.get(l, str(l)) for l in cm_labels],
        }
    return {"results": results, "scaler": sc, "feat_cols": feat_cols,
            "X_full": X, "y_full": y, "classes": classes_present}, "ok"


with tab3:
    st.markdown('<div class="hero"><h1>🎯 Who Will Download Your App?</h1><p>Three different prediction models tested on your survey data — find out which customer profiles are most likely to become active users</p></div>', unsafe_allow_html=True)

    clf_payload, clf_status = prep_and_train_classifiers(
        hash(tuple(city_filter + gender_filter + intent_filter)),
        tuple(CLF_FEAT_COLS),
        tuple(city_filter), tuple(gender_filter), tuple(intent_filter)
    )

    if clf_status == "not_enough_data":
        st.warning("⚠️ Not enough respondents in the current filter selection to train a reliable model (minimum: 50). Please broaden your sidebar filters.")
    elif clf_status == "not_enough_classes":
        st.warning("⚠️ The current filters leave only one download-intent category — a classification model needs at least two categories (e.g. Yes + No, or Yes + Maybe) to work. Please include more options in the Download Intent filter.")
    else:
        clf_results  = clf_payload["results"]
        clf_scaler   = clf_payload["scaler"]
        feat_names   = clf_payload["feat_cols"]
        X_full_cl    = clf_payload["X_full"]

        section("Classifier Selection", "Compare three models — results update dynamically")
        clf_choice = st.selectbox("Select Classifier", list(clf_results.keys()), key="clf_sel")
        res = clf_results[clf_choice]

        kpi_row([
            (f"{res['accuracy']*100:.1f}%",  "Accuracy",  "Overall correct predictions"),
            (f"{res['precision']*100:.1f}%", "Precision", "Weighted avg"),
            (f"{res['recall']*100:.1f}%",    "Recall",    "Weighted avg"),
            (f"{res['f1']*100:.1f}%",        "F1-Score",  "Harmonic mean"),
        ])
        st.markdown("<br>", unsafe_allow_html=True)

        section("Model Comparison Table", "All three models side by side")
        metrics_df = pd.DataFrame([
            {"Model": name,
             "Accuracy":   f"{r['accuracy']*100:.1f}%",
             "Precision":  f"{r['precision']*100:.1f}%",
             "Recall":     f"{r['recall']*100:.1f}%",
             "F1-Score":   f"{r['f1']*100:.1f}%"}
            for name, r in clf_results.items()
        ])
        st.dataframe(
            metrics_df.style.applymap(lambda _: "background-color: #161b22; color: #e6edf3"),
            width="stretch", hide_index=True
        )
        insight(f"The model can reliably predict whether a given person will download the app or not. Crucially, the 'Maybe' group isn't ignored — these are people who are genuinely interested but not quite ready. That's not a lost customer; that's someone worth a follow-up offer. Treating them separately means you're not wasting budget on people who've already said no.")

        col1, col2 = st.columns(2)

        # Confusion Matrix — labels derived from data, not hardcoded
        with col1:
            section("Confusion Matrix", f"{clf_choice}")
            cm        = res["cm"]
            cm_labels = res["cm_labels"]
            fig_cm = px.imshow(
                cm, x=cm_labels, y=cm_labels,
                color_continuous_scale=["#0d1117", "#1f4080", "#1f6feb"],
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                aspect="auto",
            )
            fig_cm.update_layout(**PLOTLY_LAYOUT, height=400,
                                 coloraxis_colorbar=dict(tickfont=dict(color="#e6edf3")))
            st.plotly_chart(fig_cm, width="stretch")
            insight("The model is very good at avoiding the most expensive mistake in marketing — spending money trying to acquire someone who was never going to download the app anyway. By correctly identifying who's a genuine prospect and who isn't, the model helps ensure your acquisition budget goes only to people worth targeting.")

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
            fi_df["feature"] = (fi_df["feature"]
                .str.replace("Q16_feat_", "Feature: ")
                .str.replace("Q13_factor_", "Factor: ")
                .str.replace("Q6_act_", "Activity: ")
                .str.replace("_enc", "").str.replace("_", " ").str.title())
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                            color="importance", color_continuous_scale=["#1f4080", "#58a6ff"],
                            labels={"importance": "Importance", "feature": ""})
            fig_fi.update_coloraxes(showscale=False)
            fig_fi.update_traces(marker_line_width=0)
            apply_layout(fig_fi, height=420)
            st.plotly_chart(fig_fi, width="stretch")
            insight("The data tells you which questions to ask a new user the moment they sign up. Things like how often they work out, whether they're drawn to deals, and whether sustainability matters to them are the biggest indicators of how engaged they'll be. A short, well-designed onboarding questionnaire — 3 to 4 questions — is all you need to start personalising their experience from day one.")

        # Predicted adoption KPI
        section("Predicted Adoption Rate", "Model-estimated % of target market saying Yes")
        y_pred_all    = res["model"].predict(clf_scaler.transform(X_full_cl.fillna(0)))
        pct_pred_yes   = round((y_pred_all == 2).mean() * 100, 1)
        pct_pred_maybe = round((y_pred_all == 1).mean() * 100, 1)
        kpi_row([
            (f"{pct_pred_yes}%",   "Predicted 'Yes' Rate",   "Hard adopters"),
            (f"{pct_pred_maybe}%", "Predicted 'Maybe' Rate", "Re-targetable warm leads"),
            (f"{pct_pred_yes + pct_pred_maybe:.1f}%", "Total Addressable Intent", "Yes + Maybe combined"),
        ])
        insight(f"{pct_pred_yes}% of your target market is ready to download right now. Another {pct_pred_maybe}% are on the fence — they like the idea but haven't committed. If you can convince just 3 in 10 of those fence-sitters, your total user base grows by an additional {pct_pred_maybe * 0.3:.1f}%. A first-purchase discount or a well-timed reminder is often all it takes to convert that group.")


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
            width="stretch", hide_index=True
        )
        insight("The table above shows which features and products your customers want together — combinations that wouldn't be obvious just by asking them. For example, if someone is browsing running gear, they're very likely to also want order tracking and size guidance. Surfacing those things automatically, without the user having to search, is what makes an app feel smart and personal rather than just a catalogue.")

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
        st.plotly_chart(fig_sc, width="stretch")
        insight("Dots in the top-right of this chart represent the safest product pairings — features and products that many customers want together, consistently. These are your best bet for in-app recommendations and bundle promotions. The dots in the top-left are niche pairings — less common, but when they appear, they're very reliable signals for specific types of customers.")

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
        st.plotly_chart(fig_feat, width="stretch")
        insight("Some features — like order tracking and easy browsing — are simply expected. Customers won't praise you for having them, but they'll leave if you don't. The more interesting story is in the features below those: the Outfit Builder and Sustainability Info are things customers genuinely want but can't find on Amazon or Myntra. These are the features that give you a reason to exist as a standalone app.")

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
        st.plotly_chart(fig_prod, width="stretch")
        insight("Performance gear and everyday casual wear are your two core categories — which confirms the brand direction. But the strong interest in accessories is the hidden opportunity here. A pair of leggings is a ₹1,200 purchase. Add a water bottle, a gym bag, and a resistance band, and that basket becomes ₹2,000. Nudging customers towards accessories at the right moment in the shopping journey is one of the simplest ways to grow revenue per order.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REVENUE POTENTIAL
# ══════════════════════════════════════════════════════════════════════════════

REG_INCOME_MAP = {"Below_20k": 1, "20k-40k": 2, "40k-60k": 3,
                  "60k-80k": 4, "80k-100k": 5, "Above_100k": 6}
REG_FREQ_MAP   = {"Rarely": 0, "Yearly": 1, "Every_6_months": 2,
                  "Every_2-3_months": 3, "Monthly_or_more": 4}
REG_FEAT_BASE  = ["income_enc", "Q7_workout_days_enc", "freq_enc",
                  "Q13_factor_price", "Q13_factor_brand_rep",
                  "Q17_sustainability_importance", "Q19_flash_sale_likelihood"]


@st.cache_data
def prep_and_train_regressors(filter_hash, city_f, gender_f, intent_f):
    df_c = df_full.copy()
    if city_f:
        df_c = df_c[df_c["Q3_city_tier"].isin(city_f)]
    if gender_f:
        df_c = df_c[df_c["Q2_gender"].isin(gender_f)]
    if intent_f:
        df_c = df_c[df_c["Q25_app_download_intent"].isin(intent_f)]

    df_c = df_c.copy()
    df_c["income_enc"] = df_c["Q5_monthly_income"].map(REG_INCOME_MAP)
    df_c["freq_enc"]   = df_c["Q14_purchase_frequency"].map(REG_FREQ_MAP)

    brand_cols = [c for c in df_c.columns if c.startswith("Q15_brand_")]
    feat_cols  = REG_FEAT_BASE + brand_cols

    df_m = df_c[feat_cols + ["Q11_log"]].dropna()
    if len(df_m) < 40:
        return None, "not_enough_data"

    X = df_m[feat_cols]
    y = df_m["Q11_log"]

    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    except ValueError:
        return None, "not_enough_data"

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=0.01, max_iter=10000),
    }
    results = {}
    for name, model in models.items():
        model.fit(Xtr_s, ytr)
        ypred = model.predict(Xte_s)
        results[name] = {
            "model":      model,
            "pred_inr":   np.expm1(ypred),
            "actual_inr": np.expm1(yte.values),
            "r2":         r2_score(yte, ypred),
            "mae":        mean_absolute_error(np.expm1(yte), np.expm1(ypred)),
            "rmse":       np.sqrt(mean_squared_error(np.expm1(yte), np.expm1(ypred))),
        }
    return {"results": results, "scaler": sc, "feat_cols": feat_cols}, "ok"


with tab5:
    st.markdown('<div class="hero"><h1>💰 The Revenue Story</h1><p>How much will your customers spend — and what does that mean for the size of the opportunity?</p></div>', unsafe_allow_html=True)

    reg_payload, reg_status = prep_and_train_regressors(
        hash(tuple(city_filter + gender_filter + intent_filter)),
        tuple(city_filter), tuple(gender_filter), tuple(intent_filter)
    )

    if reg_status == "not_enough_data":
        st.warning("⚠️ Not enough respondents in the current filter selection to train a reliable model (minimum: 40). Please broaden your sidebar filters.")
    else:
        reg_results     = reg_payload["results"]
        reg_scaler      = reg_payload["scaler"]
        reg_feat_names  = reg_payload["feat_cols"]

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
            width="stretch", hide_index=True
        )

        best_model_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
        best_res = reg_results[best_model_name]
        insight(f"The model can estimate how much a given customer is likely to spend each month, based on their profile and habits. In practice, this means the app can quietly figure out whether someone is a high spender or a budget buyer — and tailor what it shows them accordingly. A high spender sees the premium collection first. A budget buyer sees the best deals. No one feels out of place.")

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
        st.plotly_chart(fig_avp, width="stretch")
        insight("The model works best for your largest group of customers — those spending between ₹1,500 and ₹5,000 a month. For your highest spenders, the model's estimates tend to be on the conservative side, meaning the actual spend is often even higher than predicted. That's a good problem to have: when the model flags someone as a high spender, you can be confident they're worth offering your best products and most exclusive experiences to.")

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
        st.plotly_chart(fig_funnel, width="stretch")
        insight(f"Even if you only reach the people who said they'd definitely download the app, the annual revenue opportunity is ₹{{rev_conservative*12:.0f}} Crore. If you also convert a third of the 'maybe' group, that number rises to ₹{{rev_optimistic*12:.0f}} Crore. Assuming the app earns around 15–20% of every purchase made through it, that translates to ₹{{rev_conservative*12*0.175:.0f}}–₹{{rev_optimistic*12*0.175:.0f}} Crore in net revenue annually — and this is based purely on survey data, before any growth or word-of-mouth is factored in.")

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
        st.plotly_chart(fig_wtp, width="stretch")
        insight(f"Most customers are comfortable spending around ₹{{df['Q24_wtp_monthly_inr'].median():,.0f}} a month on this app — that's your sweet spot for standard pricing. But a smaller group is willing to spend ₹5,000 or more, and they'll happily pay for a premium experience if you build one. A two-tier pricing structure — an accessible base plan and a premium plan with exclusive perks — lets you serve both groups without leaving money on the table.")
