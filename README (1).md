# ⚡ SportWear D2C — Investor Intelligence Dashboard

A data-driven Streamlit dashboard built to pitch a D2C sportswear & athleisure mobile app to investors. The dashboard turns survey responses from 2,000 young urban Indians (18–35) into a compelling, interactive business story across five analytical lenses.

---

## 📁 Repository Structure

```
├── app.py                            # Main Streamlit application (991 lines)
├── sportswear_survey_cleaned.csv     # Cleaned survey dataset (2,000 rows, 92 columns)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📊 Dashboard Tabs

| Tab | Story | Techniques Used |
|-----|-------|----------------|
| **Market Overview** | Here is the market opportunity | Descriptive stats, KPI cards, grouped bar, pie, stacked bar charts |
| **Customer Segments** | Here are your 5 buyer personas | K-Means clustering (K=5), StandardScaler, elbow curve, heatmap |
| **App Adoption** | This is who will download your app | Logistic Regression, Decision Tree, Random Forest, confusion matrix, feature importance |
| **Feature & Product Insights** | Here is what your customers want | Apriori algorithm, association rules, support/confidence/lift analysis |
| **Revenue Potential** | Here is how much your customers will spend | Linear Regression, Ridge, Lasso, TAM funnel, WTP distribution |

---

## 🔧 Global Sidebar Filters

All five tabs respond dynamically to three sidebar filters:
- **City Tier** — Metro, Tier 1, Tier 2, Tier 3
- **Gender** — Male, Female, Non-binary, Prefer not to say
- **App Download Intent** — Yes, Maybe, No

Selecting a filter subset re-runs all charts and ML models on the filtered data in real time.

---

## 📋 Dataset Overview

The cleaned dataset (`sportswear_survey_cleaned.csv`) was derived from a 25-question synthetic survey designed for:

| ML Task | Target Variable | Method |
|---------|----------------|--------|
| Classification | `Q25_app_download_intent` (Yes / Maybe / No) | Logistic Regression, Decision Tree, Random Forest |
| Clustering | Persona segmentation | K-Means (K=5) |
| Regression | `Q11_current_monthly_spend_inr` (log-transformed) | Linear, Ridge, Lasso |
| Association Mining | Q16 features + Q21 products + Q6 activities | Apriori |

### Noise layers baked into raw data (cleaned before use):
- **5% binary noise** — random flips in multi-select columns (Q6, Q9, Q12, Q13, Q15, Q16)
- **5% Likert noise** — random 1–5 replacements in Q17–Q20
- **3% outliers in Q11** — luxury buyers (₹9,000–₹16,000) requiring log-transformation
- **4% outliers in Q24** — extreme WTP skeptics (₹100–₹400)
- **3% label noise in Q25** — randomly overwritten classification targets

### Cleaning applied:
| Action | Rows Affected |
|--------|--------------|
| Q6 none + non-zero workout days → corrected | 32 |
| Q9 all-zero goals → flagged | 25 |
| Q11 luxury outliers → winsorized + log-transformed | 101 |
| Q24 skeptic outliers → winsorized + log-transformed | 149 |
| Q25 label contradictions (≥9 features + "No") → corrected to "Maybe" | 25 |
| Ordinal encodings added (Q1, Q5, Q7, Q14, Q25) | All rows |

---

## 🎨 Design System

- **Theme:** Dark (`#0d1117` base, `#161b22` surface, `#1f6feb` accent)
- **Charts:** Plotly Express + Plotly Graph Objects (`plotly_dark` template)
- **Layout:** Wide mode, `st.tabs()`, custom CSS for KPI cards, section headers, and insight boxes

---

## 💡 Business Context

This dashboard was built to support a pitch for a D2C mobile application for sportswear and athleisure targeting urban Indians aged 18–35. The app concept includes:

1. Personalised activewear recommendations based on fitness goals
2. Browse and purchase sportswear, athleisure, and accessories
3. Exclusive member discounts and loyalty rewards
4. Sustainability and fabric certification information
5. Workout outfit builder (mix and match)
6. Size guide and virtual try-on
7. Order tracking and easy returns
8. Community features (fitness challenges, user reviews)
9. Flash sales and limited edition drops
10. Brand collaborations and athlete endorsements content

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
scikit-learn>=1.4.0
mlxtend>=0.23.0
```

---

## 👤 Author

Built as part of a market research and investor pitch preparation project.

---

## 📄 License

This project is for academic and demonstration purposes.
