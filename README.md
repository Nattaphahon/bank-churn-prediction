# Bank Customer Churn Prediction

> Predicting which customers will leave — so the bank can act before they do.

**Dataset:** [Kaggle — Bank Customer Churn](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) | 10,000 customers · 14 features · 20.4% overall churn rate

---

## Business Problem

Retaining a bank customer costs 5–7× less than acquiring a new one. Industry estimates (Bain & Company) put the cost of losing and replacing one retail banking customer at **$200–300 USD** in acquisition cost alone — before factoring in lost lifetime value.

With a 20.4% churn rate across 10,000 customers, this dataset represents roughly **$400,000–$600,000 USD** in annual replacement cost. Most of it is preventable with early intervention.

This project builds a full ML pipeline to **identify high-risk customers before they churn**, enabling the retention team to run targeted campaigns instead of blanket marketing.

---

## Key Findings

- **Germany is a crisis market.** German customers churn at **32.4%** — nearly 2× France (16.2%) and Spain (16.7%). A geo-specific retention strategy is critical.
- **Product overload drives churn, not loyalty.** Customers with 3+ products churn at **83–100%**. SHAP confirms `NumOfProducts` and `Age` (40–60 range) as the top two predictors — a sign of mis-bundling, not over-engagement.
- **Inactivity is the earliest warning signal.** Non-active members churn at **26.9%** vs 14.3% for active ones. Catching inactivity before intent to leave forms is the highest-leverage intervention.

---

## Recommended Actions

| Priority | Action | Target Segment |
|----------|--------|----------------|
| High | Geo-targeted retention campaign | Germany customers, non-active |
| High | Product audit & simplification outreach | Customers with 3+ products |
| Medium | Re-engagement program (60-day inactivity trigger) | Non-active members aged 40–60 |
| Medium | CRM early-warning dashboard | All 1,822 High-Risk customers |

---

## Model Performance

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.387 | 0.700 | 0.499 | 0.777 |
| Random Forest (Baseline) | 0.787 | 0.445 | 0.568 | 0.855 |
| **Random Forest (Tuned)** | **0.576** | **0.693** | **0.629** | **0.864** |

Threshold is calibrated to maximize **Recall** — catching churners before they leave (minimizing False Negatives) is the business priority. A missed churner costs ~20,000 in CLV + CAC; a false alarm costs ~500 in retention outreach.

---

## Retention ROI (Quick Estimate)

Targeting the 1,822 high-risk customers with a 500/person campaign:

| Scenario | Customers Retained | Value Preserved | Net ROI |
|----------|--------------------|-----------------|---------|
| Conservative (15% success) | ~205 | 4.1M | **4.5×** |
| Base case (25% success) | ~342 | 6.8M | **7.5×** |
| Optimistic (35% success) | ~479 | 9.6M | **10.5×** |

Full sensitivity analysis is in the notebook (Layer 5 → Retention ROI).

---

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, scipy
- **Explainability** — SHAP (TreeExplainer, beeswarm + waterfall plots)
- **Visualization** — matplotlib, seaborn
- **Frontend Dashboard** — Chart.js (vanilla JS, no build step)

---

## Project Structure

```
bank_cus_churn/
├── backend/
│   ├── index.ipynb               # Full analysis (EDA → Model → SHAP → ROI)
│   └── data/
│       ├── Churn_Modelling.csv
│       ├── high_risk_customers.csv   # 1,822 high-risk customers + SHAP reasons
│       ├── shap_global.json
│       ├── model_scores.json
│       └── feature_importance.json
├── frontend/
│   ├── index.html                # Interactive dashboard
│   └── data/                     # JSON/CSV copies served to the dashboard
└── requirements.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the analysis notebook
cd backend
jupyter notebook index.ipynb

# 3. Launch the dashboard (from the project root)
python -m http.server 8000
# then open http://localhost:8000/frontend/index.html
```

> The dashboard loads data via `fetch()` from `frontend/data/`, so it must be served over HTTP — opening `frontend/index.html` directly as a `file://` path is blocked by the browser's CORS policy.
