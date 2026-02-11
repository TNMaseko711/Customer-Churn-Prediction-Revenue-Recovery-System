# Customer Churn Prediction & Revenue Recovery System

## Business Context & Objectives
A B2B SaaS subscription business with 10,000+ customers needs an early-warning system for churn. This project builds a complete analytics workflow that:
- Predicts churn risk using behavioral, billing, and support signals.
- Segments customers into a 2×2 risk/value matrix.
- Quantifies intervention ROI and projected recovered revenue.
- Provides a portfolio-quality dashboard and executive presentation.

## Project Deliverables
- **Notebook:** `notebooks/churn_prediction_recovery.ipynb` — data generation, EDA, feature engineering, modeling, ROI simulation.
- **Dashboard:** `app.py` — Streamlit app with churn trend, segments, high-risk list, ROI, and model metrics.
- **Presentation:** `reports/executive_presentation.md` — executive slide-by-slide content (easy export to PDF/PPT in your preferred tool).
- **Data & Models:** stored in `data/` and `models/` (generated when the notebook runs).

## Data Dictionary (Synthetic)
| Category | Fields |
| --- | --- |
| Customer Demographics | `company_size`, `industry`, `region`, `signup_date` |
| Usage Metrics | `monthly_active_users`, `feature_adoption_rate`, `login_frequency`, `api_calls` |
| Billing Data | `subscription_tier`, `mrr`, `payment_delay`, `failed_payments`, `contract_length` |
| Support Data | `support_tickets`, `support_response_time`, `support_satisfaction`, `feature_requests` |
| Engagement | `email_opens`, `webinar_attendance`, `documentation_views` |
| Target | `churned` (`Yes`/`No`) |

## Methodology & Model Selection
1. **Data generation:** 24 months of data with a realistic churn rate (8–12%) and declining engagement 2–3 months before churn.
2. **Feature engineering:** rolling trends, recency metrics, engagement score, health score.
3. **Modeling:** Logistic Regression, Random Forest, XGBoost.
4. **Imbalance handling:** class weighting.
5. **Evaluation:** precision, recall, F1, AUC-ROC, and confusion matrices.

## Key Insights & Recommendations
- **Churn risk drivers** typically include declining usage trends, rising support delays, and payment issues.
- **High Value/High Risk** customers justify personal outreach and incentives.
- **Projected annual recovery** exceeds **$1.5M** with **ROI > 5:1** (see notebook ROI calculator).

## How to Run
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the notebook**
   ```bash
   jupyter notebook notebooks/churn_prediction_recovery.ipynb
   ```
3. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```

## Technologies Used
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, plotly, Streamlit.

---

> **Note:** The notebook generates the dataset, charts, model artifacts, and ROI summaries. Run it once to populate the `data/`, `models/`, and `assets/` folders used by the dashboard and presentation.
