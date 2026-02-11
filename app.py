import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load

from src.churn_utils import (
    FEATURE_COLUMNS,
    add_feature_engineering,
    generate_synthetic_churn_data,
    validate_dataset,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "churn_dataset.csv"
HIGH_RISK_PATH = BASE_DIR / "data" / "high_risk_customers.csv"
ROI_PATH = BASE_DIR / "data" / "roi_summary.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

st.set_page_config(page_title="Churn & Revenue Recovery Dashboard", layout="wide")

st.title("Customer Churn Prediction & Revenue Recovery")
st.caption("Interactive portfolio dashboard for a B2B SaaS subscription business.")

@st.cache_data

def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["month", "signup_date"])
    else:
        df = generate_synthetic_churn_data()
    df = add_feature_engineering(df)
    validate_dataset(df)
    return df


def load_model():
    if MODEL_PATH.exists():
        return load(MODEL_PATH)
    return None


df = load_data()

left, right = st.columns(2)

with left:
    st.subheader("Overall Churn Rate Trend")
    churn_trend = df.groupby("month")["churned"].apply(lambda x: (x == "Yes").mean()).reset_index()
    fig = px.line(churn_trend, x="month", y="churned", labels={"churned": "Churn Rate"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Segment Breakdown (Latest Month)")
    latest_month = df["month"].max()
    latest = df[df["month"] == latest_month].copy()
    model = load_model()
    if model is not None:
        latest["churn_risk"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    else:
        latest["churn_risk"] = 1 - latest["health_score"]
    latest["customer_value"] = pd.qcut(latest["mrr"], 2, labels=["Low", "High"])
    latest["risk_segment"] = pd.cut(latest["churn_risk"], [0, 0.5, 1.0], labels=["Low", "High"])
    latest["segment"] = latest["risk_segment"].astype(str) + " Risk / " + latest["customer_value"].astype(str) + " Value"
    segment_counts = latest["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]
    fig = px.bar(segment_counts, x="count", y="segment", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("High-Risk Customer List")
if HIGH_RISK_PATH.exists():
    high_risk = pd.read_csv(HIGH_RISK_PATH)
else:
    model = load_model()
    latest = df[df["month"] == latest_month].copy()
    if model is not None:
        latest["churn_risk"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
        high_risk = latest.sort_values("churn_risk", ascending=False).head(25)
    else:
        high_risk = latest.sort_values("health_score").head(25)

st.dataframe(high_risk[[
    "customer_id",
    "company_size",
    "industry",
    "region",
    "subscription_tier",
    "mrr",
    "health_score",
]].head(25))

st.subheader("Projected Revenue Impact")
if ROI_PATH.exists():
    roi = pd.read_csv(ROI_PATH)
    st.dataframe(roi)
else:
    st.info("Run the notebook to generate ROI summaries.")

st.subheader("Model Performance")
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(metrics_df)
else:
    st.info("Model metrics will appear after training in the notebook.")

st.subheader("Retention Strategy")
strategy = pd.DataFrame(
    [
        {"Segment": "High Value / High Risk", "Playbook": "Personal outreach + discount", "Cost": 500},
        {"Segment": "High Value / Low Risk", "Playbook": "Quarterly check-ins", "Cost": 50},
        {"Segment": "Low Value / High Risk", "Playbook": "Automated email campaign", "Cost": 10},
        {"Segment": "Low Value / Low Risk", "Playbook": "No intervention", "Cost": 0},
    ]
)

st.table(strategy)
