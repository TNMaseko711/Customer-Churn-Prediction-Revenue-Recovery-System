import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

INDUSTRIES = [
    "FinTech",
    "HealthTech",
    "E-Commerce",
    "SaaS",
    "Manufacturing",
    "EdTech",
    "Logistics",
]
REGIONS = ["North America", "Europe", "APAC", "LATAM"]
SUBSCRIPTION_TIERS = ["Basic", "Pro", "Enterprise"]
COMPANY_SIZES = ["SMB", "Mid-Market", "Enterprise"]

SIZE_TO_MRR = {"SMB": (300, 800), "Mid-Market": (800, 2500), "Enterprise": (2500, 12000)}

FEATURE_COLUMNS = [
    "company_size",
    "industry",
    "region",
    "subscription_tier",
    "contract_length",
    "mrr",
    "monthly_active_users",
    "feature_adoption_rate",
    "login_frequency",
    "api_calls",
    "payment_delay",
    "failed_payments",
    "support_tickets",
    "support_response_time",
    "support_satisfaction",
    "feature_requests",
    "email_opens",
    "webinar_attendance",
    "documentation_views",
    "monthly_active_users_3m_trend",
    "feature_adoption_rate_3m_trend",
    "login_frequency_3m_trend",
    "api_calls_3m_trend",
    "email_opens_3m_trend",
    "documentation_views_3m_trend",
    "recency_days",
    "engagement_score",
    "health_score",
]


def generate_synthetic_churn_data(
    n_customers: int = 12000,
    months: int = 24,
    start_date: datetime | None = None,
    churn_rate: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic churn dataset with realistic decline patterns before churn."""
    rng = np.random.default_rng(random_state)
    if start_date is None:
        start_date = datetime(2023, 1, 1)

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]
    signup_dates = pd.to_datetime(
        rng.choice(pd.date_range("2021-01-01", "2023-12-31"), size=n_customers)
    )
    base_company_size = rng.choice(COMPANY_SIZES, size=n_customers, p=[0.55, 0.3, 0.15])
    base_industry = rng.choice(INDUSTRIES, size=n_customers)
    base_region = rng.choice(REGIONS, size=n_customers, p=[0.45, 0.25, 0.2, 0.1])
    base_tier = rng.choice(SUBSCRIPTION_TIERS, size=n_customers, p=[0.5, 0.35, 0.15])

    churn_flags = rng.binomial(1, churn_rate, size=n_customers)
    churn_months = np.where(churn_flags == 1, rng.integers(12, months, size=n_customers), months)

    records: list[dict[str, object]] = []
    for idx, cust_id in enumerate(customer_ids):
        size = base_company_size[idx]
        industry = base_industry[idx]
        region = base_region[idx]
        tier = base_tier[idx]
        mrr_low, mrr_high = SIZE_TO_MRR[size]
        base_mrr = rng.uniform(mrr_low, mrr_high)
        contract_length = rng.choice([12, 24, 36], p=[0.5, 0.35, 0.15])
        churn_month = churn_months[idx]

        for month_idx in range(months):
            month_date = start_date + relativedelta(months=month_idx)
            active_users = rng.poisson(lam=25 if size == "SMB" else 60 if size == "Mid-Market" else 120)
            feature_adoption = np.clip(rng.normal(0.6, 0.15), 0.1, 0.95)
            login_frequency = rng.poisson(lam=12 if tier == "Basic" else 18 if tier == "Pro" else 26)
            api_calls = rng.poisson(lam=500 if tier == "Basic" else 1500 if tier == "Pro" else 4000)
            email_opens = rng.binomial(10, 0.4)
            webinar_attendance = rng.binomial(2, 0.25)
            doc_views = rng.poisson(lam=6)
            ticket_count = rng.poisson(lam=2)
            response_time = np.clip(rng.normal(8, 3), 1, 24)
            satisfaction_score = np.clip(rng.normal(4.1, 0.6), 1, 5)
            feature_requests = rng.poisson(lam=1)
            payment_delays = rng.binomial(1, 0.15)
            failed_payments = rng.binomial(1, 0.06)
            mrr = base_mrr * (1 + rng.normal(0.01, 0.03))

            if churn_flags[idx] == 1 and month_idx >= churn_month - 3:
                active_users *= rng.uniform(0.4, 0.7)
                feature_adoption *= rng.uniform(0.5, 0.75)
                login_frequency *= rng.uniform(0.5, 0.7)
                api_calls *= rng.uniform(0.4, 0.7)
                email_opens *= rng.uniform(0.4, 0.7)
                webinar_attendance *= rng.uniform(0.3, 0.6)
                doc_views *= rng.uniform(0.4, 0.7)
                ticket_count *= rng.uniform(1.1, 1.6)
                response_time *= rng.uniform(1.2, 1.6)
                satisfaction_score *= rng.uniform(0.7, 0.9)
                payment_delays = 1
                failed_payments = rng.binomial(1, 0.2)

            churned = "Yes" if churn_flags[idx] == 1 and month_idx >= churn_month else "No"

            records.append(
                {
                    "customer_id": cust_id,
                    "month": month_date,
                    "company_size": size,
                    "industry": industry,
                    "region": region,
                    "signup_date": signup_dates[idx],
                    "subscription_tier": tier,
                    "contract_length": contract_length,
                    "mrr": round(float(mrr), 2),
                    "monthly_active_users": int(active_users),
                    "feature_adoption_rate": round(float(feature_adoption), 3),
                    "login_frequency": int(login_frequency),
                    "api_calls": int(api_calls),
                    "payment_delay": int(payment_delays),
                    "failed_payments": int(failed_payments),
                    "support_tickets": int(ticket_count),
                    "support_response_time": round(float(response_time), 2),
                    "support_satisfaction": round(float(satisfaction_score), 2),
                    "feature_requests": int(feature_requests),
                    "email_opens": int(email_opens),
                    "webinar_attendance": int(webinar_attendance),
                    "documentation_views": int(doc_views),
                    "churned": churned,
                }
            )

    return pd.DataFrame(records)


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling trend, recency, engagement, and health scores."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df.sort_values(["customer_id", "month"], inplace=True)

    rolling_cols = [
        "monthly_active_users",
        "feature_adoption_rate",
        "login_frequency",
        "api_calls",
        "email_opens",
        "documentation_views",
    ]

    for col in rolling_cols:
        df[f"{col}_3m_trend"] = (
            df.groupby("customer_id")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

    latest_month = df["month"].max()
    df["recency_days"] = (latest_month - df["month"]).dt.days

    engagement_features = [
        "monthly_active_users",
        "feature_adoption_rate",
        "login_frequency",
        "api_calls",
        "email_opens",
        "documentation_views",
    ]
    df["engagement_score"] = df[engagement_features].rank(pct=True).mean(axis=1)

    df["health_score"] = (
        0.35 * df["engagement_score"]
        + 0.2 * df["support_satisfaction"] / 5
        + 0.15 * (1 - df["payment_delay"])
        + 0.15 * (1 - df["failed_payments"])
        + 0.15 * df["feature_adoption_rate"]
    )

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Basic data validation checks."""
    required_columns = {
        "customer_id",
        "month",
        "company_size",
        "industry",
        "region",
        "subscription_tier",
        "mrr",
        "monthly_active_users",
        "feature_adoption_rate",
        "login_frequency",
        "api_calls",
        "payment_delay",
        "failed_payments",
        "support_tickets",
        "support_response_time",
        "support_satisfaction",
        "email_opens",
        "documentation_views",
        "churned",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    if df["churned"].isna().any():
        raise ValueError("Churn label contains missing values.")

    churn_rate = (df["churned"] == "Yes").mean()
    if not 0.08 <= churn_rate <= 0.12:
        raise ValueError(f"Churn rate {churn_rate:.2%} outside expected 8-12% range.")
