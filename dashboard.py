# dashboard.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NeuralRetail | Executive Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================
DATA_PATH = Path("feature_master_online_retail_ii.csv")
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CHURN_ENDPOINT = "/predict/churn"
DEFAULT_LOG_PATH = Path("logs/predictions.jsonl")

EPS = 1e-9


# ============================================================
# UTILS
# ============================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_value(value):
    """Convert numpy/pandas values to JSON-safe Python types."""
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        return value.item()
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def to_jsonable_dict(row: pd.Series, exclude_cols: set[str] | None = None) -> dict:
    exclude_cols = exclude_cols or set()
    payload = {}
    for col, value in row.items():
        if col in exclude_cols:
            continue
        payload[col] = safe_json_value(value)
    return payload


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


@st.cache_data(show_spinner=True, ttl=60)
def load_feature_master(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature Master file not found: {path.resolve()}")

    df = pd.read_csv(path, low_memory=False)
    df = standardize_columns(df)

    # Normalize common date columns if present
    for col in ["snapshot_date", "invoice_date", "first_purchase_date", "last_purchase_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False, ttl=15)
def load_prediction_logs(log_path: str | Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame()

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

    if "input_data" in df.columns:
        input_df = pd.json_normalize(df["input_data"])
        input_df.columns = [f"input__{c}" for c in input_df.columns]
        df = pd.concat([df.drop(columns=["input_data"]), input_df], axis=1)

    if "prediction_data" in df.columns:
        pred_df = pd.json_normalize(df["prediction_data"])
        pred_df.columns = [f"pred__{c}" for c in pred_df.columns]
        df = pd.concat([df.drop(columns=["prediction_data"]), pred_df], axis=1)

    return df


def build_api_payload_from_row(row: pd.Series) -> dict:
    """
    Use all model-friendly columns from the selected customer row.
    Exclude identifiers, dates, and raw text fields that are not typically fed into XGBoost directly.
    """
    exclude_prefixes = (
        "invoice_",
        "snapshot_date",
        "first_purchase_date",
        "last_purchase_date",
        "description",
        "country",
        "stock_code",
        "product_category",
    )
    exclude_exact = {
        "customer_id",
        "request_id",
    }

    payload = {}
    for col, value in row.items():
        col_l = str(col).lower()

        if col_l in exclude_exact:
            continue
        if any(col_l.startswith(p) for p in exclude_prefixes):
            continue

        if isinstance(value, pd.Timestamp):
            continue

        # Keep numeric / boolean fields; stringify anything else only if needed
        if pd.api.types.is_numeric_dtype(type(value)) or isinstance(value, (int, float, np.number, bool)):
            payload[col] = safe_json_value(value)
        elif isinstance(value, (str,)) and value.strip() == "":
            continue
        elif isinstance(value, (str,)):
            # Usually not sent to numeric model; skip raw text/categorical fields
            continue

    return payload


def call_churn_api(
    api_base_url: str,
    feature_dict: dict,
    request_id: str | None = None,
    timeout: int = 20,
) -> dict:
    url = api_base_url.rstrip("/") + DEFAULT_CHURN_ENDPOINT
    body = {
        "request_id": request_id,
        "features": feature_dict,
    }

    response = requests.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_prediction(result_json: dict) -> tuple[float | None, int | None]:
    """
    Expected response shape:
    {
      "prediction": {
        "churn_probability": 0.82,
        "churn_label": 1
      },
      ...
    }
    """
    pred = result_json.get("prediction", {}) if isinstance(result_json, dict) else {}
    prob = pred.get("churn_probability", None)
    label = pred.get("churn_label", None)

    try:
        prob = float(prob) if prob is not None else None
    except Exception:
        prob = None

    try:
        label = int(label) if label is not None else None
    except Exception:
        label = None

    return prob, label


def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Population Stability Index for numeric columns.
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if len(expected) < 10 or len(actual) < 10:
        return np.nan
    if expected.nunique() <= 1 or actual.nunique() <= 1:
        return np.nan

    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(breakpoints) < 3:
        return np.nan

    # Make sure bins are valid
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True, duplicates="drop")
    actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True, duplicates="drop")

    expected_pct = expected_bins.value_counts(normalize=True).sort_index()
    actual_pct = actual_bins.value_counts(normalize=True).sort_index()

    aligned = pd.concat([expected_pct, actual_pct], axis=1).fillna(EPS)
    aligned.columns = ["expected", "actual"]

    value = np.sum((aligned["actual"] - aligned["expected"]) * np.log(aligned["actual"] / aligned["expected"]))
    return float(value)


def compute_drift_table(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_candidates: list[str],
) -> pd.DataFrame:
    rows = []
    for col in feature_candidates:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        score = psi(reference_df[col], current_df[col])
        if pd.notna(score):
            rows.append({"feature": col, "psi": score})

    if not rows:
        return pd.DataFrame(columns=["feature", "psi"])

    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def bucket(v):
        if pd.isna(v):
            return "Unknown"
        if v < 0.1:
            return "Stable"
        if v < 0.25:
            return "Moderate Drift"
        return "High Drift"

    out["status"] = out["psi"].apply(bucket)
    return out


def make_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E84E1B"},
                "steps": [
                    {"range": [0, 35], "color": "#E7F7EF"},
                    {"range": [35, 65], "color": "#FFF4D6"},
                    {"range": [65, 100], "color": "#FDE8E8"},
                ],
                "threshold": {
                    "line": {"color": "#111111", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ============================================================
# PAGE RENDERERS
# ============================================================
def render_executive_overview(df: pd.DataFrame) -> None:
    st.subheader("Executive Overview")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    total_customers = df["customer_id"].nunique() if "customer_id" in df.columns else len(df)
    total_revenue = float(df["monetary"].sum()) if "monetary" in df.columns else float(np.nansum(df[numeric_cols].sum())) if numeric_cols else 0.0
    avg_recency = float(df["recency_days"].mean()) if "recency_days" in df.columns else np.nan
    avg_frequency = float(df["frequency"].mean()) if "frequency" in df.columns else np.nan
    avg_diversity = float(df["product_diversity_score"].mean()) if "product_diversity_score" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Revenue", f"{total_revenue:,.2f}")
    c3.metric("Avg Frequency", f"{avg_frequency:.2f}" if pd.notna(avg_frequency) else "N/A")
    c4.metric("Avg Recency (days)", f"{avg_recency:.2f}" if pd.notna(avg_recency) else "N/A")

    c5, c6 = st.columns(2)
    c5.metric("Avg Product Diversity", f"{avg_diversity:.2f}" if pd.notna(avg_diversity) else "N/A")
    if "avg_days_between_purchases" in df.columns:
        c6.metric(
            "Avg Days Between Purchases",
            f"{float(df['avg_days_between_purchases'].mean()):.2f}",
        )
    else:
        c6.metric("Avg Days Between Purchases", "N/A")

    st.divider()

    left, right = st.columns(2)

    with left:
        if "monetary" in df.columns:
            fig = px.histogram(
                df,
                x="monetary",
                nbins=40,
                title="Customer Monetary Value Distribution",
                labels={"monetary": "Monetary Value"},
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if "product_diversity_score" in df.columns:
            diversity_counts = (
                df["product_diversity_score"]
                .fillna(0)
                .round(0)
                .astype(int)
                .value_counts()
                .sort_index()
                .reset_index()
            )
            diversity_counts.columns = ["diversity_score", "customers"]

            fig = px.bar(
                diversity_counts,
                x="diversity_score",
                y="customers",
                title="Product Diversity Score Distribution",
                labels={"diversity_score": "Unique Categories per Customer", "customers": "Customers"},
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        if {"frequency", "monetary"}.issubset(df.columns):
            plot_df = df.copy()
            if "recency_days" in plot_df.columns:
                fig = px.scatter(
                    plot_df,
                    x="frequency",
                    y="monetary",
                    color="recency_days",
                    hover_data=["customer_id"] if "customer_id" in plot_df.columns else None,
                    title="Customer Value Map",
                    labels={
                        "frequency": "Purchase Frequency",
                        "monetary": "Monetary Value",
                        "recency_days": "Recency (days)",
                    },
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x="frequency",
                    y="monetary",
                    hover_data=["customer_id"] if "customer_id" in plot_df.columns else None,
                    title="Customer Value Map",
                    labels={"frequency": "Purchase Frequency", "monetary": "Monetary Value"},
                )
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with bottom_right:
        if "recency_days" in df.columns:
            recency_bins = pd.cut(
                df["recency_days"].fillna(df["recency_days"].median()),
                bins=[-0.1, 30, 60, 90, 180, 365, np.inf],
                labels=["0-30", "31-60", "61-90", "91-180", "181-365", "365+"],
            )
            recency_counts = recency_bins.value_counts().sort_index().reset_index()
            recency_counts.columns = ["recency_bucket", "customers"]

            fig = px.bar(
                recency_counts,
                x="recency_bucket",
                y="customers",
                title="Recency Buckets",
                labels={"recency_bucket": "Days Since Last Purchase", "customers": "Customers"},
            )
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.caption("Overview metrics are computed from feature_master_online_retail_ii.csv.")


def render_customer_hub(df: pd.DataFrame, api_base_url: str) -> None:
    st.subheader("Customer Hub (Churn Prediction)")

    if "customer_id" not in df.columns:
        st.error("The dataset does not contain customer_id.")
        return

    customer_ids = df["customer_id"].dropna().astype(str).sort_values().tolist()
    selected_customer_id = st.selectbox("Select Customer ID", customer_ids)

    customer_mask = df["customer_id"].astype(str) == str(selected_customer_id)
    customer_row = df.loc[customer_mask].iloc[0].copy()
    display_row = customer_row.to_frame(name="value")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Recency", f"{customer_row.get('recency_days', np.nan):.0f}" if pd.notna(customer_row.get("recency_days", np.nan)) else "N/A")
    top2.metric("Frequency", f"{customer_row.get('frequency', np.nan):.0f}" if pd.notna(customer_row.get("frequency", np.nan)) else "N/A")
    top3.metric("Monetary", f"{customer_row.get('monetary', np.nan):,.2f}" if pd.notna(customer_row.get("monetary", np.nan)) else "N/A")
    top4.metric("Diversity", f"{customer_row.get('product_diversity_score', np.nan):.0f}" if pd.notna(customer_row.get("product_diversity_score", np.nan)) else "N/A")

    st.divider()

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("**Customer Snapshot**")
        show_cols = [
            c for c in [
                "customer_id",
                "snapshot_date",
                "last_purchase_date",
                "first_purchase_date",
                "recency_days",
                "frequency",
                "monetary",
                "total_items",
                "unique_skus",
                "unique_categories",
                "avg_days_between_purchases",
                "product_diversity_score",
                "tenure_days",
            ]
            if c in df.columns
        ]
        if show_cols:
            st.dataframe(display_row.loc[show_cols], use_container_width=True, height=360)
        else:
            st.dataframe(display_row.head(20), use_container_width=True, height=360)

    with right:
        st.markdown("**Prediction Control**")
        feature_payload = build_api_payload_from_row(customer_row)

        st.code(
            json.dumps(
                {
                    "request_id": f"cust-{selected_customer_id}",
                    "features": {k: feature_payload[k] for k in list(feature_payload)[:10]},
                },
                indent=2,
                default=str,
            ),
            language="json",
        )

        predict_btn = st.button("Score Churn Risk", use_container_width=True)

        if predict_btn:
            with st.spinner("Sending customer data to FastAPI..."):
                try:
                    result = call_churn_api(
                        api_base_url=api_base_url,
                        feature_dict=feature_payload,
                        request_id=f"cust-{selected_customer_id}",
                    )
                    churn_prob, churn_label = extract_prediction(result)

                    if churn_prob is not None:
                        st.metric("Churn Probability", f"{churn_prob:.2%}")
                        st.plotly_chart(make_gauge(churn_prob), use_container_width=True)

                        if churn_label == 1:
                            st.error("High churn risk detected.")
                        else:
                            st.success("Low churn risk detected.")
                    else:
                        st.warning("API response did not contain churn_probability.")

                    with st.expander("Raw API Response", expanded=False):
                        st.json(result)

                except requests.exceptions.RequestException as e:
                    st.error(f"FastAPI request failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected scoring error: {e}")

    st.divider()

    if "customer_sales_rolling_7d_avg" in df.columns or "sku_sales_rolling_7d_avg" in df.columns:
        chart_cols = [c for c in ["customer_sales_rolling_7d_avg", "customer_sales_rolling_30d_avg", "sku_sales_rolling_7d_avg", "sku_sales_rolling_30d_avg"] if c in df.columns]
        if chart_cols:
            chart_data = pd.DataFrame({"feature": chart_cols, "value": [float(customer_row[c]) if pd.notna(customer_row[c]) else np.nan for c in chart_cols]})
            fig = px.bar(
                chart_data,
                x="feature",
                y="value",
                title="Recent Trend Features for This Customer",
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)


def render_drift_monitor(df: pd.DataFrame, log_path: str | Path) -> None:
    st.subheader("Data Drift Monitor")

    logs_df = load_prediction_logs(log_path)

    if logs_df.empty:
        st.info("No prediction logs found yet. Once the FastAPI endpoint receives traffic, drift monitoring will appear here.")
        st.caption(f"Expected log file: {Path(log_path).resolve()}")
        return

    latest_ts = logs_df["timestamp_utc"].max() if "timestamp_utc" in logs_df.columns else pd.NaT
    total_calls = len(logs_df)
    recent_window = logs_df.copy()
    if "timestamp_utc" in recent_window.columns:
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=7)
        recent_window = recent_window[recent_window["timestamp_utc"] >= cutoff]

    avg_prob = np.nan
    if "pred__churn_probability" in logs_df.columns:
        avg_prob = pd.to_numeric(logs_df["pred__churn_probability"], errors="coerce").mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Logged Predictions", f"{total_calls:,}")
    c2.metric("7-Day Calls", f"{len(recent_window):,}")
    c3.metric("Avg Churn Probability", f"{avg_prob:.2%}" if pd.notna(avg_prob) else "N/A")
    c4.metric("Last Log Time", str(latest_ts) if pd.notna(latest_ts) else "N/A")

    st.divider()

    # Reference vs recent live inputs
    input_cols = [c for c in logs_df.columns if c.startswith("input__")]
    if not input_cols:
        st.warning("No serialized input features were found in the prediction logs.")
        return

    # Build live input frame
    live_df = logs_df[input_cols].copy()
    live_df.columns = [c.replace("input__", "") for c in live_df.columns]

    # Reference features from training data
    reference = df.copy()

    # Candidates: numeric columns shared between feature master and live logs
    feature_candidates = []
    for col in live_df.columns:
        if col in reference.columns:
            if pd.api.types.is_numeric_dtype(reference[col]) and pd.api.types.is_numeric_dtype(live_df[col]):
                feature_candidates.append(col)

    if not feature_candidates:
        st.warning("No overlapping numeric features were found between the feature master and live request logs.")
        return

    drift_df = compute_drift_table(reference, live_df, feature_candidates)

    if drift_df.empty:
        st.info("Not enough data to compute drift reliably yet.")
        return

    left, right = st.columns([0.55, 0.45])

    with left:
        st.markdown("**Top Drifted Features**")
        st.dataframe(drift_df.head(15), use_container_width=True, height=420)

    with right:
        chart_df = drift_df.head(10).copy().sort_values("psi", ascending=True)
        fig = px.bar(
            chart_df,
            x="psi",
            y="feature",
            orientation="h",
            color="status",
            title="Highest PSI Features",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Prediction trend
    if "pred__churn_probability" in logs_df.columns and "timestamp_utc" in logs_df.columns:
        trend_df = logs_df[["timestamp_utc", "pred__churn_probability"]].copy()
        trend_df["pred__churn_probability"] = pd.to_numeric(trend_df["pred__churn_probability"], errors="coerce")
        trend_df = trend_df.dropna().sort_values("timestamp_utc")

        if not trend_df.empty:
            fig = px.line(
                trend_df,
                x="timestamp_utc",
                y="pred__churn_probability",
                title="Churn Probability Over Time",
                markers=True,
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw Prediction Log Sample", expanded=False):
        st.dataframe(logs_df.tail(20), use_container_width=True)


# ============================================================
# APP SHELL
# ============================================================
def main() -> None:
    st.title("📈 NeuralRetail Executive Sales Intelligence Platform")
    st.caption("Streamlit dashboard connected to your local FastAPI serving layer.")

    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            [
                "Executive Overview",
                "Customer Hub (Churn Prediction)",
                "Data Drift Monitor",
            ],
        )

        st.divider()
        api_base_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_BASE_URL)
        log_path = st.text_input("Prediction Log Path", value=str(DEFAULT_LOG_PATH))
        st.caption("Endpoint used by the dashboard: /predict/churn")

    try:
        df = load_feature_master(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load feature master data: {e}")
        st.stop()

    if page == "Executive Overview":
        render_executive_overview(df)
    elif page == "Customer Hub (Churn Prediction)":
        render_customer_hub(df, api_base_url=api_base_url)
    elif page == "Data Drift Monitor":
        render_drift_monitor(df, log_path=log_path)


if __name__ == "__main__":
    main()