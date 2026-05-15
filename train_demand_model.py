from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "demand_prophet.pkl"
SERIES_PATH = OUTPUT_DIR / "demand_training_series.csv"
METADATA_PATH = OUTPUT_DIR / "demand_training_metadata.json"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    return out


def load_dataframe(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    try:
        return pd.read_csv(input_path, encoding="latin1", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(input_path, low_memory=False)


def detect_mode(df: pd.DataFrame, preferred: str = "auto") -> Literal["raw", "feature_master"]:
    if preferred in {"raw", "feature_master"}:
        return preferred  # type: ignore[return-value]

    cols = set(df.columns)

    raw_ready = {"invoice_date", "quantity", "unit_price"}.issubset(cols)
    raw_revenue_ready = {"invoice_date", "revenue"}.issubset(cols)
    feature_ready = {"snapshot_date", "monetary"}.issubset(cols)

    if raw_ready or raw_revenue_ready:
        return "raw"
    if feature_ready:
        return "feature_master"

    raise ValueError(
        "Could not auto-detect a valid demand source. "
        "Expected either raw transaction columns like invoice_date/quantity/unit_price "
        "or feature-master columns like snapshot_date/monetary."
    )


def build_daily_series(df: pd.DataFrame, mode: Literal["raw", "feature_master"]) -> pd.DataFrame:
    df = normalize_columns(df).copy()

    if mode == "raw":
        if "invoice_date" not in df.columns:
            raise ValueError("Raw mode requires invoice_date.")
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

        # Clean Online Retail II transaction data
        if "invoice_no" in df.columns:
            df = df[~df["invoice_no"].astype(str).str.startswith("C", na=False)]

        if "quantity" in df.columns:
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        if "unit_price" in df.columns:
            df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

        if {"quantity", "unit_price"}.issubset(df.columns):
            df = df.dropna(subset=["invoice_date", "quantity", "unit_price"])
            df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)].copy()
            df["sales_value"] = df["quantity"] * df["unit_price"]
        elif "revenue" in df.columns:
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
            df = df.dropna(subset=["invoice_date", "revenue"])
            df = df[df["revenue"] > 0].copy()
            df["sales_value"] = df["revenue"]
        else:
            raise ValueError("Raw mode requires either quantity/unit_price or revenue.")

        daily = (
            df.assign(ds=df["invoice_date"].dt.floor("D"))
            .groupby("ds", as_index=False)["sales_value"]
            .sum()
            .rename(columns={"sales_value": "y"})
            .sort_values("ds")
        )

    else:
        # Feature Master fallback: aggregate total monetary value by snapshot_date
        if "snapshot_date" not in df.columns:
            raise ValueError("Feature master mode requires snapshot_date.")
        if "monetary" not in df.columns and "revenue" not in df.columns:
            raise ValueError("Feature master mode requires monetary or revenue.")

        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")

        value_col = "monetary" if "monetary" in df.columns else "revenue"
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        df = df.dropna(subset=["snapshot_date", value_col])
        df = df[df[value_col] > 0].copy()

        daily = (
            df.assign(ds=df["snapshot_date"].dt.floor("D"))
            .groupby("ds", as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "y"})
            .sort_values("ds")
        )

    if daily.empty:
        raise ValueError("No valid daily series could be constructed from the input data.")

    # Make the series dense so Prophet sees a continuous daily timeline
    full_idx = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_idx, fill_value=0.0).rename_axis("ds").reset_index()
    daily["y"] = pd.to_numeric(daily["y"], errors="coerce").fillna(0.0)

    return daily


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def fit_prophet(train_df: pd.DataFrame) -> Prophet:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(train_df)
    return model


def evaluate_model(model: Prophet, test_df: pd.DataFrame) -> dict:
    future = test_df[["ds"]].copy()
    forecast = model.predict(future)

    y_true = test_df["y"].to_numpy(dtype=float)
    y_pred = forecast["yhat"].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    metric_mape = mape(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": metric_mape,
        "n_test_days": int(len(test_df)),
    }


def train_demand_model(
    input_path: str | Path,
    mode: str = "auto",
    holdout_days: int = 30,
) -> dict:
    raw_df = load_dataframe(input_path)
    normalized = normalize_columns(raw_df)
    detected_mode = detect_mode(normalized, preferred=mode)

    daily_series = build_daily_series(normalized, detected_mode)
    daily_series.to_csv(SERIES_PATH, index=False)

    if len(daily_series) < max(90, holdout_days + 30):
        raise ValueError(
            f"Not enough daily history to train a stable Prophet model. "
            f"Found only {len(daily_series)} daily rows."
        )

    # Hold out the last N days for validation
    train_df = daily_series.iloc[:-holdout_days].copy()
    test_df = daily_series.iloc[-holdout_days:].copy()

    train_prophet_model = fit_prophet(train_df)
    eval_metrics = evaluate_model(train_prophet_model, test_df)

    # Refit on full history for production artifact
    final_model = fit_prophet(daily_series)
    joblib.dump(final_model, MODEL_PATH)

    metadata = {
        "input_path": str(Path(input_path).resolve()),
        "source_mode": detected_mode,
        "train_start": str(daily_series["ds"].min().date()),
        "train_end": str(daily_series["ds"].max().date()),
        "n_days_total": int(len(daily_series)),
        "n_train_days": int(len(train_df)),
        "n_test_days": int(len(test_df)),
        "model_path": str(MODEL_PATH.resolve()),
        "series_path": str(SERIES_PATH.resolve()),
        "metrics": eval_metrics,
        "prophet_params": {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "monthly_seasonality": {"period": 30.5, "fourier_order": 5},
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save a Prophet demand forecasting model.")
    parser.add_argument(
        "--input",
        type=str,
        default="feature_master_online_retail_ii.csv",
        help="Path to feature_master_online_retail_ii.csv or OnlineRetailII.csv",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "feature_master"],
        help="Data source mode. auto is recommended.",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=30,
        help="Number of final days to use for validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_demand_model(
        input_path=args.input,
        mode=args.mode,
        holdout_days=args.holdout_days,
    )

    print("\nDemand model training complete.")
    print(f"Saved model: {result['model_path']}")
    print(f"Saved training series: {result['series_path']}")
    print("Validation metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")