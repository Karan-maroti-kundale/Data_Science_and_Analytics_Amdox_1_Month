import re
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
EPS = 1e-9

# Expected Online Retail II columns (case-insensitive normalization supported)
# Typical columns:
# InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country


# ============================================================
# UTILITIES
# ============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so downstream code is robust to small schema variations.
    """
    rename_map = {}
    for c in df.columns:
        c2 = c.strip().lower().replace(" ", "_").replace("-", "_")
        rename_map[c] = c2
    return df.rename(columns=rename_map)


def infer_product_category(description: str, stock_code: str = "") -> str:
    """
    Lightweight rule-based product category mapper.
    Online Retail II does not have a true category column, so we derive one.
    """
    if pd.isna(description):
        description = ""

    desc = str(description).upper()

    category_rules = {
        "ELECTRONICS": ["USB", "BATTERY", "RADIO", "CAMERA", "LAMP", "CLOCK", "SPEAKER", "CORD", "HEADPHONE"],
        "KITCHEN": ["MUG", "CUP", "BOWL", "PLATE", "JAR", "BOTTLE", "TEAPOT", "KETTLE", "CUTLERY", "KNIFE", "FORK"],
        "HOME_DECOR": ["FRAME", "VASE", "CANDLE", "ORNAMENT", "SIGN", "BANNER", "DECOR", "HANGING"],
        "STATIONERY": ["PAPER", "NOTEBOOK", "PAD", "PEN", "PENCIL", "FOLDER", "CLIP", "ENVELOPE", "STICKER"],
        "TOYS": ["TOY", "GAME", "PUZZLE", "DOLL", "BALLOON"],
        "TEXTILES": ["TOWEL", "CLOTH", "CURTAIN", "APRON", "BAG"],
        "STORAGE": ["BOX", "BASKET", "HOLDER", "RACK", "SHELF", "TIN", "CASE"],
        "SEASONAL": ["CHRISTMAS", "EASTER", "HALLOWEEN", "BIRTHDAY", "PARTY"],
    }

    for cat, keywords in category_rules.items():
        if any(k in desc for k in keywords):
            return cat

    # Fallback: use first meaningful token or stock code family
    cleaned = re.sub(r"[^A-Z0-9 ]+", " ", desc).strip()
    tokens = [t for t in cleaned.split() if len(t) > 2]

    if tokens:
        return f"OTHER_{tokens[0][:12]}"
    if stock_code and str(stock_code).strip():
        return f"SKU_{str(stock_code).strip()[:8].upper()}"

    return "OTHER"


def add_time_seasonality_features(df: pd.DataFrame, date_col: str, prefix: str) -> pd.DataFrame:
    """
    Adds seasonality indicators from a date column.
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])

    out[f"{prefix}_dayofweek"] = dt.dt.dayofweek
    out[f"{prefix}_weekofyear"] = dt.dt.isocalendar().week.astype("int64")
    out[f"{prefix}_month"] = dt.dt.month
    out[f"{prefix}_quarter"] = dt.dt.quarter
    out[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out[f"{prefix}_is_month_start"] = dt.dt.is_month_start.astype(int)
    out[f"{prefix}_is_month_end"] = dt.dt.is_month_end.astype(int)

    # Cyclical encodings
    day_of_year = dt.dt.dayofyear.astype(float)
    dow = dt.dt.dayofweek.astype(float)

    out[f"{prefix}_sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out[f"{prefix}_cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)
    out[f"{prefix}_sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    out[f"{prefix}_cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    return out


def dense_daily_series(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    value_col: str,
    prefix: str
) -> pd.DataFrame:
    """
    Build a dense daily panel for each group, fill missing dates with 0,
    and generate rolling/lag/seasonality features.
    """
    frames = []

    for key, g in df[[group_col, date_col, value_col]].copy().groupby(group_col):
        g = g.copy()
        g[date_col] = pd.to_datetime(g[date_col]).dt.floor("D")
        daily = g.groupby(date_col, as_index=True)[value_col].sum().sort_index()

        full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        dense = daily.reindex(full_idx, fill_value=0).to_frame(name=value_col)
        dense.index.name = date_col
        dense = dense.reset_index()

        dense[group_col] = key

        # Rolling averages and lag features
        dense[f"{prefix}_rolling_7d_avg"] = dense[value_col].rolling(window=7, min_periods=1).mean()
        dense[f"{prefix}_rolling_30d_avg"] = dense[value_col].rolling(window=30, min_periods=1).mean()
        dense[f"{prefix}_lag_1"] = dense[value_col].shift(1).fillna(0)
        dense[f"{prefix}_lag_7"] = dense[value_col].shift(7).fillna(0)

        # Optional extra trend features
        dense[f"{prefix}_rolling_7d_sum"] = dense[value_col].rolling(window=7, min_periods=1).sum()
        dense[f"{prefix}_rolling_30d_sum"] = dense[value_col].rolling(window=30, min_periods=1).sum()

        dense = add_time_seasonality_features(dense, date_col, prefix)
        frames.append(dense)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# CORE FEATURE ENGINEERING
# ============================================================
def build_feature_master(
    retail_df: pd.DataFrame,
    current_stock_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Build a clean, merged customer snapshot Feature Master dataframe.

    Parameters
    ----------
    retail_df : raw Online Retail II dataframe
    current_stock_df : optional dataframe with columns:
        - stockcode
        - currentstock
      used for inventory remaining logic
    """
    df = normalize_columns(retail_df).copy()

    # -----------------------------------------------------------------
    # Column harmonization
    # -----------------------------------------------------------------
    # Common Online Retail II variants
    col_map = {
        "invoiceno": "invoice_no",
        "invoice": "invoice_no",      # Added to fix Kaggle 'Invoice' name
        "stockcode": "stock_code",
        "description": "description",
        "quantity": "quantity",
        "invoicedate": "invoice_date",
        "unitprice": "unit_price",
        "price": "unit_price",        # Added to fix Kaggle 'Price' name
        "customerid": "customer_id",
        "country": "country",
    }

    # If columns already normalized differently, align them
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # CustomerID is often float because of NaNs
    if "customer_id" in df.columns:
        df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    else:
        raise ValueError("Missing customer_id column. Customer-level features require CustomerID.")

    # -----------------------------------------------------------------
    # Cleaning
    # -----------------------------------------------------------------
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date", "customer_id", "stock_code", "quantity", "unit_price"])

    # Remove cancellations / invalid sales
    df = df[~df["invoice_no"].astype(str).str.startswith("C", na=False)]
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]

    # Revenue
    df["revenue"] = df["quantity"] * df["unit_price"]

    # Normalize text fields
    if "description" not in df.columns:
        df["description"] = ""
    df["description"] = df["description"].fillna("").astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip()

    # Derived category
    df["product_category"] = df.apply(
        lambda r: infer_product_category(r["description"], r["stock_code"]),
        axis=1
    )

    # Dense daily date for transaction grain
    df["invoice_day"] = df["invoice_date"].dt.floor("D")

    # -----------------------------------------------------------------
    # CUSTOMER STATIC FEATURES
    # -----------------------------------------------------------------
    customer_last_purchase = df.groupby("customer_id")["invoice_date"].max().rename("last_purchase_date")
    customer_first_purchase = df.groupby("customer_id")["invoice_date"].min().rename("first_purchase_date")

    customer_rfm = df.groupby("customer_id").agg(
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
        total_items=("quantity", "sum"),
        unique_skus=("stock_code", "nunique"),
        unique_categories=("product_category", "nunique"),
        unique_days=("invoice_day", "nunique"),
        avg_unit_price=("unit_price", "mean"),
        median_unit_price=("unit_price", "median"),
    )

    # Recency
    snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    customer_rfm["recency_days"] = (snapshot_date - customer_last_purchase).dt.days
    customer_rfm["tenure_days"] = (customer_last_purchase - customer_first_purchase).dt.days.clip(lower=0)

    # Average days between purchases
    def avg_days_between_purchases(s: pd.Series) -> float:
        dates = pd.to_datetime(s).dropna().sort_values().drop_duplicates()
        if len(dates) < 2:
            return np.nan
        return dates.diff().dt.days.iloc[1:].mean()

    avg_gap = (
        df[["customer_id", "invoice_date"]]
        .groupby("customer_id")["invoice_date"]
        .apply(avg_days_between_purchases)
        .rename("avg_days_between_purchases")
    )

    # Product diversity score
    product_diversity = (
        df.groupby("customer_id")["product_category"]
        .nunique()
        .rename("product_diversity_score")
    )

    customer_static = customer_rfm.join(avg_gap).join(product_diversity).reset_index()

    # -----------------------------------------------------------------
    # CUSTOMER DAILY TIME-SERIES FEATURES
    # -----------------------------------------------------------------
    customer_daily = (
        df.groupby(["customer_id", "invoice_day"], as_index=False)
        .agg(
            customer_daily_revenue=("revenue", "sum"),
            customer_daily_qty=("quantity", "sum"),
            customer_daily_orders=("invoice_no", "nunique"),
        )
    )

    customer_daily_ts = dense_daily_series(
        customer_daily.rename(columns={"customer_daily_revenue": "value"}),
        group_col="customer_id",
        date_col="invoice_day",
        value_col="value",
        prefix="customer_sales"
    )

    # Also bring the daily qty/orders if useful
    customer_qty_ts = dense_daily_series(
        customer_daily.rename(columns={"customer_daily_qty": "value"}),
        group_col="customer_id",
        date_col="invoice_day",
        value_col="value",
        prefix="customer_qty"
    )

    customer_orders_ts = dense_daily_series(
        customer_daily.rename(columns={"customer_daily_orders": "value"}),
        group_col="customer_id",
        date_col="invoice_day",
        value_col="value",
        prefix="customer_orders"
    )

    # Merge customer time-series blocks
    customer_ts = (
        customer_daily_ts
        .merge(
            customer_qty_ts[
                [
                    "customer_id", "invoice_day",
                    "customer_qty_rolling_7d_avg", "customer_qty_rolling_30d_avg",
                    "customer_qty_lag_1", "customer_qty_lag_7",
                    "customer_qty_sin_doy", "customer_qty_cos_doy",
                    "customer_qty_sin_dow", "customer_qty_cos_dow",
                    "customer_qty_dayofweek", "customer_qty_weekofyear",
                    "customer_qty_month", "customer_qty_quarter",
                    "customer_qty_is_weekend", "customer_qty_is_month_start", "customer_qty_is_month_end"
                ]
            ],
            on=["customer_id", "invoice_day"],
            how="left"
        )
        .merge(
            customer_orders_ts[
                [
                    "customer_id", "invoice_day",
                    "customer_orders_rolling_7d_avg", "customer_orders_rolling_30d_avg",
                    "customer_orders_lag_1", "customer_orders_lag_7",
                    "customer_orders_sin_doy", "customer_orders_cos_doy",
                    "customer_orders_sin_dow", "customer_orders_cos_dow",
                    "customer_orders_dayofweek", "customer_orders_weekofyear",
                    "customer_orders_month", "customer_orders_quarter",
                    "customer_orders_is_weekend", "customer_orders_is_month_start", "customer_orders_is_month_end"
                ]
            ],
            on=["customer_id", "invoice_day"],
            how="left"
        )
    )

    # -----------------------------------------------------------------
    # SKU DAILY TIME-SERIES FEATURES
    # -----------------------------------------------------------------
    sku_daily = (
        df.groupby(["stock_code", "invoice_day"], as_index=False)
        .agg(
            sku_daily_qty=("quantity", "sum"),
            sku_daily_revenue=("revenue", "sum"),
        )
    )

    sku_ts = dense_daily_series(
        sku_daily.rename(columns={"sku_daily_qty": "value"}),
        group_col="stock_code",
        date_col="invoice_day",
        value_col="value",
        prefix="sku_sales"
    )

    # Inventory logic
    if current_stock_df is not None:
        current_stock = normalize_columns(current_stock_df).copy()

        # Align columns
        if "stockcode" in current_stock.columns and "stock_code" not in current_stock.columns:
            current_stock = current_stock.rename(columns={"stockcode": "stock_code"})
        if "currentstock" in current_stock.columns and "current_stock" not in current_stock.columns:
            current_stock = current_stock.rename(columns={"currentstock": "current_stock"})

        if "stock_code" not in current_stock.columns or "current_stock" not in current_stock.columns:
            raise ValueError("current_stock_df must contain stock_code and current_stock columns.")

        current_stock["stock_code"] = current_stock["stock_code"].astype(str).str.strip()
        current_stock["current_stock"] = pd.to_numeric(current_stock["current_stock"], errors="coerce")

        # Burn rate = 30-day average daily unit sales
        sku_inventory = sku_ts[["stock_code", "invoice_day", "sku_sales_rolling_30d_avg"]].copy()
        sku_inventory = sku_inventory.merge(current_stock, on="stock_code", how="left")

        sku_inventory["days_inventory_remaining"] = (
            sku_inventory["current_stock"] /
            sku_inventory["sku_sales_rolling_30d_avg"].replace(0, np.nan)
        )
        sku_inventory["days_inventory_remaining"] = sku_inventory["days_inventory_remaining"].replace([np.inf, -np.inf], np.nan)

        sku_ts = sku_ts.merge(
            sku_inventory[["stock_code", "invoice_day", "current_stock", "days_inventory_remaining"]],
            on=["stock_code", "invoice_day"],
            how="left"
        )
    else:
        sku_ts["current_stock"] = np.nan
        sku_ts["days_inventory_remaining"] = np.nan

    # -----------------------------------------------------------------
    # TRANSACTION-LEVEL MERGES
    # -----------------------------------------------------------------
    tx = df[
        [
            "customer_id", "invoice_no", "invoice_date", "invoice_day",
            "stock_code", "description", "product_category",
            "quantity", "unit_price", "revenue", "country"
        ]
    ].copy()

    # Merge customer snapshot features
    tx = tx.merge(customer_static, on="customer_id", how="left")

    # Merge customer daily TS features by customer/day
    tx = tx.merge(customer_ts, on=["customer_id", "invoice_day"], how="left")

    # Merge SKU TS features by stock/day
    tx = tx.merge(sku_ts, on=["stock_code", "invoice_day"], how="left")

    # -----------------------------------------------------------------
    # FINAL FEATURE MASTER
    # -----------------------------------------------------------------
    # Create a customer snapshot table: use latest transaction row per customer
    tx = tx.sort_values(["customer_id", "invoice_date", "invoice_no"])
    feature_master = tx.groupby("customer_id", as_index=False).tail(1).reset_index(drop=True)

    # Keep only training-friendly columns and clean names
    feature_master = feature_master.rename(columns={
        "invoice_day": "snapshot_date",
        "customer_daily_revenue": "customer_day_revenue",
        "customer_daily_qty": "customer_day_qty",
        "customer_daily_orders": "customer_day_orders",
    })

    # Optional: sort columns for readability
    preferred_order = [
        "customer_id", "snapshot_date", "invoice_no", "invoice_date",
        "stock_code", "description", "product_category", "country",
        "recency_days", "frequency", "monetary", "total_items", "unique_skus",
        "unique_categories", "unique_days", "tenure_days",
        "avg_days_between_purchases", "product_diversity_score",
        "customer_sales_rolling_7d_avg", "customer_sales_rolling_30d_avg",
        "customer_sales_lag_1", "customer_sales_lag_7",
        "customer_qty_rolling_7d_avg", "customer_qty_rolling_30d_avg",
        "customer_qty_lag_1", "customer_qty_lag_7",
        "customer_orders_rolling_7d_avg", "customer_orders_rolling_30d_avg",
        "customer_orders_lag_1", "customer_orders_lag_7",
        "sku_sales_rolling_7d_avg", "sku_sales_rolling_30d_avg",
        "sku_sales_lag_1", "sku_sales_lag_7",
        "current_stock", "days_inventory_remaining"
    ]

    existing = [c for c in preferred_order if c in feature_master.columns]
    remaining = [c for c in feature_master.columns if c not in existing]
    feature_master = feature_master[existing + remaining]

    return feature_master


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Replace with your actual file paths
    retail_path = Path("OnlineRetailII.csv")
    stock_path = Path("current_stock.csv")  # optional

    retail_df = pd.read_csv(retail_path, encoding="latin1")

    current_stock_df = None
    if stock_path.exists():
        current_stock_df = pd.read_csv(stock_path)

    feature_master = build_feature_master(retail_df, current_stock_df=current_stock_df)

    print("Feature Master shape:", feature_master.shape)
    print(feature_master.head(10))

    # Save for downstream training
    feature_master.to_csv("feature_master_online_retail_ii.csv", index=False)