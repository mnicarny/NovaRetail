# app.py
"""
NovaRetail Customer Insights Dashboard
Single-file Streamlit app for Streamlit Cloud deployment.

IMPORTANT (Streamlit Cloud):
- To read .xlsx files, pandas requires the 'openpyxl' dependency.
- Ensure requirements.txt includes: openpyxl

Suggested requirements.txt:
    streamlit
    pandas
    numpy
    plotly
    openpyxl

Dataset:
- NR_dataset.xlsx must be in the SAME directory as this app.py (repo root).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(layout="wide", page_title="NovaRetail Customer Insights Dashboard")


# -----------------------------
# Constants
# -----------------------------
DATA_FILE = "NR_dataset.xlsx"

# Canonical (internal) columns we will use throughout the app:
CANON = {
    "CustomerID": "customer_id",
    "TransactionID": "transaction_id",
    "TransactionDate": "transaction_date",
    "Label": "label",
    "PurchaseAmount": "purchase_amount",
    "CustomerSatisfaction": "customer_satisfaction",
    "CustomerRegion": "customer_region",
    "RetailChannel": "retail_channel",
    "ProductCategory": "product_category",
    "CustomerAgeGroup": "customer_age_group",
    "CustomerGender": "customer_gender",
}

REQUIRED_LOGICAL = list(CANON.keys())

# Used only for ordering if those labels exist in the data (NOT a hardcoded filter list)
LABEL_ORDER_HINT = ["Promising", "Growth", "Stable", "Decline"]


@dataclass(frozen=True)
class FilterState:
    label: List[str]
    customer_region: List[str]
    retail_channel: List[str]
    product_category: List[str]
    customer_age_group: List[str]
    customer_gender: List[str]
    year: List[str]   # keep as strings so "All" is naturally supported
    month: List[str]  # keep as strings so "All" is naturally supported


# -----------------------------
# Utilities
# -----------------------------
def normalize_col_name(col: str) -> str:
    """Normalize a column name: strip whitespace, lowercase, replace spaces with underscores."""
    return str(col).strip().lower().replace(" ", "_")


def squeeze_key(s: str) -> str:
    """Normalize further for matching: remove underscores and non-alphanumerics."""
    s2 = normalize_col_name(s)
    return "".join(ch for ch in s2 if ch.isalnum())


def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Coerce a series to datetime, safely."""
    return pd.to_datetime(series, errors="coerce")


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, safely."""
    return pd.to_numeric(series, errors="coerce")


def as_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV bytes for a dataframe."""
    return df.to_csv(index=False).encode("utf-8")


def add_all_option(options: List) -> List:
    """Return options list prefixed with 'All'."""
    return ["All"] + options


def apply_multiselect_filter(df: pd.DataFrame, col: str, selected: List[str]) -> pd.DataFrame:
    """Apply a multiselect filter. If 'All' selected, ignore filter."""
    if not selected or "All" in selected:
        return df
    return df[df[col].isin(selected)].copy()


def order_labels_if_possible(labels: List[str]) -> List[str]:
    """Order labels in a friendly way when common segments exist, otherwise sort."""
    labels_list = sorted(set([str(x) for x in labels if pd.notna(x)]))
    ordered: List[str] = []
    for x in LABEL_ORDER_HINT:
        if x in labels_list:
            ordered.append(x)
    for x in labels_list:
        if x not in ordered:
            ordered.append(x)
    return ordered


# -----------------------------
# Data loading & validation
# -----------------------------
@st.cache_data(show_spinner=False)
def load_raw_excel(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load the Excel file with robust try/except.

    NOTE:
    - Pandas reading .xlsx requires openpyxl (commonly on Streamlit Cloud).
    - If openpyxl is missing, we return a clear actionable error.
    """
    try:
        # We purposely *don't* hardcode engine here; pandas will choose.
        # If openpyxl is missing, the error will be caught and surfaced nicely.
        df = pd.read_excel(path)
        return df, None
    except ImportError as exc:
        return None, (
            "Missing dependency required to read Excel files.\n\n"
            "Install **openpyxl** by adding it to requirements.txt:\n"
            "  - openpyxl\n\n"
            f"Underlying error: {exc}"
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to read '{path}'. Error: {exc}"


def map_required_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """
    Dynamically match required logical fields to actual columns.

    Matching approach:
    1) exact normalized match (strip/lower/space->underscore)
    2) "squeezed" match (remove underscores & non-alphanumerics)
    """
    original_cols = list(df.columns)
    norm_to_original: Dict[str, str] = {}
    squeeze_to_original: Dict[str, str] = {}

    for oc in original_cols:
        norm_to_original[normalize_col_name(oc)] = oc
        squeeze_to_original[squeeze_key(oc)] = oc

    mapping: Dict[str, str] = {}
    missing: List[str] = []

    for logical in REQUIRED_LOGICAL:
        logical_norm = normalize_col_name(logical)
        logical_squeeze = squeeze_key(logical)

        if logical_norm in norm_to_original:
            mapping[logical] = norm_to_original[logical_norm]
        elif logical_squeeze in squeeze_to_original:
            mapping[logical] = squeeze_to_original[logical_squeeze]
        else:
            missing.append(logical)

    return mapping, missing


def select_and_rename_required(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Select only required variables and rename to canonical snake_case names."""
    cols = [mapping[k] for k in REQUIRED_LOGICAL]
    df2 = df.loc[:, cols].copy()

    rename_map = {mapping[k]: CANON[k] for k in REQUIRED_LOGICAL}
    df2 = df2.rename(columns=rename_map)
    df2.columns = [normalize_col_name(c) for c in df2.columns]
    return df2


# -----------------------------
# Preprocessing & feature engineering
# -----------------------------
def preprocess_transactions(df_txn: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Preprocess transaction-level data:
    - Parse transaction_date to datetime
    - Ensure purchase_amount is numeric
    - Derive year, month, year_month
    - Derive recency_days (relative to dataset max date) at transaction level:
        recency_days = (max_transaction_date_in_dataset - transaction_date).days
    """
    df = df_txn.copy()

    df["transaction_date"] = safe_to_datetime(df["transaction_date"])
    invalid_date = int(df["transaction_date"].isna().sum())

    df["purchase_amount"] = safe_to_numeric(df["purchase_amount"])
    invalid_amount = int(df["purchase_amount"].isna().sum())

    before = len(df)
    df = df.dropna(subset=["transaction_date", "purchase_amount"]).copy()
    dropped = int(before - len(df))

    df["year"] = df["transaction_date"].dt.year.astype(int)
    df["month"] = df["transaction_date"].dt.month.astype(int)
    df["year_month"] = df["transaction_date"].dt.to_period("M").astype(str)

    max_date = df["transaction_date"].max()
    df["recency_days"] = (max_date - df["transaction_date"]).dt.days.astype(int)

    stats = {
        "invalid_date": invalid_date,
        "invalid_amount": invalid_amount,
        "dropped_rows": dropped,
        "remaining_rows": int(len(df)),
    }
    return df, stats


def compute_customer_aggregates(df_txn: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate customer-level metrics (from the provided transaction-level dataframe):
      - total_revenue
      - purchase_count
      - avg_purchase_value
      - avg_satisfaction
      - last_purchase_date
      - recency_days (relative to dataset max date)
      - dominant (most common) dimensions for label/region/channel/category/age/gender
    """
    if df_txn.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "label",
                "customer_region",
                "retail_channel",
                "product_category",
                "customer_age_group",
                "customer_gender",
                "total_revenue",
                "purchase_count",
                "avg_purchase_value",
                "avg_satisfaction",
                "last_purchase_date",
                "recency_days",
            ]
        )

    max_date = df_txn["transaction_date"].max()

    def mode_or_na(s: pd.Series):
        s2 = s.dropna().astype(str)
        if s2.empty:
            return np.nan
        return s2.value_counts().index[0]

    agg = (
        df_txn.groupby("customer_id", as_index=False)
        .agg(
            label=("label", mode_or_na),
            customer_region=("customer_region", mode_or_na),
            retail_channel=("retail_channel", mode_or_na),
            product_category=("product_category", mode_or_na),
            customer_age_group=("customer_age_group", mode_or_na),
            customer_gender=("customer_gender", mode_or_na),
            total_revenue=("purchase_amount", "sum"),
            purchase_count=("transaction_id", "nunique"),
            avg_purchase_value=("purchase_amount", "mean"),
            avg_satisfaction=("customer_satisfaction", "mean"),
            last_purchase_date=("transaction_date", "max"),
        )
        .copy()
    )

    agg["recency_days"] = (max_date - agg["last_purchase_date"]).dt.days.astype(int)
    return agg


def compute_kpis(df_txn_filtered: pd.DataFrame) -> Dict[str, float]:
    """
    Compute KPI metrics on filtered transaction-level data.

    Top 10% revenue share formula:
      1) Compute revenue per customer within the filtered data.
      2) Rank customers by revenue (descending).
      3) Let N = number of unique customers.
      4) Let K = max(1, ceil(0.10 * N)).
      5) Top10% revenue = sum of revenues for top K customers.
      6) Share = Top10% revenue / Total revenue (if total revenue > 0).
    """
    if df_txn_filtered.empty:
        return {
            "total_revenue": 0.0,
            "active_customers": 0.0,
            "avg_purchase_value": 0.0,
            "top10_revenue_value": 0.0,
            "top10_revenue_share": 0.0,
        }

    total_revenue = float(df_txn_filtered["purchase_amount"].sum())
    active_customers = int(df_txn_filtered["customer_id"].nunique())
    avg_purchase_value = float(df_txn_filtered["purchase_amount"].mean()) if len(df_txn_filtered) else 0.0

    rev_by_cust = (
        df_txn_filtered.groupby("customer_id", as_index=False)["purchase_amount"]
        .sum()
        .rename(columns={"purchase_amount": "customer_revenue"})
        .sort_values("customer_revenue", ascending=False)
    )

    n = len(rev_by_cust)
    k = int(np.ceil(0.10 * n)) if n > 0 else 0
    k = max(1, k) if n > 0 else 0

    top10_value = float(rev_by_cust.head(k)["customer_revenue"].sum()) if k > 0 else 0.0
    top10_share = float(top10_value / total_revenue) if total_revenue > 0 else 0.0

    return {
        "total_revenue": total_revenue,
        "active_customers": float(active_customers),
        "avg_purchase_value": avg_purchase_value,
        "top10_revenue_value": top10_value,
        "top10_revenue_share": top10_share,
    }


# -----------------------------
# Early warning rules
# -----------------------------
def compute_early_warning_flags(df_txn_filtered: pd.DataFrame, df_cust_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based early-warning flags (simple + explainable):

    Rule A (Revenue decline + recency):
      - For each customer, compute monthly revenue for the last 2 months present in filtered data.
      - decline = (rev_last_month - rev_prev_month) / rev_prev_month (only if prev_month > 0)
      - Flag if decline <= -0.30 (>=30% drop) AND recency_days > 30.

    Rule B (Low satisfaction + falling purchase frequency):
      - avg_satisfaction <= 2.0
      - purchases in last 60 days < purchases in previous 60 days
        (relative to dataset max date in filtered data)

    Returns flagged customer table with a flag_reason string.
    """
    if df_txn_filtered.empty or df_cust_filtered.empty:
        return pd.DataFrame(columns=["customer_id", "flag_reason"])

    max_date = df_txn_filtered["transaction_date"].max()

    # Determine last two months present in data
    ym_sorted = (
        df_txn_filtered[["year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
        .assign(ym=lambda x: x["year"].astype(str) + "-" + x["month"].astype(str).str.zfill(2))
    )
    last_two = ym_sorted["ym"].tail(2).tolist()

    df_txn = df_txn_filtered.copy()
    df_txn["ym"] = df_txn["year"].astype(str) + "-" + df_txn["month"].astype(str).str.zfill(2)

    mrev = (
        df_txn.groupby(["customer_id", "ym"], as_index=False)["purchase_amount"]
        .sum()
        .rename(columns={"purchase_amount": "monthly_revenue"})
    )

    rule_a_ids = set()
    if len(last_two) == 2:
        prev, last = last_two[0], last_two[1]
        pivot = mrev[mrev["ym"].isin([prev, last])].pivot_table(
            index="customer_id", columns="ym", values="monthly_revenue", aggfunc="sum", fill_value=0.0
        ).reset_index()

        prev_rev = pivot.get(prev, pd.Series([0.0] * len(pivot)))
        last_rev = pivot.get(last, pd.Series([0.0] * len(pivot)))
        pivot["decline_pct"] = np.where(prev_rev > 0, (last_rev - prev_rev) / prev_rev, np.nan)

        pivot = pivot.merge(df_cust_filtered[["customer_id", "recency_days"]], on="customer_id", how="left")
        rule_a = (pivot["decline_pct"] <= -0.30) & (pivot["recency_days"] > 30)
        rule_a_ids = set(pivot.loc[rule_a, "customer_id"].astype(str).tolist())

    # Rule B: satisfaction <=2 + falling purchase frequency
    df_txn["days_from_max"] = (max_date - df_txn["transaction_date"]).dt.days.astype(int)
    last_60 = df_txn[df_txn["days_from_max"] <= 60]
    prev_60 = df_txn[(df_txn["days_from_max"] > 60) & (df_txn["days_from_max"] <= 120)]

    p_last = last_60.groupby("customer_id")["transaction_id"].nunique().rename("purchases_last_60")
    p_prev = prev_60.groupby("customer_id")["transaction_id"].nunique().rename("purchases_prev_60")
    freq = pd.concat([p_last, p_prev], axis=1).fillna(0).reset_index()
    freq["customer_id"] = freq["customer_id"].astype(str)

    cust = df_cust_filtered.copy()
    cust["customer_id"] = cust["customer_id"].astype(str)
    freq = freq.merge(cust[["customer_id", "avg_satisfaction"]], on="customer_id", how="left")

    rule_b = (freq["avg_satisfaction"] <= 2.0) & (freq["purchases_last_60"] < freq["purchases_prev_60"])
    rule_b_ids = set(freq.loc[rule_b, "customer_id"].astype(str).tolist())

    flagged_ids = sorted(rule_a_ids.union(rule_b_ids))
    if not flagged_ids:
        return pd.DataFrame(columns=["customer_id", "flag_reason"])

    flagged = cust[cust["customer_id"].isin(flagged_ids)].copy()

    def build_reason(cid: str) -> str:
        reasons: List[str] = []
        if cid in rule_a_ids:
            reasons.append("Revenue decline ≥30% (last month vs prior) + Recency > 30d")
        if cid in rule_b_ids:
            reasons.append("Low satisfaction (≤2) + Falling purchase frequency (last 60d vs prior 60d)")
        return " | ".join(reasons)

    flagged["flag_reason"] = flagged["customer_id"].apply(build_reason)
    flagged = flagged.sort_values(["total_revenue", "recency_days"], ascending=[False, False])
    return flagged


# -----------------------------
# Charts (Plotly Express; interactive)
# -----------------------------
def chart_revenue_by_label(df_txn: pd.DataFrame):
    if df_txn.empty:
        return None

    by = (
        df_txn.groupby("label", as_index=False)
        .agg(
            revenue=("purchase_amount", "sum"),
            customers=("customer_id", "nunique"),
            avg_purchase=("purchase_amount", "mean"),
        )
        .copy()
    )
    by["label"] = by["label"].astype(str)
    label_order = order_labels_if_possible(by["label"].tolist())
    by["label"] = pd.Categorical(by["label"], categories=label_order, ordered=True)
    by = by.sort_values("label")

    fig = px.bar(
        by,
        x="label",
        y="revenue",
        hover_data={"customers": True, "avg_purchase": ":.2f", "revenue": ":.2f"},
        title="Revenue by Segment (Label)",
    )
    fig.update_layout(yaxis_title="Total Revenue", xaxis_title="Segment")
    return fig


def chart_region_channel(df_txn: pd.DataFrame):
    if df_txn.empty:
        return None

    by = (
        df_txn.groupby(["customer_region", "retail_channel"], as_index=False)
        .agg(revenue=("purchase_amount", "sum"))
        .copy()
    )
    fig = px.bar(
        by,
        x="customer_region",
        y="revenue",
        color="retail_channel",
        barmode="stack",
        title="Revenue by Region × Retail Channel",
    )
    fig.update_layout(yaxis_title="Total Revenue", xaxis_title="Region", legend_title="Channel")
    return fig


def chart_revenue_trend_by_label(df_txn: pd.DataFrame):
    """Average revenue per customer over time by Label (simple cohort/retention proxy)."""
    if df_txn.empty:
        return None

    by = (
        df_txn.groupby(["year_month", "label"], as_index=False)
        .agg(revenue=("purchase_amount", "sum"), customers=("customer_id", "nunique"))
        .copy()
    )
    by["avg_rev_per_customer"] = np.where(by["customers"] > 0, by["revenue"] / by["customers"], 0.0)
    by["year_month"] = pd.to_datetime(by["year_month"] + "-01")

    label_order = order_labels_if_possible(by["label"].astype(str).tolist())
    fig = px.line(
        by,
        x="year_month",
        y="avg_rev_per_customer",
        color="label",
        category_orders={"label": label_order},
        markers=True,
        title="Average Revenue per Customer Over Time (by Label)",
    )
    fig.update_layout(yaxis_title="Avg Revenue / Customer", xaxis_title="Month", legend_title="Segment")
    return fig


def chart_category_heatmap(df_txn: pd.DataFrame):
    """Heatmap: ProductCategory × Label revenue contribution."""
    if df_txn.empty:
        return None

    pivot = (
        df_txn.groupby(["product_category", "label"], as_index=False)
        .agg(revenue=("purchase_amount", "sum"))
        .copy()
    )
    if pivot.empty:
        return None

    heat = pivot.pivot_table(
        index="product_category", columns="label", values="revenue", aggfunc="sum", fill_value=0.0
    )
    col_order = order_labels_if_possible(list(map(str, heat.columns.tolist())))
    heat = heat.reindex(columns=col_order)

    fig = px.imshow(
        heat,
        aspect="auto",
        title="Revenue Heatmap: Product Category × Label",
        labels=dict(x="Segment (Label)", y="Product Category", color="Revenue"),
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def chart_risky_customers_scatter(df_cust: pd.DataFrame):
    """Scatter: recency_days vs avg_purchase_value, size=total_revenue, color=Label."""
    if df_cust.empty:
        return None

    size = df_cust["total_revenue"].clip(lower=0)
    fig = px.scatter(
        df_cust,
        x="recency_days",
        y="avg_purchase_value",
        size=size,
        color="label",
        hover_data={
            "customer_id": True,
            "total_revenue": ":.2f",
            "purchase_count": True,
            "avg_satisfaction": ":.2f",
            "recency_days": True,
        },
        title="Risk / Opportunity Scatter: Recency vs Avg Purchase Value",
    )
    fig.update_layout(xaxis_title="Recency (days since last purchase)", yaxis_title="Avg Purchase Value")
    return fig


# -----------------------------
# Sidebar filters
# -----------------------------
def build_filters(df_txn: pd.DataFrame) -> FilterState:
    st.sidebar.header("Filters")

    if df_txn.empty:
        return FilterState(
            label=["All"],
            customer_region=["All"],
            retail_channel=["All"],
            product_category=["All"],
            customer_age_group=["All"],
            customer_gender=["All"],
            year=["All"],
            month=["All"],
        )

    label_options = add_all_option(order_labels_if_possible(df_txn["label"].dropna().astype(str).unique().tolist()))
    region_options = add_all_option(sorted(df_txn["customer_region"].dropna().astype(str).unique().tolist()))
    channel_options = add_all_option(sorted(df_txn["retail_channel"].dropna().astype(str).unique().tolist()))
    category_options = add_all_option(sorted(df_txn["product_category"].dropna().astype(str).unique().tolist()))
    age_options = add_all_option(sorted(df_txn["customer_age_group"].dropna().astype(str).unique().tolist()))
    gender_options = add_all_option(sorted(df_txn["customer_gender"].dropna().astype(str).unique().tolist()))

    # Store year/month as strings so "All" works consistently in multiselect
    years = sorted(df_txn["year"].dropna().astype(int).unique().tolist())
    months = sorted(df_txn["month"].dropna().astype(int).unique().tolist())
    year_options = add_all_option([str(y) for y in years])
    month_options = add_all_option([str(m) for m in months])

    return FilterState(
        label=st.sidebar.multiselect("Label", options=label_options, default=["All"]),
        customer_region=st.sidebar.multiselect("Customer Region", options=region_options, default=["All"]),
        retail_channel=st.sidebar.multiselect("Retail Channel", options=channel_options, default=["All"]),
        product_category=st.sidebar.multiselect("Product Category", options=category_options, default=["All"]),
        customer_age_group=st.sidebar.multiselect("Customer Age Group", options=age_options, default=["All"]),
        customer_gender=st.sidebar.multiselect("Customer Gender", options=gender_options, default=["All"]),
        year=st.sidebar.multiselect("Year", options=year_options, default=["All"]),
        month=st.sidebar.multiselect("Month", options=month_options, default=["All"]),
    )


def apply_filters(df_txn: pd.DataFrame, fs: FilterState) -> pd.DataFrame:
    df = df_txn.copy()

    df = apply_multiselect_filter(df, "label", fs.label)
    df = apply_multiselect_filter(df, "customer_region", fs.customer_region)
    df = apply_multiselect_filter(df, "retail_channel", fs.retail_channel)
    df = apply_multiselect_filter(df, "product_category", fs.product_category)
    df = apply_multiselect_filter(df, "customer_age_group", fs.customer_age_group)
    df = apply_multiselect_filter(df, "customer_gender", fs.customer_gender)

    if fs.year and "All" not in fs.year:
        years = [int(y) for y in fs.year]
        df = df[df["year"].isin(years)].copy()

    if fs.month and "All" not in fs.month:
        months = [int(m) for m in fs.month]
        df = df[df["month"].isin(months)].copy()

    return df


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.title("NovaRetail Customer Insights Dashboard")
    st.caption(
        "Stakeholder: Director of Customer Intelligence • Dataset: NR_dataset.xlsx • "
        "Purpose: Explore customer behavior, segment health, and revenue patterns for NovaRetail."
    )

    # ---- Load data ----
    path = DATA_FILE
    if not os.path.exists(path):
        st.error(
            f"Could not find '{DATA_FILE}' in the current directory.\n\n"
            "Make sure the repo root contains:\n"
            "- app.py\n- NR_dataset.xlsx\n- requirements.txt"
        )
        st.stop()

    df_raw, err = load_raw_excel(path)
    if err or df_raw is None:
        st.error(err or "Unknown error loading the dataset.")
        st.stop()

    # ---- Normalize + validate columns ----
    mapping, missing = map_required_columns(df_raw)
    if missing:
        st.error("Missing required fields: " + ", ".join(missing))
        st.write("Available columns (debug):")
        st.write(list(df_raw.columns))
        st.stop()

    # Keep ONLY required variables, rename to canonical
    df_req = select_and_rename_required(df_raw, mapping)

    # ---- Preprocess ----
    df_txn, stats = preprocess_transactions(df_req)

    if stats["invalid_date"] > 0 or stats["invalid_amount"] > 0 or stats["dropped_rows"] > 0:
        with st.expander("Data quality notes (click to expand)", expanded=False):
            st.write(
                {
                    "rows_loaded": int(len(df_req)),
                    "invalid_transaction_date_rows": stats["invalid_date"],
                    "non_numeric_purchase_amount_rows": stats["invalid_amount"],
                    "rows_dropped_for_core_metrics": stats["dropped_rows"],
                    "rows_remaining": stats["remaining_rows"],
                }
            )
            st.caption("Rows are dropped only when TransactionDate or PurchaseAmount cannot be parsed.")

    # ---- Sidebar filters ----
    fs = build_filters(df_txn)
    df_txn_f = apply_filters(df_txn, fs)

    # Keep both transaction-level and aggregated customer-level dataframes in memory
    df_cust_f = compute_customer_aggregates(df_txn_f)

    # ---- KPIs ----
    kpis = compute_kpis(df_txn_f)
    flagged = compute_early_warning_flags(df_txn_f, df_cust_f)

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Revenue", f"${kpis['total_revenue']:,.2f}")
    kpi_cols[1].metric("Active Customers", f"{int(kpis['active_customers']):,}")
    kpi_cols[2].metric("Avg Purchase Value", f"${kpis['avg_purchase_value']:,.2f}")
    kpi_cols[3].metric(
        "Top 10% Revenue",
        f"${kpis['top10_revenue_value']:,.2f}",
        help=(
            "Computed within filtered data: revenue per customer → sort desc → "
            "top K=ceil(0.10*N) customers → sum."
        ),
    )
    kpi_cols[4].metric(
        "Top 10% Share",
        f"{kpis['top10_revenue_share']*100:,.1f}%",
        help="Top10% share = (Top10% revenue) / (Total revenue) within filtered data.",
    )

    # ---- Early warning summary ----
    warn_cols = st.columns([1, 2])
    with warn_cols[0]:
        st.subheader("Early Warning")
        st.metric("Flagged Customers", f"{len(flagged):,}")
        st.caption("Rule-based flags to spot declining or at-risk customers (rules documented in code).")

    with warn_cols[1]:
        if len(flagged) > 0:
            show_cols = ["customer_id", "label", "total_revenue", "recency_days", "avg_satisfaction", "flag_reason"]
            show_cols = [c for c in show_cols if c in flagged.columns]
            st.dataframe(flagged[show_cols].head(12), use_container_width=True, hide_index=True)
        else:
            st.info("No customers are currently flagged under the selected filters.")

    st.divider()

    # ---- Charts (2-column wide layout) ----
    left, right = st.columns(2)

    fig1 = chart_revenue_by_label(df_txn_f)
    with left:
        if fig1 is None:
            st.info("No data available for 'Revenue by Segment' under current filters.")
        else:
            st.plotly_chart(fig1, use_container_width=True)

    fig2 = chart_region_channel(df_txn_f)
    with right:
        if fig2 is None:
            st.info("No data available for 'Revenue by Region × Channel' under current filters.")
        else:
            st.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2)

    fig3 = chart_revenue_trend_by_label(df_txn_f)
    with left2:
        if fig3 is None:
            st.info("Not enough data to build the time trend chart under current filters.")
        else:
            st.plotly_chart(fig3, use_container_width=True)

    fig4 = chart_category_heatmap(df_txn_f)
    with right2:
        if fig4 is None:
            st.info("No data available for the category heatmap under current filters.")
        else:
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ---- Risk scatter + tables ----
    scatter_col, tables_col = st.columns([1.1, 0.9])

    with scatter_col:
        st.subheader("Risk / Opportunity Explorer")
        fig5 = chart_risky_customers_scatter(df_cust_f)
        if fig5 is None:
            st.info("No customer-level data available under current filters.")
        else:
            st.plotly_chart(fig5, use_container_width=True)
        st.caption("Hint: High recency (right) + low avg purchase value (bottom) can indicate risk.")

    with tables_col:
        st.subheader("Top Customers (by Revenue)")
        if df_cust_f.empty:
            st.info("No customers under current filters.")
        else:
            top_n = st.slider("Top N", min_value=10, max_value=200, value=25, step=5)
            top_customers = df_cust_f.sort_values("total_revenue", ascending=False).head(top_n).copy()

            display_cols = [
                "customer_id",
                "label",
                "customer_region",
                "retail_channel",
                "product_category",
                "total_revenue",
                "purchase_count",
                "avg_purchase_value",
                "avg_satisfaction",
                "recency_days",
            ]
            display_cols = [c for c in display_cols if c in top_customers.columns]

            st.dataframe(top_customers[display_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Top Customers (CSV)",
                data=as_csv_bytes(top_customers[display_cols]),
                file_name="top_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()

    # ---- Transactions table + download ----
    st.subheader("Filtered Transactions")
    st.caption("This table reflects the current filters and includes derived year/month/recency_days.")
    if df_txn_f.empty:
        st.info("No transactions under current filters.")
    else:
        txn_cols = [
            "customer_id",
            "transaction_id",
            "transaction_date",
            "label",
            "purchase_amount",
            "customer_satisfaction",
            "customer_region",
            "retail_channel",
            "product_category",
            "customer_age_group",
            "customer_gender",
            "year",
            "month",
            "recency_days",
        ]
        txn_cols = [c for c in txn_cols if c in df_txn_f.columns]
        st.dataframe(df_txn_f[txn_cols], use_container_width=True, hide_index=True)

        dl_cols = st.columns([1, 1, 2])
        with dl_cols[0]:
            st.download_button(
                "Download Filtered Transactions (CSV)",
                data=as_csv_bytes(df_txn_f[txn_cols]),
                file_name="filtered_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "Download Flagged Customers (CSV)",
                data=as_csv_bytes(flagged) if len(flagged) else b"customer_id,flag_reason\n",
                file_name="flagged_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_cols[2]:
            st.caption("Exports match exactly what you’re seeing under the current filters.")

    with st.expander("How to interpret the dashboard (quick notes)", expanded=False):
        st.markdown(
            """
- **Revenue concentration:** Use *Top 10% Share* to understand how dependent revenue is on a small group.
- **Early warning:** Prioritize flagged customers with high recency and/or low satisfaction.
- **Investments:** Look for segments/regions/channels with a rising *avg revenue per customer* trend.
            """.strip()
        )

    st.divider()

    # Deployment instructions (required)
    # 1) Put app.py, NR_dataset.xlsx, and requirements.txt in the repo root, then git push to GitHub.
    # 2) In Streamlit Cloud, create a new app from the repo and set the entrypoint to app.py.
    # 3) Ensure requirements.txt includes: streamlit, pandas, numpy, plotly, openpyxl.


if __name__ == "__main__":
    main()
