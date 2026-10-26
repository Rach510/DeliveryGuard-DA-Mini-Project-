"""
analytics.py
------------
Pure functions that compute statistics from cleaned DataFrames.
No I/O, no side effects — easy to test and reuse.
"""

import pandas as pd
import numpy as np



def compute_overview(order_df: pd.DataFrame) -> dict:
    """
    Return headline metrics for the Overview section.

    Parameters
    ----------
    order_df : DataFrame
        Order-level cleaned dataset (output of data_preprocessing.py).

    Returns
    -------
    dict with keys:
        total_orders, late_orders, on_time_orders,
        late_rate_pct, avg_delivery_days, median_delivery_days,
        avg_delay_days (among late orders only),
        avg_delay_days_all (positive = late, negative = early),
        pct_early, pct_on_time, pct_late
    """
    n = len(order_df)
    if n == 0:
        return {}

    late = int(order_df["is_late"].sum())
    exact_on_time = int((order_df["delay_days"] == 0).sum())

    late_orders = order_df[order_df["is_late"] == 1]
    avg_late_delay = float(late_orders["delay_days"].mean()) if len(late_orders) else 0.0

    early   = (order_df["delay_days"] < 0).sum()
    exact   = (order_df["delay_days"] == 0).sum()

    return {
        "total_orders":      n,
        "late_orders":       late,
        "on_time_orders":    exact_on_time,
        "not_late_orders":   n - late,
        "late_rate_pct":     round(late / n * 100, 2),
        "avg_delivery_days": round(float(order_df["delivery_days"].mean()), 2),
        "median_delivery_days": round(float(order_df["delivery_days"].median()), 2),
        "avg_delay_days":    round(avg_late_delay, 2),
        "avg_delay_days_all": round(float(order_df["delay_days"].mean()), 2),
        "pct_early":   round(early / n * 100, 2),
        "pct_on_time": round(exact / n * 100, 2),
        "pct_late":    round(late / n * 100, 2),
    }



def late_rate_by_category(
    item_df: pd.DataFrame,
    min_items: int = 100,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Calculate late-delivery rates per product category.

    Parameters
    ----------
    item_df    : Item-level DataFrame (one row per order-item).
    min_items  : Exclude categories with fewer observations.
    top_n      : Return at most this many rows (sorted by late rate descending).

    Returns
    -------
    DataFrame with columns:
        category, total_items, late_items, late_rate_pct
    """
    if item_df.empty:
        return pd.DataFrame()

    stats = (
        item_df.groupby("category")
        .agg(
            total_items = ("order_id", "size"),
            late_items  = ("is_late",  "sum"),
        )
        .reset_index()
    )
    stats["late_rate_pct"] = (stats["late_items"] / stats["total_items"] * 100).round(2)

    stats = stats[stats["total_items"] >= min_items]
    stats = stats.sort_values("late_rate_pct", ascending=False).head(top_n)
    return stats.reset_index(drop=True)


def delivery_days_by_category(
    item_df: pd.DataFrame,
    min_items: int = 100,
) -> pd.DataFrame:
    """Average and median delivery days per category."""
    if item_df.empty:
        return pd.DataFrame()

    stats = (
        item_df.groupby("category")
        .agg(
            total_items   = ("order_id",       "size"),
            avg_days      = ("delivery_days",  "mean"),
            median_days   = ("delivery_days",  "median"),
        )
        .reset_index()
    )
    stats = stats[stats["total_items"] >= min_items]
    stats["avg_days"]    = stats["avg_days"].round(2)
    stats["median_days"] = stats["median_days"].round(2)
    return stats.sort_values("avg_days", ascending=False).reset_index(drop=True)



def late_rate_by_seller(
    order_df: pd.DataFrame,
    min_orders: int = 20,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Late-delivery rate per seller (using primary_seller_id).

    Parameters
    ----------
    order_df   : Order-level DataFrame.
    min_orders : Exclude sellers with fewer orders (avoids noise).
    top_n      : Return the top N sellers by late rate.

    Returns
    -------
    DataFrame with columns:
        primary_seller_id, total_orders, late_orders, late_rate_pct
    """
    if order_df.empty or "primary_seller_id" not in order_df.columns:
        return pd.DataFrame()

    stats = (
        order_df.groupby("primary_seller_id")
        .agg(
            total_orders = ("order_id", "count"),
            late_orders  = ("is_late",  "sum"),
        )
        .reset_index()
    )
    stats["late_rate_pct"] = (stats["late_orders"] / stats["total_orders"] * 100).round(2)
    stats = stats[stats["total_orders"] >= min_orders]
    stats = stats.sort_values("late_rate_pct", ascending=False).head(top_n)
    return stats.reset_index(drop=True)



def late_rate_by_month(order_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly late-delivery rate (ordered chronologically)."""
    if order_df.empty:
        return pd.DataFrame()

    df = order_df.copy()
    df["year_month"] = (
        df["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    stats = (
        df.groupby("year_month")
        .agg(
            total_orders = ("order_id", "count"),
            late_orders  = ("is_late",  "sum"),
        )
        .reset_index()
    )
    stats["late_rate_pct"] = (stats["late_orders"] / stats["total_orders"] * 100).round(2)
    return stats.sort_values("year_month").reset_index(drop=True)


def late_rate_by_dayofweek(order_df: pd.DataFrame) -> pd.DataFrame:
    """Late rate by day of week the order was placed."""
    if order_df.empty:
        return pd.DataFrame()

    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    stats = (
        order_df.groupby("purchase_dayofweek")
        .agg(
            total_orders = ("order_id", "count"),
            late_orders  = ("is_late",  "sum"),
        )
        .reset_index()
    )
    stats["day_name"] = stats["purchase_dayofweek"].map(day_names)
    stats["late_rate_pct"] = (stats["late_orders"] / stats["total_orders"] * 100).round(2)
    return stats.sort_values("purchase_dayofweek").reset_index(drop=True)



def accuracy_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    """
    Breakdown of deliveries into early / on-time / late buckets
    and the average days for each bucket.
    """
    if order_df.empty:
        return pd.DataFrame()

    def bucket(d):
        if d < 0:
            return "Early"
        elif d == 0:
            return "On time"
        else:
            return "Late"

    df = order_df.copy()
    df["bucket"] = df["delay_days"].apply(bucket)

    stats = (
        df.groupby("bucket")
        .agg(
            count         = ("order_id",    "count"),
            avg_delay_days = ("delay_days", "mean"),
        )
        .reset_index()
    )
    stats["pct"] = (stats["count"] / stats["count"].sum() * 100).round(2)
    stats["avg_delay_days"] = stats["avg_delay_days"].round(2)
    return stats


def delay_distribution(order_df: pd.DataFrame) -> pd.Series:
    """Return the delay_days series for plotting (all orders)."""
    return order_df["delay_days"].dropna()



def late_rate_by_state(order_df: pd.DataFrame, col: str = "customer_state") -> pd.DataFrame:
    """Late-delivery rate grouped by a state column (customer or seller)."""
    if order_df.empty or col not in order_df.columns:
        return pd.DataFrame()

    stats = (
        order_df.groupby(col)
        .agg(
            total_orders = ("order_id", "count"),
            late_orders  = ("is_late",  "sum"),
        )
        .reset_index()
    )
    stats["late_rate_pct"] = (stats["late_orders"] / stats["total_orders"] * 100).round(2)
    return stats.sort_values("late_rate_pct", ascending=False).reset_index(drop=True)
