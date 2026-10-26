"""
data_preprocessing.py
---------------------
Loads, cleans, merges, and enriches the raw Olist CSV files.
Produces two outputs:
  - data/processed/cleaned_orders.csv   (order-level)
  - data/processed/item_level.csv       (item-level, for category analysis)
"""

import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

ORDERS_FILE   = RAW_DIR / "olist_orders_dataset.csv"
ITEMS_FILE    = RAW_DIR / "olist_order_items_dataset.csv"
PRODUCTS_FILE = RAW_DIR / "olist_products_dataset.csv"
SELLERS_FILE  = RAW_DIR / "olist_sellers_dataset.csv"
CATEGORY_TRANSLATION_FILE = RAW_DIR / "product_category_name_translation.csv"
GEOLOCATION_FILE = RAW_DIR / "olist_geolocation_dataset.csv"
CUSTOMERS_FILE   = RAW_DIR / "olist_customers_dataset.csv"



DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date string columns to datetime."""
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _select_completed_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only delivered orders that have both an actual delivery date
    and an estimated delivery date.  Document clearly what is dropped.
    """
    original_count = len(orders)

    completed = orders[
        orders["order_status"].eq("delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    dropped = original_count - len(completed)
    print(
        f"  Orders in raw file:          {original_count:,}\n"
        f"  Delivered with known dates:  {len(completed):,}\n"
        f"  Dropped (other statuses):    {dropped:,}"
    )
    return completed


def _engineer_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add delivery-duration and delay columns."""

    df["delivery_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400

    df["delay_days"] = (
        df["order_delivered_customer_date"]
        - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86_400

    df["is_late"] = (df["delay_days"] > 0).astype(int)

    df["purchase_month"]      = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"]  = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"]       = df["order_purchase_timestamp"].dt.hour
    df["purchase_year"]       = df["order_purchase_timestamp"].dt.year

    df["estimated_window_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400

    return df


def _aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the order-items file to one row per order.
    Avoids double-counting when an order has multiple items.
    """
    agg = (
        items.groupby("order_id")
        .agg(
            total_price   = ("price",         "sum"),
            total_freight = ("freight_value",  "sum"),
            item_count    = ("order_item_id",  "count"),
            seller_count  = ("seller_id",      "nunique"),
            primary_seller_id = ("seller_id", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )
    return agg



def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full pipeline.

    Returns
    -------
    order_df : pd.DataFrame
        Order-level dataset (one row per order).
    item_df : pd.DataFrame
        Item-level dataset (one row per order-item, joined with outcomes).
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


    print("\n[1/6] Loading raw CSV files …")

    orders   = pd.read_csv(ORDERS_FILE)
    items    = pd.read_csv(ITEMS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    print(f"  orders:   {orders.shape}")
    print(f"  items:    {items.shape}")
    print(f"  products: {products.shape}")

    sellers = None
    if SELLERS_FILE.exists():
        sellers = pd.read_csv(SELLERS_FILE)
        print(f"  sellers:  {sellers.shape}")

    cat_trans = None
    if CATEGORY_TRANSLATION_FILE.exists():
        cat_trans = pd.read_csv(CATEGORY_TRANSLATION_FILE)
        print(f"  category translation loaded")

    customers = None
    if CUSTOMERS_FILE.exists():
        customers = pd.read_csv(CUSTOMERS_FILE)
        print(f"  customers: {customers.shape}")


    print("\n[2/6] Parsing date columns …")
    orders = _parse_dates(orders)


    print("\n[3/6] Filtering completed orders …")
    completed = _select_completed_orders(orders)


    print("\n[4/6] Engineering delivery features …")
    completed = _engineer_delivery_features(completed)

    neg = (completed["delivery_days"] < 0).sum()
    ext = (completed["delivery_days"] > 365).sum()
    if neg:
        print(f"  WARNING: {neg} orders have negative delivery_days — dropped")
        completed = completed[completed["delivery_days"] >= 0]
    if ext:
        print(f"  NOTE: {ext} orders have delivery_days > 365 (kept, investigate if needed)")


    print("\n[5/6] Merging datasets …")

    # Translate category names to English if available
    if cat_trans is not None:
        products = products.merge(
            cat_trans,
            on="product_category_name",
            how="left",
        )
        products["category"] = products["product_category_name_english"].fillna(
            products["product_category_name"]
        )
    else:
        products["category"] = products["product_category_name"]

    order_items_agg = _aggregate_items(items)

    order_df = completed.merge(order_items_agg, on="order_id", how="left")

    if customers is not None:
        order_df = order_df.merge(
            customers[["customer_id", "customer_state", "customer_city"]],
            on="customer_id",
            how="left",
        )

    if sellers is not None:
        order_df = order_df.merge(
            sellers[["seller_id", "seller_state", "seller_city"]].rename(
                columns={"seller_id": "primary_seller_id"}
            ),
            on="primary_seller_id",
            how="left",
        )

    item_df = items.merge(
        completed[
            [
                "order_id",
                "is_late",
                "delay_days",
                "delivery_days",
                "order_purchase_timestamp",
                "purchase_month",
                "purchase_dayofweek",
                "purchase_year",
            ]
        ],
        on="order_id",
        how="inner",
    )
    item_df = item_df.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )
    item_df["category"] = item_df["category"].fillna("unknown")


    print("\n[6/6] Saving processed files …")

    order_out = PROCESSED_DIR / "cleaned_orders.csv"
    item_out  = PROCESSED_DIR / "item_level.csv"

    order_df.to_csv(order_out, index=False)
    item_df.to_csv(item_out,  index=False)

    print(f"  Order-level rows saved: {len(order_df):,}  → {order_out}")
    print(f"  Item-level rows saved:  {len(item_df):,}  → {item_out}")
    print("\nPreprocessing complete.\n")

    return order_df, item_df


if __name__ == "__main__":
    run_preprocessing()
