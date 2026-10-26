"""
exploration.py
--------------
Run this in Jupyter or as a plain Python script to explore the raw
dataset before running the preprocessing pipeline.

Usage in Jupyter:
  jupyter notebook  →  open notebooks/exploration.ipynb
  (or convert this file: jupytext --to notebook exploration.py)

Usage as script:
  cd DeliveryGuard && python notebooks/exploration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
RAW_DIR = Path("data/raw")



print("=" * 60)
print("STEP 1 — Raw data overview")
print("=" * 60)

orders   = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
items    = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")

for name, df in [("orders", orders), ("items", items), ("products", products)]:
    print(f"\n{name.upper()}")
    print(f"  Shape:   {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Nulls:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")



print("\n" + "=" * 60)
print("STEP 2 — Order status distribution")
print("=" * 60)
print(orders["order_status"].value_counts())



print("\n" + "=" * 60)
print("STEP 3 — Delivery duration for delivered orders")
print("=" * 60)

DATE_COLS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for c in DATE_COLS:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")

delivered = orders[
    orders["order_status"].eq("delivered")
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
].copy()

delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86_400

delivered["delay_days"] = (
    delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86_400

delivered["is_late"] = (delivered["delay_days"] > 0).astype(int)

print(f"  Delivered orders with full dates: {len(delivered):,}")
print(f"  Late delivery rate: {delivered['is_late'].mean() * 100:.2f}%")
print(f"\n  Delivery duration (days):")
print(delivered["delivery_days"].describe().round(2).to_string())



fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(delivered["delivery_days"].clip(0, 60), bins=40, color="#2563EB", alpha=0.8)
axes[0].axvline(delivered["delivery_days"].mean(),   color="red",  linestyle="--", label=f"Mean {delivered['delivery_days'].mean():.1f}d")
axes[0].axvline(delivered["delivery_days"].median(), color="grey", linestyle=":",  label=f"Median {delivered['delivery_days'].median():.1f}d")
axes[0].set_title("Delivery duration distribution")
axes[0].set_xlabel("Days")
axes[0].legend()

axes[1].hist(delivered["delay_days"].clip(-30, 60), bins=50, color="#7C3AED", alpha=0.8)
axes[1].axvline(0, color="red", linewidth=2, label="Promised date")
axes[1].set_title("Delay distribution (−early, +late)")
axes[1].set_xlabel("Days relative to estimated delivery")
axes[1].legend()

counts = delivered["is_late"].value_counts().rename({0: "On time / Early", 1: "Late"})
axes[2].bar(counts.index, counts.values, color=["#16A34A", "#DC2626"])
axes[2].set_title("On-time vs late deliveries")
axes[2].set_xticks(range(len(counts)))
axes[2].set_xticklabels(counts.index)
for i, v in enumerate(counts.values):
    axes[2].text(i, v + 50, f"{v:,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("notebooks/exploration_plots.png", dpi=120, bbox_inches="tight")
print("\n  Plot saved to notebooks/exploration_plots.png")
plt.show()



cat_trans_path = RAW_DIR / "product_category_name_translation.csv"
if cat_trans_path.exists():
    cat_trans = pd.read_csv(cat_trans_path)
    products = products.merge(cat_trans, on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(products["product_category_name"])
else:
    products["category"] = products["product_category_name"]

item_delivered = items.merge(
    delivered[["order_id", "is_late", "delay_days"]],
    on="order_id",
    how="inner",
).merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
)
item_delivered["category"] = item_delivered["category"].fillna("unknown")

cat_stats = (
    item_delivered.groupby("category")
    .agg(total_items=("order_id", "size"), late_items=("is_late", "sum"))
    .assign(late_rate_pct=lambda d: d["late_items"] / d["total_items"] * 100)
    .query("total_items >= 100")
    .sort_values("late_rate_pct", ascending=False)
    .head(15)
)

print("\n" + "=" * 60)
print("STEP 5 — Top 15 categories by late rate (≥100 items)")
print("=" * 60)
print(cat_stats.to_string())

fig2, ax = plt.subplots(figsize=(9, 6))
ax.barh(cat_stats["category"][::-1], cat_stats["late_rate_pct"][::-1], color="#2563EB", alpha=0.85)
ax.set_xlabel("Late rate (%)")
ax.set_title("Top 15 product categories — late delivery rate")
plt.tight_layout()
plt.savefig("notebooks/category_late_rate.png", dpi=120, bbox_inches="tight")
print("  Plot saved to notebooks/category_late_rate.png")
plt.show()

print("\nExploration complete.")
