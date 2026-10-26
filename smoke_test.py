"""Quick end-to-end checks for DeliveryGuard. Run: python smoke_test.py"""
from pathlib import Path
import pandas as pd
from src import analytics, prediction

PROJECT_DIR = Path(__file__).resolve().parent
ORDER_PATH = PROJECT_DIR / "data" / "processed" / "cleaned_orders.csv"
ITEM_PATH = PROJECT_DIR / "data" / "processed" / "item_level.csv"


def main():
    assert ORDER_PATH.exists(), f"Missing {ORDER_PATH}; run python -m src.data_preprocessing first."
    assert ITEM_PATH.exists(), f"Missing {ITEM_PATH}; run python -m src.data_preprocessing first."

    orders = pd.read_csv(ORDER_PATH, parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(ITEM_PATH, parse_dates=["order_purchase_timestamp"])

    assert not orders.empty, "Processed order data is empty."
    assert not items.empty, "Processed item data is empty."
    assert orders["order_id"].is_unique, "cleaned_orders.csv must have one row per order."
    assert orders["is_late"].isin([0, 1]).all(), "is_late must contain only 0 and 1."
    assert (orders["delivery_days"] >= 0).all(), "Negative delivery duration found."
    assert items["category"].notna().all(), "Item categories contain missing values."

    overview = analytics.compute_overview(orders)
    assert overview["total_orders"] == len(orders)
    assert 0 <= overview["late_rate_pct"] <= 100
    exact_example = pd.DataFrame({
        "order_id": ["early", "on_time", "late"],
        "is_late": [0, 0, 1],
        "delay_days": [-1.0, 0.0, 2.0],
        "delivery_days": [4.0, 5.0, 7.0],
    })
    exact_stats = analytics.compute_overview(exact_example)
    assert exact_stats["on_time_orders"] == 1
    assert exact_stats["not_late_orders"] == 2
    print("PASS exact on-time metric definition")

    for label, result in [
        ("category", analytics.late_rate_by_category(items)),
        ("delivery by category", analytics.delivery_days_by_category(items)),
        ("seller", analytics.late_rate_by_seller(orders)),
        ("month", analytics.late_rate_by_month(orders)),
        ("day of week", analytics.late_rate_by_dayofweek(orders)),
        ("accuracy", analytics.accuracy_summary(orders)),
        ("state", analytics.late_rate_by_state(orders)),
    ]:
        assert isinstance(result, pd.DataFrame), f"{label} analytics did not return a DataFrame."
        print(f"PASS analytics: {label} ({len(result)} rows)")

    dates = pd.date_range("2024-01-01", periods=10, freq="10D")
    sample = pd.DataFrame({
        "order_purchase_timestamp": dates,
        "order_delivered_customer_date": dates + pd.Timedelta(days=2),
        "primary_seller_id": ["same_seller"] * 10,
        "is_late": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    })
    rated = prediction.add_seller_historical_rate(sample)
    assert rated["seller_historical_rate"].notna().all()
    assert rated["seller_historical_rate"].iloc[0] == 0.0
    assert abs(rated["seller_historical_rate"].iloc[5] - 0.4) < 1e-9
    print("PASS seller history uses only outcomes available before purchase")

    model = prediction.train_model(orders)
    if "error" in model:
        print("MODEL NOT TRAINED:", model["error"])
    else:
        print("PASS model training")
        print("Model metrics:", model["metrics"])
        guess = prediction.predict_single(model, 10, 2, 20, 100, 15, 1, 1, 0.3)
        assert "probability_late" in guess, guess
        print("PASS single-order prediction:", guess)

    print("\nALL AVAILABLE SMOKE CHECKS PASSED")
    print(f"Orders: {len(orders):,}; items: {len(items):,}; late rate: {overview['late_rate_pct']}%")


if __name__ == "__main__":
    main()
