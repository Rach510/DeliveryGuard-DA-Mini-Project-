"""Train and evaluate the optional late-delivery baseline model."""

from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

FEATURE_COLUMNS = [
    "purchase_month",
    "purchase_dayofweek",
    "estimated_window_days",
    "total_price",
    "total_freight",
    "item_count",
    "seller_count",
    "seller_historical_rate",
]
TARGET = "is_late"


def add_seller_historical_rate(order_df: pd.DataFrame) -> pd.DataFrame:
    """Add seller history using only deliveries completed before each order."""
    df = order_df.sort_values("order_purchase_timestamp", kind="stable").copy()
    if df.empty:
        df["seller_historical_rate"] = pd.Series(dtype=float)
        return df

    purchase_col = "order_purchase_timestamp"
    delivery_col = "order_delivered_customer_date"
    if delivery_col not in df.columns:
        # Without delivery timestamps, availability of previous outcomes cannot be established.
        df["seller_historical_rate"] = 0.0
        return df

    history_events = []
    seller_total: dict[str, int] = {}
    seller_late: dict[str, int] = {}
    total_seen = 0
    late_seen = 0
    rates = np.zeros(len(df), dtype=float)

    df[purchase_col] = pd.to_datetime(df[purchase_col], errors="coerce")
    df[delivery_col] = pd.to_datetime(df[delivery_col], errors="coerce")
    rows = df[[purchase_col, delivery_col, "primary_seller_id", TARGET]].reset_index(drop=True)
    for i, row in rows.iterrows():
        purchase_time = row[purchase_col]
        if pd.isna(purchase_time):
            continue
        while history_events and history_events[0][0] < purchase_time:
            _, _, seller, outcome = heappop(history_events)
            total_seen += 1
            late_seen += outcome
            if seller is not None:
                seller_total[seller] = seller_total.get(seller, 0) + 1
                seller_late[seller] = seller_late.get(seller, 0) + outcome

        seller_value = row["primary_seller_id"]
        seller = None if pd.isna(seller_value) else str(seller_value)
        prior_seller_orders = seller_total.get(seller, 0) if seller is not None else 0
        if prior_seller_orders >= 5:
            rates[i] = seller_late[seller] / prior_seller_orders
        elif total_seen:
            rates[i] = late_seen / total_seen

        delivered_at = row[delivery_col]
        if pd.notna(delivered_at):
            heappush(history_events, (delivered_at, i, seller, int(row[TARGET])))

    df["seller_historical_rate"] = rates
    return df


def build_feature_matrix(order_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return chronologically ordered model features and target labels."""
    df = add_seller_historical_rate(order_df)
    missing = [col for col in FEATURE_COLUMNS + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {', '.join(missing)}")

    usable = df[FEATURE_COLUMNS + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    return usable[FEATURE_COLUMNS], usable[TARGET].astype(int)


def train_model(
    order_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Fit a chronological Logistic Regression baseline and report test metrics."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn is not installed. Run: python -m pip install scikit-learn"}
    if not 0.05 <= test_size <= 0.5:
        return {"error": "test_size must be between 0.05 and 0.50."}

    try:
        X, y = build_feature_matrix(order_df)
    except (KeyError, ValueError) as exc:
        return {"error": str(exc)}

    if len(X) < 200:
        return {"error": "At least 200 usable orders are needed to train this model."}

    split_idx = int(len(X) * (1 - test_size))
    if split_idx < 2 or len(X) - split_idx < 2:
        return {"error": "Not enough data for a chronological train/test split."}

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if y_train.nunique() < 2:
        return {"error": "The training period contains only one outcome class; the model cannot learn both outcomes."}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    majority_class = int(y_train.mode().iloc[0])
    baseline_accuracy = float((y_test == majority_class).mean())
    auc = float(roc_auc_score(y_test, y_prob)) if y_test.nunique() == 2 else float("nan")
    metrics = {
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(auc, 4) if np.isfinite(auc) else float("nan"),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "model_accuracy": round(float((y_pred == y_test).mean()), 4),
    }

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": model.coef_[0].round(4),
    }).sort_values("coefficient", key=lambda col: col.abs(), ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["On time", "Late"], zero_division=0
        ),
        "feature_importance": importance,
        "baseline_accuracy": baseline_accuracy,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_prob": y_prob,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def predict_single(
    model_result: dict,
    purchase_month: int,
    purchase_dayofweek: int,
    estimated_window_days: float,
    total_price: float,
    total_freight: float,
    item_count: int,
    seller_count: int,
    seller_historical_rate: float,
) -> dict:
    """Estimate a hypothetical order's late-delivery probability."""
    if "model" not in model_result:
        return {"error": model_result.get("error", "Model is not available.")}

    values = {
        "purchase_month": purchase_month,
        "purchase_dayofweek": purchase_dayofweek,
        "estimated_window_days": estimated_window_days,
        "total_price": total_price,
        "total_freight": total_freight,
        "item_count": item_count,
        "seller_count": seller_count,
        "seller_historical_rate": seller_historical_rate,
    }
    X_new = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    scaled = model_result["scaler"].transform(X_new)
    probability = float(model_result["model"].predict_proba(scaled)[0, 1])
    return {
        "probability_late": round(probability, 4),
        "prediction": "Late" if probability >= 0.5 else "On time",
    }
