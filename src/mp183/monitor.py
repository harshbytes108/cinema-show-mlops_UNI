from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.mp183.clean import clean_data

FEATURE_COLS = [
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km",
    "language",
    "location",
]
NUMERIC_COLS = FEATURE_COLS[:6]
TARGET_COL = "occupancy_pct"


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected.dropna(), quantiles))
    if len(breakpoints) < 2:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts, _ = np.histogram(expected.dropna(), bins=breakpoints)
    act_counts, _ = np.histogram(actual.dropna(), bins=breakpoints)

    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, 1.0)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, 1.0)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def run_monitoring() -> None:
    train_df = clean_data(pd.read_csv(Path("data/raw/train.csv")))
    later_df = pd.read_csv(Path("data/raw/later.csv"))
    model = joblib.load(Path("models/model.joblib"))

    print("=== FEATURE DRIFT (PSI: train vs later) ===")
    psi_scores = {}
    for col in NUMERIC_COLS:
        psi = compute_psi(train_df[col], later_df[col])
        psi_scores[col] = psi
        flag = "DRIFT" if psi >= 0.2 else ("SLIGHT" if psi >= 0.1 else "OK")
        print(f"  {col:22s}: PSI = {psi:.4f} [{flag}]")

    train_mae = float(mean_absolute_error(train_df[TARGET_COL], model.predict(train_df[FEATURE_COLS])))
    later_mae = float(mean_absolute_error(later_df[TARGET_COL], model.predict(later_df[FEATURE_COLS])))
    print("\n=== MODEL PERFORMANCE (v1) ===")
    print(f"  train_mae = {train_mae:.4f}")
    print(f"  later_mae = {later_mae:.4f}")

    top_feature = max(psi_scores, key=psi_scores.get)
    print(f"\nTop drifted feature: {top_feature} (PSI={psi_scores[top_feature]:.4f})")


if __name__ == "__main__":
    run_monitoring()
