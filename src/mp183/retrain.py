import hashlib
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.mp183.clean import clean_data
from src.mp183.monitor import NUMERIC_COLS, compute_psi
from src.mp183.train import FEATURE_COLS, TARGET_COL, build_pipeline


def find_date_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "date" in col.lower():
            return col
    raise ValueError(f"No date column found in later.csv columns: {list(df.columns)}")


def run_wave3_monitoring_and_retrain() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "mp183_occupancy"
    model_name = "mp183_occupancy_model"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    train_df = clean_data(pd.read_csv(Path("data/raw/train.csv")))
    later_df = pd.read_csv(Path("data/raw/later.csv"))

    # 1. Load v1 model explicitly from MLflow Registry
    v1_model = mlflow.sklearn.load_model(f"models:/{model_name}/1")

    # 2. 10-bin PSI across all 6 numeric features
    psi_scores = {col: round(compute_psi(train_df[col], later_df[col]), 4) for col in NUMERIC_COLS}
    top_feature = max(psi_scores, key=psi_scores.get)

    # 3. v1 MAE per location on later.csv
    group_mae = {}
    for loc in ["City Centre", "Mall", "Highway"]:
        sub = later_df[later_df["location"] == loc]
        if not sub.empty:
            group_mae[loc] = round(float(mean_absolute_error(sub[TARGET_COL], v1_model.predict(sub[FEATURE_COLS]))), 4)
    worst_served_group = max(group_mae, key=group_mae.get)

    # 4. Sort later.csv by date -> 70% train / 30% eval split
    date_col = find_date_col(later_df)
    later_sorted = later_df.assign(_dt=pd.to_datetime(later_df[date_col])).sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)
    split_idx = int(len(later_sorted) * 0.70)
    later_70 = later_sorted.iloc[:split_idx]
    later_30 = later_sorted.iloc[split_idx:]

    # 5. Train v2 on cleaned train.csv + first 70% of later.csv
    combined = pd.concat([train_df, later_70], ignore_index=True)
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(combined[FEATURE_COLS + [TARGET_COL]], index=False).values.tobytes()
    ).hexdigest()

    v2_pipe = build_pipeline(
        GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    )
    v2_pipe.fit(combined[FEATURE_COLS], combined[TARGET_COL])

    # 6. Evaluate v1 vs v2 on last 30% of later.csv
    mae_v1_eval = float(mean_absolute_error(later_30[TARGET_COL], v1_model.predict(later_30[FEATURE_COLS])))
    mae_v2_eval = float(mean_absolute_error(later_30[TARGET_COL], v2_pipe.predict(later_30[FEATURE_COLS])))

    # 7. Apply 12% improvement rule
    promoted = mae_v2_eval <= 0.88 * mae_v1_eval
    decision = "promote" if promoted else "keep"

    with mlflow.start_run(run_name="gbr_v2_70_30") as run:
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_params({"stage": "v2_retrain_70_30", "split": "70/30_by_date", "decision": decision})
        mlflow.log_metric("mae_v1_last30", mae_v1_eval)
        mlflow.log_metric("mae_v2_last30", mae_v2_eval)
        try:
            mlflow.sklearn.log_model(v2_pipe, name="model", serialization_format="cloudpickle")
        except TypeError:
            mlflow.sklearn.log_model(v2_pipe, "model")
        v2_run_id = run.info.run_id

    reg = mlflow.register_model(model_uri=f"runs:/{v2_run_id}/model", name=model_name)

    Path("models").mkdir(exist_ok=True)
    if promoted:
        client.set_registered_model_alias(model_name, "champion", reg.version)
        joblib.dump(v2_pipe, "models/model.joblib")
        champion_version = reg.version
    else:
        client.set_registered_model_alias(model_name, "champion", "1")
        joblib.dump(v1_model, "models/model.joblib")
        champion_version = 1

    print("\n=== WAVE 3 MONITORING & RETRAIN RESULTS ===")
    print(f"date_col: {date_col}")
    print(f"psi_scores: {psi_scores}")
    print(f"top_drifted_feature: {top_feature} (PSI={psi_scores[top_feature]})")
    print(f"location_mae_v1: {group_mae}")
    print(f"worst_served_group: {worst_served_group}")
    print(f"mae_v1_last30: {mae_v1_eval:.4f}")
    print(f"mae_v2_last30: {mae_v2_eval:.4f}")
    print(f"threshold_12pct (0.88 * mae_v1): {0.88 * mae_v1_eval:.4f}")
    print(f"decision: {decision}")
    print(f"champion_version: {champion_version}")
    print(f"v2_run_id: {v2_run_id}")


if __name__ == "__main__":
    run_wave3_monitoring_and_retrain()
