import hashlib
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.mp183.clean import clean_data
from src.mp183.train import FEATURE_COLS, TARGET_COL, build_pipeline


def run_retrain() -> None:
    train_df = clean_data(pd.read_csv(Path("data/raw/train.csv")))
    later_df = pd.read_csv(Path("data/raw/later.csv"))
    combined = pd.concat([train_df, later_df], ignore_index=True)

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(combined[FEATURE_COLS + [TARGET_COL]], index=False).values.tobytes()
    ).hexdigest()

    X = combined[FEATURE_COLS]
    y = combined[TARGET_COL]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "mp183_occupancy"
    model_name = "mp183_occupancy_model"
    mlflow.set_experiment(experiment_name)

    estimator = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    val_pipe = build_pipeline(estimator)
    val_pipe.fit(X_tr, y_tr)
    mae_val = float(mean_absolute_error(y_val, val_pipe.predict(X_val)))

    full_pipe = build_pipeline(estimator)
    full_pipe.fit(X, y)
    mae_later = float(mean_absolute_error(later_df[TARGET_COL], full_pipe.predict(later_df[FEATURE_COLS])))

    with mlflow.start_run(run_name="gbr_v2_retrained") as run:
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_params({"model_type": "GradientBoosting", "stage": "v2_retrain", "rows": len(combined)})
        mlflow.log_metric("mae_val", mae_val)
        mlflow.log_metric("mae_later", mae_later)
        try:
            mlflow.sklearn.log_model(full_pipe, name="model", serialization_format="cloudpickle")
        except TypeError:
            mlflow.sklearn.log_model(full_pipe, "model")
        v2_run_id = run.info.run_id

    reg = mlflow.register_model(model_uri=f"runs:/{v2_run_id}/model", name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", reg.version)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(full_pipe, "models/model.joblib")

    print(f"v2_run_id: {v2_run_id}")
    print(f"champion_version: {reg.version}")
    print(f"v2_mae_val: {mae_val:.4f}")
    print(f"v2_mae_later: {mae_later:.4f}")


if __name__ == "__main__":
    run_retrain()
