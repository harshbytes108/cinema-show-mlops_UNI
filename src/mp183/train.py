import hashlib
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
CAT_COLS = FEATURE_COLS[6:]
TARGET_COL = "occupancy_pct"


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )
    return Pipeline([("prep", pre), ("model", model)])


def run_training() -> None:
    params = load_params()
    data_cfg = params.get("data", {})
    train_path = Path(data_cfg.get("train_path", "data/raw/train.csv"))
    later_path = Path(data_cfg.get("later_path", "data/raw/later.csv"))

    data_hash = sha256_file(train_path)
    raw_train = pd.read_csv(train_path)
    cleaned_train = clean_data(raw_train)
    later_df = pd.read_csv(later_path)

    X = cleaned_train[FEATURE_COLS]
    y = cleaned_train[TARGET_COL]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_later = later_df[FEATURE_COLS]
    y_later = later_df[TARGET_COL]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "mp183_occupancy"
    model_name = "mp183_occupancy_model"
    mlflow.set_experiment(experiment_name)

    candidates = [
        ("ridge_alpha_1", Ridge(alpha=1.0), {"model_type": "Ridge", "alpha": 1.0}),
        (
            "rf_100_d10",
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 10},
        ),
        (
            "gbr_150_lr005",
            GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42),
            {"model_type": "GradientBoosting", "n_estimators": 150, "learning_rate": 0.05, "max_depth": 4},
        ),
    ]

    best_mae = float("inf")
    best_run_id = None
    best_pipe = None

    for run_name, estimator, hp in candidates:
        pipe = build_pipeline(estimator)
        pipe.fit(X_tr, y_tr)
        mae_val = float(mean_absolute_error(y_val, pipe.predict(X_val)))

        full_pipe = build_pipeline(estimator)
        full_pipe.fit(X, y)
        mae_later = float(mean_absolute_error(y_later, full_pipe.predict(X_later)))

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_params(hp)
            mlflow.log_metric("mae_val", mae_val)
            mlflow.log_metric("mae_later", mae_later)

            try:
                mlflow.sklearn.log_model(full_pipe, name="model", serialization_format="cloudpickle")
            except TypeError:
                mlflow.sklearn.log_model(full_pipe, "model")

            print(f"{run_name}: run_id={run.info.run_id} | mae_val={mae_val:.4f} | mae_later={mae_later:.4f}")

            if mae_val < best_mae:
                best_mae = mae_val
                best_run_id = run.info.run_id
                best_pipe = full_pipe

    model_uri = f"runs:/{best_run_id}/model"
    reg = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", reg.version)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_pipe, "models/model.joblib")

    print("\n--- COPY INTO reports/findings.yaml (tracking) ---")
    print(f"experiment_name: \"{experiment_name}\"")
    print(f"best_run_id: \"{best_run_id}\"")
    print(f"registered_model_name: \"{model_name}\"")
    print(f"champion_version: {reg.version}")


if __name__ == "__main__":
    run_training()
