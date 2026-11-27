from pathlib import Path

import joblib
import pandas as pd

from src.mp183.train import FEATURE_COLS


def find_holdout_path() -> Path:
    for candidate in [Path("data/holdout_features.csv"), Path("data/raw/holdout_features.csv")]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find holdout_features.csv in data/ or data/raw/")


def run_predict() -> None:
    holdout_path = find_holdout_path()
    df = pd.read_csv(holdout_path)
    model = joblib.load(Path("models/model.joblib"))

    preds = model.predict(df[FEATURE_COLS])
    id_col = "record_id" if "record_id" in df.columns else df.columns[0]

    out = pd.DataFrame({
        "record_id": df[id_col],
        "prediction": preds,
    })

    out_path = Path("predictions/holdout_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Input: {holdout_path} ({len(df)} rows)")
    print(f"Saved: {out_path} ({len(out)} rows)")
    print(f"Columns: {list(out.columns)}")
    print(f"Nulls: {out.isna().sum().to_dict()}")
    print(out.head())


if __name__ == "__main__":
    run_predict()
