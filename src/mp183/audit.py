from pathlib import Path
import pandas as pd
import yaml

NUMERIC_COLS = [
    "ticket_price_rs",
    "days_since_release",
    "star_rating",
    "show_hour",
    "screen_seats",
    "distance_to_metro_km",
]
CAT_COLS = ["language", "location"]
TARGET = "occupancy_pct"


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_audit() -> None:
    params = load_params()
    data_cfg = params.get("data", {})
    train_path = Path(data_cfg.get("train_path", "data/raw/train.csv"))
    later_path = Path(data_cfg.get("later_path", "data/raw/later.csv"))
    holdout_path = Path(data_cfg.get("holdout_path", "data/raw/holdout_features.csv"))

    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    print(f"Shapes -> train: {train.shape}, later: {later.shape}, holdout: {holdout.shape}\n")

    # 1. Leaking column
    extra_cols = [c for c in train.columns if c not in holdout.columns and c != TARGET]
    print("1. LEAKING COLUMN")
    print("   Columns in train but not holdout:", extra_cols)
    for c in extra_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            print(f"   corr({c}, {TARGET}) = {train[c].corr(train[TARGET]):.4f}")

    # 2. Unit mixup (compare min/mean/max per location)
    print("\n2. UNIT MIXUP (by location)")
    for col in NUMERIC_COLS:
        stats = train.groupby("location")[col].agg(["min", "mean", "max"])
        ratios = stats["mean"].max() / max(abs(stats["mean"].min()), 1e-9)
        if ratios > 3:
            print(f"   Candidate column: {col}")
            print(stats, "\n")

    # 3. Sentinel value
    print("3. SENTINEL VALUE (top repeated / negative / extreme values)")
    for col in NUMERIC_COLS:
        vc = train[col].value_counts().head(3)
        neg_cnt = (train[col] < 0).sum()
        print(f"   {col}: neg={neg_cnt}, top3={vc.to_dict()}")

    # 4. Duplicate rows
    dup_count = int(train.duplicated().sum())
    print(f"\n4. DUPLICATE ROWS: {dup_count} extra copies")

    # 5. Stuck sensor (longest contiguous run by date)
    print("\n5. STUCK SENSOR (longest contiguous runs sorted by date)")
    t_sorted = train.sort_values("date").reset_index(drop=True)
    for col in NUMERIC_COLS:
        s = t_sorted[col]
        run_id = (s != s.shift()).cumsum()
        runs = t_sorted.groupby(run_id).agg(
            val=(col, "first"),
            length=(col, "size"),
            start_date=("date", "first"),
            end_date=("date", "last"),
        )
        best = runs.sort_values("length", ascending=False).iloc[0]
        if best["length"] >= 10:
            print(f"   {col}: val={best['val']}, len={best['length']}, {best['start_date']} -> {best['end_date']}")

    # 6. Target typos (~10x too big)
    typos = train[train[TARGET] > 100]
    print(f"\n6. TARGET TYPOS ({TARGET} > 100): count={len(typos)}")
    if len(typos) > 0:
        print("   Sample values:", sorted(typos[TARGET].tolist())[:10])
    print("   Top 10 target values:", sorted(train[TARGET].dropna().tolist(), reverse=True)[:10])

    # 7. Spelling variants (train vs clean later.csv)
    print("\n7. SPELLING VARIANTS")
    for col in CAT_COLS:
        train_vals = train[col].value_counts(dropna=False).to_dict()
        later_vals = sorted(later[col].dropna().unique().tolist())
        print(f"   {col}:")
        print(f"     later (clean, {len(later_vals)} levels): {later_vals}")
        print(f"     train: {train_vals}")


if __name__ == "__main__":
    run_audit()
