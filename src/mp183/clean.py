import numpy as np
import pandas as pd

LEAKING_COL = "tickets_scanned_pct"
TARGET_COL = "occupancy_pct"
STUCK_START = "2026-03-15"
STUCK_END = "2026-04-02"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # 1. Drop post-event leaking column
    if LEAKING_COL in cleaned.columns:
        cleaned = cleaned.drop(columns=[LEAKING_COL])

    # 2. Replace sentinel value (999) in screen_seats with median of valid values
    if "screen_seats" in cleaned.columns:
        valid_seats = cleaned.loc[cleaned["screen_seats"] != 999, "screen_seats"]
        median_seats = float(valid_seats.median()) if not valid_seats.empty else np.nan
        cleaned.loc[cleaned["screen_seats"] == 999, "screen_seats"] = median_seats

    # 3. Fix stuck sensor in ticket_price_rs (2026-03-15 to 2026-04-02, stuck at 368)
    if "date" in cleaned.columns and "ticket_price_rs" in cleaned.columns:
        stuck_mask = (
            (cleaned["date"] >= STUCK_START)
            & (cleaned["date"] <= STUCK_END)
            & (cleaned["ticket_price_rs"] == 368)
        )
        valid_prices = cleaned.loc[~stuck_mask, "ticket_price_rs"]
        median_price = float(valid_prices.median()) if not valid_prices.empty else np.nan
        cleaned.loc[stuck_mask, "ticket_price_rs"] = median_price

    # 4. Fix 10x target typos (occupancy_pct > 100)
    if TARGET_COL in cleaned.columns:
        typo_mask = cleaned[TARGET_COL] > 100
        cleaned.loc[typo_mask, TARGET_COL] = cleaned.loc[typo_mask, TARGET_COL] / 10.0

    return cleaned
