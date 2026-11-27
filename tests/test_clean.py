import pandas as pd

from src.mp183.clean import clean_data


def test_clean_data_fixes_all_confirmed_issues():
    df = pd.DataFrame(
        {
            "date": ["2026-03-10", "2026-03-20", "2026-04-10"],
            "ticket_price_rs": [200.0, 368.0, 250.0],
            "screen_seats": [150, 999, 200],
            "tickets_scanned_pct": [60.0, 70.0, 80.0],
            "occupancy_pct": [55.0, 650.0, 75.0],
        }
    )

    out = clean_data(df)

    assert "tickets_scanned_pct" not in out.columns
    assert out.loc[1, "screen_seats"] == 175.0
    assert out.loc[1, "ticket_price_rs"] == 225.0
    assert out.loc[1, "occupancy_pct"] == 65.0
