"""
src/diagnosis/diagnosis_agent.py

Deterministic diagnosis logic: for each (cell, hour) where the forecast
predicts congestion risk, classifies it as ROUTINE (safe to hand to the
solver) or ANOMALOUS (escalate to a human).

UPDATED: now loads the FORECAST from generate_forecasts.py's saved
LightGBM predictions (data/raw/cell_forecasts.parquet), instead of
computing an internal seasonal-naive forecast. This reflects the real,
validated finding that LightGBM beats seasonal-naive given the full
~62-day dataset. Everything else -- the congestion threshold comparison,
neighbor-correlation logic, and holiday checks -- is UNCHANGED; only
where the forecast number comes from changed.

REQUIREMENTS:
    pip install pandas numpy pyarrow

INPUT:
    data/raw/cdr_with_congestion_flags.parquet (or .csv)
    data/raw/cell_forecasts.parquet (or .csv) -- from generate_forecasts.py

RUN:
    python src/diagnosis/diagnosis_agent.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ACTIVITY_COLS = ["smsin", "smsout", "callin", "callout", "internet"]

GRID_SIZE = 100
NEIGHBOR_FRACTION_THRESHOLD = 0.3

KNOWN_HOLIDAYS = {
    pd.Timestamp("2013-11-01").date(),
    pd.Timestamp("2013-12-08").date(),
    pd.Timestamp("2013-12-25").date(),
    pd.Timestamp("2013-12-26").date(),
    pd.Timestamp("2014-01-01").date(),
}


def load_data() -> pd.DataFrame:
    parquet_path = DATA_DIR / "cdr_with_congestion_flags.parquet"
    csv_path = DATA_DIR / "cdr_with_congestion_flags.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Couldn't find cdr_with_congestion_flags file in {DATA_DIR}.")
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
    present_cols = [c for c in ACTIVITY_COLS if c in df.columns]
    if "total_activity" not in df.columns:
        df["total_activity"] = df[present_cols].sum(axis=1)
    return df


def load_forecasts() -> pd.DataFrame:
    parquet_path = DATA_DIR / "cell_forecasts.parquet"
    csv_path = DATA_DIR / "cell_forecasts.csv"
    if parquet_path.exists():
        forecasts = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        forecasts = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Couldn't find cell_forecasts file in {DATA_DIR}. Run generate_forecasts.py first.")
    dt_cols = ["datetime"] + [c for c in forecasts.columns if c.startswith("target_datetime_")]
    for col in dt_cols:
        if not pd.api.types.is_datetime64_any_dtype(forecasts[col]):
            forecasts[col] = pd.to_datetime(forecasts[col])
    return forecasts


def cell_id_to_xy(cell_id: int) -> tuple:
    y, x = divmod(cell_id - 1, GRID_SIZE)
    return x + 1, y + 1


def xy_to_cell_id(x: int, y: int) -> int:
    return (y - 1) * GRID_SIZE + x


def get_neighbors(cell_id: int) -> list:
    x, y = cell_id_to_xy(cell_id)
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
                neighbors.append(xy_to_cell_id(nx, ny))
    return neighbors


def attach_forecast_and_risk(df: pd.DataFrame, forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    cols_needed = ["CellID", "datetime", f"forecast_{horizon}h", f"target_datetime_{horizon}h"]
    df = df.merge(forecasts[cols_needed], on=["CellID", "datetime"], how="left")
    df[f"predicted_risk_{horizon}h"] = df[f"forecast_{horizon}h"] > df["congestion_threshold"]
    return df


def diagnose(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    risk_col = f"predicted_risk_{horizon}h"
    target_dt_col = f"target_datetime_{horizon}h"

    flagged = df[df[risk_col] == True].copy()
    print(f"\n+{horizon}h horizon: {len(flagged):,} (cell, hour) combinations predicted at risk "
          f"({len(flagged)/len(df):.1%} of all rows).")

    if flagged.empty:
        return flagged

    flagged["target_hour_key"] = flagged[target_dt_col].values.astype("datetime64[h]")
    at_risk_by_hour = flagged.groupby("target_hour_key")["CellID"].apply(set).to_dict()

    def classify(row):
        target_date = row[target_dt_col].date()
        source_date = (row[target_dt_col] - pd.to_timedelta(24, unit="h")).date()

        if target_date in KNOWN_HOLIDAYS:
            return "ROUTINE", "target date is a known holiday"

        hour_key = np.datetime64(row[target_dt_col], "h")
        at_risk_cells_this_hour = at_risk_by_hour.get(hour_key, set())
        neighbors = get_neighbors(row["CellID"])
        n_neighbors_at_risk = sum(1 for n in neighbors if n in at_risk_cells_this_hour)
        neighbor_fraction = n_neighbors_at_risk / len(neighbors) if neighbors else 0

        if neighbor_fraction >= NEIGHBOR_FRACTION_THRESHOLD:
            return "ROUTINE", f"{n_neighbors_at_risk}/{len(neighbors)} neighbors also at risk"

        if source_date in KNOWN_HOLIDAYS:
            return "ANOMALOUS", "isolated, AND forecast is based on a holiday -- basis may be unreliable"

        return "ANOMALOUS", f"isolated -- only {n_neighbors_at_risk}/{len(neighbors)} neighbors at risk"

    results = flagged.apply(classify, axis=1, result_type="expand")
    flagged["classification"] = results[0]
    flagged["reason"] = results[1]

    n_routine = (flagged["classification"] == "ROUTINE").sum()
    n_anomalous = (flagged["classification"] == "ANOMALOUS").sum()
    print(f"  ROUTINE (auto-handle via solver): {n_routine:,} ({n_routine/len(flagged):.1%})")
    print(f"  ANOMALOUS (escalate to human): {n_anomalous:,} ({n_anomalous/len(flagged):.1%})")

    return flagged


if __name__ == "__main__":
    df = load_data()
    forecasts = load_forecasts()
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")
    print(f"Loaded {len(forecasts):,} pre-computed LightGBM forecast rows.")

    all_diagnoses = {}
    for h in [1, 2, 3, 4]:
        df = attach_forecast_and_risk(df, forecasts, h)
        diagnosed = diagnose(df, h)
        all_diagnoses[h] = diagnosed

    print("\n--- Sample of ANOMALOUS cases (worth a manual sanity check) ---")
    sample = all_diagnoses[1][all_diagnoses[1]["classification"] == "ANOMALOUS"]
    if not sample.empty:
        print(sample[["CellID", "target_datetime_1h", "forecast_1h", "congestion_threshold", "reason"]].head(10).to_string())
    else:
        print("(none found at +1h)")

    out_path = DATA_DIR / "diagnosis_results_1h.parquet"
    try:
        all_diagnoses[1].to_parquet(out_path, index=False)
        print(f"\nSaved +1h diagnosis results to {out_path}")
    except ImportError:
        csv_out = DATA_DIR / "diagnosis_results_1h.csv"
        all_diagnoses[1].to_csv(csv_out, index=False)
        print(f"\n'pyarrow' not installed -- saved as CSV instead: {csv_out}")

    print("\nDone. Paste the printed output back so we can sanity-check the classification logic together.")