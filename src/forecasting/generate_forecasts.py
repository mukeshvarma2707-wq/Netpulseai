"""
src/forecasting/generate_forecasts.py

Trains the FINAL production LightGBM models (the same architecture just
validated in lightgbm_model.py, which now beats the seasonal-naive
baseline given the full ~62-day dataset) and saves real forecasts for
every (cell, hour) to a file that diagnosis_agent.py and solver.py load
instead of computing their own internal naive forecast.

WHY A SEPARATE SCRIPT, NOT JUST REUSING lightgbm_model.py DIRECTLY:
lightgbm_model.py's job is EVALUATION -- it deliberately holds out a test
set to measure real, honest performance. This script's job is
PRODUCTION -- once the architecture is validated, the final deployed model
is retrained on ALL available real data (standard practice: more real
data can only help a model that's already been shown to generalize), and
used to generate the actual forecasts the rest of the pipeline consumes.
Keeping these separate means the evaluation script's numbers are never
confused with production behavior.

STILL USES THE SEASONAL-NAIVE VALUE AS AN INPUT FEATURE (not a
contradiction): the validated architecture's key insight was that
LightGBM wins by learning a CORRECTION on top of the naive forecast, not
by ignoring it. That's preserved here.

REQUIREMENTS:
    pip install pandas numpy lightgbm pyarrow

INPUT:
    data/raw/cdr_with_congestion_flags.parquet (or .csv)

OUTPUT:
    data/raw/cell_forecasts.parquet -- CellID, datetime, target_datetime_1h..4h,
    forecast_1h..4h (the actual number diagnosis_agent.py and solver.py use)

RUN:
    python src/forecasting/generate_forecasts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

LAGS = [1, 2, 3, 24]
HORIZONS = [1, 2, 3, 4]
GRID_SIZE = 100


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
    return df


def add_spatial_neighbor_feature(df: pd.DataFrame) -> pd.DataFrame:
    unique_cells = df["CellID"].unique()
    neighbor_map = {c: get_neighbors(c) for c in unique_cells}
    pivot = df.pivot_table(index="datetime", columns="CellID", values="total_activity", aggfunc="first")
    neighbor_avg = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for c in unique_cells:
        real_neighbors = [n for n in neighbor_map[c] if n in pivot.columns]
        neighbor_avg[c] = pivot[real_neighbors].mean(axis=1) if real_neighbors else np.nan
    stacked = neighbor_avg.stack(future_stack=True).rename("neighbor_avg_activity").reset_index()
    df = df.merge(stacked, on=["datetime", "CellID"], how="left")
    return df


def engineer_features(df: pd.DataFrame):
    df = df.sort_values(["CellID", "datetime"]).reset_index(drop=True)
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    print("Computing spatial neighbor features...")
    df = add_spatial_neighbor_feature(df)

    grouped = df.groupby("CellID")["total_activity"]
    for lag in LAGS:
        df[f"lag_{lag}h"] = grouped.shift(lag)
    neighbor_grouped = df.groupby("CellID")["neighbor_avg_activity"]
    df["neighbor_avg_lag1h"] = neighbor_grouped.shift(1)

    for h in HORIZONS:
        df[f"target_datetime_{h}h"] = df["datetime"] + pd.to_timedelta(h, unit="h")
        df[f"naive_feature_{h}h"] = grouped.shift(24 - h)
        df[f"target_{h}h"] = grouped.shift(-h)

    df["cell_historical_mean"] = df.groupby("CellID")["total_activity"].transform("mean")

    feature_cols = [f"lag_{l}h" for l in LAGS] + ["hour_of_day", "day_of_week", "is_weekend", "neighbor_avg_lag1h"]
    return df, feature_cols


if __name__ == "__main__":
    import lightgbm as lgb

    df = load_data()
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")

    df, feature_cols = engineer_features(df)
    model_feature_cols = feature_cols + ["CellID", "cell_historical_mean"]

    result = df[["CellID", "datetime"]].copy()

    for h in HORIZONS:
        cols_for_h = model_feature_cols + [f"naive_feature_{h}h"]

        # Training rows need complete features AND a real target to learn from
        train_mask = df[cols_for_h + [f"target_{h}h"]].notna().all(axis=1)
        X_train = df.loc[train_mask, cols_for_h]
        y_train = np.log1p(df.loc[train_mask, f"target_{h}h"])

        model = lgb.LGBMRegressor(
            n_estimators=60, learning_rate=0.05, num_leaves=15,
            min_child_samples=50, random_state=42, verbosity=-1,
        )
        model.fit(X_train, y_train, categorical_feature=["CellID"])

        # PREDICTION rows only need complete features (a real target isn't
        # needed to make a forecast -- this deliberately includes the most
        # recent rows, where the real future value isn't known yet, which
        # is exactly the production use case).
        predict_mask = df[cols_for_h].notna().all(axis=1)
        preds = np.expm1(model.predict(df.loc[predict_mask, cols_for_h]))
        preds = np.clip(preds, 0, None)

        result.loc[predict_mask, f"forecast_{h}h"] = preds
        result[f"target_datetime_{h}h"] = df[f"target_datetime_{h}h"]

        print(f"+{h}h: trained on {len(X_train):,} rows, generated {predict_mask.sum():,} real forecasts.")

    out_path = DATA_DIR / "cell_forecasts.parquet"
    try:
        result.to_parquet(out_path, index=False)
        print(f"\nSaved production forecasts to {out_path}")
    except ImportError:
        result.to_csv(DATA_DIR / "cell_forecasts.csv", index=False)
        print(f"\n'pyarrow' not installed - saved as CSV instead.")

    print("\nDone. Paste the printed output back so we can confirm the forecasts before wiring them into diagnosis_agent.py.")