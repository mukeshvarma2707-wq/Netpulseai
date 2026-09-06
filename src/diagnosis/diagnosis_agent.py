"""
src/diagnosis/diagnosis_agent.py

Deterministic diagnosis logic: for each (cell, hour) where the seasonal-naive
forecast predicts congestion risk, classifies it as ROUTINE (safe to hand to
the solver for automatic reallocation) or ANOMALOUS (escalate to a human).

WHY DETERMINISTIC FIRST, NOT AN LLM AGENT YET: the core classification logic
(grid adjacency, neighbor-correlation threshold, calendar check) needs to be
correct and empirically validated on its own before wrapping it in an
agent/LLM layer. Adding natural-language explanation generation on top of
validated logic is a separate, lower-risk step — this script is that
foundation.

TWO SIGNALS, BOTH GROUNDED IN DOCUMENTED FACTS (not guesses):
1. NEIGHBORING-CELL CORRELATION: uses the grid's actual documented indexing
   formula from the official dataset paper (Barlacchi et al. 2015):
   CellID = y*100 + x. We invert this to get (x, y) grid coordinates for
   any CellID, then find its real 8-connected spatial neighbors.
2. CALENDAR CHECK: November 1, 2013 (Ognissanti) is the one real, documented
   public holiday in our 7-day window.

CLASSIFICATION RULE:
    - If the forecasted congestion hour falls on a known holiday, OR a high
      fraction of the cell's real spatial neighbors are ALSO forecasted at
      risk that same hour (suggesting an area-wide, explainable pattern
      rather than an isolated glitch) -> ROUTINE.
    - Otherwise (an isolated single-cell spike with no calendar explanation)
      -> ANOMALOUS, escalate to a human.

REQUIREMENTS:
    pip install pandas numpy pyarrow

INPUT:
    data/raw/cdr_with_congestion_flags.parquet (or .csv)

RUN:
    python src/diagnosis/diagnosis_agent.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ACTIVITY_COLS = ["smsin", "smsout", "callin", "callout", "internet"]

GRID_SIZE = 100  # 100 x 100 grid, per the dataset paper
NEIGHBOR_FRACTION_THRESHOLD = 0.3  # tuned down from 0.4 after testing: corner cells of a
# spike area only have 3/8 real neighbors inside the affected block (37.5%), which a 0.4
# threshold wrongly excluded from "routine" — 0.3 correctly includes them while still
# correctly flagging a genuinely isolated single-cell spike (0/8 neighbors) as anomalous.

# The one real, documented holiday in our 7-day window (Barlacchi et al. 2015
# / general knowledge: Nov 1 = Ognissanti, an Italian public holiday)
KNOWN_HOLIDAYS = {pd.Timestamp("2013-11-01").date()}


def load_data() -> pd.DataFrame:
    parquet_path = DATA_DIR / "cdr_with_congestion_flags.parquet"
    csv_path = DATA_DIR / "cdr_with_congestion_flags.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Couldn't find cdr_with_congestion_flags.parquet or .csv in {DATA_DIR}. "
            "Run src/features/congestion_threshold.py first."
        )
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
    present_cols = [c for c in ACTIVITY_COLS if c in df.columns]
    if "total_activity" not in df.columns:
        df["total_activity"] = df[present_cols].sum(axis=1)
    return df


def cell_id_to_xy(cell_id: int) -> tuple:
    """Inverts the documented grid formula (CellID = y*100 + x) to get (x, y)."""
    y, x = divmod(cell_id - 1, GRID_SIZE)  # -1 for 1-indexed CellID
    return x + 1, y + 1


def xy_to_cell_id(x: int, y: int) -> int:
    return (y - 1) * GRID_SIZE + x


def get_neighbors(cell_id: int) -> list:
    """Real 8-connected spatial neighbors, respecting grid boundaries."""
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


def compute_naive_forecast_and_risk(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Seasonal-naive forecast for a given horizon (same hour, 24h before the
    target), and whether that forecast crosses the cell's own congestion
    threshold — i.e. PREDICTED risk, not historical/retrospective risk."""
    df = df.sort_values(["CellID", "datetime"]).reset_index(drop=True)
    grouped = df.groupby("CellID")["total_activity"]
    df[f"naive_forecast_{horizon}h"] = grouped.shift(24 - horizon)
    df[f"predicted_risk_{horizon}h"] = df[f"naive_forecast_{horizon}h"] > df["congestion_threshold"]
    df[f"target_datetime_{horizon}h"] = df["datetime"] + pd.to_timedelta(horizon, unit="h")
    return df


def diagnose(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    risk_col = f"predicted_risk_{horizon}h"
    target_dt_col = f"target_datetime_{horizon}h"

    flagged = df[df[risk_col] == True].copy()
    print(f"\n+{horizon}h horizon: {len(flagged):,} (cell, hour) combinations predicted at risk "
          f"({len(flagged)/len(df):.1%} of all rows).")

    if flagged.empty:
        return flagged

    # Build a fast lookup: for each (target hour), which cells are flagged at risk
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

        # SECOND CALENDAR CHECK, distinct from the first: the naive forecast's
        # basis value (the "yesterday" it copied from) may itself have come
        # from an atypical day, like a holiday. That doesn't mean today is
        # genuinely busy — it means the forecast basis is unreliable, which
        # is a reason for EXTRA caution, not confident auto-handling.
        if source_date in KNOWN_HOLIDAYS:
            return "ANOMALOUS", "isolated, AND forecast is based on a holiday — basis may be unreliable"

        return "ANOMALOUS", f"isolated — only {n_neighbors_at_risk}/{len(neighbors)} neighbors at risk"

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
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")

    all_diagnoses = {}
    for h in [1, 2, 3, 4]:
        df = compute_naive_forecast_and_risk(df, h)
        diagnosed = diagnose(df, h)
        all_diagnoses[h] = diagnosed

    print("\n--- Sample of ANOMALOUS cases (worth a manual sanity check) ---")
    sample = all_diagnoses[1][all_diagnoses[1]["classification"] == "ANOMALOUS"]
    if not sample.empty:
        print(sample[["CellID", "target_datetime_1h", "naive_forecast_1h", "congestion_threshold", "reason"]].head(10).to_string())
    else:
        print("(none found at +1h)")

    out_path = DATA_DIR / "diagnosis_results_1h.parquet"
    try:
        all_diagnoses[1].to_parquet(out_path, index=False)
        print(f"\nSaved +1h diagnosis results to {out_path}")
    except ImportError:
        csv_out = DATA_DIR / "diagnosis_results_1h.csv"
        all_diagnoses[1].to_csv(csv_out, index=False)
        print(f"\n'pyarrow' not installed — saved as CSV instead: {csv_out}")

    print("\nDone. Paste the printed output back so we can sanity-check the classification logic together.")