"""
src/features/loubar_threshold.py

Implements the Loubar method (Louail et al. 2014, "From mobile phone data
to the spatial structure of cities") for defining a congestion-risk
threshold per cell — replacing an arbitrary percentile cutoff (e.g. "95th
percentile") with a threshold derived from each cell's own activity
distribution shape.

ADAPTATION BEING MADE EXPLICIT: the original Loubar method ranks locations
(cells) against each other AT ONE POINT IN TIME to find spatial hotspots.
We're applying the same underlying math across TIME for a single cell
instead — i.e., "is this hour unusually high for THIS cell's own history,"
not "is this cell unusually high compared to other cells right now." The
math is identical; the axis it's applied to is different. This is a
deliberate, defensible adaptation, not a misuse of the method — worth
stating this explicitly in review rather than letting it look like the
textbook use case.

THE METHOD (closed-form version, confirmed against multiple published
papers using it this way):
    Given a set of activity values:
        mu       = mean(values)
        rho_max  = max(values)
        F_star   = 1 - (mu / rho_max)   <- fraction of the distribution
                                            considered "below threshold"
        threshold = percentile(values, F_star * 100)
    Values above `threshold` are flagged as hotspots / congestion-risk.

WHY THIS IS BETTER THAN A FIXED PERCENTILE: F* is derived from each cell's
own mean-to-max ratio, not an arbitrary round number chosen by us — a cell
with a very spiky, unequal activity pattern gets a different (typically
higher) effective percentile than a cell with flatter, more evenly
distributed activity. This matches how the literature actually defines
"hotspot" for this exact kind of data.

REQUIREMENTS:
    pip install pandas numpy pyarrow

INPUT:
    data/raw/cdr_activity_aggregated.parquet (or .csv)

RUN:
    python src/features/loubar_threshold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ACTIVITY_COLS = ["smsin", "smsout", "callin", "callout", "internet"]

BOCCONI_CELL = 4259
NAVIGLI_CELL = 4456
DUOMO_CELL = 5060


def load_aggregated() -> pd.DataFrame:
    parquet_path = DATA_DIR / "cdr_activity_aggregated.parquet"
    csv_path = DATA_DIR / "cdr_activity_aggregated.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Couldn't find cdr_activity_aggregated.parquet or .csv in {DATA_DIR}. "
            "Run src/ingestion/cdr_loader.py first."
        )
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        try:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        except (ValueError, TypeError):
            df["datetime"] = pd.to_datetime(df["datetime"])
    present_cols = [c for c in ACTIVITY_COLS if c in df.columns]
    df["total_activity"] = df[present_cols].sum(axis=1)
    return df


def loubar_threshold(values: np.ndarray) -> tuple[float, float]:
    """Returns (threshold_value, F_star) for a 1D array of activity values.

    Handles the degenerate case where mu == rho_max (i.e. every value is
    identical, e.g. a cell with zero activity all week) by returning the
    max itself as the threshold — nothing in a perfectly flat series should
    be flagged as a spike.
    """
    mu = float(np.mean(values))
    rho_max = float(np.max(values))
    if rho_max == 0 or mu == rho_max:
        return rho_max, 1.0
    f_star = 1.0 - (mu / rho_max)
    f_star = min(max(f_star, 0.0), 1.0)  # guard against any numerical edge case
    threshold = float(np.percentile(values, f_star * 100))
    return threshold, f_star


def apply_loubar_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Computes a Loubar threshold per cell (over that cell's own hourly
    history) and flags each row as above/below its cell's threshold."""
    results = []
    thresholds = {}
    for cell_id, group in df.groupby("CellID"):
        threshold, f_star = loubar_threshold(group["total_activity"].values)
        thresholds[cell_id] = (threshold, f_star)

    df["loubar_threshold"] = df["CellID"].map(lambda c: thresholds[c][0])
    df["loubar_f_star"] = df["CellID"].map(lambda c: thresholds[c][1])
    df["is_congestion_risk"] = df["total_activity"] > df["loubar_threshold"]
    return df


def summarize(df: pd.DataFrame) -> None:
    n_cells = df["CellID"].nunique()
    flagged_pct = df["is_congestion_risk"].mean() * 100
    print(f"\nCells processed: {n_cells:,}")
    print(f"Rows flagged as congestion-risk overall: {flagged_pct:.2f}%")

    f_star_summary = df.drop_duplicates("CellID")["loubar_f_star"]
    print("\nF* distribution across cells (how much this varies vs. a fixed percentile):")
    print(f_star_summary.describe())

    print("\n--- Sanity check against the known cells ---")
    for name, cell_id in [("Bocconi", BOCCONI_CELL), ("Navigli", NAVIGLI_CELL), ("Duomo", DUOMO_CELL)]:
        cell_df = df[df["CellID"] == cell_id]
        if cell_df.empty:
            continue
        pct_flagged = cell_df["is_congestion_risk"].mean() * 100
        f_star = cell_df["loubar_f_star"].iloc[0]
        threshold = cell_df["loubar_threshold"].iloc[0]
        print(
            f"{name} (cell {cell_id}): F*={f_star:.3f}, threshold={threshold:.1f}, "
            f"{pct_flagged:.1f}% of its hours flagged as congestion-risk"
        )


if __name__ == "__main__":
    df = load_aggregated()
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")
    df = apply_loubar_per_cell(df)
    summarize(df)

    out_path = DATA_DIR / "cdr_with_loubar.parquet"
    try:
        df.to_parquet(out_path, index=False)
        print(f"\nSaved data with Loubar flags to {out_path}")
    except ImportError:
        csv_out = DATA_DIR / "cdr_with_loubar.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n'pyarrow' not installed — saved as CSV instead: {csv_out}")
    print("\nDone. Paste the printed output back so we can review the threshold behavior together.")