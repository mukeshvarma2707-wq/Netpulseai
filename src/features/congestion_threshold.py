"""
src/features/congestion_threshold.py

Defines a per-cell congestion-risk threshold using a percentile-based
approach, applied to each cell's own hourly activity history.

WHY NOT LOUBAR (this replaces loubar_threshold.py — keep this history,
it's a real finding worth being able to explain in review):
We tested the Loubar method (Louail et al. 2014) for this exact purpose
first. On real data, it flagged 48.95% of all cell-hours as
"congestion-risk" overall (F* averaged 0.51 across cells) — far too
permissive to be a useful "this is unusual" signal. The reason: Loubar was
designed for cross-sectional spatial distributions (ranking many locations
against each other, where a few genuine hotspots dominate total activity —
a skewed, power-law-like shape that gives a strict threshold). Our use case
is different: one cell's own activity across a smooth daily/weekly cycle,
which doesn't have that same extreme skew, so Loubar's threshold collapsed
toward the median instead of toward a genuine rare extreme.
CONCLUSION: Loubar remains the right tool for spatial hotspot ranking
(which is what we already validated cleanly against Bocconi/Navigli/Duomo);
it's the wrong tool for defining a rare temporal event per cell. This
script reverts to a percentile-based threshold for that specific purpose,
based on that empirical evidence.

METHOD:
    For each cell, threshold = the Nth percentile of ITS OWN hourly
    activity history (default N=90, i.e. flag roughly the top 10% of
    hours for that cell as congestion-risk). This is still per-cell and
    data-driven (not a single global cutoff), just without the Loubar
    formula's assumption of a skewed cross-sectional distribution.

REQUIREMENTS:
    pip install pandas numpy pyarrow

INPUT:
    data/raw/cdr_activity_aggregated.parquet (or .csv)

RUN:
    python src/features/congestion_threshold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ACTIVITY_COLS = ["smsin", "smsout", "callin", "callout", "internet"]

BOCCONI_CELL = 4259
NAVIGLI_CELL = 4456
DUOMO_CELL = 5060

PERCENTILE = 90  # tunable — flag roughly the top (100-PERCENTILE)% of hours per cell


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


def apply_percentile_threshold(df: pd.DataFrame, percentile: float = PERCENTILE) -> pd.DataFrame:
    thresholds = df.groupby("CellID")["total_activity"].quantile(percentile / 100)
    df["congestion_threshold"] = df["CellID"].map(thresholds)
    df["is_congestion_risk"] = df["total_activity"] > df["congestion_threshold"]
    return df


def summarize(df: pd.DataFrame) -> None:
    n_cells = df["CellID"].nunique()
    flagged_pct = df["is_congestion_risk"].mean() * 100
    print(f"\nCells processed: {n_cells:,}")
    print(f"Percentile used: {PERCENTILE}th")
    print(f"Rows flagged as congestion-risk overall: {flagged_pct:.2f}% (expected roughly {100-PERCENTILE}%)")

    print("\n--- Sanity check against the known cells ---")
    for name, cell_id in [("Bocconi", BOCCONI_CELL), ("Navigli", NAVIGLI_CELL), ("Duomo", DUOMO_CELL)]:
        cell_df = df[df["CellID"] == cell_id]
        if cell_df.empty:
            continue
        pct_flagged = cell_df["is_congestion_risk"].mean() * 100
        threshold = cell_df["congestion_threshold"].iloc[0]
        print(f"{name} (cell {cell_id}): threshold={threshold:.1f}, {pct_flagged:.1f}% of its hours flagged")


if __name__ == "__main__":
    df = load_aggregated()
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")
    df = apply_percentile_threshold(df)
    summarize(df)

    out_path = DATA_DIR / "cdr_with_congestion_flags.parquet"
    try:
        df.to_parquet(out_path, index=False)
        print(f"\nSaved data with congestion flags to {out_path}")
    except ImportError:
        csv_out = DATA_DIR / "cdr_with_congestion_flags.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n'pyarrow' not installed — saved as CSV instead: {csv_out}")
    print("\nDone. Paste the printed output back so we can confirm this behaves sensibly on real data.")