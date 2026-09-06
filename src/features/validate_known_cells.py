"""
src/features/validate_known_cells.py

Sanity-checks our loaded, aggregated Milan CDR data against real, documented
behavior from the dataset's original paper (Barlacchi et al. 2015,
"A multi-source dataset of urban life in the city of Milan and the
Province of Trentino"). If our data reproduces the patterns the original
authors themselves documented, that's real evidence our pipeline is reading
and aggregating the data correctly before we build anything more complex on
top of it.

DOCUMENTED PATTERNS WE'RE CHECKING (from the paper, Section "Spatial aspects"):
  - Bocconi University (cell 4259): connections should DROP on weekends
    (it's a university -> weekday-driven activity)
  - Navigli nightlife district (cell 4456): internet activity should RISE
    in the evening (nightlife area -> evening-driven activity)
  - Duomo, city centre (cell 5060): should show high overall activity
    relative to other cells (busiest, most central area of Milan)

IMPORTANT CAVEAT FROM THE PAPER: the activity values are NOT raw call/SMS
counts — they're scaled by an undisclosed constant k that Telecom Italia
uses to obfuscate true volume. This is fine for everything we're doing,
since we only ever compare a cell's activity against itself or against
relative rankings — never claiming an absolute count. This script's checks
are deliberately relative/comparative for that exact reason.

REQUIREMENTS:
    pip install pandas pyarrow

INPUT:
    Expects data/raw/cdr_activity_aggregated.parquet (or .csv), produced by
    src/ingestion/cdr_loader.py

RUN:
    python src/features/validate_known_cells.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

BOCCONI_CELL = 4259
NAVIGLI_CELL = 4456
DUOMO_CELL = 5060

ACTIVITY_COLS = ["smsin", "smsout", "callin", "callout", "internet"]


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

    # datetime may have been saved as string; make sure it's a real datetime for feature extraction
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        # Handles both epoch-ms and already-parsed string cases, without assuming which one it is
        try:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        except (ValueError, TypeError):
            df["datetime"] = pd.to_datetime(df["datetime"])

    present_cols = [c for c in ACTIVITY_COLS if c in df.columns]
    df["total_activity"] = df[present_cols].sum(axis=1)
    df["hour"] = df["datetime"].dt.hour
    df["is_weekend"] = df["datetime"].dt.dayofweek >= 5  # 5=Sat, 6=Sun
    return df


def check_bocconi(df: pd.DataFrame) -> None:
    print("\n--- Bocconi University (cell 4259): expect LOWER activity on weekends ---")
    cell_df = df[df["CellID"] == BOCCONI_CELL]
    if cell_df.empty:
        print(f"WARNING: no rows found for cell {BOCCONI_CELL} — check CellID range in your data.")
        return
    weekday_avg = cell_df[~cell_df["is_weekend"]]["total_activity"].mean()
    weekend_avg = cell_df[cell_df["is_weekend"]]["total_activity"].mean()
    print(f"Weekday avg total activity: {weekday_avg:.2f}")
    print(f"Weekend avg total activity: {weekend_avg:.2f}")
    result = "PASS" if weekend_avg < weekday_avg else "UNEXPECTED"
    print(f"Result: {result} (expected weekend < weekday)")


def check_navigli(df: pd.DataFrame) -> None:
    print("\n--- Navigli nightlife district (cell 4456): expect internet activity to RISE in the evening ---")
    cell_df = df[df["CellID"] == NAVIGLI_CELL]
    if cell_df.empty:
        print(f"WARNING: no rows found for cell {NAVIGLI_CELL} — check CellID range in your data.")
        return
    if "internet" not in cell_df.columns:
        print("WARNING: no 'internet' column found — can't check this pattern.")
        return
    evening = cell_df[cell_df["hour"].between(18, 23)]["internet"].mean()
    daytime = cell_df[cell_df["hour"].between(6, 17)]["internet"].mean()
    print(f"Daytime (6-17h) avg internet activity: {daytime:.2f}")
    print(f"Evening (18-23h) avg internet activity: {evening:.2f}")
    result = "PASS" if evening > daytime else "UNEXPECTED"
    print(f"Result: {result} (expected evening > daytime)")


def check_duomo(df: pd.DataFrame) -> None:
    print("\n--- Duomo, city centre (cell 5060): expect HIGH overall activity relative to other cells ---")
    cell_avg = df.groupby("CellID")["total_activity"].mean()
    if DUOMO_CELL not in cell_avg.index:
        print(f"WARNING: no rows found for cell {DUOMO_CELL} — check CellID range in your data.")
        return
    duomo_value = cell_avg.loc[DUOMO_CELL]
    percentile = (cell_avg < duomo_value).mean() * 100
    print(f"Duomo's average total activity: {duomo_value:.2f}")
    print(f"Duomo ranks at the {percentile:.1f}th percentile across all {len(cell_avg)} cells")
    result = "PASS" if percentile >= 90 else "WEAKER THAN EXPECTED"
    print(f"Result: {result} (expected roughly top 10%, i.e. >=90th percentile)")


if __name__ == "__main__":
    df = load_aggregated()
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")
    check_bocconi(df)
    check_navigli(df)
    check_duomo(df)
    print("\nDone. Paste the printed output back so we can confirm the pipeline is reading real patterns correctly.")