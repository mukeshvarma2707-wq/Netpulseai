"""
src/ingestion/cdr_loader.py

Loads the Milan telecom ACTIVITY dataset (sms-call-internet-mi-*.csv files),
one day at a time, summing across CountryCode WITHIN each file before
combining across days.

WHY THIS MATTERS (memory safety): each daily file contains multiple rows per
(CellID, timestamp) — one per country code active in that cell that
interval. Loading all 7 days of raw, un-aggregated rows into memory at once
before doing anything with them is unnecessarily heavy on a constrained
laptop. Since we already know we need to sum across country codes eventually
(to get each cell's TOTAL activity per interval), doing that sum per-file,
immediately after loading it, means we only ever hold one day's raw data in
memory at a time — the combined result across all 7 days is much smaller
because it's already aggregated.

REQUIREMENTS:
    pip install pandas pyarrow

RUN:
    python src/ingestion/cdr_loader.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ACTIVITY_PATTERN = "sms-call-internet-mi-*.csv"

# Columns we expect to sum across country codes. If your real file has
# different column names, this script will tell you exactly what it found
# so we can adjust — it does NOT silently guess.
ACTIVITY_COLS_GUESS = ["smsin", "smsout", "callin", "callout", "internet"]


def find_activity_files() -> list[Path]:
    files = sorted(RAW_DIR.glob(ACTIVITY_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{ACTIVITY_PATTERN}' found in {RAW_DIR}. "
            f"Files currently in that folder: {[p.name for p in RAW_DIR.glob('*.csv')]}"
        )
    return files


def load_one_day_aggregated(path: Path, cell_col: str, time_col: str, activity_cols: list[str]) -> pd.DataFrame:
    """Loads a single day's file and immediately collapses country-code rows
    into one row per (cell, timestamp) by summing activity columns."""
    print(f"  Loading {path.name} ...", flush=True)
    day_df = pd.read_csv(path, sep=None, engine="python")

    # Downcast numeric columns to reduce memory before aggregating
    for col in activity_cols:
        if col in day_df.columns:
            day_df[col] = pd.to_numeric(day_df[col], errors="coerce", downcast="float")

    agg = (
        day_df.groupby([cell_col, time_col], as_index=False)[activity_cols]
        .sum()
    )
    agg["_source_file"] = path.name
    print(f"    -> {len(day_df):,} raw rows collapsed to {len(agg):,} (cell, time) rows")
    return agg


def load_and_validate() -> pd.DataFrame:
    files = find_activity_files()
    print(f"Found {len(files)} activity file(s).")

    # Peek at the first file only, to confirm real column names before processing all 7
    print(f"\nPeeking at {files[0].name} to confirm real column names ...")
    peek = pd.read_csv(files[0], sep=None, engine="python", nrows=5)
    print("Real columns found:", list(peek.columns))

    cell_col = next((c for c in peek.columns if "cell" in c.lower()), None)
    time_col = next(
        (c for c in peek.columns if any(k in c.lower() for k in ("time", "date"))), None
    )
    activity_cols = [c for c in ACTIVITY_COLS_GUESS if c in peek.columns]

    if not cell_col or not time_col:
        raise ValueError(
            f"Couldn't auto-detect cell/time columns from: {list(peek.columns)}. "
            "Paste this column list back so we can fix the detection logic."
        )
    if not activity_cols:
        raise ValueError(
            f"None of the expected activity columns {ACTIVITY_COLS_GUESS} were found in "
            f"{list(peek.columns)}. Paste the real column list back so we can fix this."
        )

    print(f"Using cell_col='{cell_col}', time_col='{time_col}', activity_cols={activity_cols}")

    print("\nProcessing each day (loading + aggregating one at a time)...")
    daily_frames = [
        load_one_day_aggregated(f, cell_col, time_col, activity_cols) for f in files
    ]

    print("\nCombining all aggregated days...")
    df = pd.concat(daily_frames, ignore_index=True)

    print("\n--- VALIDATION (combined, already aggregated across country codes) ---")
    print("Final shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nSample rows:")
    print(df.head(10).to_string())

    if pd.api.types.is_numeric_dtype(df[time_col]):
        parsed = pd.to_datetime(df[time_col], unit="ms", errors="coerce")
        print(f"\n'{time_col}' parsed as epoch milliseconds.")
    else:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        print(f"\n'{time_col}' parsed as a normal datetime string.")
    print("Real date range:", parsed.min(), "to", parsed.max())
    unique_steps = sorted(parsed.dropna().unique())
    if len(unique_steps) > 1:
        step_diff = pd.Series(unique_steps).diff().dropna().mode()
        print("Most common interval between timestamps:", step_diff.iloc[0] if not step_diff.empty else "unknown")

    print("\nNumber of distinct cells:", df[cell_col].nunique())

    return df


if __name__ == "__main__":
    df = load_and_validate()
    out_path = RAW_DIR / "cdr_activity_aggregated.parquet"
    try:
        df.to_parquet(out_path, index=False)
        print(f"\nSaved aggregated activity data to {out_path}")
    except ImportError:
        csv_out = RAW_DIR / "cdr_activity_aggregated.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n'pyarrow' not installed — saved as CSV instead: {csv_out}")
    print("\nDone. Paste the printed output back so we can confirm the real structure together.")