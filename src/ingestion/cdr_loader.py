"""
src/ingestion/cdr_loader.py

Loads the FULL Milan Grid CDR dataset from Harvard Dataverse - real,
raw, tab-separated .txt files at native 10-minute granularity - and
aggregates them into the same hourly format the rest of this pipeline
was built and validated against.

REAL FORMAT, CONFIRMED DIRECTLY FROM THE RAW FILES (not assumed):
    - Tab-separated, NO header row
    - 8 fields per row: CellID, timestamp_ms, countrycode, smsin,
      smsout, callin, callout, internet
    - timestamp_ms is milliseconds since Unix epoch, UTC - but Milan's
      local time (what the filenames and everything downstream is
      labeled in) is UTC+1 in this period, so it must be explicitly
      converted via the Europe/Rome timezone, not just parsed naively.
    - Empty fields mean "no activity that interval" - treated as 0 when
      summing, not as missing data.
    - Real interval confirmed: 10 minutes (the earlier Kaggle CSV mirror
      of this same dataset had already pre-aggregated this to hourly).

DESIGN DECISION (flagging explicitly): every downstream script in this
pipeline (congestion thresholds, forecasting, diagnosis, the solver) was
built and validated at HOURLY granularity. Rather than redesign all of
that for native 10-minute data - a large rewrite for uncertain benefit,
given our actual use case is hour-ahead capacity planning, not
minute-level reaction - this loader aggregates the raw 10-minute data
UP to hourly during ingestion, exactly reproducing what the Kaggle
mirror had already done. Every downstream script keeps working
unchanged.

REQUIREMENTS:
    pip install pandas pyarrow

RUN:
    python src/ingestion/cdr_loader.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

COLUMN_NAMES = ["CellID", "timestamp_ms", "countrycode", "smsin", "smsout", "callin", "callout", "internet"]
ACTIVITY_COLS = ["smsin", "smsout", "callin", "callout", "internet"]


def find_activity_files() -> list:
    files = sorted(RAW_DIR.glob("sms-call-internet-mi-*.txt"))
    if not files:
        files = sorted(RAW_DIR.glob("sms-call-internet-mi-*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No files matching 'sms-call-internet-mi-*.txt' or '*.csv' found in {RAW_DIR}. "
            f"Files currently in that folder: {[f.name for f in RAW_DIR.iterdir()]}"
        )
    return files


def load_and_aggregate_one_file(path: Path) -> pd.DataFrame:
    """Loads one raw file, converts timestamps to real local Milan time,
    and aggregates to hourly totals per cell - done per-file (not all
    files at once) to stay memory-safe, the same pattern used from the
    start of this project."""
    is_raw_txt = path.suffix == ".txt"

    if is_raw_txt:
        df = pd.read_csv(path, sep="\t", header=None, names=COLUMN_NAMES)
        df["datetime"] = (
            pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            .dt.tz_convert("Europe/Rome")
            .dt.tz_localize(None)
        )
        for col in ACTIVITY_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])

    raw_row_count = len(df)

    per_cell_hour = df.groupby(["CellID", "datetime"])[ACTIVITY_COLS].sum(min_count=0).reset_index()

    if is_raw_txt:
        per_cell_hour["datetime"] = per_cell_hour["datetime"].dt.floor("h")
        per_cell_hour = per_cell_hour.groupby(["CellID", "datetime"])[ACTIVITY_COLS].sum(min_count=0).reset_index()

    print(f"    -> {raw_row_count:,} raw rows collapsed to {len(per_cell_hour):,} (cell, hour) rows")
    return per_cell_hour


if __name__ == "__main__":
    files = find_activity_files()
    print(f"Found {len(files)} activity file(s).")

    if files[0].suffix == ".txt":
        print("\nDetected raw Harvard Dataverse .txt files (tab-separated, 10-minute granularity, no header).")
    else:
        print("\nDetected pre-aggregated Kaggle .csv files (hourly, comma-separated, with header).")

    print("\nProcessing each day (loading + aggregating one at a time)...")
    all_days = []
    for f in files:
        print(f"  Loading {f.name} ...")
        all_days.append(load_and_aggregate_one_file(f))

    print("\nCombining all aggregated days...")
    combined = pd.concat(all_days, ignore_index=True)

    # REAL GAP FOUND AND FIXED: some (cell, hour) combinations have ZERO
    # rows in the raw data at all (not a zero-value row — no row exists),
    # when a cell had no activity across every country code that hour.
    # Confirmed by day-by-day row counts drifting below the expected
    # 240,000 (10,000 cells x 24 hours) as the dataset progresses.
    # Left unfixed, this would silently corrupt lag-based features
    # downstream (a shift(1) would grab the wrong hour's value across a
    # gap, with no error raised). Reindex to guarantee every cell has
    # exactly one row per hour across the full range, filling any truly
    # missing hour with 0 — consistent with how empty raw fields are
    # already treated.
    full_hours = pd.date_range(combined["datetime"].min(), combined["datetime"].max(), freq="h")
    all_cells = combined["CellID"].unique()
    full_index = pd.MultiIndex.from_product([all_cells, full_hours], names=["CellID", "datetime"])
    before_reindex = len(combined)
    combined = (
        combined.set_index(["CellID", "datetime"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    gap_rows_filled = len(combined) - before_reindex
    print(f"Reindexed to guarantee complete (cell, hour) coverage: "
          f"filled {gap_rows_filled:,} genuinely missing hours with 0 "
          f"({gap_rows_filled/len(combined):.3%} of the full grid).")

    print("\n--- VALIDATION (combined, already aggregated across country codes and to hourly) ---")
    print("Final shape:", combined.shape)
    print("Columns:", list(combined.columns))
    print("\nMissing values:")
    print(combined.isna().sum())
    print("\nSample rows:")
    print(combined.head(10).to_string())

    print("\nReal date range:", combined["datetime"].min(), "to", combined["datetime"].max())
    intervals = combined.sort_values(["CellID", "datetime"]).groupby("CellID")["datetime"].diff().dropna()
    print("Most common interval between timestamps:", intervals.mode()[0] if len(intervals) else "n/a")
    print("\nNumber of distinct cells:", combined["CellID"].nunique())

    out_path = RAW_DIR / "cdr_activity_aggregated.parquet"
    try:
        combined.to_parquet(out_path, index=False)
        print(f"\nSaved aggregated activity data to {out_path}")
    except ImportError:
        csv_out = RAW_DIR / "cdr_activity_aggregated.csv"
        combined.to_csv(csv_out, index=False)
        print(f"\n'pyarrow' not installed - saved as CSV instead: {csv_out}")

    print("\nDone. Paste the printed output back so we can confirm the real structure together.")