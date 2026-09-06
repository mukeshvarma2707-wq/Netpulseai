"""Load event and holiday calendar data."""
from pathlib import Path

import pandas as pd


def load_calendar(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Load a calendar CSV or Parquet file and normalize dates."""
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if date_column in frame.columns:
        frame[date_column] = pd.to_datetime(frame[date_column]).dt.date
    return frame
