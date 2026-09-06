"""Feature engineering for time-series demand forecasting."""
import pandas as pd


def add_lag_and_seasonal_features(
    frame: pd.DataFrame,
    target: str,
    timestamp_column: str = "timestamp",
    lags: tuple[int, ...] = (1, 24, 168),
) -> pd.DataFrame:
    """Add calendar signals and lag values while preserving input row order."""
    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column])
    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    for lag in lags:
        result[f"{target}_lag_{lag}"] = result[target].shift(lag)
    return result
