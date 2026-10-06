"""
src/evaluation/common.py

Shared, memory-lean data preparation for the out-of-sample evaluation
scripts (flag_quality.py, capacity_check.py).

It reproduces the feature set and row selection of
src/forecasting/lightgbm_model.py EXACTLY (same values, same row order), so
the 60-tree replication can match that script's printed MAE. It differs only
in how memory is used:

  - The data is a complete (cell x hour) grid (cdr_loader.py reindexes it),
    so total_activity is held as one 10,000 x 1,488 array ("panel") instead
    of a 14.88M-row DataFrame with ~25 columns.
  - Feature frames are built for one horizon and one split at a time.

Exactness notes (why some steps use pandas rather than numpy):
  - LightGBM samples rows to build its histogram bins, so row ORDER matters.
    Rows are produced cell-major, hour-ascending, which is the order
    lightgbm_model.py has after sort_values(["CellID", "datetime"]).
  - The neighbour average and the per-cell training mean are computed with
    the same pandas calls as the original, because a mathematically equal
    numpy reduction can differ in the last bit of a float64.

Constants match lightgbm_model.py: lags 1/2/3/24 h, horizons 1-4,
TRAIN_DAYS = 45 counted from the first complete row (2013-11-02 00:00),
giving a cutoff of 2013-12-17 00:00.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

GRID_SIZE = 100
LAGS = [1, 2, 3, 24]
HORIZONS = [1, 2, 3, 4]
TRAIN_DAYS = 45                       # same as lightgbm_model.py (uncommitted change from 5)
MAX_LAG = max(LAGS)                   # first usable origin = hour index 24 (2013-11-02 00:00)
MAX_HORIZON = max(HORIZONS)           # original drops origins lacking the +4h target, for every horizon
HOTSPOT_SHARE = 0.02                  # top 2% of cells
CONGESTION_PERCENTILE = 90            # same as congestion_threshold.py

# Test-window holidays (target dates). Dec 8 is a holiday too, but it falls in the training period.
HOLIDAYS = {pd.Timestamp("2013-12-25").date(), pd.Timestamp("2013-12-26").date(), pd.Timestamp("2014-01-01").date()}

# Original lightgbm_model.py hyper-parameters (the "production" 60-tree model).
PRODUCTION_PARAMS = dict(n_estimators=60, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                         random_state=42, verbosity=-1)

FEATURE_ORDER = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "hour_of_day", "day_of_week", "is_weekend",
                 "neighbor_avg_lag1h", "CellID", "cell_historical_mean"]  # + naive_feature_{h}h last


# ----------------------------------------------------------------------------- grid helpers
def cell_id_to_xy(cell_id: int) -> tuple:
    y, x = divmod(cell_id - 1, GRID_SIZE)
    return x + 1, y + 1


def xy_to_cell_id(x: int, y: int) -> int:
    return (y - 1) * GRID_SIZE + x


def get_neighbors(cell_id: int) -> list:
    """Same neighbour ORDER as lightgbm_model.py (dx outer, dy inner)."""
    x, y = cell_id_to_xy(cell_id)
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
                out.append(xy_to_cell_id(nx, ny))
    return out


# ----------------------------------------------------------------------------- memory tracking
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_k32, _psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
_k32.GetCurrentProcess.restype = wt.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]


def _memory_now() -> tuple[float, float]:
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1e9, c.PagefileUsage / 1e9


class Step:
    """Context manager: wall time and peak RAM / committed memory of one step,
    sampled every 50 ms (Windows' own peak counters are process-lifetime only)."""

    log: list[dict] = []

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self._stop = threading.Event()
        self.peak_ws, self.peak_commit = _memory_now()

        def sample():
            while not self._stop.wait(0.05):
                ws, commit = _memory_now()
                self.peak_ws, self.peak_commit = max(self.peak_ws, ws), max(self.peak_commit, commit)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self._stop.set()
        self._thread.join()
        ws, commit = _memory_now()
        self.peak_ws, self.peak_commit = max(self.peak_ws, ws), max(self.peak_commit, commit)
        Step.log.append(dict(step=self.name, seconds=round(self.seconds, 1),
                             peak_ram_gb=round(self.peak_ws, 2), peak_commit_gb=round(self.peak_commit, 2)))
        print(f"  [{self.name}] {self.seconds:,.1f}s, peak RAM {self.peak_ws:.2f} GB, peak commit {self.peak_commit:.2f} GB")
        return False


# ----------------------------------------------------------------------------- data
@dataclass
class Panel:
    cells: np.ndarray          # (n_cells,) sorted CellIDs
    hours: pd.DatetimeIndex    # (n_hours,) every hour, ascending
    activity: np.ndarray       # (n_cells, n_hours) float64 total_activity
    neighbor_avg: np.ndarray   # (n_cells, n_hours) float64 mean CURRENT activity of the 8-neighbourhood

    @property
    def first_origin(self) -> int:
        return MAX_LAG

    @property
    def last_origin(self) -> int:
        return len(self.hours) - 1 - MAX_HORIZON

    @property
    def cutoff_idx(self) -> int:
        """Index of the first TEST origin: first complete row + TRAIN_DAYS (same as the original)."""
        cutoff = self.hours[self.first_origin] + pd.Timedelta(days=TRAIN_DAYS)
        return int(self.hours.get_loc(cutoff))

    def split_hours(self, split: str) -> np.ndarray:
        if split == "train":
            return np.arange(self.first_origin, self.cutoff_idx)
        if split == "test":
            return np.arange(self.cutoff_idx, self.last_origin + 1)
        raise ValueError(split)


def load_panel() -> Panel:
    """Reads CellID/datetime/total_activity and checks the grid is complete and sorted."""
    df = pd.read_parquet(RAW_DIR / "cdr_with_congestion_flags.parquet", columns=["CellID", "datetime", "total_activity"])
    df = df.sort_values(["CellID", "datetime"], kind="stable").reset_index(drop=True)
    cells = np.sort(df["CellID"].unique())
    hours = pd.DatetimeIndex(np.sort(df["datetime"].unique()))
    if len(df) != len(cells) * len(hours):
        raise ValueError(f"Grid not complete: {len(df):,} rows vs {len(cells)} x {len(hours)}")
    expected = pd.date_range(hours[0], hours[-1], freq="h")
    if not hours.equals(expected):
        raise ValueError("Hours are not contiguous.")
    activity = df["total_activity"].to_numpy(dtype=np.float64).reshape(len(cells), len(hours))
    if not (df["CellID"].to_numpy().reshape(len(cells), len(hours))[:, 0] == cells).all():
        raise ValueError("Rows not cell-major after sort.")
    del df

    # Neighbour average: identical computation to lightgbm_model.add_spatial_neighbor_feature
    pivot = pd.DataFrame(activity.T, index=hours, columns=cells)
    neighbor_avg = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    present = set(cells.tolist())
    for c in cells:
        real_neighbors = [n for n in get_neighbors(int(c)) if n in present]
        neighbor_avg[c] = pivot[real_neighbors].mean(axis=1) if real_neighbors else np.nan
    nb = neighbor_avg.to_numpy(dtype=np.float64).T.copy()
    del pivot, neighbor_avg
    return Panel(cells=cells, hours=hours, activity=activity, neighbor_avg=nb)


def cell_training_mean(panel: Panel) -> pd.Series:
    """Per-cell mean of total_activity over the training ORIGIN rows, computed with the same
    pandas groupby as lightgbm_model.chronological_split (bit-identical)."""
    idx = panel.split_hours("train")
    values = panel.activity[:, idx].ravel()
    cell_col = np.repeat(panel.cells, len(idx))
    return pd.Series(values).groupby(cell_col).mean()


def training_thresholds(panel: Panel) -> pd.Series:
    """Per-cell congestion threshold from the TRAINING period only: the 90th percentile of each
    cell's total_activity over every hour observed before the test cutoff
    (2013-11-01 00:00 to 2013-12-16 23:00). Same rule as congestion_threshold.py, restricted to
    data available at training time.

    Nov 1 is included (it has no 24h lag, so it is not a training ROW, but its activity is known
    before the cutoff). Checked against the other laptop: including it reproduces the reference
    +1h F1 0.438 and flagged share 3.5% (excluding it gives 0.437 / 3.45%)."""
    idx = np.arange(0, panel.cutoff_idx)
    values = panel.activity[:, idx].ravel()
    cell_col = np.repeat(panel.cells, len(idx))
    return pd.Series(values).groupby(cell_col).quantile(CONGESTION_PERCENTILE / 100)


def hotspot_cells_training(cell_means: pd.Series) -> np.ndarray:
    """Top 2% of cells (200 of 10,000) by training-period mean activity. Used for all new results."""
    n = int(round(len(cell_means) * HOTSPOT_SHARE))
    return cell_means.sort_values(ascending=False, kind="stable").index[:n].to_numpy()


def build_frame(panel: Panel, h: int, split: str, cell_means: pd.Series, dtype=np.float64):
    """Feature frame for one horizon and split, in lightgbm_model.py's column and row order.

    Returns (X, y, info), where info holds per-row arrays used for evaluation:
    cell, origin/target times, seasonal-naive and weekly-naive forecasts.
    """
    idx = panel.split_hours(split)
    n_cells, n_t = len(panel.cells), len(idx)
    A = panel.activity

    def take(arr, offset):
        return arr[:, idx + offset].ravel()

    origin_times = panel.hours[idx]
    hod = np.tile(origin_times.hour.to_numpy(), n_cells)
    dow = np.tile(origin_times.dayofweek.to_numpy(), n_cells)
    X = pd.DataFrame({
        "lag_1h": take(A, -1), "lag_2h": take(A, -2), "lag_3h": take(A, -3), "lag_24h": take(A, -24),
        "hour_of_day": hod, "day_of_week": dow, "is_weekend": (dow >= 5).astype(np.int64),
        "neighbor_avg_lag1h": take(panel.neighbor_avg, -1),
        "CellID": np.repeat(panel.cells, n_t),
        "cell_historical_mean": np.repeat(cell_means.reindex(panel.cells).to_numpy(), n_t),
        f"naive_feature_{h}h": take(A, h - 24),
    })
    if dtype != np.float64:
        X = X.astype(dtype)
    y = take(A, h)

    weekly_idx = idx + h - 168
    weekly = np.full(n_cells * n_t, np.nan)
    ok = weekly_idx >= 0
    if ok.any():
        weekly = A[:, np.where(ok, weekly_idx, 0)].ravel()
        weekly[~np.tile(ok, n_cells)] = np.nan

    info = {
        "cell": np.repeat(panel.cells, n_t),
        "target_time": np.tile((panel.hours[idx] + pd.Timedelta(hours=h)).to_numpy(), n_cells),
        "seasonal_naive": take(A, h - 24),
        "weekly_naive": weekly,
    }
    return X, y, info


def fit_predict(X_train, y_train, X_test, params: dict, n_jobs=None):
    """Fits LightGBM on log1p(target), returns (model, clipped expm1 predictions, fit seconds)."""
    import lightgbm as lgb

    kw = dict(params)
    if n_jobs is not None:
        kw["n_jobs"] = n_jobs
    model = lgb.LGBMRegressor(**kw)
    t0 = time.perf_counter()
    model.fit(X_train, np.log1p(y_train), categorical_feature=["CellID"])
    fit_s = time.perf_counter() - t0
    preds = np.clip(np.expm1(model.predict(X_test)), 0, None)
    return model, preds, fit_s


# ----------------------------------------------------------------------------- metrics
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mape_clipped(y, p):
    """Same convention as lightgbm_model.py: actual and forecast clipped below at 1."""
    yc, pc = np.clip(y, 1, None), np.clip(p, 1, None)
    return float(np.mean(np.abs(yc - pc) / yc))


def flag_metrics(actual_flag: np.ndarray, pred_flag: np.ndarray) -> dict:
    tp = int(np.sum(actual_flag & pred_flag))
    fp = int(np.sum(~actual_flag & pred_flag))
    fn = int(np.sum(actual_flag & ~pred_flag))
    tn = int(np.sum(~actual_flag & ~pred_flag))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (tp + fp and tp + fn and precision + recall) else float("nan")
    n = tp + fp + fn + tn
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn,
                flagged_share=(tp + fp) / n if n else float("nan"),
                actual_share=(tp + fn) / n if n else float("nan"))
