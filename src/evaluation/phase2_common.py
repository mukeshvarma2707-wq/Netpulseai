"""
src/evaluation/phase2_common.py

Shared machinery for Phase 2 (tune_trees.py, feature_ablation.py):
evaluation windows, feature groups G1-G4, stratified cell samples, model
fitting with a validation curve, the size-selection rule, prediction
storage, and slice / bootstrap evaluation.

Builds on common.py (panel, production features, metrics). Every design
constant is defined here with its reason; see docs/BALANCEGRID_IMPROVEMENTS.md
(Phase 2) for the full list.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.common import (
    CONGESTION_PERCENTILE, GRID_SIZE, HOTSPOT_SHARE, MAX_HORIZON, MAX_LAG, PRODUCTION_PARAMS, ROOT,
    Panel, Step, flag_metrics, get_neighbors,
)

OUT_DIR = ROOT / "data" / "experiments" / "phase2"
PRED_DIR = OUT_DIR / "preds"
MODEL_DIR = OUT_DIR / "models"

ALL_HOLIDAYS = {pd.Timestamp(d).date() for d in ("2013-11-01", "2013-12-08", "2013-12-25", "2013-12-26", "2014-01-01")}

# ----------------------------------------------------------------------------- model size
LEARNING_RATE = 0.1
LEAF_GRID = [15, 31, 63, 127]
TREE_CAP = 3000
WITHIN_BEST = 0.005          # smallest tree count whose val loss <= (1 + 0.5%) x best loss
MIN_CHILD_SAMPLES = 50       # kept from production
N_JOBS = 12
SEED = 42
FEATURE_DTYPE = np.float32
MIN_HEADROOM_GB = 6.0

# ----------------------------------------------------------------------------- sample
SAMPLE_CELLS = 1000          # 10% of the grid
SAMPLE_SEED = 0

# ----------------------------------------------------------------------------- bootstrap
BOOT_N = 5000
BOOT_SEED = 0
BOOT_CI = (5, 95)
INDICATIVE_DAYS = 5


# ============================================================================= memory guard
class _MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def commit_headroom_gb() -> float:
    """System-wide commit headroom (commit limit minus committed), as in Task Manager."""
    s = _MEMSTATUS()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPageFile / 1e9


def require_headroom(what: str, min_gb: float = MIN_HEADROOM_GB):
    gb = commit_headroom_gb()
    print(f"  headroom before {what}: {gb:.1f} GB")
    if gb < min_gb:
        raise SystemExit(f"STOP: commit headroom {gb:.1f} GB < {min_gb} GB before {what}. Free memory and re-run.")


# ============================================================================= windows
@dataclass(frozen=True)
class Window:
    """All split boundaries are TARGET times except test, which uses origins (as in Phase 1:
    the same origin rows for every horizon, so +h never reaches past the window)."""
    name: str
    train_target_end: str        # train: targets < this (and origins >= first complete row)
    val_target_start: str        # validation: targets in [start, end)
    val_target_end: str
    test_origin_start: str       # test: origins in [start, end_inclusive]
    test_origin_end: str
    threshold_end: str           # evaluation thresholds: activity hours < this (all data known at test time)


W1 = Window("w1", train_target_end="2013-12-10", val_target_start="2013-12-10", val_target_end="2013-12-17",
            test_origin_start="2013-12-17 00:00", test_origin_end="2014-01-01 19:00", threshold_end="2013-12-17")
W2 = Window("w2", train_target_end="2013-11-18", val_target_start="2013-11-18", val_target_end="2013-11-25",
            test_origin_start="2013-11-25 00:00", test_origin_end="2013-12-07 19:00", threshold_end="2013-11-25")
WINDOWS = {"w1": W1, "w2": W2}


def origin_index(panel: Panel, win: Window, split: str, h: int) -> np.ndarray:
    H = panel.hours
    t = np.arange(MAX_LAG, len(H) - MAX_HORIZON)   # every origin with full lags and a +4h target
    target = H[t] + pd.Timedelta(hours=h)
    if split == "train":
        keep = target < pd.Timestamp(win.train_target_end)
    elif split == "val":
        keep = (target >= pd.Timestamp(win.val_target_start)) & (target < pd.Timestamp(win.val_target_end))
    elif split == "test":
        keep = (H[t] >= pd.Timestamp(win.test_origin_start)) & (H[t] <= pd.Timestamp(win.test_origin_end))
    else:
        raise ValueError(split)
    return t[np.asarray(keep)]


def training_cell_means(panel: Panel, win: Window) -> pd.Series:
    """Per-cell mean activity over the training hours (first complete row up to train_target_end).
    Used for the cell_historical_mean feature and to pick hotspot cells."""
    end = int(panel.hours.get_loc(pd.Timestamp(win.train_target_end)))
    idx = np.arange(MAX_LAG, end)
    return pd.Series(panel.activity[:, idx].ravel()).groupby(np.repeat(panel.cells, len(idx))).mean()


def eval_thresholds(panel: Panel, win: Window) -> np.ndarray:
    """Per-cell 90th percentile of activity over every hour before the test window (Phase 1 rule)."""
    end = int(panel.hours.get_loc(pd.Timestamp(win.threshold_end)))
    idx = np.arange(0, end)
    q = pd.Series(panel.activity[:, idx].ravel()).groupby(np.repeat(panel.cells, len(idx))).quantile(CONGESTION_PERCENTILE / 100)
    return q.reindex(panel.cells).to_numpy()


def hotspots(cell_means: pd.Series) -> np.ndarray:
    n = int(round(len(cell_means) * HOTSPOT_SHARE))
    return cell_means.sort_values(ascending=False, kind="stable").index[:n].to_numpy()


def stratified_sample(panel: Panel, hot: np.ndarray, n_cells: int = SAMPLE_CELLS, seed: int = SAMPLE_SEED) -> np.ndarray:
    """Cell sample that keeps the hotspot share at HOTSPOT_SHARE (20 of 1,000)."""
    rng = np.random.default_rng(seed)
    n_hot = int(round(n_cells * HOTSPOT_SHARE))
    typical = np.setdiff1d(panel.cells, hot)
    chosen = np.concatenate([rng.choice(hot, n_hot, replace=False), rng.choice(typical, n_cells - n_hot, replace=False)])
    return np.sort(chosen)


# ============================================================================= feature groups
GROUPS = {
    "G1": ["lag_168h", "weekly_naive"],
    "G2": ["roll_mean_3h", "roll_std_3h", "roll_mean_6h", "roll_std_6h", "roll_mean_24h", "roll_std_24h"],
    "MOM": ["momentum_1h", "momentum_3h"],
    "G3": ["neighbor_max_lag1h", "ring2_mean_lag1h"],
    "G4": ["target_is_holiday", "naive_source_is_holiday"],
}


class FeatureStore:
    """Per-hour arrays (n_cells x n_hours) for every engineered feature, computed once.
    All features at origin t use activity up to t-1 only (same convention as the production lags)."""

    def __init__(self, panel: Panel):
        self.panel = panel
        A = panel.activity
        n_cells, n_h = A.shape
        if not (panel.cells == np.arange(1, n_cells + 1)).all() or n_cells != GRID_SIZE * GRID_SIZE:
            raise ValueError("Expected cells 1..10000 for the 2-D grid operations.")

        # Rolling mean / std over A[t-w .. t-1]  (value stored at index t)
        c1 = np.concatenate([np.zeros((n_cells, 1)), np.cumsum(A, axis=1)], axis=1)        # c1[:, k] = sum A[:, :k]
        c2 = np.concatenate([np.zeros((n_cells, 1)), np.cumsum(A * A, axis=1)], axis=1)
        self.roll = {}
        for w in (3, 6, 24):
            mean = np.full_like(A, np.nan)
            std = np.full_like(A, np.nan)
            s1 = c1[:, w:n_h] - c1[:, : n_h - w]            # sum of A[t-w .. t-1] for t = w .. n_h-1
            s2 = c2[:, w:n_h] - c2[:, : n_h - w]
            m = s1 / w
            mean[:, w:] = m
            std[:, w:] = np.sqrt(np.clip(s2 / w - m * m, 0, None))
            self.roll[w] = (mean, std)
        del c1, c2

        # Spatial: neighbour max and 2-step ring mean of CURRENT activity (lagged by 1 when used)
        grid = A.reshape(GRID_SIZE, GRID_SIZE, n_h)          # axis0 = y-1, axis1 = x-1
        pad_max = np.pad(grid, ((1, 1), (1, 1), (0, 0)), constant_values=-np.inf)
        nmax = np.full_like(grid, -np.inf)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    nmax = np.maximum(nmax, pad_max[1 + dy: 1 + dy + GRID_SIZE, 1 + dx: 1 + dx + GRID_SIZE])
        del pad_max
        pad_sum = np.pad(grid, ((2, 2), (2, 2), (0, 0)))
        ones = np.pad(np.ones((GRID_SIZE, GRID_SIZE)), 2)
        rsum = np.zeros_like(grid)
        rcnt = np.zeros((GRID_SIZE, GRID_SIZE))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy or dx:
                    rsum += pad_sum[2 + dy: 2 + dy + GRID_SIZE, 2 + dx: 2 + dx + GRID_SIZE]
                    rcnt += ones[2 + dy: 2 + dy + GRID_SIZE, 2 + dx: 2 + dx + GRID_SIZE]
        del pad_sum
        self.nmax = nmax.reshape(n_cells, n_h)
        self.ring2 = (rsum / rcnt[:, :, None]).reshape(n_cells, n_h)
        del rsum, grid
        self.hour_dates = panel.hours.date
        gc.collect()


def check_spatial(store: FeatureStore, cells=(1, 4259, 5060, 10000), t: int = 500):
    """Spot-check the vectorised spatial features against a direct per-cell computation."""
    A = store.panel.activity
    for c in cells:
        nb = get_neighbors(c)
        x, y = (c - 1) % GRID_SIZE + 1, (c - 1) // GRID_SIZE + 1
        ring = [(yy - 1) * GRID_SIZE + xx for yy in range(y - 2, y + 3) for xx in range(x - 2, x + 3)
                if 1 <= xx <= GRID_SIZE and 1 <= yy <= GRID_SIZE and (xx, yy) != (x, y)]
        assert np.isclose(store.nmax[c - 1, t], max(A[n - 1, t] for n in nb)), c
        assert np.isclose(store.ring2[c - 1, t], np.mean([A[n - 1, t] for n in ring])), c
        assert len(ring) == {1: 8, 10000: 8}.get(c, len(ring))


def build_rows(store: FeatureStore, h: int, t_idx: np.ndarray, cells_idx: np.ndarray, cell_means: pd.Series,
               groups=(), dtype=FEATURE_DTYPE):
    """Feature frame (production features + chosen groups) for origins t_idx and the cells at
    positions cells_idx, in cell-major order. Returns X, y, meta(cell, target_time)."""
    P = store.panel
    A = P.activity

    def take(arr, offset=0):
        return arr[cells_idx][:, t_idx + offset].ravel()

    n_c, n_t = len(cells_idx), len(t_idx)
    origin_times = P.hours[t_idx]
    dow = np.tile(origin_times.dayofweek.to_numpy(), n_c)
    cols = {
        "lag_1h": take(A, -1), "lag_2h": take(A, -2), "lag_3h": take(A, -3), "lag_24h": take(A, -24),
        "hour_of_day": np.tile(origin_times.hour.to_numpy(), n_c), "day_of_week": dow,
        "is_weekend": (dow >= 5).astype(np.int64), "neighbor_avg_lag1h": take(P.neighbor_avg, -1),
        "CellID": np.repeat(P.cells[cells_idx], n_t),
        "cell_historical_mean": np.repeat(cell_means.reindex(P.cells[cells_idx]).to_numpy(), n_t),
        f"naive_feature_{h}h": take(A, h - 24),
    }
    if "G1" in groups:
        def back(offset):
            ok = (t_idx + offset) >= 0
            v = A[cells_idx][:, np.where(ok, t_idx + offset, 0)]
            v[:, ~ok] = np.nan
            return v.ravel()
        cols["lag_168h"] = back(-168)
        cols["weekly_naive"] = back(h - 168)
    if "G2" in groups:
        for w in (3, 6, 24):
            mean, std = store.roll[w]
            cols[f"roll_mean_{w}h"] = take(mean)
            cols[f"roll_std_{w}h"] = take(std)
    if "MOM" in groups:
        cols["momentum_1h"] = take(A, -1) - take(A, -2)
        cols["momentum_3h"] = take(A, -1) - take(A, -4)
    if "G3" in groups:
        cols["neighbor_max_lag1h"] = take(store.nmax, -1)
        cols["ring2_mean_lag1h"] = take(store.ring2, -1)
    if "G4" in groups:
        tgt_dates = (P.hours[t_idx] + pd.Timedelta(hours=h)).date
        src_dates = (P.hours[t_idx] + pd.Timedelta(hours=h - 24)).date
        cols["target_is_holiday"] = np.tile(np.isin(tgt_dates, list(ALL_HOLIDAYS)).astype(np.int64), n_c)
        cols["naive_source_is_holiday"] = np.tile(np.isin(src_dates, list(ALL_HOLIDAYS)).astype(np.int64), n_c)
    # Convert column by column so no full float64 copy of the frame is ever held.
    X = pd.DataFrame({k: np.asarray(v, dtype=dtype) for k, v in cols.items()})
    del cols
    y = take(A, h)
    meta = {"cell": np.repeat(P.cells[cells_idx], n_t),
            "target_time": np.tile((P.hours[t_idx] + pd.Timedelta(hours=h)).to_numpy(), n_c),
            "seasonal_naive": take(A, h - 24)}
    return X, y, meta


# ============================================================================= fitting
def lgbm_params(num_leaves: int, n_estimators: int) -> dict:
    return dict(n_estimators=n_estimators, learning_rate=LEARNING_RATE, num_leaves=num_leaves,
                min_child_samples=MIN_CHILD_SAMPLES, random_state=SEED, n_jobs=N_JOBS, verbosity=-1)


def fit_with_curve(X_tr, y_tr, X_val, y_val, params: dict):
    """Fits on log1p(target) and records the validation L2 (on the log1p scale) after every tree."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params)
    record: dict = {}
    t0 = time.perf_counter()
    model.fit(X_tr, np.log1p(y_tr), categorical_feature=["CellID"],
              eval_X=(X_val,), eval_y=(np.log1p(y_val),), eval_metric="l2",
              callbacks=[lgb.record_evaluation(record)])
    fit_s = time.perf_counter() - t0
    curve = np.asarray(record["valid_0"]["l2"])
    return model, curve, fit_s


def fit_plain(X_tr, y_tr, params: dict):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params)
    t0 = time.perf_counter()
    model.fit(X_tr, np.log1p(y_tr), categorical_feature=["CellID"])
    return model, time.perf_counter() - t0


def choose_trees(curve: np.ndarray) -> dict:
    """Size rule: the smallest tree count whose validation loss is within 0.5% of the best."""
    best_i = int(np.argmin(curve))
    k = int(np.argmax(curve <= curve[best_i] * (1 + WITHIN_BEST))) + 1     # 1-based tree count
    return {"best_trees": best_i + 1, "best_loss": float(curve[best_i]), "chosen_trees": k,
            "chosen_loss": float(curve[k - 1]), "cap_reached": best_i + 1 == len(curve),
            "loss_at_100": float(curve[min(99, len(curve) - 1)])}


def predict(model, X, n_trees: int | None = None) -> np.ndarray:
    return np.clip(np.expm1(model.predict(X, num_iteration=n_trees)), 0, None).astype(np.float32)


def save_preds(window: str, variant: str, h: int, split: str, preds: np.ndarray):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PRED_DIR / f"{window}__{variant}__{h}h__{split}.npy", preds.astype(np.float32))


def load_preds(window: str, variant: str, h: int, split: str) -> np.ndarray:
    return np.load(PRED_DIR / f"{window}__{variant}__{h}h__{split}.npy")


def has_preds(window: str, variant: str, h: int, split: str = "test") -> bool:
    return (PRED_DIR / f"{window}__{variant}__{h}h__{split}.npy").exists()


def save_json(name: str, obj):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(name: str):
    with open(OUT_DIR / name) as fh:
        return json.load(fh)


# ============================================================================= evaluation
def slice_masks(meta: dict, hot: set) -> dict:
    target = pd.DatetimeIndex(meta["target_time"])
    dates = target.date
    is_hot = np.isin(meta["cell"], list(hot))
    weekend = target.dayofweek.to_numpy() >= 5
    holiday = np.isin(dates, list(ALL_HOLIDAYS))
    return {"all": np.ones(len(dates), bool), "typical": ~is_hot, "hotspot": is_hot,
            "weekday": ~weekend, "weekend": weekend, "non-holiday": ~holiday, "holiday": holiday}, dates


def evaluate(forecasts: dict, y: np.ndarray, thr_rows: np.ndarray, meta: dict, hot: set,
             slices=("all", "typical", "hotspot", "weekday", "weekend", "non-holiday", "holiday")) -> pd.DataFrame:
    masks, dates = slice_masks(meta, hot)
    actual = y > thr_rows
    out = []
    for s in slices:
        m = masks[s]
        if not m.any():
            continue
        n_days = len(np.unique(dates[m]))
        for name, f in forecasts.items():
            fm = flag_metrics(actual[m], f[m] > thr_rows[m])
            out.append({"slice": s, "forecaster": name, "days": n_days, "cell_hours": int(m.sum()),
                        "indicative": n_days < INDICATIVE_DAYS,
                        "mae": float(np.mean(np.abs(y[m] - f[m]))), "bias": float(np.mean(f[m] - y[m])), **fm})
    return pd.DataFrame(out)


def paired_bootstrap(forecasts: dict, reference: str, y: np.ndarray, thr_rows: np.ndarray, meta: dict, hot: set,
                     slices=("all", "hotspot", "non-holiday")) -> pd.DataFrame:
    """For each forecaster vs `reference`: difference (forecaster - reference) in F1 and MAE with a
    paired day-level bootstrap (resample target days with replacement; same days for both)."""
    masks, dates = slice_masks(meta, hot)
    actual = y > thr_rows
    rng = np.random.default_rng(BOOT_SEED)
    _, day_index = np.unique(dates, return_inverse=True)
    n_days_total = day_index.max() + 1
    out = []
    for s in slices:
        m = masks[s]
        days_in = np.unique(day_index[m])
        d = day_index[m]
        draws = rng.integers(0, len(days_in), size=(BOOT_N, len(days_in)))

        def per_day(f):
            flag = f[m] > thr_rows[m]
            a = actual[m]
            return [np.bincount(d, v, minlength=n_days_total)[days_in] for v in
                    (a & flag, ~a & flag, a & ~flag, np.abs(y[m] - f[m]), np.ones(m.sum()))]

        def stats(parts, idx=None):
            tp, fp, fn, ae, n = (p.sum() if idx is None else p[idx].sum(axis=1) for p in parts)
            with np.errstate(invalid="ignore", divide="ignore"):
                f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
            return f1, ae / n

        ref = per_day(forecasts[reference])
        f1_r, mae_r = stats(ref)
        f1_rb, mae_rb = stats(ref, draws)
        for name, f in forecasts.items():
            if name == reference:
                continue
            parts = per_day(f)
            f1_o, mae_o = stats(parts)
            f1_ob, mae_ob = stats(parts, draws)
            df1, dmae = f1_ob - f1_rb, mae_ob - mae_rb
            out.append({"slice": s, "forecaster": name, "vs": reference, "days": len(days_in),
                        "f1": float(f1_o), "f1_ref": float(f1_r), "f1_diff": float(f1_o - f1_r),
                        "f1_lo": float(np.nanpercentile(df1, BOOT_CI[0])), "f1_hi": float(np.nanpercentile(df1, BOOT_CI[1])),
                        "mae": float(mae_o), "mae_ref": float(mae_r), "mae_diff": float(mae_o - mae_r),
                        "mae_lo": float(np.percentile(dmae, BOOT_CI[0])), "mae_hi": float(np.percentile(dmae, BOOT_CI[1]))})
    res = pd.DataFrame(out)
    if len(res):
        res["f1_verdict"] = np.select([res.f1_lo > 0, res.f1_hi < 0], ["better", "worse"], "n.s.")
        res["mae_verdict"] = np.select([res.mae_hi < 0, res.mae_lo > 0], ["better", "worse"], "n.s.")
    return res
