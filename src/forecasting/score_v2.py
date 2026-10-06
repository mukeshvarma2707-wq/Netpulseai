"""
src/forecasting/score_v2.py  (v2: score one chosen origin hour from raw activity)

Scores ONE origin hour from raw activity with the saved v2 models, instead of reading saved
prediction files, and runs the unchanged v2 rules on the result:
    forecasts (+1h .. +4h) -> watch flags -> diagnosis (ROUTINE / ANOMALOUS) -> V0 solver moves (+1h).

THIS REPLAYS HOURS INSIDE THE 62-DAY DATASET (Nov 1 2013 - Jan 1 2014). IT IS NOT LIVE FORECASTING.

INFORMATION SET: features at origin t use activity up to t-1 only; hour t itself is never used.
The lead time from the last observed hour is therefore h+1 hours (2, 3, 4, 5 hours for +1h..+4h).
score_origin() slices the panel to hours < t before computing anything.

INPUTS (read-only):
    data/raw/cdr_with_congestion_flags.parquet  - ONLY CellID, datetime, total_activity
                                                  (via evaluation.common.load_panel; no threshold
                                                  or flag column of that file is read)
    data/processed/v2/models/*.txt              - LightGBM models (copied from data/experiments/phase2/models,
                                                  SHA-256 checked against models_manifest.json)
    data/processed/v2/thresholds_v2.parquet     - thresholds (Nov 1 - Dec 9), hotspot set,
                                                  training_mean (= cell_historical_mean)
OUTPUTS: data/processed/v2/scored/<YYYYMMDD_HHMM>/
    forecasts.parquet, watch_flags.parquet, diagnosis.parquet, solver_moves.parquet,
    coverage.parquet (+1h), manifest.json. Refuses to overwrite unless --force.

FORECAST_KIND (by target T = origin + h; also carried by watch flags, diagnosis and solver outputs):
    in-sample      T < 2013-12-10 00:00: a training target of the models; thresholds_v2 and
                   cell_historical_mean also use data after the origin (look-ahead). Not a measure of
                   prediction quality.
    validation     2013-12-10 00:00 <= T < 2013-12-17 00:00: used for model size and margin
                   selection; out-of-training, but not an untouched test.
    unevaluated    T >= 2013-12-17 00:00 from an origin before 2013-12-17 00:00: between validation
                   and test, no saved reference.
    test           T >= 2013-12-17 00:00 from an origin >= 2013-12-17 00:00, T <= 2014-01-01 23:00:
                   not used for size or margins; used for feature-group selection in Phase 2.
    forecast       T > 2014-01-01 23:00: forecast only, no actuals available.
    Origins before 2013-11-02 00:00 (fewer than 24 hours of history) are rejected; the latest
    origin is 2014-01-02 00:00 (uses data through Jan 1 23:00).

RUN:
    python src/forecasting/score_v2.py install-models            (once: copy models + write manifest)
    python src/forecasting/score_v2.py score --origin "2013-12-18 11:00" [--horizons 1 2 3 4] [--no-solver] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.common import GRID_SIZE, Step  # noqa: E402

V2 = ROOT / "data" / "processed" / "v2"
MODELS_DIR = V2 / "models"
SCORED_DIR = V2 / "scored"
SRC_MODELS = ROOT / "data" / "experiments" / "phase2" / "models"
POOLED = {h: f"w1__final_63l__{h}h" for h in (1, 2, 3, 4)}
HOTSPOT = {1: "w1__hotspot_only_final__1h"}
LAG_MIN = 24                                 # earliest origin index (lag_24h, roll_24h)
TRAIN_END = pd.Timestamp("2013-12-10")       # training targets < this
VAL_END = pd.Timestamp("2013-12-17")         # validation targets < this; test origins >= this
DATA_END = pd.Timestamp("2014-01-01 23:00")  # last hour with actuals
KIND_LABEL = {
    "in-sample": "in-sample: a training target of the models; thresholds_v2 and cell_historical_mean use data "
                 "after the origin (look-ahead); not a measure of prediction quality",
    "validation": "validation (used for model size and margin selection): out-of-training, but not an untouched test",
    "unevaluated": "unevaluated: between validation and test, no saved reference",
    "test": "test (Dec 17 onward): not used for model size or margins; used for feature-group selection in Phase 2",
    "forecast": "forecast: no actuals available",
}
INFO_SET = ("features at origin t use activity up to t-1 only; hour t is never used; lead time from the last "
            "observed hour is h+1 hours (2, 3, 4, 5 hours for +1h..+4h)")


# ============================================================================= helpers
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def forecast_kind(origin: pd.Timestamp, target: pd.Timestamp) -> str:
    if target < TRAIN_END:
        return "in-sample"
    if target < VAL_END:
        return "validation"
    if target > DATA_END:
        return "forecast"
    if origin < VAL_END:
        return "unevaluated"
    return "test"


def _model_info(path: Path) -> dict:
    head, params, trees = {}, {}, 0
    with open(path) as fh:
        in_params = False
        for line in fh:
            if line.startswith("Tree="):
                trees += 1
            elif line.startswith(("version=", "feature_names=", "max_feature_idx=", "objective=")):
                k, v = line.rstrip("\n").split("=", 1)
                head[k] = v
            elif line.startswith("parameters:"):
                in_params = True
            elif line.startswith("end of parameters"):
                in_params = False
            elif in_params and line.startswith("["):
                k, v = line.strip()[1:-1].split(": ", 1) if ": " in line else (line.strip()[1:-1].rstrip(":"), "")
                params[k] = v
    return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size, "format": head.get("version"),
            "objective": head.get("objective"), "features": head.get("feature_names", "").split(),
            "trees": trees, "num_leaves": int(params.get("num_leaves", 0)),
            "categorical_feature_index": params.get("categorical_feature"), "learning_rate": params.get("learning_rate"),
            "min_data_in_leaf": params.get("min_data_in_leaf"), "seed": params.get("seed")}


def install_models():
    """Copies (not moves) the five model files into data/processed/v2/models and writes the manifest."""
    import lightgbm
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    thr_path = V2 / "thresholds_v2.parquet"
    thr = pd.read_parquet(thr_path)
    models = {}
    for name in list(POOLED.values()) + list(HOTSPOT.values()):
        src, dst = SRC_MODELS / f"{name}.txt", MODELS_DIR / f"{name}.txt"
        if not dst.exists():
            shutil.copy2(src, dst)
        info = _model_info(dst)
        assert info["sha256"] == sha256(src), f"copy of {name} does not match the original"
        info["source"] = str(src.relative_to(ROOT))
        models[name] = info
    feats = {m["file"]: m["features"] for m in models.values()}
    for m in models.values():
        assert m["features"][8] == "CellID" and m["categorical_feature_index"] == "8"
    manifest = {
        "lightgbm_version": lightgbm.__version__,
        "models": models,
        "pooled_by_horizon": POOLED, "hotspot_model_by_horizon": HOTSPOT,
        "feature_order": "as stored in each model file (feature_names); 23 features; CellID (index 8) is categorical",
        "feature_dtype": "float32 (every feature, including CellID, as in build_rows)",
        "target_transform": "trained on log1p(activity); prediction = clip(expm1(raw), 0) cast to float32",
        "cell_historical_mean": {"source": str(thr_path.relative_to(ROOT)), "column": "training_mean",
                                 "sha256_of_thresholds_v2": sha256(thr_path),
                                 "definition": "per-cell mean of total_activity over training target hours (index 24 = Nov 2 00:00 .. Dec 9 23:00)"},
        "hotspot_set": {"rule": "top 2% (200 cells) by training mean (Nov 2 - Dec 9)",
                        "cell_ids": sorted(int(c) for c in thr.loc[thr.is_hotspot, "CellID"])},
        "thresholds": "per-cell 90th percentile of total_activity, 2013-11-01 00:00 .. 2013-12-09 23:00 (thresholds_v2)",
        "training_period": "targets 2013-11-02 .. 2013-12-09 23:00 (window 1); sizes chosen on Dec 10-16",
        "information_set": INFO_SET,
        "forecast_kind_rules": KIND_LABEL,
        "origin_range": "2013-11-02 00:00 .. 2014-01-02 00:00",
        "feature_lists_identical_across_pooled_models_except_naive_feature": len({tuple(f[:10] + f[11:]) for f in feats.values()}) == 1,
    }
    with open(MODELS_DIR / "models_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def load_models(verify: bool = True) -> dict:
    import lightgbm as lgb
    man = json.loads((MODELS_DIR / "models_manifest.json").read_text())
    boosters = {}
    for name, info in man["models"].items():
        p = MODELS_DIR / info["file"]
        if verify:
            assert sha256(p) == info["sha256"], f"{p} changed since install"
        boosters[name] = lgb.Booster(model_file=str(p))
    thr = pd.read_parquet(V2 / "thresholds_v2.parquet")
    return {"boosters": boosters, "manifest": man, "thr": thr}


# ============================================================================= features
def features_at(A: np.ndarray, nb_avg: np.ndarray, hours_before: pd.DatetimeIndex, origin: pd.Timestamp,
                h: int, cell_mean: np.ndarray, cells: np.ndarray) -> pd.DataFrame:
    """Feature frame for ONE origin (all cells), from activity strictly before the origin.
    A, nb_avg: (n_cells, t) with t = number of hours before the origin. Same arithmetic as
    phase2_common.FeatureStore + build_rows (cumulative sums from the first hour, so rolling
    statistics are bit-identical)."""
    t = A.shape[1]
    n = len(cells)

    def back(off):                        # A[t + off] or NaN if before the data
        return A[:, t + off] if t + off >= 0 else np.full(n, np.nan)

    c1 = np.concatenate([np.zeros((n, 1)), np.cumsum(A, axis=1)], axis=1)
    c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(A * A, axis=1)], axis=1)
    roll = {}
    for w in (3, 6, 24):
        s1 = c1[:, t] - c1[:, t - w]
        s2 = c2[:, t] - c2[:, t - w]
        m = s1 / w
        roll[w] = (m, np.sqrt(np.clip(s2 / w - m * m, 0, None)))
    del c1, c2
    last = A[:, t - 1].reshape(GRID_SIZE, GRID_SIZE, 1)
    pad_max = np.pad(last, ((1, 1), (1, 1), (0, 0)), constant_values=-np.inf)
    nmax = np.full_like(last, -np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nmax = np.maximum(nmax, pad_max[1 + dy: 1 + dy + GRID_SIZE, 1 + dx: 1 + dx + GRID_SIZE])
    pad_sum = np.pad(last, ((2, 2), (2, 2), (0, 0)))
    ones = np.pad(np.ones((GRID_SIZE, GRID_SIZE)), 2)
    rsum = np.zeros_like(last)
    rcnt = np.zeros((GRID_SIZE, GRID_SIZE))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy or dx:
                rsum += pad_sum[2 + dy: 2 + dy + GRID_SIZE, 2 + dx: 2 + dx + GRID_SIZE]
                rcnt += ones[2 + dy: 2 + dy + GRID_SIZE, 2 + dx: 2 + dx + GRID_SIZE]
    ring2 = (rsum / rcnt[:, :, None]).reshape(n)
    dow = origin.dayofweek
    cols = {
        "lag_1h": A[:, t - 1], "lag_2h": A[:, t - 2], "lag_3h": A[:, t - 3], "lag_24h": A[:, t - 24],
        "hour_of_day": np.full(n, origin.hour), "day_of_week": np.full(n, dow),
        "is_weekend": np.full(n, int(dow >= 5), dtype=np.int64), "neighbor_avg_lag1h": nb_avg[:, t - 1],
        "CellID": cells, "cell_historical_mean": cell_mean, f"naive_feature_{h}h": A[:, t + h - 24],
        "lag_168h": back(-168), "weekly_naive": back(h - 168),
    }
    for w in (3, 6, 24):
        cols[f"roll_mean_{w}h"], cols[f"roll_std_{w}h"] = roll[w]
    cols["momentum_1h"] = A[:, t - 1] - A[:, t - 2]
    cols["momentum_3h"] = A[:, t - 1] - A[:, t - 4]
    cols["neighbor_max_lag1h"] = nmax.reshape(n)
    cols["ring2_mean_lag1h"] = ring2
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})


# ============================================================================= scoring
def score_origin(panel, models: dict, origin, horizons=(1, 2, 3, 4), solver: bool = True, return_features: bool = False):
    """THE single scoring path (used by the command line and by the parity harness)."""
    from src.diagnosis.diagnosis_agent_v2 import diagnose
    from src.optimization import solver_v2

    origin = pd.Timestamp(origin)
    first = panel.hours[0]
    t = int((origin - first) / pd.Timedelta(hours=1))
    if origin != first + pd.Timedelta(hours=t):
        raise ValueError("origin must be on the hour")
    if t < LAG_MIN:
        raise ValueError(f"origin {origin} rejected: needs 24 hours of history (earliest 2013-11-02 00:00)")
    if t > len(panel.hours):
        raise ValueError(f"origin {origin} rejected: no data before it beyond {panel.hours[-1]}")
    # information set: only hours strictly before the origin are passed on
    A = panel.activity[:, :t]
    nb = panel.neighbor_avg[:, :t]
    hours_before = panel.hours[:t]
    thr_df = models["thr"].set_index("CellID").reindex(panel.cells)
    cell_mean = thr_df["training_mean"].to_numpy()
    is_hot = thr_df["is_hotspot"].to_numpy()
    thr = thr_df["congestion_threshold"]
    boosters = models["boosters"]
    parts, feats = [], {}
    for h in horizons:
        X = features_at(A, nb, hours_before, origin, h, cell_mean, panel.cells)
        b = boosters[POOLED[h]]
        X = X[b.feature_name()]
        f = np.clip(np.expm1(b.predict(X)), 0, None).astype(np.float32)
        model_version = np.full(len(f), "final_63l", dtype=object)
        if h in HOTSPOT:
            bh = boosters[HOTSPOT[h]]
            fh = np.clip(np.expm1(bh.predict(X[bh.feature_name()][is_hot])), 0, None).astype(np.float32)
            f[is_hot] = fh
            model_version[is_hot] = "hotspot_only_final"
        target = origin + pd.Timedelta(hours=h)
        parts.append(pd.DataFrame({"CellID": panel.cells.astype(np.int64), "origin_datetime": origin, "horizon": np.int8(h),
                                   "target_datetime": target, "forecast": f.astype(np.float64),
                                   "segment": np.where(is_hot, "hotspot", "typical"), "model_version": model_version,
                                   "forecast_kind": forecast_kind(origin, target)}))
        if return_features:
            feats[h] = X
    fc = pd.concat(parts, ignore_index=True)
    fc["origin_datetime"] = fc["origin_datetime"].astype("datetime64[ns]")
    fc["target_datetime"] = fc["target_datetime"].astype("datetime64[ns]")
    diag, watch = [], []
    for h in horizons:
        d_, w_ = diagnose(fc[fc.horizon == h].reset_index(drop=True), thr)
        diag.append(d_)
        watch.append(w_)
    diag = pd.concat(diag, ignore_index=True)
    watch = pd.concat(watch, ignore_index=True)
    moves = cov = None
    if solver and 1 in horizons:
        moves, cov, _ = solver_v2.run(fc, thr, diag)
    out = {"forecasts": fc, "diagnosis": diag, "watch_flags": watch, "solver_moves": moves, "coverage": cov}
    if return_features:
        out["features"] = feats
    return out


def write_outputs(res: dict, origin, models: dict, timings: dict, force: bool) -> Path:
    origin = pd.Timestamp(origin)
    d = SCORED_DIR / origin.strftime("%Y%m%d_%H%M")
    if d.exists() and not force:
        raise SystemExit(f"{d} exists; use --force to overwrite")
    d.mkdir(parents=True, exist_ok=True)
    for k in ("forecasts", "diagnosis", "watch_flags", "solver_moves", "coverage"):
        if res[k] is not None:
            res[k].to_parquet(d / f"{k}.parquet", index=False)
    fc = res["forecasts"]
    man = {"origin": str(origin), "information_set": INFO_SET,
           "note": "replays an hour inside the 62-day dataset; not live forecasting. Solver: V0, +1h, one-for-one "
                   "activity-unit assumption (moves between grid squares are not physical capacity moves).",
           "targets": {f"+{h}h": {"target": str(t), "forecast_kind": k, "label": KIND_LABEL[k]}
                       for h, t, k in fc.drop_duplicates("horizon")[["horizon", "target_datetime", "forecast_kind"]].itertuples(index=False)},
           "kind_applies_to": "forecasts, watch flags, diagnosis and solver outputs",
           "inputs": {"activity": "data/raw/cdr_with_congestion_flags.parquet (CellID, datetime, total_activity only)",
                      "thresholds_v2_sha256": models["manifest"]["cell_historical_mean"]["sha256_of_thresholds_v2"],
                      "models": {k: v["sha256"] for k, v in models["manifest"]["models"].items()}},
           "lightgbm_version": models["manifest"]["lightgbm_version"],
           "counts": {"forecast_rows": len(fc), "flags": len(res["diagnosis"]),
                      "routine": int((res["diagnosis"].classification == "ROUTINE").sum()),
                      "watch_flags": len(res["watch_flags"]),
                      "solver_moves": 0 if res["solver_moves"] is None else len(res["solver_moves"])},
           "timings_s": timings}
    (d / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install-models")
    s = sub.add_parser("score")
    s.add_argument("--origin", required=True)
    s.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    s.add_argument("--no-solver", action="store_true")
    s.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "install-models":
        with Step("install models"):
            m = install_models()
        print(json.dumps({k: (v["sha256"][:12], v["trees"], v["num_leaves"]) for k, v in m["models"].items()}, indent=1))
        return
    from src.evaluation.common import load_panel
    origin = pd.Timestamp(a.origin)
    d = SCORED_DIR / origin.strftime("%Y%m%d_%H%M")
    if d.exists() and not a.force:
        raise SystemExit(f"{d} exists; use --force to overwrite")
    timings = {}
    t0 = time.perf_counter()
    with Step("load panel (CellID, datetime, total_activity)"):
        panel = load_panel()
    timings["load_panel"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    with Step("load models"):
        models = load_models()
    timings["load_models"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    with Step(f"score origin {origin}"):
        try:
            res = score_origin(panel, models, origin, tuple(a.horizons), solver=not a.no_solver)
        except ValueError as e:
            raise SystemExit(str(e))
    timings["score"] = round(time.perf_counter() - t0, 1)
    out = write_outputs(res, origin, models, timings, a.force)
    print(json.loads((out / "manifest.json").read_text())["targets"])
    print(f"written to {out}")


if __name__ == "__main__":
    main()
