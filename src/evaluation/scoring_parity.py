"""
src/evaluation/scoring_parity.py  (Stage B parity harness for src/forecasting/score_v2.py)

Loads the panel and the models ONCE and calls score_v2.score_origin() (the same function the
command line uses) for:
  - 30 parity origins (10 validation, 20 test): features vs build_rows on the full panel
    (bit-identical float32), forecasts vs the saved cell_forecasts_v2 (bit-identical, <= 1 float32
    ULP allowed and reported), flags, watch flags, diagnosis (class and reason) and V0 solver moves
    vs diagnosis_results_v2 / watch_flags_v2 / reallocation_results_v2 (0 mismatches required);
  - 6 weekly-boundary origins (Nov 2 00:00; Nov 7 20:00 .. Nov 8 00:00), no saved reference:
    features vs build_rows and forecasts vs the original evaluation path (phase2_common.predict on
    build_rows features) with the same model files;
  - a leakage test (extra): activity at hours >= origin replaced by garbage must not change anything;
  - v1 database SHA-256 / mtime and data/raw file list / sizes / mtimes before and after.
Outputs: data/experiments/scoring/.

RUN: python src/evaluation/scoring_parity.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.common import Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import W1, FeatureStore, build_rows, load_json, predict, training_cell_means  # noqa: E402
from src.forecasting import score_v2  # noqa: E402

OUT = ROOT / "data" / "experiments" / "scoring"
V2 = ROOT / "data" / "processed" / "v2"
RAW = ROOT / "data" / "raw"
VAL_ORIGINS = ["2013-12-10 02:00", "2013-12-10 08:00", "2013-12-11 12:00", "2013-12-12 17:00", "2013-12-13 20:00",
               "2013-12-14 11:00", "2013-12-15 15:00", "2013-12-15 19:00", "2013-12-16 07:00", "2013-12-16 19:00"]
TEST_ORIGINS = ["2013-12-18 02:00", "2013-12-18 07:00", "2013-12-18 11:00", "2013-12-18 14:00", "2013-12-20 17:00",
                "2013-12-20 20:00", "2013-12-21 02:00", "2013-12-21 11:00", "2013-12-29 07:00", "2013-12-29 14:00",
                "2013-12-29 20:00", "2013-12-25 11:00", "2013-12-25 17:00", "2014-01-01 14:00",
                "2013-12-17 00:00", "2013-12-24 10:00", "2013-12-27 15:00", "2013-12-31 22:00", "2014-01-01 01:00",
                "2014-01-01 19:00"]
BOUNDARY_ORIGINS = ["2013-11-02 00:00", "2013-11-07 20:00", "2013-11-07 21:00", "2013-11-07 22:00",
                    "2013-11-07 23:00", "2013-11-08 00:00"]
LEAK_ORIGINS = ["2013-12-13 20:00", "2013-12-29 14:00"]


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_state():
    return {"v1_db_sha256": _sha(RAW / "balancegrid.db"), "v1_db_mtime_ns": (RAW / "balancegrid.db").stat().st_mtime_ns,
            "raw_files": sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in RAW.iterdir())}


def ulp_diff(a, b):
    a32 = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b32 = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    return np.abs(a32 - b32)


def bits_equal(X, Y):
    """Bit-identical float32 frames (NaN == NaN); returns (equal, number of differing cells)."""
    a = X.to_numpy(np.float32).view(np.uint32)
    b = Y.to_numpy(np.float32).view(np.uint32)
    diff = (a != b).any(axis=1)
    return bool(not diff.any()), int(diff.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected_state()
    with Step("load panel"):
        panel = load_panel()
    with Step("feature store (original evaluation path)"):
        store = FeatureStore(panel)
        cm = training_cell_means(panel, W1)
        groups = tuple(load_json("final_features.json")["final_groups"])
    with Step("load models"):
        models = score_v2.load_models()
    thr_df = models["thr"].set_index("CellID").reindex(panel.cells)
    assert np.array_equal(thr_df.training_mean.to_numpy(), cm.reindex(panel.cells).to_numpy()), "training_mean != cell_historical_mean"
    thr = thr_df.congestion_threshold.to_numpy()
    with Step("load saved v2 outputs"):
        fc_v2 = pd.read_parquet(V2 / "cell_forecasts_v2.parquet")
        diag_v2 = pd.read_parquet(V2 / "diagnosis_results_v2.parquet")
        watch_v2 = pd.read_parquet(V2 / "watch_flags_v2.parquet")
        mv_v2 = pd.read_parquet(V2 / "reallocation_results_v2.parquet")
    fc_key = fc_v2.set_index(["origin_datetime", "horizon"]).sort_index()
    cells_idx = np.arange(len(panel.cells))
    hot = thr_df.is_hotspot.to_numpy()
    rows, crow = [], []
    held = {}

    with Step("parity: 30 origins"):
        for kind, origins in (("validation", VAL_ORIGINS), ("test", TEST_ORIGINS)):
            for o in origins:
                o = pd.Timestamp(o)
                t = int(panel.hours.get_loc(o))
                res = score_v2.score_origin(panel, models, o, return_features=True)
                if str(o) == "2013-12-18 11:00:00":
                    held = res
                fc = res["forecasts"]
                for h in (1, 2, 3, 4):
                    b = models["boosters"][score_v2.POOLED[h]]
                    Xref = build_rows(store, h, np.array([t]), cells_idx, cm, groups=groups)[0][b.feature_name()]
                    feq, fdiff = bits_equal(res["features"][h], Xref)
                    mine = fc[fc.horizon == h].sort_values("CellID")
                    saved = fc_key.loc[(o, h)].sort_values("CellID")
                    assert (mine.CellID.to_numpy() == saved.CellID.to_numpy()).all()
                    u = ulp_diff(mine.forecast.to_numpy(), saved.forecast.to_numpy())
                    near = int((np.abs(mine.forecast.to_numpy() - thr) / thr < 1e-6).sum())
                    T = o + pd.Timedelta(hours=h)
                    dm = res["diagnosis"][res["diagnosis"].horizon == h]
                    ds = diag_v2[(diag_v2.horizon == h) & (diag_v2.target_datetime == T)]
                    m = dm[["CellID", "classification", "reason"]].merge(ds[["CellID", "classification", "reason"]], on="CellID",
                                                                        how="outer", suffixes=("_s", "_v2"), indicator=True)
                    both = m._merge == "both"
                    wm = set(res["watch_flags"][res["watch_flags"].horizon == h].CellID)
                    ws = set(watch_v2[(watch_v2.horizon == h) & (watch_v2.target_datetime == T)].CellID)
                    row = {"origin": str(o), "period": kind, "horizon": h, "target": str(T),
                           "forecast_kind_scored": mine.forecast_kind.iloc[0], "forecast_kind_saved": str(saved.forecast_kind.iloc[0]),
                           "features_bit_identical": feq, "feature_rows_differing": fdiff,
                           "forecast_ulp_max": int(u.max()), "forecast_cells_differing": int((u > 0).sum()),
                           "cells_within_1e-6_of_threshold": near,
                           "flags_scored": len(dm), "flags_saved": len(ds),
                           "flag_set_mismatch": int((m._merge != "both").sum()),
                           "class_mismatch": int((m.classification_s[both] != m.classification_v2[both]).sum()),
                           "reason_mismatch": int((m.reason_s[both] != m.reason_v2[both]).sum()),
                           "watch_scored": len(wm), "watch_mismatch": len(wm ^ ws)}
                    if h == 1:
                        ms = res["solver_moves"]
                        mvs = mv_v2[mv_v2.target_datetime == T]
                        a = {(int(c), int(n)): v for c, n, v in zip(ms.congested_cell, ms.source_cell, ms.amount_moved)}
                        bb = {(int(c), int(n)): v for c, n, v in zip(mvs.congested_cell, mvs.source_cell, mvs.amount_moved)}
                        row.update({"moves_scored": len(a), "moves_saved": len(bb), "move_key_mismatch": len(set(a) ^ set(bb)),
                                    "move_max_abs_diff": max([abs(a[k] - bb[k]) for k in set(a) & set(bb)] or [0.0])})
                    rows.append(row)
    par = pd.DataFrame(rows)
    par.to_csv(OUT / "parity_30_origins.csv", index=False)

    with Step("boundary origins (no saved reference)"):
        for o in BOUNDARY_ORIGINS:
            o = pd.Timestamp(o)
            t = int(panel.hours.get_loc(o))
            res = score_v2.score_origin(panel, models, o, return_features=True, solver=True)
            fc = res["forecasts"]
            for h in (1, 2, 3, 4):
                b = models["boosters"][score_v2.POOLED[h]]
                Xref = build_rows(store, h, np.array([t]), cells_idx, cm, groups=groups)[0][b.feature_name()]
                feq, fdiff = bits_equal(res["features"][h], Xref)
                ref = predict(b, Xref)                                   # original evaluation path function
                if h in score_v2.HOTSPOT:
                    bh = models["boosters"][score_v2.HOTSPOT[h]]
                    ref[hot] = predict(bh, Xref[bh.feature_name()][hot])
                mine = fc[fc.horizon == h].sort_values("CellID").forecast.to_numpy()
                u = ulp_diff(mine, ref)
                crow.append({"origin": str(o), "horizon": h, "forecast_kind": fc[fc.horizon == h].forecast_kind.iloc[0],
                             "features_bit_identical": feq, "feature_rows_differing": fdiff,
                             "lag_168h_nan_share": float(Xref["lag_168h"].isna().mean()),
                             "weekly_naive_nan_share": float(Xref["weekly_naive"].isna().mean()),
                             "forecast_ulp_max": int(u.max()), "forecast_cells_differing": int((u > 0).sum()),
                             "flags": int((res["diagnosis"].horizon == h).sum())})
    bnd = pd.DataFrame(crow)
    bnd.to_csv(OUT / "parity_boundary_origins.csv", index=False)

    with Step("leakage test"):
        leak = []
        for o in LEAK_ORIGINS:
            o = pd.Timestamp(o)
            t = int(panel.hours.get_loc(o))
            base = score_v2.score_origin(panel, models, o)
            p2 = copy.copy(panel)
            p2.activity = panel.activity.copy()
            p2.neighbor_avg = panel.neighbor_avg.copy()
            rng = np.random.default_rng(1)
            p2.activity[:, t:] = rng.uniform(0, 5e4, size=p2.activity[:, t:].shape)
            p2.neighbor_avg[:, t:] = rng.uniform(0, 5e4, size=p2.neighbor_avg[:, t:].shape)
            alt = score_v2.score_origin(p2, models, o)
            same = all(base[k].reset_index(drop=True).equals(alt[k].reset_index(drop=True))
                       for k in ("forecasts", "diagnosis", "watch_flags", "solver_moves"))
            leak.append({"origin": str(o), "outputs_identical_when_future_is_garbage": same})
            del p2
    pd.DataFrame(leak).to_csv(OUT / "leakage_test.csv", index=False)
    if held:
        held["forecasts"].to_parquet(OUT / "harness_20131218_1100_forecasts.parquet", index=False)
        held["solver_moves"].to_parquet(OUT / "harness_20131218_1100_moves.parquet", index=False)

    after = protected_state()
    prot = {"v1_db_unchanged": before["v1_db_sha256"] == after["v1_db_sha256"] and before["v1_db_mtime_ns"] == after["v1_db_mtime_ns"],
            "raw_unchanged": before["raw_files"] == after["raw_files"], "raw_file_count": len(after["raw_files"]),
            "v1_db_sha256": after["v1_db_sha256"]}
    summary = {
        "parity_rows": len(par), "origins": par.origin.nunique(),
        "features_bit_identical_all": bool(par.features_bit_identical.all()),
        "forecast_ulp_max": int(par.forecast_ulp_max.max()), "forecast_cells_differing": int(par.forecast_cells_differing.sum()),
        "cells_within_1e-6_of_threshold": int(par["cells_within_1e-6_of_threshold"].sum()),
        "flag_set_mismatch": int(par.flag_set_mismatch.sum()), "class_mismatch": int(par.class_mismatch.sum()),
        "reason_mismatch": int(par.reason_mismatch.sum()), "watch_mismatch": int(par.watch_mismatch.sum()),
        "move_key_mismatch": int(par.move_key_mismatch.sum()), "move_max_abs_diff": float(par.move_max_abs_diff.max()),
        "moves_compared": int(par.moves_scored.sum()), "flags_compared": int(par.flags_scored.sum()),
        "kind_labels_agree": bool((par.forecast_kind_scored == par.forecast_kind_saved).all()),
        "boundary_features_bit_identical": bool(bnd.features_bit_identical.all()),
        "boundary_forecast_ulp_max": int(bnd.forecast_ulp_max.max()),
        "leakage_test_passed": all(r["outputs_identical_when_future_is_garbage"] for r in leak),
        **prot,
    }
    (OUT / "parity_summary.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(Step.log).to_csv(OUT / "timing_parity.csv", index=False)
    pd.set_option("display.width", 250)
    print(json.dumps(summary, indent=1))
    print(bnd.to_string(index=False))
    print(par.groupby(["period", "horizon"])[["flags_scored", "watch_scored", "forecast_ulp_max", "flag_set_mismatch"]].sum().to_string())


if __name__ == "__main__":
    with Step("TOTAL scoring parity"):
        main()
