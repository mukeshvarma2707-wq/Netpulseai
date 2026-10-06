"""
src/evaluation/capacity_check.py  (Phase 1, +1h only)

Is the production model under-trained? Re-trains the +1h model with 60 and
with 500 trees, every other setting identical to lightgbm_model.py
(learning rate 0.05, 15 leaves, min_child_samples 50, log1p target, same
features, same train/test split), and compares accuracy and flagging.

Reports, on the test window (origins 2013-12-17 00:00 to 2014-01-01 19:00):
  MAE (all cells, hotspot cells), precision, recall, F1, flagged share vs
  actual share, mean bias (forecast - actual) overall and on hotspots, and
  the fit time / peak memory of each fit.

Settings for these new fits: float32 features and n_jobs = 12 (the effect of
both on the 60-tree model is measured in flag_quality.py replicate, see
step1_knobs_1h.csv). Thresholds and hotspots come from the training period
(see common.training_thresholds / hotspot_cells_training).

RUN:
    python src/evaluation/capacity_check.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import (  # noqa: E402
    PRODUCTION_PARAMS, ROOT, Step, build_frame, cell_training_mean, fit_predict, flag_metrics,
    hotspot_cells_training, load_panel, mae, training_thresholds,
)

OUT_DIR = ROOT / "data" / "experiments" / "phase1"
H = 1
TREE_COUNTS = [60, 500]   # 60 = production; 500 = the comparison point used on the other laptop
N_JOBS = 12
DTYPE = np.float32

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with Step("load panel + thresholds"):
        panel = load_panel()
        cell_means = cell_training_mean(panel)
        thr = training_thresholds(panel)
    hot = set(hotspot_cells_training(cell_means).tolist())
    with Step("build +1h frames (float32)"):
        X_tr, y_tr, _ = build_frame(panel, H, "train", cell_means, dtype=DTYPE)
        X_te, y_te, info = build_frame(panel, H, "test", cell_means, dtype=DTYPE)
    is_hot = np.isin(info["cell"], list(hot))
    n_t = len(y_te) // len(panel.cells)
    cell_thr = np.repeat(thr.reindex(panel.cells).to_numpy(), n_t)
    actual = y_te > cell_thr
    del panel
    gc.collect()

    rows = []
    for n_trees in TREE_COUNTS:
        params = {**PRODUCTION_PARAMS, "n_estimators": n_trees}
        with Step(f"+1h fit {n_trees} trees") as s:
            _, preds, fit_s = fit_predict(X_tr, y_tr, X_te, params, n_jobs=N_JOBS)
        m = flag_metrics(actual, preds > cell_thr)
        mh = flag_metrics(actual[is_hot], preds[is_hot] > cell_thr[is_hot])
        rows.append({
            "trees": n_trees, "mae_all": mae(y_te, preds), "mae_hotspot": mae(y_te[is_hot], preds[is_hot]),
            "mae_typical": mae(y_te[~is_hot], preds[~is_hot]),
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "flagged_share": m["flagged_share"], "actual_share": m["actual_share"],
            "hotspot_f1": mh["f1"], "hotspot_flagged_share": mh["flagged_share"], "hotspot_actual_share": mh["actual_share"],
            "bias_all": float(np.mean(preds - y_te)), "bias_hotspot": float(np.mean(preds[is_hot] - y_te[is_hot])),
            "bias_typical": float(np.mean(preds[~is_hot] - y_te[~is_hot])),
            "fit_seconds": fit_s, "step_seconds": s.seconds,
            "peak_ram_gb": s.peak_ws, "peak_commit_gb": s.peak_commit,
        })
        del preds
        gc.collect()

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "step3_capacity_check.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nCAPACITY CHECK (+1h, test window; training-period thresholds and hotspots)")
    print(res.T.to_string(header=[f"{t} trees" for t in TREE_COUNTS], float_format=lambda v: f"{v:.4f}"))
    pd.DataFrame(Step.log).to_csv(OUT_DIR / "timing_capacity_check.csv", index=False)
