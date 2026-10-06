"""
src/evaluation/flag_quality.py  (Phase 1)

Out-of-sample check of the forecasting model as a CONGESTION FLAGGER:
does "forecast above threshold" agree with "actually above threshold"?

Steps (run one at a time; outputs go to data/experiments/phase1/):

  replicate  Re-trains the original 60-tree LightGBM (lightgbm_model.py settings, train origins
             before 2013-12-17) for each horizon, prints the same MAE table as the original, and
             saves the test predictions for the later steps. Also reports hotspot MAE under both
             hotspot definitions, and (+1h only) the effect of float32 features and n_jobs=12.
  flags      Precision / recall / F1 / confusion counts / flagged vs actual share / MAE / MAPE on
             the test window for LightGBM, seasonal-naive (t-24h) and weekly-naive (t-168h),
             sliced by cell type, weekday/weekend and holiday/non-holiday (by TARGET date).
  bootstrap  Paired day-level bootstrap (90% intervals) of LightGBM minus each naive baseline,
             for F1 and MAE, per horizon and slice.
  insample   Analysis of the existing (in-sample) data/raw/cell_forecasts.parquet.
  channels   Each channel's share of total_activity, overall and for cells 4259, 4456, 5060.
  holiday    Read-only: on Dec 25/26, how many flagged cells would be ANOMALOUS under the
             30% neighbour rule alone (no holiday exemption).

Design decisions (also recorded in docs/BALANCEGRID_IMPROVEMENTS.md):
  - Thresholds: per-cell 90th percentile of total_activity over every hour before the test
    cutoff (2013-11-01 00:00 to 2013-12-16 23:00), i.e. all data known at training time.
  - Exceedance uses a strict ">" (same as congestion_threshold.py).
  - Hotspots for all new results: top 200 cells by training-period mean.
  - Slices use the TARGET date (the hour being forecast). A slice with fewer than 5 target days is
    marked "indicative".
  - MAPE uses the original script's convention: actual and forecast clipped below at 1.

RUN:
    python src/evaluation/flag_quality.py replicate
    python src/evaluation/flag_quality.py flags
    python src/evaluation/flag_quality.py bootstrap
    python src/evaluation/flag_quality.py insample
    python src/evaluation/flag_quality.py channels
    python src/evaluation/flag_quality.py holiday
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import (  # noqa: E402
    HORIZONS, HOLIDAYS, PRODUCTION_PARAMS, RAW_DIR, ROOT, Step, build_frame, cell_training_mean,
    fit_predict, flag_metrics, get_neighbors, hotspot_cells_training, load_panel, mae, mape_clipped,
    training_thresholds,
)

OUT_DIR = ROOT / "data" / "experiments" / "phase1"
INDICATIVE_DAYS = 5
N_JOBS_NEW = 12


def pred_path(h: int) -> Path:
    return OUT_DIR / f"lgbm60_test_preds_{h}h.npy"


# ============================================================================ replicate
def step_replicate():
    with Step("load panel + neighbour average"):
        panel = load_panel()
        cell_means = cell_training_mean(panel)
    hot_train = set(hotspot_cells_training(cell_means).tolist())
    print(f"\nCutoff (first test origin): {panel.hours[panel.cutoff_idx]}  "
          f"train origins {panel.hours[panel.first_origin]} .. {panel.hours[panel.cutoff_idx - 1]}  "
          f"test origins {panel.hours[panel.cutoff_idx]} .. {panel.hours[panel.last_origin]}")

    rows, knob_rows = [], []
    for h in HORIZONS:
        with Step(f"+{h}h build frames"):
            X_tr, y_tr, _ = build_frame(panel, h, "train", cell_means)
            X_te, y_te, info = build_frame(panel, h, "test", cell_means)
        if h == 1:
            print(f"  train rows {len(X_tr):,}  test rows {len(X_te):,}")
        with Step(f"+{h}h fit 60 trees (float64, default threads = original)"):
            model, preds, fit_s = fit_predict(X_tr, y_tr, X_te, PRODUCTION_PARAMS)
        np.save(pred_path(h), preds)
        base = info["seasonal_naive"]

        # Original hotspot definition: 98th percentile of cell_historical_mean taken over TEST ROWS.
        test_means = pd.Series(X_te["cell_historical_mean"].to_numpy())
        cutoff_orig = test_means.quantile(0.98)
        hot_orig = (test_means >= cutoff_orig).to_numpy()
        hot_tr = np.isin(info["cell"], list(hot_train))
        n_hot_orig_cells = len(np.unique(info["cell"][hot_orig]))

        row = {
            "horizon": f"+{h}h", "mae_lgbm": mae(y_te, preds), "mae_naive": mae(y_te, base),
            "mape_lgbm": mape_clipped(y_te, preds), "mape_naive": mape_clipped(y_te, base),
            "hot_orig_cells": n_hot_orig_cells,
            "hot_orig_mae_lgbm": mae(y_te[hot_orig], preds[hot_orig]), "hot_orig_mae_naive": mae(y_te[hot_orig], base[hot_orig]),
            "typ_orig_mae_lgbm": mae(y_te[~hot_orig], preds[~hot_orig]), "typ_orig_mae_naive": mae(y_te[~hot_orig], base[~hot_orig]),
            "hot_train_mae_lgbm": mae(y_te[hot_tr], preds[hot_tr]), "hot_train_mae_naive": mae(y_te[hot_tr], base[hot_tr]),
            "fit_seconds": fit_s,
        }
        row["improvement"] = (row["mae_naive"] - row["mae_lgbm"]) / row["mae_naive"]
        rows.append(row)
        imp = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
        print(f"\n+{h}h  naive MAE {row['mae_naive']:.2f} MAPE {row['mape_naive']:.1%} | LightGBM MAE {row['mae_lgbm']:.2f} "
              f"MAPE {row['mape_lgbm']:.1%} | improvement {row['improvement']:.1%}")
        print(f"     hotspot (original def, {n_hot_orig_cells} cells, cutoff {cutoff_orig:.2f}): LightGBM {row['hot_orig_mae_lgbm']:.2f}  naive {row['hot_orig_mae_naive']:.2f}")
        print(f"     typical (original def): LightGBM {row['typ_orig_mae_lgbm']:.2f}  naive {row['typ_orig_mae_naive']:.2f}")
        print(f"     hotspot (training def, 200 cells): LightGBM {row['hot_train_mae_lgbm']:.2f}  naive {row['hot_train_mae_naive']:.2f}")
        print(f"     top 6 importances: {dict(imp.head(6))}")
        if h == 1:
            print(f"     cells in original hotspot set but not training top-200: "
                  f"{len(set(np.unique(info['cell'][hot_orig])) - hot_train)}; reverse: "
                  f"{len(hot_train - set(np.unique(info['cell'][hot_orig])))}")

            # Effect of the memory/speed settings used for new runs, measured not assumed.
            for label, dtype, n_jobs in [("float64, n_jobs=12", np.float64, N_JOBS_NEW),
                                         ("float32, n_jobs=12", np.float32, N_JOBS_NEW)]:
                Xa, Xb = (X_tr, X_te) if dtype == np.float64 else (X_tr.astype(np.float32), X_te.astype(np.float32))
                with Step(f"+1h fit 60 trees ({label})"):
                    _, p2, fit2 = fit_predict(Xa, y_tr, Xb, PRODUCTION_PARAMS, n_jobs=n_jobs)
                knob_rows.append({"variant": label, "mae_lgbm": mae(y_te, p2),
                                  "delta_vs_original": mae(y_te, p2) - row["mae_lgbm"],
                                  "max_abs_pred_diff": float(np.max(np.abs(p2 - preds))),
                                  "hot_train_mae": mae(y_te[hot_tr], p2[hot_tr]), "fit_seconds": fit2})
                del Xa, Xb, p2
                gc.collect()
        del X_tr, y_tr, X_te, y_te, info, model, preds
        gc.collect()

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "step1_replication.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nREPLICATION TABLE")
    print(table.round(3).to_string(index=False))
    if knob_rows:
        knobs = pd.DataFrame(knob_rows)
        knobs.to_csv(OUT_DIR / "step1_knobs_1h.csv", index=False)
        print("\n+1h effect of dtype / threads (vs original float64, default threads)")
        print(knobs.to_string(index=False))


# ============================================================================ flags
def _slices(info: dict, hot_cells: set) -> dict:
    target = pd.DatetimeIndex(info["target_time"])
    dates = target.date
    is_hot = np.isin(info["cell"], list(hot_cells))
    weekend = target.dayofweek.to_numpy() >= 5
    holiday = np.isin(dates, list(HOLIDAYS))
    every = np.ones(len(dates), dtype=bool)
    return {"all": every, "typical cells": ~is_hot, "hotspot cells": is_hot,
            "weekday": ~weekend, "weekend": weekend, "non-holiday": ~holiday, "holiday": holiday,
            "_dates": dates}


def step_flags():
    with Step("load panel + thresholds"):
        panel = load_panel()
        cell_means = cell_training_mean(panel)
        thr = training_thresholds(panel)
    hot = set(hotspot_cells_training(cell_means).tolist())
    thr_arr = thr.reindex(panel.cells).to_numpy()
    print(f"Training-period thresholds: mean {thr_arr.mean():.1f}; Bocconi {thr.loc[4259]:.1f}, "
          f"Navigli {thr.loc[4456]:.1f}, Duomo {thr.loc[5060]:.1f}")

    records = []
    for h in HORIZONS:
        preds = np.load(pred_path(h))
        _, y, info = build_frame(panel, h, "test", cell_means)
        n_t = len(y) // len(panel.cells)
        cell_thr = np.repeat(thr_arr, n_t)
        actual = y > cell_thr
        weekly_missing = int(np.isnan(info["weekly_naive"]).sum())
        print(f"\n+{h}h: weekly-naive missing for {weekly_missing} of {len(y):,} test rows")
        sl = _slices(info, hot)
        dates = sl.pop("_dates")
        forecasters = {"LightGBM-60": preds, "seasonal-naive": info["seasonal_naive"], "weekly-naive": info["weekly_naive"]}
        for sname, mask in sl.items():
            n_days = len(np.unique(dates[mask]))
            for fname, f in forecasters.items():
                m = flag_metrics(actual[mask], f[mask] > cell_thr[mask])
                records.append({"horizon": h, "slice": sname, "forecaster": fname, "days": n_days,
                                "cell_hours": int(mask.sum()),
                                "indicative": n_days < INDICATIVE_DAYS,
                                "mae": mae(y[mask], f[mask]), "mape": mape_clipped(y[mask], f[mask]), **m})
        del preds, y, info, actual, cell_thr
        gc.collect()

    res = pd.DataFrame(records)
    res.to_csv(OUT_DIR / "step2_flag_quality.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    show = res.copy()
    show["slice"] = np.where(show["indicative"], show["slice"] + " (indicative)", show["slice"])
    cols = ["horizon", "slice", "forecaster", "days", "cell_hours", "precision", "recall", "f1",
            "flagged_share", "actual_share", "tp", "fp", "fn", "tn", "mae", "mape"]
    print("\nFLAG QUALITY (test window, training-period thresholds and hotspots)")
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ============================================================================ bootstrap
BOOT_N = 5000      # resamples
BOOT_SEED = 0
BOOT_CI = (5, 95)  # 90% percentile interval


def step_bootstrap():
    """Paired day-level bootstrap of LightGBM minus seasonal-naive (and minus weekly-naive):
    differences in F1 and MAE, resampling the TARGET days of the test window with replacement.
    Days are the unit because errors within a day are strongly correlated across cells and hours;
    pairing (same resampled days for both forecasters) removes the shared day-to-day variation."""
    panel = load_panel()
    cell_means = cell_training_mean(panel)
    thr_arr = training_thresholds(panel).reindex(panel.cells).to_numpy()
    hot = set(hotspot_cells_training(cell_means).tolist())
    rng = np.random.default_rng(BOOT_SEED)
    records = []
    for h in HORIZONS:
        preds = np.load(pred_path(h))
        _, y, info = build_frame(panel, h, "test", cell_means)
        cell_thr = np.repeat(thr_arr, len(y) // len(panel.cells))
        actual = y > cell_thr
        sl = _slices(info, hot)
        dates = sl.pop("_dates")
        day_codes, day_index = np.unique(dates, return_inverse=True)
        for sname in ("all", "typical cells", "hotspot cells", "weekday", "non-holiday"):
            mask = sl[sname]
            days_in = np.unique(day_index[mask])
            stats = {}
            for fname, f in (("lgbm", preds), ("naive", info["seasonal_naive"]), ("weekly", info["weekly_naive"])):
                flag = f > cell_thr
                d = day_index[mask]
                tp = np.bincount(d, (actual & flag)[mask], minlength=len(day_codes))[days_in]
                fp = np.bincount(d, (~actual & flag)[mask], minlength=len(day_codes))[days_in]
                fn = np.bincount(d, (actual & ~flag)[mask], minlength=len(day_codes))[days_in]
                ae = np.bincount(d, np.abs(y - f)[mask], minlength=len(day_codes))[days_in]
                n = np.bincount(d, minlength=len(day_codes))[days_in]
                stats[fname] = (tp, fp, fn, ae, n)
            draws = rng.integers(0, len(days_in), size=(BOOT_N, len(days_in)))

            def f1_mae(s, idx=None):
                tp, fp, fn, ae, n = (a if idx is None else a[idx].sum(axis=1) for a in s)
                if idx is None:
                    tp, fp, fn, ae, n = tp.sum(), fp.sum(), fn.sum(), ae.sum(), n.sum()
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan), ae / n

            f1_l, mae_l = f1_mae(stats["lgbm"])
            f1_lb, mae_lb = f1_mae(stats["lgbm"], draws)
            for other in ("naive", "weekly"):
                f1_o, mae_o = f1_mae(stats[other])
                f1_ob, mae_ob = f1_mae(stats[other], draws)
                df1, dmae = f1_lb - f1_ob, mae_lb - mae_ob
                records.append({
                    "horizon": h, "slice": sname, "vs": other, "days": len(days_in),
                    "f1_lgbm": float(f1_l), "f1_other": float(f1_o), "f1_diff": float(f1_l - f1_o),
                    "f1_diff_lo": np.nanpercentile(df1, BOOT_CI[0]), "f1_diff_hi": np.nanpercentile(df1, BOOT_CI[1]),
                    "mae_lgbm": float(mae_l), "mae_other": float(mae_o), "mae_diff": float(mae_l - mae_o),
                    "mae_diff_lo": np.percentile(dmae, BOOT_CI[0]), "mae_diff_hi": np.percentile(dmae, BOOT_CI[1]),
                })
        del preds, y, info, actual
        gc.collect()
    res = pd.DataFrame(records)
    res["f1_verdict"] = np.select([res.f1_diff_lo > 0, res.f1_diff_hi < 0], ["LightGBM better", "LightGBM worse"], "not significant")
    res["mae_verdict"] = np.select([res.mae_diff_hi < 0, res.mae_diff_lo > 0], ["LightGBM better", "LightGBM worse"], "not significant")
    res.to_csv(OUT_DIR / "step2_bootstrap.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    print(f"\nPAIRED DAY-LEVEL BOOTSTRAP ({BOOT_N} resamples, seed {BOOT_SEED}, 90% intervals): LightGBM-60 minus other")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ============================================================================ insample
def step_insample():
    """The existing cell_forecasts.parquet is IN-SAMPLE (trained and predicted on the same rows),
    compared against the full-period thresholds and flags it was built with."""
    with Step("load forecasts + flags"):
        fc = pd.read_parquet(RAW_DIR / "cell_forecasts.parquet", columns=["CellID", "datetime", "forecast_1h"])
        fl = pd.read_parquet(RAW_DIR / "cdr_with_congestion_flags.parquet",
                             columns=["CellID", "datetime", "congestion_threshold", "is_congestion_risk"])
    for name, d in (("forecasts", fc), ("flags", fl)):
        ok = d["CellID"].is_monotonic_increasing and (d.groupby("CellID")["datetime"].is_monotonic_increasing.all())
        if not ok:
            raise ValueError(f"{name} not sorted by CellID, datetime")
    n_cells = fc["CellID"].nunique()
    n_hours = len(fc) // n_cells
    f = fc["forecast_1h"].to_numpy().reshape(n_cells, n_hours)
    thr = fl["congestion_threshold"].to_numpy().reshape(n_cells, n_hours)
    actual = fl["is_congestion_risk"].to_numpy().reshape(n_cells, n_hours)
    del fc, fl

    # The forecast made at origin t is for target t+1; align on the TARGET hour grid.
    pred_on_target = np.full_like(f, np.nan)
    pred_on_target[:, 1:] = f[:, :-1]
    has = ~np.isnan(pred_on_target)
    flagged = has & (pred_on_target > thr)
    N = actual.size
    missing_hours_per_cell = int((~has).sum(axis=1)[0])
    m_all = flag_metrics(actual.ravel(), flagged.ravel())
    m_has = flag_metrics(actual[has], flagged[has])
    gap_pp = (m_all["actual_share"] - m_all["flagged_share"]) * 100
    missing_contrib_pp = actual[~has].sum() / N * 100
    out = {
        "target_hours_total": int(N), "target_hours_without_forecast": int((~has).sum()),
        "missing_target_hours_per_cell": missing_hours_per_cell,
        "actual_share_pct": m_all["actual_share"] * 100, "flagged_share_pct": m_all["flagged_share"] * 100,
        "gap_pp": gap_pp, "gap_from_missing_forecasts_pp": missing_contrib_pp,
        "gap_from_rows_with_forecast_pp": gap_pp - missing_contrib_pp,
        "recall_all_target_hours": m_all["recall"], "precision_all": m_all["precision"], "f1_all": m_all["f1"],
        "recall_rows_with_forecast": m_has["recall"], "flagged_share_rows_with_forecast_pct": m_has["flagged_share"] * 100,
        "actual_share_rows_with_forecast_pct": m_has["actual_share"] * 100,
    }
    with open(OUT_DIR / "step2_insample.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nIN-SAMPLE cell_forecasts.parquet (+1h, full-period thresholds)")
    for k, v in out.items():
        print(f"  {k:42s} {v:,.4f}" if isinstance(v, float) else f"  {k:42s} {v:,}")


# ============================================================================ channels
def step_channels():
    cols = ["smsin", "smsout", "callin", "callout", "internet"]
    with Step("load aggregated activity"):
        d = pd.read_parquet(RAW_DIR / "cdr_activity_aggregated.parquet", columns=["CellID"] + cols)
    rows = []
    for label, mask in [("all cells", slice(None)), ("Bocconi 4259", d["CellID"] == 4259),
                        ("Navigli 4456", d["CellID"] == 4456), ("Duomo 5060", d["CellID"] == 5060)]:
        sums = d.loc[mask, cols].sum()
        rows.append({"scope": label, **(sums / sums.sum() * 100).round(2).to_dict()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "step2_channel_shares.csv", index=False)
    print("\nCHANNEL SHARE OF total_activity (%), full 62 days")
    print(res.to_string(index=False))


# ============================================================================ holiday rule
def step_holiday():
    d = pd.read_parquet(RAW_DIR / "diagnosis_results_1h.parquet",
                        columns=["CellID", "target_datetime_1h", "classification", "reason"])
    d["t"] = pd.to_datetime(d["target_datetime_1h"])
    at_risk = d.groupby("t")["CellID"].apply(set).to_dict()
    rows = []
    for day in ("2013-12-25", "2013-12-26"):
        sub = d[d["t"].dt.date == pd.Timestamp(day).date()]
        would_be_anomalous = 0
        for cell, t in zip(sub["CellID"].to_numpy(), sub["t"]):
            nbrs = get_neighbors(int(cell))
            frac = sum(n in at_risk[t] for n in nbrs) / len(nbrs)
            would_be_anomalous += frac < 0.3
        rows.append({"target_date": day, "flagged": len(sub),
                     "currently_routine": int((sub["classification"] == "ROUTINE").sum()),
                     "anomalous_without_holiday_exemption": int(would_be_anomalous),
                     "share": would_be_anomalous / len(sub) if len(sub) else float("nan")})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "step_optional_holiday_rule.csv", index=False)
    print("\nHOLIDAY EXEMPTION (read-only, existing diagnosis_results_1h.parquet)")
    print(res.to_string(index=False))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = {"replicate": step_replicate, "flags": step_flags, "bootstrap": step_bootstrap, "insample": step_insample,
             "channels": step_channels, "holiday": step_holiday}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT_DIR / f"timing_{sys.argv[1]}.csv", index=False)
