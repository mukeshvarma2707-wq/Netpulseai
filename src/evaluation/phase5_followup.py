"""
src/evaluation/phase5_followup.py  (review follow-ups to Phase 5; reuses phase5_transfer)

  rates     Actual / flagged exceedance shares per window (overall, hotspot, typical) and the
            weekly exceedance share over Nov 25 - Dec 7 and Dec 17 - Jan 1 (each window's thresholds).
  train     Training days, rows and weekly-lag coverage of the window-2 models (Nov 2-17).
  fit2b     Extra +1h pair trained on targets Nov 2-24 (frozen size), tested Nov 25 - Dec 7:
            Milan-trained (predicted on Trentino) and Trentino-trained. NOTE: Nov 18-24 was used
            before for window-2 sizing (Milan tree counts, Phase 3 check 5); here it is training data.
  cmp2b     New pair vs the previous (Nov 2-17) pair: MAE and F1, paired day bootstrap.
  margins   Window 2 margin gain (m = 1 vs Milan's margin), paired day bootstrap, both models.

RUN: python src/evaluation/phase5_followup.py {rates|train|fit2b|cmp2b|margins}
"""

from __future__ import annotations

import dataclasses
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import MAX_LAG, ROOT, Step  # noqa: E402
from src.evaluation.phase2_common import BOOT_SEED, W1, W2, lgbm_params, origin_index, require_headroom  # noqa: E402
from src.evaluation.margin_tuning import flag_counts  # noqa: E402
from src.evaluation.phase5_transfer import (  # noqa: E402
    FROZEN_TREES, LEAVES, MILAN_MARGINS, City, _boot, _compare, from_sf, pred_path, sf_target,
)

OUT = ROOT / "data" / "experiments" / "phase5" / "followup"
W2B = dataclasses.replace(W2, name="w2b", train_target_end="2013-11-25")   # train targets Nov 2-24; same test
THR_END_W2 = "2013-11-25"


def _w(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def step_rates():
    m = pd.read_csv(ROOT / "data" / "experiments" / "phase5" / "transfer" / "metrics.csv")
    x = m[(m.eval_city == "trentino") & (m.margin == 1.0) & m.slice.isin(["all", "hotspot", "typical"])
          & m.forecaster.isin(["seasonal-naive", "Milan-trained SF (transfer)", "Trentino-trained SF (in-domain, frozen size)"])]
    tab = x.pivot_table(index=["window", "horizon", "slice"], columns="forecaster", values="flagged_share", sort=False)
    tab["actual_share"] = x.groupby(["window", "horizon", "slice"], sort=False)["actual_share"].first()
    tab = tab.reset_index()
    _w(tab, "base_rates_flagged.csv")
    tn = City("trentino")
    rows = []
    for name, thr_end, start, end in (("w2", "2013-11-25", "2013-11-25", "2013-12-08"), ("w1", "2013-12-10", "2013-12-17", "2014-01-02")):
        e = int(tn.hours.get_loc(pd.Timestamp(thr_end)))
        thr = np.quantile(tn.A[:, :e], 0.9, axis=1)
        hot = np.argsort(-tn.scale(W2 if name == "w2" else W1), kind="stable")[:125]
        is_hot = np.zeros(len(tn.cells), bool); is_hot[hot] = True
        i0, i1 = int(tn.hours.searchsorted(pd.Timestamp(start))), int(tn.hours.searchsorted(pd.Timestamp(end)))
        ex = tn.A[:, i0:i1] > thr[:, None]
        hrs = tn.hours[i0:i1]
        wk = (hrs - pd.to_timedelta(hrs.dayofweek, unit="D")).normalize()
        for w in np.unique(wk):
            cols = wk == w
            rows.append({"window": name, "week_start (Mon)": pd.Timestamp(w).date(), "days_in_window": int(cols.sum() / 24),
                         "all": ex[:, cols].mean(), "hotspot": ex[is_hot][:, cols].mean(), "typical": ex[~is_hot][:, cols].mean()})
        # daily persistence: share of exceeding cell-hours whose cell also exceeded at the same hour the day before
        prev = tn.A[:, i0 - 24:i1 - 24] > thr[:, None]
        rows.append({"window": name, "week_start (Mon)": "persistence", "days_in_window": int(ex.shape[1] / 24),
                     "all": (ex & prev).sum() / ex.sum(), "hotspot": np.nan, "typical": np.nan})
    _w(pd.DataFrame(rows), "weekly_exceedance.csv")
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def step_train():
    rows = []
    for city in ("milan", "trentino"):
        C = City(city)
        for win in (W2, W2B):
            for h in (1, 4):
                t = origin_index(C, win, "train", h)
                lag168_missing = float(np.mean(t - 168 < 0))
                weekly_missing = float(np.mean(t + h - 168 < 0))
                rows.append({"city": city, "window": win.name, "horizon": h, "cells": len(C.cells), "origin_hours": len(t),
                             "first_origin": C.hours[t[0]], "last_origin": C.hours[t[-1]], "days": round(len(t) / 24, 2),
                             "rows": len(t) * len(C.cells), "lag_168h_missing_share": lag168_missing,
                             "weekly_naive_missing_share": weekly_missing,
                             "origins_without_lag168": f"{C.hours[t[0]]} .. {C.hours[t[t - 168 < 0][-1]] if (t - 168 < 0).any() else '-'}"})
        del C
        gc.collect()
    r = pd.DataFrame(rows)
    _w(r, "training_spans.csv")
    pd.set_option("display.width", 250)
    print(r.to_string(index=False))


def step_fit2b():
    import lightgbm as lgb
    PRED = OUT / "preds"
    PRED.mkdir(parents=True, exist_ok=True)
    h, n = 1, FROZEN_TREES[1]
    log = []
    booster_file = OUT / "milanSF_w2b_1h.txt"
    if not (PRED / "milanSF_w2b_1h__on_trentino_test.npy").exists():
        if not booster_file.exists():
            C = City("milan")
            s = C.scale(W2B)
            X, y, meta = C.rows(W2B, "train", h, s, full_meta=False)
            ytr = sf_target(y, s[meta["cell_pos"]]); rows_n = len(y); del y, meta; gc.collect()
            require_headroom("milan w2b +1h fit")
            with Step(f"fit milanSF w2b +1h ({rows_n:,} rows, {n} trees)") as st:
                model = lgb.LGBMRegressor(**lgbm_params(LEAVES, n)).fit(X, ytr)
            model.booster_.save_model(str(booster_file))
            log.append({"model": "milanSF w2b +1h", "rows": rows_n, "seconds": st.seconds, "peak_commit_gb": st.peak_commit})
            del X, model, C
            gc.collect()
        tn = City("trentino")
        booster = lgb.Booster(model_file=str(booster_file))
        se = tn.scale(W2B)
        with Step("predict milanSF w2b +1h on trentino"):
            Xe, _, me = tn.rows(W2B, "test", h, se)
            np.save(PRED / "milanSF_w2b_1h__on_trentino_test.npy", from_sf(booster.predict(Xe), se[me["cell_pos"]]).astype(np.float32))
            del Xe
    else:
        tn = City("trentino")
    if not (PRED / "trentinoSF_w2b_1h__on_trentino_test.npy").exists():
        s = tn.scale(W2B)
        X, y, meta = tn.rows(W2B, "train", h, s, full_meta=False)
        ytr = sf_target(y, s[meta["cell_pos"]]); rows_n = len(y); del y, meta; gc.collect()
        require_headroom("trentino w2b +1h fit")
        with Step(f"fit trentinoSF w2b +1h ({rows_n:,} rows, {n} trees)") as st:
            model = lgb.LGBMRegressor(**lgbm_params(LEAVES, n)).fit(X, ytr)
        log.append({"model": "trentinoSF w2b +1h", "rows": rows_n, "seconds": st.seconds, "peak_commit_gb": st.peak_commit})
        del X
        Xe, _, me = tn.rows(W2B, "test", h, s)
        np.save(PRED / "trentinoSF_w2b_1h__on_trentino_test.npy", from_sf(model.predict(Xe), s[me["cell_pos"]]).astype(np.float32))
    if log:
        old = json.load(open(OUT / "fit2b_log.json")) if (OUT / "fit2b_log.json").exists() else []
        json.dump(old + log, open(OUT / "fit2b_log.json", "w"), indent=2, default=float)
    print(log)


def _test_context(tn, h=1):
    """Test rows of window 2 (identical for w2 and w2b: same test origins and thresholds)."""
    s = tn.scale(W2)
    _, y, meta = tn.rows(W2, "test", h, s)
    e = int(tn.hours.get_loc(pd.Timestamp(THR_END_W2)))
    thr = np.quantile(tn.A[:, :e], 0.9, axis=1)
    cp = meta["cell_pos"]
    hot = np.isin(cp, np.argsort(-s, kind="stable")[:125])
    low = (thr <= 1.0)[cp]
    _, day_idx = np.unique(pd.DatetimeIndex(meta["target_time"]).date, return_inverse=True)
    return y, thr[cp], hot, low, day_idx, meta


def step_cmp2b():
    rng = np.random.default_rng(BOOT_SEED)
    tn = City("trentino")
    y, thr_rows, hot, low, day_idx, meta = _test_context(tn)
    P = OUT / "preds"
    f = {"Milan-trained, train Nov 2-17": np.load(pred_path("milan", "w2", 1, "trentino", "test")).astype(float),
         "Trentino-trained, train Nov 2-17": np.load(pred_path("trentino", "w2", 1, "trentino", "test")).astype(float),
         "Milan-trained, train Nov 2-24": np.load(P / "milanSF_w2b_1h__on_trentino_test.npy").astype(float),
         "Trentino-trained, train Nov 2-24": np.load(P / "trentinoSF_w2b_1h__on_trentino_test.npy").astype(float),
         "seasonal-naive": meta["seasonal_naive"]}
    slices = {"all": np.ones(len(y), bool), "excl. thr<=1": ~low, "hotspot": hot}
    rows, boots = [], []
    for name, fc in f.items():
        for sl, msk in slices.items():
            c = flag_counts((fc > thr_rows)[msk], (y > thr_rows)[msk])
            rows.append({"forecaster": name, "slice": sl, "mae": float(np.mean(np.abs(y - fc)[msk])),
                         **{k: c[k] for k in ("precision", "recall", "f1", "flagged_share", "actual_share")}})
    pairs = [("Milan-trained, train Nov 2-24", "Milan-trained, train Nov 2-17"),
             ("Trentino-trained, train Nov 2-24", "Trentino-trained, train Nov 2-17"),
             ("Milan-trained, train Nov 2-24", "Trentino-trained, train Nov 2-24"),
             ("Milan-trained, train Nov 2-17", "Trentino-trained, train Nov 2-17"),
             ("Trentino-trained, train Nov 2-24", "seasonal-naive"), ("Milan-trained, train Nov 2-24", "seasonal-naive")]
    for a, b in pairs:
        for sl, msk in slices.items():
            r = _compare(f[a], f[b], y, thr_rows, msk, day_idx, rng)
            boots.append({"a": a, "b": b, "slice": sl, **r})
    _w(pd.DataFrame(rows), "w2b_metrics.csv")
    _w(pd.DataFrame(boots), "w2b_bootstrap.csv")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(boots)[["a", "b", "slice", "mae_diff", "mae_lo", "mae_hi", "mae_verdict", "f1_diff", "f1_lo", "f1_hi", "f1_verdict"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def step_margins():
    rng = np.random.default_rng(BOOT_SEED)
    tn = City("trentino")
    rows = []
    for h in (1, 4):
        y, thr_rows, hot, low, day_idx, meta = _test_context(tn, h)
        act = y > thr_rows
        days = np.unique(day_idx)
        for label, mc in (("Milan-trained SF (transfer)", "milan"), ("Trentino-trained SF (in-domain)", "trentino")):
            fc = np.load(pred_path(mc, "w2", h, "trentino", "test")).astype(float)
            for m in MILAN_MARGINS[h]:
                for sl, msk in (("all", np.ones(len(y), bool)), ("hotspot", hot)):
                    d = day_idx[msk]; a = act[msk]
                    def parts(mm):
                        fl = (fc > mm * thr_rows)[msk]
                        tp = np.bincount(d, a & fl, len(days)); fp = np.bincount(d, ~a & fl, len(days)); fn = np.bincount(d, a & ~fl, len(days))
                        return 2 * tp, 2 * tp + fp + fn, tp, tp + fp
                    n1, d1, tp1, f1_ = parts(m)
                    n0, d0, tp0, f0_ = parts(1.0)
                    lo, hi = _boot((n1, d1, n0, d0), rng)
                    c0 = flag_counts((fc > thr_rows)[msk], a); c1 = flag_counts((fc > m * thr_rows)[msk], a)
                    rows.append({"horizon": h, "model": label, "margin": m, "slice": sl,
                                 "f1_m1": c0["f1"], "f1_margin": c1["f1"], "f1_gain": c1["f1"] - c0["f1"], "lo": lo, "hi": hi,
                                 "verdict": "gain" if lo > 0 else "loss" if hi < 0 else "n.s.",
                                 "precision_m1": c0["precision"], "precision_margin": c1["precision"],
                                 "recall_m1": c0["recall"], "recall_margin": c1["recall"],
                                 "flagged_m1": c0["flagged_share"], "flagged_margin": c1["flagged_share"], "actual": c0["actual_share"]})
    r = pd.DataFrame(rows)
    _w(r, "w2_margin_gain.csv")
    pd.set_option("display.width", 260)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    steps = {"rates": step_rates, "train": step_train, "fit2b": step_fit2b, "cmp2b": step_cmp2b, "margins": step_margins}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
