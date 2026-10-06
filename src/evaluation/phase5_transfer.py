"""
src/evaluation/phase5_transfer.py  (Phase 5: Milan -> Trentino transfer, scale-free)

MODEL TRANSFER WITH LOCAL HISTORY (not zero-shot): each cell's scale s_c and its congestion
thresholds come from that city's own training-period history; only the model is transferred.

Scale-free formulation
  s_c     = mean activity of the cell over the model's training target hours (floored at 1)
  target  = log1p(A[t+h]) - log1p(s_c)
  levels  = log1p(x) - log1p(s_c)    for lags, seasonal / weekly naive, rolling means,
                                        neighbour mean / max, 2-step ring mean
  spread  = log1p(std) - log1p(s_c)  for rolling standard deviations
  momenta = difference / s_c
  calendar features unchanged; CellID and cell_historical_mean EXCLUDED (21 features)
Model size FROZEN from the final configuration (63 leaves; 2,153 / 2,181 trees at +1h / +4h),
chosen on Milan Dec 10-16 for the UNSCALED target; never tuned for this target.

Windows (same calendar in both cities)
  w1: train targets < 2013-12-10; test origins Dec 17 00:00 .. Jan 1 19:00
  w2: train targets < 2013-11-18; test origins Nov 25 00:00 .. Dec 7 19:00
      (the Milan w2 model's frozen size comes from Dec 10-16, AFTER this test window)
Thresholds: per-cell 90th percentile of activity before the test window (w1: before Dec 10,
w2: before Nov 25). Hotspots: top 2% of cells by s_c.

Steps:
  fit   all models listed in FITS (resumable; headroom check before each)
  eval  every table (Trentino transfer, Milan reformulation control, sensitivities)

RUN: python src/evaluation/phase5_transfer.py {fit|eval}
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import MAX_HORIZON, MAX_LAG, ROOT, Step, get_neighbors, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    ALL_HOLIDAYS, BOOT_CI, BOOT_N, BOOT_SEED, INDICATIVE_DAYS, TREE_CAP, W1, W2, choose_trees, lgbm_params,
    load_preds, origin_index, require_headroom,
)
from src.evaluation.margin_tuning import f1_best, flag_counts, pr_curve, ratio  # noqa: E402

P5 = ROOT / "data" / "experiments" / "phase5"
OUT = P5 / "transfer"
PRED = OUT / "preds"
FROZEN_TREES = {1: 2153, 4: 2181}
LEAVES = 63
WINDOWS = {"w1": W1, "w2": W2}
THR_END = {"w1": "2013-12-10", "w2": "2013-11-25"}
MILAN_MARGINS = {1: (0.95, 0.955, 0.96), 4: (0.94, 0.945)}
LOW_THR = 1.0
# (train city, window, horizon, curve?)  curve = fit to 3,000 trees with a validation curve on the
# training city's own Dec 10-16 (sizing sensitivity D; only Trentino w1 +1h)
FITS = [("milan", "w1", 1, False), ("milan", "w1", 4, False), ("milan", "w2", 1, False), ("milan", "w2", 4, False),
        ("trentino", "w1", 1, True), ("trentino", "w1", 4, False), ("trentino", "w2", 1, False), ("trentino", "w2", 4, False)]


# ============================================================================= panels
class City:
    def __init__(self, name: str):
        self.name = name
        if name == "milan":
            p = load_panel()
            self.cells, self.hours, self.A = p.cells, p.hours, p.activity
            del p.neighbor_avg                                    # not used in Phase 5 (memory)
            nb1 = [np.array([n - 1 for n in get_neighbors(int(c))]) for c in self.cells]
        else:
            d = pd.read_parquet(P5 / "cdr_activity_aggregated_tn.parquet").sort_values(["CellID", "datetime"])
            self.cells = np.sort(d["CellID"].unique())
            self.hours = pd.DatetimeIndex(np.sort(d["datetime"].unique()))
            self.A = d[["smsin", "smsout", "callin", "callout", "internet"]].sum(axis=1).to_numpy().reshape(len(self.cells), len(self.hours))
            del d
            pos = {int(c): i for i, c in enumerate(self.cells)}
            e = pd.read_csv(P5 / "trentino_adjacency.csv")
            nbs = {i: [] for i in range(len(self.cells))}
            for a, b in zip(e.a, e.b):
                if a in pos and b in pos:                        # reporting neighbours only
                    nbs[pos[a]].append(pos[b]); nbs[pos[b]].append(pos[a])
            nb1 = [np.array(sorted(nbs[i])) for i in range(len(self.cells))]
        n = len(self.cells)
        W = sp.lil_matrix((n, n))
        for i, nb in enumerate(nb1):
            if len(nb):
                W[i, nb] = 1.0
        W = W.tocsr()
        W2 = ((W + W @ W) > 0).astype(float).tolil()             # within 2 steps (king moves)
        W2.setdiag(0)
        W2 = W2.tocsr()
        W2.eliminate_zeros()
        A = self.A
        deg1, deg2 = np.asarray(W.sum(1)).ravel(), np.asarray(W2.sum(1)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            self.nb_avg = np.asarray(W @ A) / deg1[:, None]
            self.ring2 = np.asarray(W2 @ A) / deg2[:, None]
        width = max(len(x) for x in nb1)
        idx = np.full((n, width), n)
        for i, nb in enumerate(nb1):
            idx[i, :len(nb)] = nb
        Aext = np.vstack([A, np.full((1, A.shape[1]), -np.inf)])
        nmax = np.full_like(A, -np.inf)
        for k in range(width):
            nmax = np.maximum(nmax, Aext[idx[:, k]])
        nmax[np.isinf(nmax)] = np.nan
        self.nmax = nmax.astype(np.float32)
        self.nb_avg = self.nb_avg.astype(np.float32)
        self.ring2 = self.ring2.astype(np.float32)
        del Aext, nmax
        H = A.shape[1]
        c1 = np.concatenate([np.zeros((n, 1)), np.cumsum(A, 1)], 1)
        c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(A * A, 1)], 1)
        self.roll = {}
        for w in (3, 6, 24):
            mean = np.full_like(A, np.nan); std = np.full_like(A, np.nan)
            s1, s2 = c1[:, w:H] - c1[:, :H - w], c2[:, w:H] - c2[:, :H - w]
            m = s1 / w
            mean[:, w:] = m
            std[:, w:] = np.sqrt(np.clip(s2 / w - m * m, 0, None))
            self.roll[w] = (mean.astype(np.float32), std.astype(np.float32))
            del mean, std
        del c1, c2
        gc.collect()

    def scale(self, win) -> np.ndarray:
        end = int(self.hours.get_loc(pd.Timestamp(win.train_target_end)))
        return np.maximum(self.A[:, MAX_LAG:end].mean(axis=1), 1.0)

    def thresholds(self, window: str) -> np.ndarray:
        e = int(self.hours.get_loc(pd.Timestamp(THR_END[window])))
        return np.quantile(self.A[:, :e], 0.9, axis=1)

    def rows(self, win, split: str, h: int, s: np.ndarray, full_meta: bool = True):
        t = origin_index(self, win, split, h)
        A, n = self.A, len(self.cells)
        ls = np.log1p(s)[:, None]

        def lvl(arr, off=0):
            v = arr[:, t + off]
            return (np.log1p(np.clip(v, 0, None)) - ls).ravel()

        def back(off):
            ok = (t + off) >= 0
            v = np.log1p(A[:, np.where(ok, t + off, 0)]) - ls
            v[:, ~ok] = np.nan
            return v.ravel()

        ot = self.hours[t]
        dow = np.tile(ot.dayofweek.to_numpy(), n)
        cols = {
            "lag_1h": lvl(A, -1), "lag_2h": lvl(A, -2), "lag_3h": lvl(A, -3), "lag_24h": lvl(A, -24),
            "hour_of_day": np.tile(ot.hour.to_numpy(), n), "day_of_week": dow, "is_weekend": (dow >= 5).astype(np.int8),
            "neighbor_avg_lag1h": lvl(self.nb_avg, -1), f"naive_feature_{h}h": lvl(A, h - 24),
            "lag_168h": back(-168), "weekly_naive": back(h - 168),
            "momentum_1h": ((A[:, t - 1] - A[:, t - 2]) / s[:, None]).ravel(),
            "momentum_3h": ((A[:, t - 1] - A[:, t - 4]) / s[:, None]).ravel(),
            "neighbor_max_lag1h": lvl(self.nmax, -1), "ring2_mean_lag1h": lvl(self.ring2, -1),
        }
        for w in (3, 6, 24):
            mean, std = self.roll[w]
            cols[f"roll_mean_{w}h"] = lvl(mean)
            cols[f"roll_std_{w}h"] = lvl(std)
        X = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in cols.items()})
        del cols
        y = A[:, t + h].ravel()
        if not full_meta:
            return X, y, {"cell_pos": np.repeat(np.arange(n, dtype=np.int32), len(t)), "n_t": len(t)}
        meta = {"cell_pos": np.repeat(np.arange(n), len(t)), "target_time": np.tile((ot + pd.Timedelta(hours=h)).to_numpy(), n),
                "seasonal_naive": A[:, t + h - 24].ravel(), "weekly_naive": A[:, t + h - 168].ravel(), "n_t": len(t)}
        return X, y, meta


def sf_target(y, s_rows):
    return np.log1p(y) - np.log1p(s_rows)


def from_sf(p, s_rows):
    return np.clip(np.expm1(p + np.log1p(s_rows)), 0, None)


def pred_path(model_city, window, h, eval_city, split, tag=""):
    return PRED / f"{model_city}SF_{window}_{h}h__on_{eval_city}_{split}{tag}.npy"


# ============================================================================= fit
def step_fit():
    import lightgbm as lgb
    PRED.mkdir(parents=True, exist_ok=True)
    cities = {}
    log = json.load(open(OUT / "fit_log.json")) if (OUT / "fit_log.json").exists() else []
    done = {(r["city"], r["window"], r["horizon"]) for r in log}
    MODELS = OUT / "models"
    MODELS.mkdir(parents=True, exist_ok=True)

    def need(city):
        """Hold only one city's feature store at a time (memory)."""
        if city not in cities:
            for other in list(cities):
                del cities[other]
            gc.collect()
            with Step(f"build {city} panel + features"):
                cities[city] = City(city)
        return cities[city]

    # Milan models first (fit + predict on Milan), then predict them on Trentino from saved boosters,
    # then the Trentino models.
    for city, window, h, curve in FITS:
        if city == "trentino" and any(c == "milan" and not pred_path("milan", w, hh, "trentino", "test").exists()
                                      for c, w, hh, _ in FITS):
            tn = need("trentino")
            for c, w, hh, _ in FITS:
                if c == "milan" and not pred_path("milan", w, hh, "trentino", "test").exists():
                    booster = lgb.Booster(model_file=str(MODELS / f"milanSF_{w}_{hh}h.txt"))
                    se = tn.scale(WINDOWS[w])
                    with Step(f"predict milanSF {w} +{hh}h on trentino"):
                        for split in ("val", "test"):
                            Xe, _, me = tn.rows(WINDOWS[w], split, hh, se)
                            np.save(pred_path("milan", w, hh, "trentino", split),
                                    from_sf(booster.predict(Xe), se[me["cell_pos"]]).astype(np.float32))
                            del Xe
        if (city, window, h) in done:
            continue
        C, win = need(city), WINDOWS[window]
        s = C.scale(win)
        X, y, meta = C.rows(win, "train", h, s, full_meta=False)
        ytr = sf_target(y, s[meta["cell_pos"]])
        n_rows = len(y)
        del y, meta
        gc.collect()
        n_trees = TREE_CAP if curve else FROZEN_TREES[h]
        params = lgbm_params(LEAVES, n_trees)
        require_headroom(f"{city} {window} +{h}h fit")
        record = {}
        with Step(f"fit {city}SF {window} +{h}h ({n_rows:,} rows, {n_trees} trees)") as st:
            model = lgb.LGBMRegressor(**params)
            if curve:
                Xv, yv, mv = C.rows(win, "val", h, s)
                model.fit(X, ytr, eval_X=(Xv,), eval_y=(sf_target(yv, s[mv["cell_pos"]]),), eval_metric="l2",
                          callbacks=[lgb.record_evaluation(record)])
            else:
                model.fit(X, ytr)
        entry = {"city": city, "window": window, "horizon": h, "rows": n_rows, "trees_fitted": n_trees,
                 "fit_seconds": st.seconds, "peak_ram_gb": st.peak_ws, "peak_commit_gb": st.peak_commit}
        sizes = {"": FROZEN_TREES[h]}
        if curve:
            c_arr = np.asarray(record["valid_0"]["l2"])
            np.save(OUT / f"curve_{city}_{window}_{h}h.npy", c_arr)
            ch = choose_trees(c_arr)
            entry["curve"] = ch
            sizes["_trentinosized"] = ch["chosen_trees"]
        del X
        gc.collect()
        model.booster_.save_model(str(MODELS / f"{city}SF_{window}_{h}h.txt"), num_iteration=FROZEN_TREES[h] if not curve else None)
        # predictions on the training city (its own local scale), val and test splits;
        # Milan models are predicted on Trentino later from the saved booster
        se = s
        for split in ("val", "test"):
            Xe, _, me = C.rows(win, split, h, se)
            for tag, n in sizes.items():
                raw = model.predict(Xe, num_iteration=n)
                np.save(pred_path(city, window, h, city, split, tag), from_sf(raw, se[me["cell_pos"]]).astype(np.float32))
            del Xe
        log.append(entry)
        json.dump(log, open(OUT / "fit_log.json", "w"), indent=2, default=float)
        print(json.dumps(entry, default=float))
        del model
        gc.collect()


# ============================================================================= eval
def _boot(diff_parts, rng):
    """Paired day bootstrap of a difference of per-day ratios; diff_parts = (num_a, den_a, num_b, den_b) per day."""
    na, da, nb, db = diff_parts
    idx = rng.integers(0, len(na), size=(BOOT_N, len(na)))
    d = na[idx].sum(1) / np.maximum(da[idx].sum(1), 1e-12) - nb[idx].sum(1) / np.maximum(db[idx].sum(1), 1e-12)
    lo, hi = np.percentile(d, BOOT_CI)
    return lo, hi


def _compare(fa, fb, y, thr_rows, mask, day_idx, rng, margin_a=1.0, margin_b=1.0):
    """a minus b: MAE (lower better) and F1, paired day bootstrap on rows in mask."""
    d = day_idx[mask]
    days = np.unique(d)
    nd = d.max() + 1
    act = (y > thr_rows)[mask]
    out = {}
    ae_a = np.bincount(d, np.abs(y - fa)[mask], nd)[days]
    ae_b = np.bincount(d, np.abs(y - fb)[mask], nd)[days]
    cnt = np.bincount(d, minlength=nd)[days].astype(float)
    lo, hi = _boot((ae_a, cnt, ae_b, cnt), rng)
    out["mae_diff"], out["mae_lo"], out["mae_hi"] = ae_a.sum() / cnt.sum() - ae_b.sum() / cnt.sum(), lo, hi
    def parts(f, m):
        fl = (f > m * thr_rows)[mask]
        tp = np.bincount(d, act & fl, nd)[days]; fp = np.bincount(d, ~act & fl, nd)[days]; fn = np.bincount(d, act & ~fl, nd)[days]
        return 2 * tp, 2 * tp + fp + fn
    na, da = parts(fa, margin_a); nb_, db = parts(fb, margin_b)
    lo, hi = _boot((na, da, nb_, db), rng)
    out["f1_diff"], out["f1_lo"], out["f1_hi"] = na.sum() / da.sum() - nb_.sum() / db.sum(), lo, hi
    out["mae_verdict"] = "a better" if out["mae_hi"] < 0 else "a worse" if out["mae_lo"] > 0 else "n.s."
    out["f1_verdict"] = "a better" if out["f1_lo"] > 0 else "a worse" if out["f1_hi"] < 0 else "n.s."
    return out


def step_eval():
    rng = np.random.default_rng(BOOT_SEED)
    tn, mi = City("trentino"), City("milan")
    metrics, boots, tuned, control = [], [], [], []
    for window in ("w1", "w2"):
        win = WINDOWS[window]
        for h in (1, 4):
            if not pred_path("milan", window, h, "trentino", "test").exists():
                continue
            for E, ename in ((tn, "trentino"), (mi, "milan")):
                s = E.scale(win)
                thr = E.thresholds(window)
                _, y, meta = E.rows(win, "test", h, s)
                cp = meta["cell_pos"]
                thr_rows = thr[cp]
                hot_cells = np.argsort(-s, kind="stable")[: int(round(len(s) * 0.02))]
                is_hot = np.isin(cp, hot_cells)
                low = (thr <= LOW_THR)[cp]
                tt = pd.DatetimeIndex(meta["target_time"])
                dates = tt.date
                _, day_idx = np.unique(dates, return_inverse=True)
                holiday = np.isin(dates, list(ALL_HOLIDAYS))
                weekend = tt.dayofweek.to_numpy() >= 5
                fc = {"seasonal-naive": meta["seasonal_naive"], "weekly-naive": meta["weekly_naive"],
                      "Milan-trained SF (transfer)" if ename == "trentino" else "Milan SF": np.load(pred_path("milan", window, h, ename, "test")).astype(float)}
                if ename == "trentino":
                    fc["Trentino-trained SF (in-domain, frozen size)"] = np.load(pred_path("trentino", window, h, "trentino", "test")).astype(float)
                    p_sized = pred_path("trentino", window, h, "trentino", "test", "_trentinosized")
                    if p_sized.exists():
                        fc["Trentino-trained SF (sized on Trentino Dec 10-16)"] = np.load(p_sized).astype(float)
                else:
                    fp = "final_63l"
                    try:
                        fc["Milan final_63l (unscaled)"] = load_preds(window, fp, h, "test").astype(float)
                    except FileNotFoundError:
                        pass
                slices = {"all": np.ones(len(y), bool), "excl. thr<=1": ~low, "hotspot": is_hot, "typical": ~is_hot,
                          "typical excl. thr<=1": ~is_hot & ~low, "non-holiday": ~holiday, "holiday": holiday,
                          "weekday": ~weekend, "weekend": weekend}
                margins = [1.0] + list(MILAN_MARGINS[h])
                for name, f in fc.items():
                    for sl, msk in slices.items():
                        if not msk.any():
                            continue
                        nd = len(np.unique(dates[msk]))
                        base = {"eval_city": ename, "window": window, "horizon": h, "forecaster": name, "slice": sl,
                                "days": nd, "indicative": nd < INDICATIVE_DAYS or sl == "holiday",
                                "cell_hours": int(msk.sum()), "mae": float(np.mean(np.abs(y - f)[msk])),
                                "log_mae": float(np.mean(np.abs(np.log1p(y) - np.log1p(f))[msk]))}
                        for m in margins:
                            fcnt = flag_counts((f > m * thr_rows)[msk], (y > thr_rows)[msk])
                            metrics.append({**base, "margin": m, **{k: fcnt[k] for k in ("precision", "recall", "f1", "flagged_share", "actual_share")}})
                # paired comparisons
                if ename == "trentino":
                    pairs = [("Milan-trained SF (transfer)", "Trentino-trained SF (in-domain, frozen size)"),
                             ("Milan-trained SF (transfer)", "seasonal-naive"),
                             ("Trentino-trained SF (in-domain, frozen size)", "seasonal-naive")]
                    if "Trentino-trained SF (sized on Trentino Dec 10-16)" in fc:
                        pairs += [("Milan-trained SF (transfer)", "Trentino-trained SF (sized on Trentino Dec 10-16)")]
                else:
                    pairs = [("Milan SF", "Milan final_63l (unscaled)"), ("Milan SF", "seasonal-naive")] if "Milan final_63l (unscaled)" in fc else [("Milan SF", "seasonal-naive")]
                for a, b in pairs:
                    for sl in ("all", "excl. thr<=1", "hotspot", "typical", "non-holiday"):
                        for m in (1.0, MILAN_MARGINS[h][1]):
                            r = _compare(fc[a], fc[b], y, thr_rows, slices[sl], day_idx, rng, m if "naive" not in a else 1.0, m if "naive" not in b else 1.0)
                            boots.append({"eval_city": ename, "window": window, "horizon": h, "a": a, "b": b, "slice": sl, "margin": m, **r})
                # Trentino-tuned margins (separate sensitivity; window 1 only, Dec 10-16, thresholds before Dec 10)
                if ename == "trentino" and window == "w1":
                    _, yv, mv = E.rows(win, "val", h, s)
                    thr_v = thr[mv["cell_pos"]]
                    for model_city, label in (("milan", "Milan-trained SF (transfer)"), ("trentino", "Trentino-trained SF (in-domain, frozen size)")):
                        fv = np.load(pred_path(model_city, window, h, "trentino", "val")).astype(float)
                        m_t = f1_best(pr_curve(ratio(fv, thr_v), yv > thr_v))
                        for sl in ("all", "excl. thr<=1", "hotspot", "typical"):
                            msk = slices[sl]
                            c1 = flag_counts((fc[label] > m_t * thr_rows)[msk], (y > thr_rows)[msk])
                            tuned.append({"window": window, "horizon": h, "forecaster": label, "margin_tuned_on_trentino": m_t,
                                          "slice": sl, **{k: c1[k] for k in ("precision", "recall", "f1", "flagged_share", "actual_share")}})
                del y, meta
                gc.collect()
    for name, data in (("metrics.csv", metrics), ("bootstrap.csv", boots), ("trentino_tuned_margins.csv", tuned)):
        pd.DataFrame(data).to_csv(OUT / name, index=False)
    print("saved", [len(metrics), len(boots), len(tuned)])


if __name__ == "__main__":
    steps = {"fit": step_fit, "eval": step_eval}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
