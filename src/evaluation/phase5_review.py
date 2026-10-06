"""
src/evaluation/phase5_review.py  (review items after the Phase 5 follow-ups; no fitting)

  bounds     Transfer vs in-domain on Trentino w2: 90% intervals as bounds (MAE relative to the
             in-domain MAE; F1), +1h (Nov 2-17 and Nov 2-24 pairs) and +4h (Nov 2-17 pair).
  marginspre Trentino w2 margin gain using only margins chosen before Nov 25 (Milan's window-2
             margins from Nov 18-24: 0.950 at +1h, 0.935 at +4h), next to Milan's w1 margins.
  thrsens    Threshold sensitivity of every F1 / precision / replay significance call in
             "What can be claimed": Phase 2 thresholds (data to Dec 16) vs Phase 3 (to Dec 9) for
             window 1; window 2 (Nov 1-24 vs Nov 1-17); Trentino w2 (before Nov 25 vs before Nov 18).
  dec31      +1h forecast error (MAE, flag F1) per test day, Dec 31 against the other days.

Saved predictions only. Outputs: data/experiments/phase5/review/.
RUN: python src/evaluation/phase5_review.py {bounds|marginspre|thrsens|dec31}
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import HORIZONS, ROOT, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    ALL_HOLIDAYS, BOOT_CI, BOOT_N, BOOT_SEED, W1, W2, FeatureStore, build_rows, eval_thresholds, hotspots,
    load_preds, origin_index, training_cell_means,
)
from src.evaluation.margin_tuning import (  # noqa: E402
    DOWNSTREAM_NONHOLIDAY, HOTSPOT_VARIANT, f1_best, flag_bootstrap, flag_counts, pr_curve, ratio, thresholds_before,
)
from src.evaluation.phase4_sensitivity import neighbour_fraction  # noqa: E402

OUT = ROOT / "data" / "experiments" / "phase5" / "review"
P5 = ROOT / "data" / "experiments" / "phase5"
MILAN_W2_MARGINS = {1: 0.950, 4: 0.935}     # chosen on Nov 18-24 (Phase 3 check 5)
MILAN_W1_MARGINS = {1: 0.955, 4: 0.945}     # middle of the recommended ranges (include Dec 10-16)


def _w(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def _ci(draws):
    lo, hi = np.percentile(draws, BOOT_CI)
    return float(lo), float(hi), ("better" if lo > 0 else "worse" if hi < 0 else "n.s.")


# ============================================================================= bounds
def step_bounds():
    b1 = pd.read_csv(P5 / "transfer" / "bootstrap.csv")
    b1 = b1[(b1.eval_city == "trentino") & (b1.window == "w2") & (b1.a == "Milan-trained SF (transfer)")
            & (b1.b == "Trentino-trained SF (in-domain, frozen size)")]
    m1 = pd.read_csv(P5 / "transfer" / "metrics.csv")
    ref_mae = {h: float(m1[(m1.eval_city == "trentino") & (m1.window == "w2") & (m1.horizon == h) & (m1.margin == 1.0)
                           & (m1.slice == "all") & (m1.forecaster == "Trentino-trained SF (in-domain, frozen size)")]["mae"].iloc[0])
               for h in (1, 4)}
    rows = []
    for _, r in b1[b1.slice.isin(["all", "excl. thr<=1", "hotspot", "typical"])].iterrows():
        rows.append({"pair": "train Nov 2-17", "horizon": r.horizon, "margin": r.margin, "slice": r.slice,
                     "mae_diff": r.mae_diff, "mae_lo": r.mae_lo, "mae_hi": r.mae_hi,
                     "mae_hi_pct_of_in_domain_all": 100 * r.mae_hi / ref_mae[r.horizon] if r.slice == "all" else np.nan,
                     "f1_diff": r.f1_diff, "f1_lo": r.f1_lo, "f1_hi": r.f1_hi})
    b2 = pd.read_csv(P5 / "followup" / "w2b_bootstrap.csv")
    b2 = b2[(b2.a == "Milan-trained, train Nov 2-24") & (b2.b == "Trentino-trained, train Nov 2-24")]
    m2 = pd.read_csv(P5 / "followup" / "w2b_metrics.csv")
    ref2 = float(m2[(m2.forecaster == "Trentino-trained, train Nov 2-24") & (m2.slice == "all")]["mae"].iloc[0])
    for _, r in b2.iterrows():
        rows.append({"pair": "train Nov 2-24", "horizon": 1, "margin": 1.0, "slice": r.slice,
                     "mae_diff": r.mae_diff, "mae_lo": r.mae_lo, "mae_hi": r.mae_hi,
                     "mae_hi_pct_of_in_domain_all": 100 * r.mae_hi / ref2 if r.slice == "all" else np.nan,
                     "f1_diff": r.f1_diff, "f1_lo": r.f1_lo, "f1_hi": r.f1_hi})
    res = pd.DataFrame(rows)
    _w(res, "transfer_bounds.csv")
    pd.set_option("display.width", 250)
    print("in-domain MAE (all):", ref_mae, "Nov 2-24 +1h:", round(ref2, 3))
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ============================================================================= marginspre
def step_marginspre():
    from src.evaluation.phase5_followup import _test_context
    from src.evaluation.phase5_transfer import City, pred_path
    rng = np.random.default_rng(BOOT_SEED)
    tn = City("trentino")
    rows = []
    for h in (1, 4):
        y, thr_rows, hot, _, day_idx, _ = _test_context(tn, h)
        act = y > thr_rows
        fc = {"transfer": np.load(pred_path("milan", "w2", h, "trentino", "test")).astype(float),
              "in-domain": np.load(pred_path("trentino", "w2", h, "trentino", "test")).astype(float)}
        for source, m in (("before Nov 25 (Milan w2, Nov 18-24)", MILAN_W2_MARGINS[h]),
                          ("Milan w1 range middle (uses Dec 10-16)", MILAN_W1_MARGINS[h])):
            fl = {f"{k} m=1": f > thr_rows for k, f in fc.items()}
            fl.update({f"{k} m={m}": f > m * thr_rows for k, f in fc.items()})
            for sl, msk in (("all", np.ones(len(y), bool)), ("hotspot", hot)):
                for k in fc:
                    d, lo, hi, v = flag_bootstrap({"a": fl[f"{k} m={m}"], "b": fl[f"{k} m=1"]}, "b", act, day_idx, msk, rng)["a"]
                    c0, c1 = flag_counts(fl[f"{k} m=1"][msk], act[msk]), flag_counts(fl[f"{k} m={m}"][msk], act[msk])
                    rows.append({"horizon": h, "margin_source": source, "margin": m, "model": k, "slice": sl,
                                 "f1_m1": c0["f1"], "f1_m": c1["f1"], "gain": d, "lo": lo, "hi": hi, "verdict": v,
                                 "precision_m1": c0["precision"], "precision_m": c1["precision"],
                                 "recall_m1": c0["recall"], "recall_m": c1["recall"],
                                 "flagged_m1": c0["flagged_share"], "flagged_m": c1["flagged_share"], "actual": c0["actual_share"]})
                d, lo, hi, v = flag_bootstrap({"a": fl[f"transfer m={m}"], "b": fl[f"in-domain m={m}"]}, "b", act, day_idx, msk, rng)["a"]
                rows.append({"horizon": h, "margin_source": source, "margin": m, "model": "transfer - in-domain (both at m)",
                             "slice": sl, "gain": d, "lo": lo, "hi": hi, "verdict": v})
    res = pd.DataFrame(rows)
    _w(res, "w2_margin_gain_pre_nov25.csv")
    pd.set_option("display.width", 280)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ============================================================================= thrsens
def _calls(rows, ctx, defn, h, sl, name, res):
    d, lo, hi, v = res
    rows.append({"context": ctx, "thresholds": defn, "horizon": h, "slice": sl, "comparison": name,
                 "diff": d, "lo": lo, "hi": hi, "verdict": v})


def _milan_w1(rows, rng):
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    hot = hotspots(cm)
    thr_defs = {"P2 (Nov 1 - Dec 16)": eval_thresholds(panel, W1), "P3 (Nov 1 - Dec 9)": thresholds_before(panel, "2013-12-10")}
    thr_p3 = thr_defs["P3 (Nov 1 - Dec 9)"]
    chosen = pd.read_csv(ROOT / "data" / "experiments" / "phase3" / "margins_chosen.csv")
    pick = lambda h, fc: float(chosen[(chosen.horizon == h) & (chosen.forecaster == fc) & (chosen.segment == "all")  # noqa: E731
                                      & (chosen.kind == "f1")]["margin"].iloc[0])
    pos = None
    for h in HORIZONS:
        sets = {}
        for split in ("val", "test"):
            t = origin_index(panel, W1, split, h)
            _, y, meta = build_rows(store, h, t, np.arange(len(panel.cells)), cm)
            n_t = len(t)
            if pos is None:
                pos = np.searchsorted(panel.cells, np.sort(hot))
            fc = {"naive": meta["seasonal_naive"].astype(float),
                  "tuned": load_preds("w1", "tuned", h, split).astype(float),
                  "final31": load_preds("w1", "final", h, split).astype(float),
                  "final63": load_preds("w1", "final_63l", h, split).astype(float)}
            comp = fc["final63"].copy()
            comp.reshape(len(panel.cells), n_t)[pos] = load_preds("w1", HOTSPOT_VARIANT[h], h, split).reshape(len(pos), n_t)
            fc["H"] = comp
            sets[split] = dict(y=y, fc=fc, n_t=n_t, hot=np.isin(meta["cell"], hot),
                               dates=pd.DatetimeIndex(meta["target_time"]).date)
        v, t = sets["val"], sets["test"]
        # each forecaster's own margin, chosen on Dec 10-16 with P3 thresholds (the only leak-free choice);
        # the same margins are applied under both threshold definitions
        thr_v = np.repeat(thr_p3, v["n_t"])
        own = {k: f1_best(pr_curve(ratio(v["fc"][k], thr_v), v["y"] > thr_v)) for k in ("naive", "tuned", "final63")}
        m_p1, m_h1 = pick(h, "final pooled"), pick(h, "pooled+hotspot")
        assert abs(own["final63"] - m_p1) < 1e-9, (own["final63"], m_p1)
        _, day_idx = np.unique(t["dates"], return_inverse=True)
        allm = np.ones(len(t["y"]), bool)
        for defn, thr in thr_defs.items():
            tr = np.repeat(thr, t["n_t"])
            act = t["y"] > tr
            f = t["fc"]
            fl = {"naive m=1": f["naive"] > tr, "naive own m": f["naive"] > own["naive"] * tr,
                  "tuned m=1": f["tuned"] > tr, "tuned own m": f["tuned"] > own["tuned"] * tr,
                  "final31 m=1": f["final31"] > tr,
                  "P0": f["final63"] > tr, "P1": f["final63"] > m_p1 * tr,
                  "H0": f["H"] > tr, "H1": f["H"] > m_h1 * tr}
            for sl, msk in (("all", allm), ("hotspot", t["hot"])):
                for a, b, name in (("tuned m=1", "naive m=1", "(i) production model vs naive, m=1"),
                                   ("tuned own m", "naive own m", "(i) production model vs naive, both tuned"),
                                   ("P0", "naive m=1", "(ii) final_63l vs naive, m=1"),
                                   ("P1", "naive own m", "(ii) final_63l vs naive, both tuned"),
                                   ("final31 m=1", "naive m=1", "Phase 2 final (31 leaves) vs naive, m=1"),
                                   ("final31 m=1", "tuned m=1", "feature groups: final (31 leaves) vs tuned, m=1")):
                    _calls(rows, "Milan w1", defn, h, sl, name, flag_bootstrap({"a": fl[a], "b": fl[b]}, "b", act, day_idx, msk, rng)["a"])
            _calls(rows, "Milan w1", defn, h, "hotspot", "hotspot model: H0 vs P0",
                   flag_bootstrap({"a": fl["H0"], "b": fl["P0"]}, "b", act, day_idx, t["hot"], rng)["a"])
            _calls(rows, "Milan w1", defn, h, "all", "margin helps: P1 vs P0",
                   flag_bootstrap({"a": fl["P1"], "b": fl["P0"]}, "b", act, day_idx, allm, rng)["a"])
            _calls(rows, "Milan w1", defn, h, "hotspot", "keep hotspots m=1: H1 vs H0",
                   flag_bootstrap({"a": fl["H1"], "b": fl["H0"]}, "b", act, day_idx, t["hot"], rng)["a"])
        print(f"  Milan w1 +{h}h done")
        del sets
        gc.collect()
    return panel, store, cm, thr_defs


def _milan_rule_and_replay(rows, rng, panel, store, cm, thr_defs):
    from src.diagnosis.diagnosis_agent import KNOWN_HOLIDAYS, NEIGHBOR_FRACTION_THRESHOLD
    from src.optimization.solver import solve_hour
    t_idx = origin_index(panel, W1, "test", 1)
    _, y, meta = build_rows(store, 1, t_idx, np.arange(len(panel.cells)), cm)
    n_t = len(t_idx)
    F = load_preds("w1", "final_63l", 1, "test").astype(np.float64).reshape(len(panel.cells), n_t)
    Y = y.reshape(len(panel.cells), n_t)
    targets = pd.DatetimeIndex(meta["target_time"][:n_t])
    holiday = np.isin(targets.date, list(ALL_HOLIDAYS))
    sets = {"24 downstream hours": np.isin(targets, pd.DatetimeIndex(DOWNSTREAM_NONHOLIDAY)), "311 non-holiday hours": ~holiday}
    cells = panel.cells
    for defn, thr in thr_defs.items():
        flags = F > thr[:, None]
        frac = neighbour_fraction(flags)
        actual = Y > thr[:, None]
        for sname, cols in sets.items():
            fl, fr, ac = flags[:, cols], frac[:, cols], actual[:, cols]
            r = fl & (fr >= NEIGHBOR_FRACTION_THRESHOLD)
            a = fl & ~(fr >= NEIGHBOR_FRACTION_THRESHOLD)
            rh, rn, ah, an = (r & ac).sum(0), r.sum(0), (a & ac).sum(0), a.sum(0)
            idx = rng.integers(0, cols.sum(), size=(BOOT_N, cols.sum()))
            diff = rh[idx].sum(1) / rn[idx].sum(1) - ah[idx].sum(1) / np.maximum(an[idx].sum(1), 1)
            lo, hi, v = _ci(diff)
            _calls(rows, "Milan w1 diagnosis rule (+1h, m=1, hour bootstrap)", defn, 1, sname,
                   f"precision ROUTINE - ANOMALOUS ({rh.sum() / rn.sum():.3f} vs {ah.sum() / an.sum():.3f})",
                   (rh.sum() / rn.sum() - ah.sum() / an.sum(), lo, hi, v))
        # outcome replay, 24 non-holiday hours, m = 0.96 (one-for-one unit assumption)
        per = {k: [] for k in ("V0", "B", "C")}
        for target in DOWNSTREAM_NONHOLIDAY:
            k = int(np.where(targets == pd.Timestamp(target))[0][0])
            fc = F[:, k]
            act = panel.activity[:, panel.hours.get_loc(pd.Timestamp(target))]
            hol = pd.Timestamp(target).date() in KNOWN_HOLIDAYS
            for var in ("V0", "B", "C"):
                m = 1.0 if var == "V0" else 0.96
                flagged = fc > m * thr
                fr_ = neighbour_fraction(flagged[:, None])[:, 0]
                routine = flagged & (hol | (fr_ >= NEIGHBOR_FRACTION_THRESHOLD))
                deficit = np.clip(fc - m * thr, 0, None)
                spare = np.clip(thr - fc, 0, None) if var == "V0" else (
                    np.clip(m * thr - fc, 0, None) if var == "B" else np.where(flagged, 0.0, np.clip(thr - fc, 0, None)))
                sol = solve_hour(pd.DataFrame({"CellID": cells[routine], "deficit": deficit[routine]}),
                                 pd.DataFrame({"CellID": cells, "spare": spare}))
                recv, give = np.zeros(len(cells)), np.zeros(len(cells))
                for (c, n), val in sol.items():
                    recv[int(c) - 1] += val
                    give[int(n) - 1] += val
                before = np.clip(act - thr, 0, None)
                after = np.clip(act - recv + give - thr, 0, None)
                harm = (np.clip(act + give - thr, 0, None) - before)[give > 1e-9].sum()
                per[var].append((before.sum() - after.sum(), harm))
        P = {k: np.array(v) for k, v in per.items()}
        for name, a, b in (("B - V0", "B", "V0"), ("C - V0", "C", "V0"), ("C - B", "C", "B")):
            for j, metric in ((0, "net reduction"), (1, "donor harm")):
                diff = P[a][:, j] - P[b][:, j]
                idx = rng.integers(0, len(diff), size=(BOOT_N, len(diff)))
                lo, hi, v = _ci(diff[idx].sum(1))
                _calls(rows, "Milan w1 outcome replay (24 h, m=0.96, hour bootstrap, one-for-one)", defn, 1, "all",
                       f"{metric}: {name}", (float(diff.sum()), lo, hi, v))
        print(f"  rule + replay done ({defn})")


def _milan_w2(rows, rng, panel, store):
    cm2 = training_cell_means(panel, W2)
    hot2 = hotspots(cm2)
    defs = {"w2 used (Nov 1-17)": thresholds_before(panel, "2013-11-18"), "w2 alt (Nov 1-24)": eval_thresholds(panel, W2)}
    w2m = pd.read_csv(ROOT / "data" / "experiments" / "phase3" / "checks" / "w2_margins.csv")
    for h in (1, 4):
        t = origin_index(panel, W2, "test", h)
        _, y, meta = build_rows(store, h, t, np.arange(len(panel.cells)), cm2)
        f = load_preds("w2", "final_63l", h, "test").astype(float)
        sel = w2m[(w2m.horizon == h) & (w2m.segment == "all") & w2m.model.str.startswith("final")]
        m_own, m_w1 = float(sel["w2_margin"].iloc[0]), float(sel["w1_margin"].iloc[0])
        is_hot = np.isin(meta["cell"], hot2)
        _, day_idx = np.unique(pd.DatetimeIndex(meta["target_time"]).date, return_inverse=True)
        for defn, thr in defs.items():
            tr = np.repeat(thr, len(t))
            act = y > tr
            fl = {"P0": f > tr, "own": f > m_own * tr, "w1": f > m_w1 * tr}
            for sl, msk in (("all", np.ones(len(y), bool)), ("hotspot", is_hot)):
                _calls(rows, "Milan w2", defn, h, sl, f"margin helps: w2 own m={m_own} vs m=1",
                       flag_bootstrap({"a": fl["own"], "b": fl["P0"]}, "b", act, day_idx, msk, rng)["a"])
                _calls(rows, "Milan w2", defn, h, sl, f"margin helps: w1 m={m_w1} vs m=1",
                       flag_bootstrap({"a": fl["w1"], "b": fl["P0"]}, "b", act, day_idx, msk, rng)["a"])
        print(f"  Milan w2 +{h}h done")


def _trentino(rows, rng):
    from src.evaluation.phase5_followup import _test_context
    from src.evaluation.phase5_transfer import City, pred_path
    tn = City("trentino")
    e18 = int(tn.hours.get_loc(pd.Timestamp("2013-11-18")))
    thr18 = np.quantile(tn.A[:, :e18], 0.9, axis=1)
    for h in (1, 4):
        y, thr25_rows, hot, _, day_idx, meta = _test_context(tn, h)
        defs = {"Trentino w2 used (before Nov 25)": thr25_rows, "Trentino w2 alt (before Nov 18)": thr18[meta["cell_pos"]]}
        fc = {"transfer": np.load(pred_path("milan", "w2", h, "trentino", "test")).astype(float),
              "in-domain": np.load(pred_path("trentino", "w2", h, "trentino", "test")).astype(float),
              "naive": meta["seasonal_naive"].astype(float)}
        m = MILAN_W1_MARGINS[h]
        for defn, tr in defs.items():
            act = y > tr
            fl = {k: f > tr for k, f in fc.items()}
            fl.update({f"{k} m": fc[k] > m * tr for k in ("transfer", "in-domain")})
            for sl, msk in (("all", np.ones(len(y), bool)), ("hotspot", hot)):
                for a, b, name in (("transfer", "in-domain", "P5 transfer vs in-domain, m=1"),
                                   ("transfer", "naive", "P5 transfer vs naive, m=1"),
                                   ("transfer m", "transfer", f"P5 margin gain, transfer (m={m})"),
                                   ("in-domain m", "in-domain", f"P5 margin gain, in-domain (m={m})")):
                    _calls(rows, "Trentino w2", defn, h, sl, name,
                           flag_bootstrap({"a": fl[a], "b": fl[b]}, "b", act, day_idx, msk, rng)["a"])
        print(f"  Trentino w2 +{h}h done")


def step_thrsens():
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    with Step("Milan w1 F1 calls"):
        panel, store, cm, thr_defs = _milan_w1(rows, rng)
    with Step("Milan w1 diagnosis rule + replay"):
        _milan_rule_and_replay(rows, rng, panel, store, cm, thr_defs)
    with Step("Milan w2"):
        _milan_w2(rows, rng, panel, store)
    del panel, store
    gc.collect()
    with Step("Trentino w2"):
        _trentino(rows, rng)
    res = pd.DataFrame(rows)
    _w(res, "threshold_sensitivity.csv")
    key = ["context", "comparison", "horizon", "slice"]
    wide = res.pivot_table(index=key, columns="thresholds", values="verdict", aggfunc="first", sort=False)
    wide = wide.reset_index()
    defs_by_ctx = res.groupby("context")["thresholds"].unique()
    changed = []
    for _, r in wide.iterrows():
        vs = [r[d] for d in defs_by_ctx[r.context]]
        if len(set(vs)) > 1:
            changed.append(r)
    ch = pd.DataFrame(changed)
    _w(ch, "threshold_sensitivity_changed.csv")
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 70)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nCALLS THAT CHANGE BETWEEN DEFINITIONS:")
    print(ch.to_string(index=False) if len(ch) else "(none)")


# ============================================================================= dec31
def step_dec31():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    thr = thresholds_before(panel, "2013-12-10")
    t = origin_index(panel, W1, "test", 1)
    _, y, meta = build_rows(store, 1, t, np.arange(len(panel.cells)), cm)
    f = load_preds("w1", "final_63l", 1, "test").astype(float)
    naive = meta["seasonal_naive"]
    tr = np.repeat(thr, len(t))
    act = y > tr
    tt = pd.DatetimeIndex(meta["target_time"])
    days = tt.normalize()
    rows = []
    for d in np.unique(days):
        m = days == d
        c, cn = flag_counts((f > tr)[m], act[m]), flag_counts((naive > tr)[m], act[m])
        dd = pd.Timestamp(d)
        rows.append({"day": dd.date(), "weekday": dd.day_name()[:3], "hours": int(m.sum() / len(panel.cells)),
                     "holiday": dd.date() in ALL_HOLIDAYS, "mean_activity": float(y[m].mean()),
                     "mae_model": float(np.abs(y - f)[m].mean()), "mae_naive": float(np.abs(y - naive)[m].mean()),
                     "mean_error_model": float((f - y)[m].mean()),
                     "f1_model": c["f1"], "f1_naive": cn["f1"], "precision_model": c["precision"], "recall_model": c["recall"],
                     "flagged_model": c["flagged_share"], "actual_share": c["actual_share"]})
    res = pd.DataFrame(rows)
    _w(res, "dec31_by_day.csv")
    pd.set_option("display.width", 260)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    other = res[(res.day != pd.Timestamp("2013-12-31").date())]
    nonhol = other[~other.holiday]
    print("\nother days: MAE model median / range", other.mae_model.median(), other.mae_model.min(), other.mae_model.max())
    print("non-holiday other days: F1 median / range", nonhol.f1_model.median(), nonhol.f1_model.min(), nonhol.f1_model.max())


if __name__ == "__main__":
    steps = {"bounds": step_bounds, "marginspre": step_marginspre, "thrsens": step_thrsens, "dec31": step_dec31}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
