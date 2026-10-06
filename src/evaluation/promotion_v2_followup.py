"""
src/evaluation/promotion_v2_followup.py  (review items before stage 2; saved predictions only)

  fairmargin  Trentino w2, Nov 2-17 models: each model's own global margin tuned on Trentino
              Nov 18-24 (RE-USED week) with thresholds from Nov 1-17, tested on Nov 25 - Dec 7.
              Transfer vs in-domain F1 at own margins and at equal flagged share; paired day bootstrap.
  tnthr       Trentino w2 threshold sensitivity (Nov 1-17 vs Nov 1-24): transfer vs in-domain vs naive.
  nohours     +1h target hours of the v2 run with no ROUTINE case.
  watchprec   Watch-set precision (share of watch flags whose target actually exceeded the
              threshold), per horizon and split; written to data/processed/v2/v2_metadata.json.

RUN: python src/evaluation/promotion_v2_followup.py {fairmargin|tnthr|nohours|watchprec}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import ROOT, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import BOOT_SEED, W2  # noqa: E402
from src.evaluation.margin_tuning import f1_best, flag_bootstrap, flag_counts, pr_curve, ratio  # noqa: E402

OUT = ROOT / "data" / "experiments" / "promotion_v2"
V2 = ROOT / "data" / "processed" / "v2"


def _w(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def _tn_split(tn, split, h):
    from src.evaluation.phase5_transfer import pred_path
    s = tn.scale(W2)
    _, y, meta = tn.rows(W2, split, h, s)
    fc = {"transfer": np.load(pred_path("milan", "w2", h, "trentino", split)).astype(float),
          "in-domain": np.load(pred_path("trentino", "w2", h, "trentino", split)).astype(float),
          "naive": meta["seasonal_naive"].astype(float)}
    hot = np.isin(meta["cell_pos"], np.argsort(-s, kind="stable")[:125])
    _, day_idx = np.unique(pd.DatetimeIndex(meta["target_time"]).date, return_inverse=True)
    return y, fc, meta["cell_pos"], hot, day_idx, pd.DatetimeIndex(meta["target_time"])


def _thr(tn, end):
    e = int(tn.hours.get_loc(pd.Timestamp(end)))
    return np.quantile(tn.A[:, :e], 0.9, axis=1)


def _margin_for_share(f, thr_rows, share):
    """Margin m with flagged share (f > m * thr) as close as possible to `share` (computed on test:
    a diagnostic that removes the operating-point difference, not a deployable choice)."""
    r = np.sort(ratio(f, thr_rows))
    k = int(round((1 - share) * len(r)))
    return float(r[min(max(k, 0), len(r) - 1)])


# ============================================================================= fairmargin
def step_fairmargin():
    from src.evaluation.phase5_transfer import City
    rng = np.random.default_rng(BOOT_SEED)
    tn = City("trentino")
    thr17, thr24 = _thr(tn, "2013-11-18"), _thr(tn, "2013-11-25")
    rows, boots = [], []
    for h in (1, 4):
        yv, fv, cpv, _, _, tv = _tn_split(tn, "val", h)
        assert tv.min() >= pd.Timestamp("2013-11-18") and tv.max() < pd.Timestamp("2013-11-25")
        yt, ft, cpt, hot, day_idx, _ = _tn_split(tn, "test", h)
        thv = thr17[cpv]
        own = {k: f1_best(pr_curve(ratio(fv[k], thv), yv > thv)) for k in fv}
        val_share = {k: float(np.mean(yv > thv)) for k in fv}
        for tdef, thr in (("test thresholds Nov 1-17 (same as tuning)", thr17), ("test thresholds Nov 1-24 (Phase 5 definition)", thr24)):
            tr = thr[cpt]
            act = yt > tr
            allm = np.ones(len(yt), bool)
            fl = {f"{k} m=1": ft[k] > tr for k in ft}
            fl.update({f"{k} own m": ft[k] > own[k] * tr for k in ft})
            ts = float(fl["transfer own m"].mean())
            eq = {"actual share": float(act.mean()), "transfer's own-margin share": ts}
            m_eq = {}
            for lab, share in eq.items():
                for k in ("transfer", "in-domain"):
                    m_eq[(lab, k)] = _margin_for_share(ft[k], tr, share)
                    fl[f"{k} eq {lab}"] = ft[k] > m_eq[(lab, k)] * tr
            for k in ft:
                for kind in ("m=1", "own m"):
                    for sl, msk in (("all", allm), ("hotspot", hot)):
                        c = flag_counts(fl[f"{k} {kind}"][msk], act[msk])
                        rows.append({"horizon": h, "test_thresholds": tdef, "model": k, "rule": kind,
                                     "margin": 1.0 if kind == "m=1" else own[k], "slice": sl, **c})
            for lab in eq:
                for k in ("transfer", "in-domain"):
                    c = flag_counts(fl[f"{k} eq {lab}"], act)
                    rows.append({"horizon": h, "test_thresholds": tdef, "model": k, "rule": f"equal share = {lab}",
                                 "margin": m_eq[(lab, k)], "slice": "all", **c})
            comps = [("transfer m=1", "in-domain m=1", "m = 1"),
                     ("transfer own m", "in-domain own m", "each at its own tuned margin"),
                     ("transfer own m", "naive own m", "transfer vs naive, own margins"),
                     ("in-domain own m", "naive own m", "in-domain vs naive, own margins")]
            comps += [(f"transfer eq {lab}", f"in-domain eq {lab}", f"equal flagged share ({lab}, margins set on test)") for lab in eq]
            for a, b, name in comps:
                for sl, msk in (("all", allm), ("hotspot", hot)):
                    if "eq" in a and sl == "hotspot":
                        continue
                    d, lo, hi, v = flag_bootstrap({"a": fl[a], "b": fl[b]}, "b", act, day_idx, msk, rng)["a"]
                    boots.append({"horizon": h, "test_thresholds": tdef, "comparison": name, "slice": sl,
                                  "f1_diff": d, "lo": lo, "hi": hi, "verdict": v})
        for k in own:
            rows.append({"horizon": h, "test_thresholds": "tuning (Nov 18-24, thresholds Nov 1-17)", "model": k,
                         "rule": "own m chosen", "margin": own[k], "slice": "all", "actual_share": val_share[k]})
        print(f"+{h}h own margins:", own)
    _w(pd.DataFrame(rows), "tn_fair_margin_results.csv")
    _w(pd.DataFrame(boots), "tn_fair_margin_bootstrap.csv")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    r = pd.DataFrame(rows)
    print(r[r.slice == "all"][["horizon", "test_thresholds", "model", "rule", "margin", "precision", "recall", "f1", "flagged_share", "actual_share"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(boots).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ============================================================================= tnthr
def step_tnthr():
    from src.evaluation.phase5_transfer import City
    rng = np.random.default_rng(BOOT_SEED)
    tn = City("trentino")
    defs = {"Nov 1-24 (used in Phase 5)": _thr(tn, "2013-11-25"), "Nov 1-17": _thr(tn, "2013-11-18")}
    rows = []
    for h in (1, 4):
        yt, ft, cpt, hot, day_idx, _ = _tn_split(tn, "test", h)
        for dname, thr in defs.items():
            tr = thr[cpt]
            act = yt > tr
            fl = {k: f > tr for k, f in ft.items()}
            for sl, msk in (("all", np.ones(len(yt), bool)), ("hotspot", hot)):
                for a, b in (("transfer", "in-domain"), ("transfer", "naive"), ("in-domain", "naive")):
                    d, lo, hi, v = flag_bootstrap({"a": fl[a], "b": fl[b]}, "b", act, day_idx, msk, rng)["a"]
                    rows.append({"horizon": h, "thresholds": dname, "comparison": f"{a} - {b}", "slice": sl,
                                 "f1_diff": d, "lo": lo, "hi": hi, "verdict": v, "actual_share": float(act[msk].mean())})
    r = pd.DataFrame(rows)
    _w(r, "tn_threshold_sensitivity.csv")
    w = r.assign(cell=r.apply(lambda x: f"{x.f1_diff:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}] {x.verdict}", axis=1)) \
         .pivot_table(index=["horizon", "comparison", "slice"], columns="thresholds", values="cell", aggfunc="first")
    pd.set_option("display.width", 220)
    print(w.to_string())


# ============================================================================= nohours
def step_nohours():
    fc = pd.read_parquet(V2 / "cell_forecasts_v2.parquet", columns=["horizon", "target_datetime", "forecast_kind"],
                         filters=[("horizon", "==", 1)])
    diag = pd.read_parquet(V2 / "diagnosis_results_v2.parquet", filters=[("horizon", "==", 1)])
    hours = fc.drop_duplicates("target_datetime").set_index("target_datetime")["forecast_kind"].astype(str).sort_index()
    rt = set(diag.loc[diag.classification == "ROUTINE", "target_datetime"])
    anyflag = diag.groupby("target_datetime").size()
    no = hours[~hours.index.isin(rt)]
    res = pd.DataFrame({"target_datetime": no.index, "forecast_kind": no.values,
                        "flags_at_hour": anyflag.reindex(no.index).fillna(0).astype(int).values})
    res["date"] = res.target_datetime.dt.date
    res["hour"] = res.target_datetime.dt.hour
    _w(res, "v2_hours_without_routine.csv")
    print(f"hours without ROUTINE: {len(res)} of {len(hours)}; validation: {(res.forecast_kind == 'validation').sum()}")
    print("flags at those hours (all ANOMALOUS or none):", res.flags_at_hour.sum())
    print(res.groupby("date")["hour"].apply(lambda s: ",".join(f"{x:02d}" for x in s)).to_string())
    print(res.groupby("hour").size().to_string())


def step_nohourscheck():
    """The 158 hours: traffic level (actual activity) vs the other +1h target hours, forecast
    coverage per hour, and an independent recount of flags (forecast > threshold) vs diagnosis_v2."""
    panel = load_panel()
    thr = pd.read_parquet(V2 / "thresholds_v2.parquet").set_index("CellID")["congestion_threshold"].reindex(panel.cells).to_numpy()
    fc = pd.read_parquet(V2 / "cell_forecasts_v2.parquet", filters=[("horizon", "==", 1)])
    diag = pd.read_parquet(V2 / "diagnosis_results_v2.parquet", filters=[("horizon", "==", 1)])
    no = set(pd.read_csv(OUT / "v2_hours_without_routine.csv", parse_dates=["target_datetime"]).target_datetime)
    fc["flag"] = fc.forecast.to_numpy() > thr[fc.CellID.to_numpy() - 1]
    per = fc.groupby("target_datetime").agg(cells=("CellID", "nunique"), rows=("CellID", "size"), flags_recount=("flag", "sum"))
    per["flags_diag"] = diag.groupby("target_datetime").size().reindex(per.index).fillna(0).astype(int)
    per["routine_diag"] = diag[diag.classification == "ROUTINE"].groupby("target_datetime").size().reindex(per.index).fillna(0).astype(int)
    ti = panel.hours.get_indexer(per.index)
    per["mean_activity"] = panel.activity[:, ti].mean(axis=0)
    per["actual_exceed_share"] = (panel.activity[:, ti] > thr[:, None]).mean(axis=0)
    per["no_routine"] = per.index.isin(no)
    per["hour"] = per.index.hour
    out = per.reset_index()
    _w(out, "v2_hours_check.csv")
    g = out.groupby("no_routine").agg(hours=("hour", "size"), mean_activity=("mean_activity", "mean"),
                                      median_activity=("mean_activity", "median"), actual_exceed_share=("actual_exceed_share", "mean"),
                                      flags=("flags_diag", "sum"))
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    nr = out[out.no_routine]
    print("hour-of-day of the 158:", nr.hour.value_counts().sort_index().to_dict())
    print("share of the 158 in 00-08:", float((nr.hour <= 8).mean()))
    print("activity percentile of each no-routine hour among all 548 (median, max):",
          float(nr.mean_activity.rank(pct=True).median()),
          float(out.mean_activity.rank(pct=True)[out.no_routine].median()), float(out.mean_activity.rank(pct=True)[out.no_routine].max()))
    print("every hour has 10,000 cells and 10,000 rows:", bool((out.cells == 10000).all() and (out.rows == 10000).all()))
    print("flag recount == diagnosis flags at every hour:", bool((out.flags_recount == out.flags_diag).all()))
    day = nr[nr.hour > 8]
    print("no-routine hours after 08:00:", len(day))
    print(day[["target_datetime", "mean_activity", "actual_exceed_share", "flags_diag"]].to_string(index=False))


# ============================================================================= watchprec
KIND_LABEL = {
    "validation": "validation (used for model size and margin selection): out-of-training, but not an untouched test",
    "test": "test (Dec 17 onward): out-of-training and not used for model size or margins; it was used for "
            "feature-group selection in Phase 2, so it is not fully untouched either",
}


def step_watchprec():
    panel = load_panel()
    thr = pd.read_parquet(V2 / "thresholds_v2.parquet").set_index("CellID")["congestion_threshold"]
    watch = pd.read_parquet(V2 / "watch_flags_v2.parquet")
    diag = pd.read_parquet(V2 / "diagnosis_results_v2.parquet", columns=["CellID", "target_datetime", "horizon", "forecast_kind"])
    rows = []
    for name, df in (("watch", watch), ("flagged (m = 1, for reference)", diag)):
        ti = panel.hours.get_indexer(pd.DatetimeIndex(df.target_datetime))
        y = panel.activity[df.CellID.to_numpy() - 1, ti]
        exceeded = y > thr.reindex(df.CellID).to_numpy()
        g = pd.DataFrame({"horizon": df.horizon.to_numpy(), "kind": df.forecast_kind.astype(str).to_numpy(), "exceeded": exceeded})
        for (h, k), x in g.groupby(["horizon", "kind"]):
            rows.append({"set": name, "horizon": int(h), "forecast_kind": k, "rows": len(x),
                         "actually_exceeded": int(x.exceeded.sum()), "precision": float(x.exceeded.mean())})
    r = pd.DataFrame(rows)
    _w(r, "watch_precision.csv")
    r[r.set == "watch"].to_csv(V2 / "watch_precision_v2.csv", index=False)
    meta = {
        "model_version": {"pooled": "final_63l", "hotspot_1h": "hotspot_only_final"},
        "thresholds": "per-cell 90th percentile of total_activity, 2013-11-01 00:00 .. 2013-12-09 23:00",
        "watch_margins": {"1": 0.955, "2": 0.95, "3": 0.95, "4": 0.94},
        "watch_definition": "typical cells only (hotspots stay at m = 1): margin x threshold < forecast <= threshold",
        "watch_status": "advisory: never counted as resolved, never in /cases, never sent to the solver",
        "watch_precision": {f"{x.horizon}h_{x.forecast_kind}": round(x.precision, 4) for x in r[r.set == "watch"].itertuples()},
        "watch_precision_note": "share of watch flags whose target hour actually exceeded the threshold (measured on "
                                "this dataset, Dec 10 - Jan 1; the test window is the holiday lull)",
        "forecast_kind_label": KIND_LABEL,
        "solver": "V0 (unchanged solve_hour), +1h; units are scaled record counts, one-for-one transfer assumption",
    }
    with open(V2 / "v2_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    steps = {"fairmargin": step_fairmargin, "tnthr": step_tnthr, "nohours": step_nohours, "nohourscheck": step_nohourscheck, "watchprec": step_watchprec}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_followup_{sys.argv[1]}.csv", index=False)
