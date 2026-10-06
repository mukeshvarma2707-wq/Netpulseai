"""
src/evaluation/phase3_checks.py  (bounded checks after Phase 3, before Phase 4)

  rates    Exceedance rate by day and by week, 2013-11-02 .. 2014-01-01, at the Phase 3
           thresholds (Nov 1 - Dec 9), for all / hotspot / typical cells, real activity.
  recount  Downstream reference recount: TP, FP, FN and actual exceedances per hour for the
           24 non-holiday hours, from the saved per-hour results AND independently from the
           raw activity.
  h3       Strategies P3 / H3: margin tuned on Dec 10-16 for TYPICAL cells, m = 1 on hotspots
           (plus recall-target typical margins), vs H0, H2, P0, P1, paired day bootstrap.
  replay   Outcome replay on the downstream hours: the solver's moves applied to ACTUAL
           activity (V0, A = V0 moves with watch-only flags, B, C at m = 0.96 / 0.98, and a
           perfect-foresight reference).
  w2fit    Window 2: final features, 63 leaves, +1h and +4h, tree count by the 0.5% rule on
           Nov 18-24 (its second use); saves validation and test predictions.
  w2       Margin transfer on window 2 (thresholds from Nov 1-17; tune on Nov 18-24; test
           Nov 25 - Dec 7).

Assumption for replay: capacity is "activity units" that transfer one-for-one between
neighbouring cells, with no radio effects (the dataset has no real capacity figures).

RUN:  python src/evaluation/phase3_checks.py {rates|recount|h3|replay|w2fit|w2}
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import CONGESTION_PERCENTILE, ROOT, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    ALL_HOLIDAYS, BOOT_SEED, INDICATIVE_DAYS, MODEL_DIR, TREE_CAP, W1, W2, FeatureStore, build_rows,
    choose_trees, fit_with_curve, has_preds, hotspots, lgbm_params, load_json, load_preds, origin_index,
    predict, require_headroom, save_preds, training_cell_means,
)
from src.evaluation.margin_tuning import (  # noqa: E402
    DOWNSTREAM_HOLIDAY, DOWNSTREAM_NONHOLIDAY, MARGINS, RECALL_TARGETS, Data, f1_best, flag_bootstrap,
    flag_counts, margin_for_recall, pooled_variant, pr_curve, ratio, thresholds_before,
)

OUT = ROOT / "data" / "experiments" / "phase3" / "checks"
P3_THR_END = "2013-12-10"
W2_THR_END = "2013-11-18"


def _write(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


# ============================================================================= 1. rates
def step_rates():
    panel = load_panel()
    thr = thresholds_before(panel, P3_THR_END)
    hot = set(hotspots(training_cell_means(panel, W1)).tolist())
    is_hot = np.isin(panel.cells, list(hot))
    exceed = panel.activity > thr[:, None]
    H = panel.hours
    keep = (H >= pd.Timestamp("2013-11-02")) & (H < pd.Timestamp("2014-01-02"))
    df = pd.DataFrame({"hour": H[keep],
                       "all": exceed[:, keep].mean(axis=0),
                       "hotspot": exceed[is_hot][:, keep].mean(axis=0),
                       "typical": exceed[~is_hot][:, keep].mean(axis=0)})
    df["date"] = df["hour"].dt.date
    daily = df.groupby("date")[["all", "hotspot", "typical"]].mean().reset_index()
    d = pd.to_datetime(daily["date"])
    daily["weekday"] = d.dt.day_name().str[:3]
    daily["weekend"] = d.dt.dayofweek >= 5
    daily["holiday"] = daily["date"].isin(ALL_HOLIDAYS)
    daily["period"] = np.select(
        [d < pd.Timestamp("2013-12-10"), d < pd.Timestamp("2013-12-17")],
        ["threshold period (in-sample)", "validation Dec 10-16"], "test Dec 17 - Jan 1")
    _write(daily, "rates_daily.csv")
    df["week_start"] = (pd.to_datetime(df["date"]) - pd.to_timedelta(pd.to_datetime(df["date"]).dt.dayofweek, unit="D")).dt.date
    weekly = df.groupby("week_start")[["all", "hotspot", "typical"]].mean().reset_index()
    weekly["days"] = df.groupby("week_start")["date"].nunique().values
    _write(weekly, "rates_weekly.csv")
    by_period = daily.groupby("period", sort=False)[["all", "hotspot", "typical"]].mean()
    nonhol = daily[~daily.holiday & ~daily.weekend].groupby("period", sort=False)[["all", "hotspot", "typical"]].mean()
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    f = lambda v: f"{v:.4f}"  # noqa: E731
    print("DAILY"); print(daily.to_string(index=False, float_format=f))
    print("\nWEEKLY (Mon start)"); print(weekly.to_string(index=False, float_format=f))
    print("\nBY PERIOD (all days)"); print(by_period.to_string(float_format=f))
    print("\nBY PERIOD (weekdays that are not holidays)"); print(nonhol.to_string(float_format=f))


# ============================================================================= 2. recount
def step_recount():
    by_hour = pd.read_csv(ROOT / "data" / "experiments" / "phase3" / "downstream_by_hour.csv")
    v0 = by_hour[(by_hour.variant == "V0 current") & (by_hour.subset == "non-holiday")].copy()
    v0["TP"] = v0["flagged_actually_exceeding"]
    v0["FP"] = v0["flagged"] - v0["TP"]
    v0["FN"] = v0["actual_exceeding"] - v0["TP"]
    # Independent recount from raw activity at the same thresholds
    panel = load_panel()
    thr = thresholds_before(panel, P3_THR_END)
    v0["actual_exceeding_raw"] = [int((panel.activity[:, panel.hours.get_loc(pd.Timestamp(t))] > thr).sum()) for t in v0["target"]]
    cols = ["target", "flagged", "TP", "FP", "FN", "actual_exceeding", "actual_exceeding_raw"]
    _write(v0[cols], "recount_nonholiday.csv")
    tot = v0[["flagged", "TP", "FP", "FN", "actual_exceeding", "actual_exceeding_raw"]].sum()
    # The other 19,203: flagged cases in the diagnosis output (+1h, v1 forecasts) on Dec 25
    diag = pd.read_parquet(ROOT / "data" / "raw" / "diagnosis_results_1h.parquet", columns=["target_datetime_1h"])
    dec25 = int((pd.to_datetime(diag["target_datetime_1h"]).dt.date == pd.Timestamp("2013-12-25").date()).sum())
    print(v0[cols].to_string(index=False))
    print("\nTOTALS:", tot.to_dict())
    print(f"Dec 25 flagged cases in diagnosis_results_1h.parquet (v1 in-sample forecasts, full-period thresholds): {dec25:,}")
    with open(OUT / "recount_totals.json", "w") as fh:
        json.dump({**{k: int(v) for k, v in tot.items()}, "dec25_diagnosis_cases": dec25}, fh, indent=2)


# ============================================================================= 3. h3
def step_h3():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    hot = hotspots(cm)
    thr = thresholds_before(panel, P3_THR_END)
    ch = pd.read_csv(ROOT / "data" / "experiments" / "phase3" / "margins_chosen.csv")

    def m(fc, seg, kind, h):
        return float(ch[(ch.horizon == h) & (ch.forecaster == fc) & (ch.segment == seg) & (ch.kind == kind)]["margin"].iloc[0])

    rng = np.random.default_rng(BOOT_SEED)
    rows, boots, used = [], [], []
    for h in (1, 2, 3, 4):
        d = Data(panel, store, cm, hot, thr, h, "test")
        def rows_m(m_hot, m_typ):
            return np.where(d.is_hot, m_hot, m_typ)
        strat = {
            "P0 pooled m=1": ("final pooled", rows_m(1.0, 1.0)),
            "P1 pooled global m": ("final pooled", rows_m(m("final pooled", "all", "f1", h), m("final pooled", "all", "f1", h))),
            "P3 pooled typical m, hotspot m=1": ("final pooled", rows_m(1.0, m("final pooled", "typical", "f1", h))),
            "H0 pooled+hotspot m=1": ("pooled+hotspot", rows_m(1.0, 1.0)),
            "H2 pooled+hotspot hot/typ m": ("pooled+hotspot", rows_m(m("pooled+hotspot", "hotspot", "f1", h), m("pooled+hotspot", "typical", "f1", h))),
            "H3 pooled+hotspot typical m, hotspot m=1": ("pooled+hotspot", rows_m(1.0, m("pooled+hotspot", "typical", "f1", h))),
        }
        for rt in RECALL_TARGETS:
            for fc, tag in (("pooled+hotspot", "H3"), ("final pooled", "P3")):
                mt = m(fc, "typical", f"recall{rt}", h)
                strat[f"{tag} typical recall>={rt} (m={mt:.3f}), hotspot m=1"] = (fc, rows_m(1.0, mt))
        for s, (fc, mr) in strat.items():
            used.append({"horizon": h, "strategy": s, "m_typical": float(mr[~d.is_hot][0]), "m_hotspot": float(mr[d.is_hot][0])})
        flags = {s: d.fc[fc] > mr * d.thr_rows for s, (fc, mr) in strat.items()}
        dates = d.target.date
        _, day_index = np.unique(dates, return_inverse=True)
        slices = {"all": np.ones(len(d.y), bool), "hotspot": d.is_hot, "typical": ~d.is_hot,
                  "non-holiday": ~d.holiday, "holiday": d.holiday}
        for s, fl in flags.items():
            for sl, msk in slices.items():
                nd = len(np.unique(dates[msk]))
                rows.append({"horizon": h, "strategy": s, "slice": sl, "days": nd, "indicative": nd < INDICATIVE_DAYS,
                             **flag_counts(fl[msk], d.actual[msk])})
        for ref in ("H0 pooled+hotspot m=1", "H2 pooled+hotspot hot/typ m", "P0 pooled m=1", "P1 pooled global m"):
            for sl in ("all", "hotspot"):
                for s, (dif, lo, hi, v) in flag_bootstrap(flags, ref, d.actual, day_index, slices[sl], rng).items():
                    boots.append({"horizon": h, "vs": ref, "strategy": s, "slice": sl, "f1_diff": dif, "lo": lo, "hi": hi, "verdict": v})
        del d, flags
        gc.collect()
    _write(pd.DataFrame(rows), "h3_results.csv")
    _write(pd.DataFrame(boots), "h3_bootstrap.csv")
    _write(pd.DataFrame(used), "h3_margins_used.csv")
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(r[r.slice.isin(["all", "hotspot"])].pivot_table(index=["strategy"], columns=["slice", "horizon"], values="f1", sort=False).to_string(float_format=lambda v: f"{v:.3f}"))


# ============================================================================= 4. replay
def _diagnose(cells, flagged, holiday, neighbors, frac):
    flagged_set = set(cells[flagged].tolist())
    routine = np.zeros(len(cells), bool)
    for i in np.where(flagged)[0]:
        if holiday:
            routine[i] = True
            continue
        nb = neighbors[int(cells[i])]
        routine[i] = sum(n in flagged_set for n in nb) / len(nb) >= frac
    return routine


def step_replay():
    from src.diagnosis.diagnosis_agent import KNOWN_HOLIDAYS, NEIGHBOR_FRACTION_THRESHOLD, get_neighbors
    from src.optimization.solver import solve_hour

    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    thr = thresholds_before(panel, P3_THR_END)
    t_idx = origin_index(panel, W1, "test", 1)
    _, _, meta = build_rows(store, 1, t_idx, np.arange(len(panel.cells)), cm)
    n_t = len(t_idx)
    F = load_preds("w1", pooled_variant(), 1, "test").astype(np.float64).reshape(len(panel.cells), n_t)
    targets = pd.DatetimeIndex(meta["target_time"][:n_t])
    cells = panel.cells
    pos_of = {int(c): i for i, c in enumerate(cells)}
    neighbors = {int(c): get_neighbors(int(c)) for c in cells}

    def moves(fc, m, variant, holiday):
        """Returns (flagged, routine, solution dict) using the unchanged solver and flag definitions."""
        if variant in ("V0", "A", "PF"):
            flagged_m1 = fc > thr
            routine = _diagnose(cells, flagged_m1, holiday, neighbors, NEIGHBOR_FRACTION_THRESHOLD)
            deficit, spare = np.clip(fc - thr, 0, None), np.clip(thr - fc, 0, None)
            watch = (fc > m * thr) & ~flagged_m1 if variant == "A" else np.zeros(len(cells), bool)
            flagged = flagged_m1
        else:
            flagged = fc > m * thr
            routine = _diagnose(cells, flagged, holiday, neighbors, NEIGHBOR_FRACTION_THRESHOLD)
            deficit = np.clip(fc - m * thr, 0, None)
            spare = np.clip(m * thr - fc, 0, None) if variant == "B" else np.where(flagged, 0.0, np.clip(thr - fc, 0, None))
            watch = np.zeros(len(cells), bool)
        t0 = time.perf_counter()
        sol = solve_hour(pd.DataFrame({"CellID": cells[routine], "deficit": deficit[routine]}),
                         pd.DataFrame({"CellID": cells, "spare": spare}))
        return flagged, routine, watch, sol, time.perf_counter() - t0

    def outcome(actual, sol):
        recv = np.zeros(len(cells)); give = np.zeros(len(cells))
        for (c, n), v in sol.items():
            recv[pos_of[int(c)]] += v
            give[pos_of[int(n)]] += v
        before = np.clip(actual - thr, 0, None)
        after_load = actual - recv + give
        after = np.clip(after_load - thr, 0, None)
        exc = actual > thr
        moved = recv.sum()
        is_recv, is_donor = recv > 1e-9, give > 1e-9
        harm = np.clip(actual + give - thr, 0, None) - before
        return {"excess_before": float(before.sum()), "excess_after": float(after.sum()),
                "excess_reduction": float(before.sum() - after.sum()),
                "moved_total": float(moved),
                "relief_on_actual_exceeders": float(np.minimum(recv, before)[exc & is_recv].sum()),
                "moved_to_non_exceeders": float(recv[is_recv & ~exc].sum()),
                "receivers": int(is_recv.sum()), "receivers_not_exceeding": int((is_recv & ~exc).sum()),
                "donors": int(is_donor.sum()),
                "donor_harm": float(harm[is_donor].sum()),
                "donors_pushed_over": int((is_donor & ~exc & (actual + give > thr)).sum()),
                "donors_already_above": int((is_donor & exc).sum()),
                "donated_by_donors_already_above": float(give[is_donor & exc].sum())}

    rows = []
    for subset, hours in (("non-holiday", DOWNSTREAM_NONHOLIDAY), ("holiday", DOWNSTREAM_HOLIDAY)):
        for target in hours:
            k = int(np.where(targets == pd.Timestamp(target))[0][0])
            fc = F[:, k]
            actual = panel.activity[:, panel.hours.get_loc(pd.Timestamp(target))]
            holiday = pd.Timestamp(target).date() in KNOWN_HOLIDAYS
            runs = [("V0 current", "V0", 1.0), ("PF perfect foresight", "PF", 1.0)]
            for mm in (0.96, 0.98):
                runs += [(f"A watch-only m={mm}", "A", mm), (f"B margin deficit+spare m={mm}", "B", mm),
                         (f"C margin deficit, flagged cannot donate m={mm}", "C", mm)]
            for name, var, mm in runs:
                use = actual if var == "PF" else fc
                flagged, routine, watch, sol, solve_s = moves(use, mm, var, holiday)
                rows.append({"subset": subset, "target": target, "variant": name, "flagged": int(flagged.sum()),
                             "watch_flags": int(watch.sum()), "routine": int(routine.sum()),
                             "solver_seconds": solve_s, **outcome(actual, sol)})
        print(f"  {subset}: {len(hours)} hours replayed")
    res = pd.DataFrame(rows)
    _write(res, "replay_by_hour.csv")
    agg = res.groupby(["subset", "variant"], sort=False).sum(numeric_only=True).reset_index()
    agg["share_moved_to_non_exceeders"] = agg["moved_to_non_exceeders"] / agg["moved_total"]
    agg["excess_reduction_share"] = agg["excess_reduction"] / agg["excess_before"]
    _write(agg, "replay_summary.csv")
    pd.set_option("display.width", 260)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


# ============================================================================= 5. window 2
def step_w2fit():
    """Deployed configuration (final features, 63 leaves) refit on window 2 for +1h and +4h."""
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W2)
    groups = tuple(load_json("final_features.json")["final_groups"])
    cidx = np.arange(len(panel.cells))
    info = []
    for h in (1, 4):
        if has_preds("w2", "final_63l", h, "test"):
            continue
        X, y, _ = build_rows(store, h, origin_index(panel, W2, "train", h), cidx, cm, groups=groups)
        Xv, yv, _ = build_rows(store, h, origin_index(panel, W2, "val", h), cidx, cm, groups=groups)
        require_headroom(f"w2 final_63l +{h}h")
        with Step(f"w2 final features +{h}h, 63 leaves, up to {TREE_CAP} trees") as s:
            model, curve, fit_s = fit_with_curve(X, y, Xv, yv, lgbm_params(63, TREE_CAP))
        del X
        choice = choose_trees(curve)
        n = choice["chosen_trees"]
        np.save(OUT / f"w2_final_63l_{h}h_curve.npy", curve)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(MODEL_DIR / f"w2__final_63l__{h}h.txt"), num_iteration=n)
        save_preds("w2", "final_63l", h, "val", predict(model, Xv, n))
        Xt, _, _ = build_rows(store, h, origin_index(panel, W2, "test", h), cidx, cm, groups=groups)
        save_preds("w2", "final_63l", h, "test", predict(model, Xt, n))
        info.append({"horizon": h, **choice, "fit_seconds_3000": fit_s, "peak_ram_gb": s.peak_ws, "peak_commit_gb": s.peak_commit})
        _write(pd.DataFrame(info), f"w2_fit_{h}h.csv")
        print(info[-1])
        del model, Xv, Xt
        gc.collect()


def step_w2():
    panel = load_panel()
    store = FeatureStore(panel)
    cm2 = training_cell_means(panel, W2)
    hot2 = hotspots(cm2)
    thr2 = thresholds_before(panel, W2_THR_END)
    rng = np.random.default_rng(BOOT_SEED)
    w1m = pd.read_csv(ROOT / "data" / "experiments" / "phase3" / "margins_chosen.csv")
    rates, margins, rows, boots = [], [], [], []
    for variant, label in (("final_63l", "final features, 63 leaves (deployed config)"),
                           ("tuned", "production features, 31 leaves (secondary)")):
        for h in (1, 4):
            if not has_preds("w2", variant, h, "val"):
                print(f"skip {variant} +{h}h: no validation predictions")
                continue
            sets = {}
            for split in ("val", "test"):
                t = origin_index(panel, W2, split, h)
                _, y, meta = build_rows(store, h, t, np.arange(len(panel.cells)), cm2)
                thr_rows = np.repeat(thr2, len(t))
                sets[split] = dict(y=y, f=load_preds("w2", variant, h, split).astype(np.float64), thr=thr_rows,
                                   act=y > thr_rows, hot=np.isin(meta["cell"], hot2),
                                   dates=pd.DatetimeIndex(meta["target_time"]).date)
                for seg, msk in (("all", slice(None)), ("hotspot", sets[split]["hot"]), ("typical", ~sets[split]["hot"])):
                    rates.append({"model": label, "horizon": h, "split": split, "segment": seg,
                                  "actual_share": float(sets[split]["act"][msk].mean())})
            v, t = sets["val"], sets["test"]
            chosen = {}
            for seg, msk in (("all", np.ones(len(v["y"]), bool)), ("hotspot", v["hot"]), ("typical", ~v["hot"])):
                chosen[seg] = f1_best(pr_curve(ratio(v["f"][msk], v["thr"][msk]), v["act"][msk]))
            w1_ref = {seg: float(w1m[(w1m.horizon == h) & (w1m.forecaster == "final pooled") & (w1m.segment == seg) & (w1m.kind == "f1")]["margin"].iloc[0])
                      for seg in ("all", "hotspot", "typical")}
            for seg in chosen:
                margins.append({"model": label, "horizon": h, "segment": seg, "w2_margin": chosen[seg], "w1_margin": w1_ref[seg]})
            def mr(mh, mt):
                return np.where(t["hot"], mh, mt)
            strat = {"P0 m=1": mr(1.0, 1.0), "P1 global m": mr(chosen["all"], chosen["all"]),
                     "P2 hot/typ m": mr(chosen["hotspot"], chosen["typical"]), "P3 typical m, hotspot m=1": mr(1.0, chosen["typical"]),
                     "P1 with window-1 global m": mr(w1_ref["all"], w1_ref["all"])}
            flags = {s: t["f"] > m * t["thr"] for s, m in strat.items()}
            _, day_index = np.unique(t["dates"], return_inverse=True)
            for s, fl in flags.items():
                for seg, msk in (("all", np.ones(len(t["y"]), bool)), ("hotspot", t["hot"]), ("typical", ~t["hot"])):
                    rows.append({"model": label, "horizon": h, "strategy": s, "slice": seg, **flag_counts(fl[msk], t["act"][msk])})
            for seg, msk in (("all", np.ones(len(t["y"]), bool)), ("hotspot", t["hot"])):
                for s, (dif, lo, hi, vv) in flag_bootstrap(flags, "P0 m=1", t["act"], day_index, msk, rng).items():
                    boots.append({"model": label, "horizon": h, "vs": "P0 m=1", "strategy": s, "slice": seg,
                                  "f1_diff": dif, "lo": lo, "hi": hi, "verdict": vv})
            del sets
            gc.collect()
    for name, data in (("w2_base_rates.csv", rates), ("w2_margins.csv", margins), ("w2_results.csv", rows), ("w2_bootstrap.csv", boots)):
        _write(pd.DataFrame(data), name)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    f = lambda v: f"{v:.3f}"  # noqa: E731
    print(pd.DataFrame(rates).pivot_table(index=["model", "horizon", "segment"], columns="split", values="actual_share").to_string(float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(margins).to_string(index=False, float_format=f))
    print(pd.DataFrame(boots).to_string(index=False, float_format=f))


if __name__ == "__main__":
    steps = {"rates": step_rates, "recount": step_recount, "h3": step_h3, "replay": step_replay,
             "w2fit": step_w2fit, "w2": step_w2}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
