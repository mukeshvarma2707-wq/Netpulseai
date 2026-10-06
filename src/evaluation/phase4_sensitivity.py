"""
src/evaluation/phase4_sensitivity.py  (Phase 4, bounded sensitivity checks + report fixes)

Steps:
  ranges    Margin plateaus: margins whose validation F1 is within 0.002 / 0.005 of the best,
            and the test F1 across that range (window 1: all horizons; window 2: +1h, +4h).
  rule      (a) Does the 30% neighbour rule mean anything? ROUTINE vs ANOMALOUS precision and
            actual excess at fractions 0.2 / 0.3 / 0.4 (full test window, non-holiday hours),
            and where moves go (24 downstream hours).
  holiday   (b) Holiday exemption on vs off: which flags would change, and their outcomes.
  pct       (c) Congestion percentile 85 / 90 / 95: flags, ROUTINE/ANOMALOUS, solver coverage,
            outcome replay, each against its own definition.
  hotcut    (d) Hotspot cutoff 1% / 2% / 5%: hotspot-only refit at +1h (size frozen), vs pooled.

Common choices: final pooled model ("final_63l"), +1h, flag rule m = 1, thresholds = per-cell
percentile over 2013-11-01 .. 12-09 (Phase 3 definition) unless the step varies it.
Outcome replay assumes activity units transfer one-for-one with no radio effects.

RUN: python src/evaluation/phase4_sensitivity.py {ranges|rule|holiday|pct|hotcut}
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
from src.evaluation.common import GRID_SIZE, ROOT, Step, get_neighbors, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    ALL_HOLIDAYS, BOOT_CI, BOOT_N, BOOT_SEED, W1, W2, FeatureStore, build_rows, fit_plain, hotspots,
    lgbm_params, load_json, load_preds, origin_index, predict, training_cell_means,
)
from src.evaluation.margin_tuning import (  # noqa: E402
    DOWNSTREAM_HOLIDAY, DOWNSTREAM_NONHOLIDAY, f1_best, flag_bootstrap, flag_counts, pooled_variant,
    pr_curve, ratio,
)

OUT = ROOT / "data" / "experiments" / "phase4"
P3_END = "2013-12-10"
FRACTIONS = (0.2, 0.3, 0.4)


def _write(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def thresholds(panel, end: str, pct: float) -> np.ndarray:
    e = int(panel.hours.get_loc(pd.Timestamp(end)))
    v = panel.activity[:, :e]
    q = pd.Series(v.ravel()).groupby(np.repeat(panel.cells, e)).quantile(pct / 100)
    return q.reindex(panel.cells).to_numpy()


def neighbour_fraction(flags: np.ndarray) -> np.ndarray:
    """flags: (10000, T) bool, cells 1..10000 in order. Fraction of each cell's existing 8-neighbours
    that are flagged, per column (same definition as diagnosis_agent.get_neighbors)."""
    T = flags.shape[1]
    g = flags.reshape(GRID_SIZE, GRID_SIZE, T).astype(np.int16)
    pg = np.pad(g, ((1, 1), (1, 1), (0, 0)))
    ones = np.pad(np.ones((GRID_SIZE, GRID_SIZE), np.int16), 1)
    cnt = np.zeros_like(g)
    nb = np.zeros((GRID_SIZE, GRID_SIZE), np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                cnt += pg[1 + dy:1 + dy + GRID_SIZE, 1 + dx:1 + dx + GRID_SIZE]
                nb += ones[1 + dy:1 + dy + GRID_SIZE, 1 + dx:1 + dx + GRID_SIZE]
    return (cnt / nb[:, :, None]).reshape(GRID_SIZE * GRID_SIZE, T)


def _verify_fraction(flags, frac, rng, n=300):
    T = flags.shape[1]
    for _ in range(n):
        i, t = int(rng.integers(0, flags.shape[0])), int(rng.integers(0, T))
        nb = get_neighbors(i + 1)
        expect = sum(flags[n_ - 1, t] for n_ in nb) / len(nb)
        assert abs(frac[i, t] - expect) < 1e-12, (i + 1, t)


class Test1h:
    """+1h test window: forecasts F, actuals Y (cells x hours), target times."""

    def __init__(self, panel, store, cm, variant=None):
        t_idx = origin_index(panel, W1, "test", 1)
        _, y, meta = build_rows(store, 1, t_idx, np.arange(len(panel.cells)), cm)
        self.n_t = len(t_idx)
        self.F = load_preds("w1", variant or pooled_variant(), 1, "test").astype(np.float64).reshape(len(panel.cells), self.n_t)
        self.Y = y.reshape(len(panel.cells), self.n_t)
        self.targets = pd.DatetimeIndex(meta["target_time"][:self.n_t])
        self.dates = self.targets.date
        self.holiday = np.isin(self.dates, list(ALL_HOLIDAYS))


def solve_and_replay(cells, fc, actual, thr, receive_mask):
    """Unchanged solver: deficit/spare from the forecast at m = 1; receivers = receive_mask.
    Returns per-cell received / given arrays, solver seconds."""
    from src.optimization.solver import solve_hour
    deficit, spare = np.clip(fc - thr, 0, None), np.clip(thr - fc, 0, None)
    t0 = time.perf_counter()
    sol = solve_hour(pd.DataFrame({"CellID": cells[receive_mask], "deficit": deficit[receive_mask]}),
                     pd.DataFrame({"CellID": cells, "spare": spare}))
    secs = time.perf_counter() - t0
    recv, give = np.zeros(len(cells)), np.zeros(len(cells))
    for (c, n), v in sol.items():
        recv[int(c) - 1] += v
        give[int(n) - 1] += v
    return recv, give, secs, deficit


def outcome(actual, thr, recv, give):
    before = np.clip(actual - thr, 0, None)
    after = np.clip(actual - recv + give - thr, 0, None)
    exc = actual > thr
    is_recv, is_don = recv > 1e-9, give > 1e-9
    harm = np.clip(actual + give - thr, 0, None) - before
    return {"excess_before": before.sum(), "net_reduction": before.sum() - after.sum(),
            "gross_relief": np.minimum(recv, before)[exc & is_recv].sum(), "moved": recv.sum(),
            "moved_to_non_exceeders": recv[is_recv & ~exc].sum(), "donor_harm": harm[is_don].sum(),
            "donors_pushed_over": int((is_don & ~exc & (actual + give > thr)).sum()),
            "donors_already_above": int((is_don & exc).sum())}


# ============================================================================= ranges
def step_ranges():
    rows = []
    c = pd.read_csv(ROOT / "data" / "experiments" / "phase3" / "pr_curves.csv")
    c = c[(c.forecaster == "final pooled") & (c.segment == "all")]
    for h in (1, 2, 3, 4):
        v = c[(c.horizon == h) & (c.split == "val")].set_index("margin")["f1"]
        t = c[(c.horizon == h) & (c.split == "test")].set_index("margin")["f1"]
        rows += _plateau("w1 (tune Dec 10-16, test Dec 17-Jan 1)", h, v, t)
    panel = load_panel()
    store = FeatureStore(panel)
    cm2 = training_cell_means(panel, W2)
    thr2 = thresholds(panel, "2013-11-18", 90)
    for h in (1, 4):
        curves = {}
        for split in ("val", "test"):
            ti = origin_index(panel, W2, split, h)
            _, y, _ = build_rows(store, h, ti, np.arange(len(panel.cells)), cm2)
            thr_rows = np.repeat(thr2, len(ti))
            f = load_preds("w2", "final_63l", h, split).astype(np.float64)
            curves[split] = pr_curve(ratio(f, thr_rows), y > thr_rows).set_index("margin")["f1"]
        rows += _plateau("w2 (tune Nov 18-24, test Nov 25-Dec 7)", h, curves["val"], curves["test"])
    res = pd.DataFrame(rows)
    _write(res, "margin_plateaus.csv")
    pd.set_option("display.width", 220)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _plateau(window, h, val_f1, test_f1):
    best_m = f1_best(pd.DataFrame({"margin": val_f1.index.to_numpy(), "f1": val_f1.to_numpy()}))
    best = val_f1.max()
    out = []
    for tol in (0.002, 0.005):
        ms = val_f1.index[val_f1 >= best - tol]
        tf = test_f1.loc[ms]
        out.append({"window": window, "horizon": h, "tolerance": tol, "val_best_margin": best_m,
                    "val_best_f1": best, "margin_lo": ms.min(), "margin_hi": ms.max(), "n_margins": len(ms),
                    "test_f1_min": tf.min(), "test_f1_max": tf.max(), "test_f1_at_val_best": test_f1.loc[best_m],
                    "test_best_margin": float(test_f1.idxmax()), "test_best_f1": test_f1.max(),
                    "test_f1_at_m1": test_f1.loc[1.0]})
    return out


# ============================================================================= (a) rule
def step_rule():
    from src.diagnosis.diagnosis_agent import NEIGHBOR_FRACTION_THRESHOLD
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    thr = thresholds(panel, P3_END, 90)
    d = Test1h(panel, store, cm)
    flags = d.F > thr[:, None]
    frac = neighbour_fraction(flags)
    rng = np.random.default_rng(BOOT_SEED)
    _verify_fraction(flags, frac, rng)
    actual = d.Y > thr[:, None]
    excess = np.clip(d.Y - thr[:, None], 0, None)
    nonhol = ~d.holiday
    _, day_idx = np.unique(d.dates, return_inverse=True)
    print(f"current rule threshold in diagnosis_agent: {NEIGHBOR_FRACTION_THRESHOLD}")

    rows, boots = [], []
    for fth in FRACTIONS:
        routine = flags & (frac >= fth)
        anomalous = flags & ~(frac >= fth)
        for name, m in (("ROUTINE", routine), ("ANOMALOUS", anomalous)):
            mm = m[:, nonhol]
            rows.append({"fraction": fth, "class": name, "flags": int(mm.sum()),
                         "share_of_flags": mm.sum() / flags[:, nonhol].sum(),
                         "precision": actual[:, nonhol][mm].mean(),
                         "mean_actual_excess": excess[:, nonhol][mm].mean(),
                         "mean_excess_given_exceeded": excess[:, nonhol][mm & actual[:, nonhol]].mean()})
        # paired day bootstrap: precision(ROUTINE) - precision(ANOMALOUS), non-holiday days
        days = np.unique(day_idx[nonhol])
        def per_day(m):
            hit = np.bincount(day_idx, (m & actual).sum(axis=0), minlength=day_idx.max() + 1)[days]
            n = np.bincount(day_idx, m.sum(axis=0), minlength=day_idx.max() + 1)[days]
            return hit, n
        rh, rn = per_day(routine)
        ah, an = per_day(anomalous)
        draws = rng.integers(0, len(days), size=(BOOT_N, len(days)))
        diff = rh[draws].sum(1) / rn[draws].sum(1) - ah[draws].sum(1) / an[draws].sum(1)
        point = rh.sum() / rn.sum() - ah.sum() / an.sum()
        lo, hi = np.percentile(diff, BOOT_CI)
        boots.append({"fraction": fth, "precision_routine_minus_anomalous": point, "lo": lo, "hi": hi,
                      "verdict": "ROUTINE better" if lo > 0 else "ROUTINE worse" if hi < 0 else "n.s.", "days": len(days)})

    # Where moves go, on the 24 downstream hours
    cells = panel.cells
    moves = []
    for target in DOWNSTREAM_NONHOLIDAY:
        k = int(np.where(d.targets == pd.Timestamp(target))[0][0])
        fc, act = d.F[:, k], d.Y[:, k]
        fl = fc > thr
        # counterfactual: every flag routed to the solver
        recv_all, _, _, _ = solve_and_replay(cells, fc, act, thr, fl)
        for fth in FRACTIONS:
            r = fl & (frac[:, k] >= fth)
            recv_rule, give_rule, secs, _ = solve_and_replay(cells, fc, act, thr, r)
            exc = act > thr
            moves.append({"target": target, "fraction": fth,
                          "rule_moved": recv_rule.sum(), "rule_moved_to_non_exceeders": recv_rule[~exc].sum(),
                          "all_moved_to_routine_class": recv_all[r].sum(),
                          "all_moved_to_routine_class_non_exceeders": recv_all[r & ~exc].sum(),
                          "all_moved_to_anomalous_class": recv_all[fl & ~r].sum(),
                          "all_moved_to_anomalous_class_non_exceeders": recv_all[fl & ~r & ~exc].sum(),
                          **{f"rule_{k2}": v for k2, v in outcome(act, thr, recv_rule, give_rule).items()},
                          "solver_seconds": secs})
    mv = pd.DataFrame(moves).groupby("fraction").sum(numeric_only=True).reset_index()
    mv["rule_share_to_non_exceeders"] = mv["rule_moved_to_non_exceeders"] / mv["rule_moved"]
    mv["cf_share_non_exceeders_routine_class"] = mv["all_moved_to_routine_class_non_exceeders"] / mv["all_moved_to_routine_class"]
    mv["cf_share_non_exceeders_anomalous_class"] = mv["all_moved_to_anomalous_class_non_exceeders"] / mv["all_moved_to_anomalous_class"]
    _write(pd.DataFrame(rows), "rule_precision.csv")
    _write(pd.DataFrame(boots), "rule_bootstrap.csv")
    _write(mv, "rule_moves.csv")
    pd.set_option("display.width", 250)
    f = lambda v: f"{v:,.3f}"  # noqa: E731
    print(pd.DataFrame(rows).to_string(index=False, float_format=f))
    print(pd.DataFrame(boots).to_string(index=False, float_format=f))
    print(mv[["fraction", "rule_moved", "rule_share_to_non_exceeders", "cf_share_non_exceeders_routine_class",
              "cf_share_non_exceeders_anomalous_class", "all_moved_to_anomalous_class", "rule_net_reduction",
              "rule_gross_relief", "rule_donor_harm"]].to_string(index=False, float_format=f))


# ============================================================================= (b) holiday
def step_holiday():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    thr = thresholds(panel, P3_END, 90)
    d = Test1h(panel, store, cm)
    flags = d.F > thr[:, None]
    frac = neighbour_fraction(flags)
    actual = d.Y > thr[:, None]
    excess = np.clip(d.Y - thr[:, None], 0, None)
    rows = []
    # Final model, every test hour on Dec 25, Dec 26, Jan 1 (exemption makes all ROUTINE today)
    for day in ("2013-12-25", "2013-12-26", "2014-01-01"):
        cols = d.dates == pd.Timestamp(day).date()
        fl, fr = flags[:, cols], frac[:, cols]
        change = fl & (fr < 0.3)
        stay = fl & (fr >= 0.3)
        for name, m in (("would become ANOMALOUS", change), ("stays ROUTINE", stay)):
            rows.append({"source": "final model, +1h, Phase 3 thresholds", "day": day, "hours": int(cols.sum()),
                         "group": name, "flags": int(m.sum()), "precision": actual[:, cols][m].mean() if m.any() else np.nan,
                         "mean_actual_excess": excess[:, cols][m].mean() if m.any() else np.nan})
    # Existing diagnosis output (v1 in-sample forecasts, full-period thresholds), Dec 25-26, read-only
    diag = pd.read_parquet(ROOT / "data" / "raw" / "diagnosis_results_1h.parquet",
                           columns=["CellID", "target_datetime_1h", "congestion_threshold"])
    diag["t"] = pd.to_datetime(diag["target_datetime_1h"])
    sub = diag[diag["t"].dt.date.isin([pd.Timestamp("2013-12-25").date(), pd.Timestamp("2013-12-26").date()])].copy()
    at_risk = sub.groupby("t")["CellID"].apply(set).to_dict()
    sub["frac"] = [sum(n in at_risk[t] for n in get_neighbors(int(c))) / len(get_neighbors(int(c)))
                   for c, t in zip(sub["CellID"], sub["t"])]
    hidx = panel.hours.get_indexer(sub["t"])
    sub["actual"] = panel.activity[sub["CellID"].to_numpy() - 1, hidx]
    sub["exceeded"] = sub["actual"] > sub["congestion_threshold"]
    sub["excess"] = np.clip(sub["actual"] - sub["congestion_threshold"], 0, None)
    for day, g in sub.groupby(sub["t"].dt.date):
        for name, m in (("would become ANOMALOUS", g["frac"] < 0.3), ("stays ROUTINE", g["frac"] >= 0.3)):
            rows.append({"source": "diagnosis_results_1h.parquet (v1 in-sample, full-period thresholds)", "day": str(day),
                         "hours": g["t"].nunique(), "group": name, "flags": int(m.sum()),
                         "precision": g.loc[m, "exceeded"].mean(), "mean_actual_excess": g.loc[m, "excess"].mean()})
    # 6 holiday downstream hours: moves under the exemption, split by group
    cells = panel.cells
    mv = []
    for target in DOWNSTREAM_HOLIDAY:
        k = int(np.where(d.targets == pd.Timestamp(target))[0][0])
        fc, act = d.F[:, k], d.Y[:, k]
        fl = fc > thr
        change = fl & (frac[:, k] < 0.3)
        recv, give, secs, _ = solve_and_replay(cells, fc, act, thr, fl)        # exemption: all flags ROUTINE
        recv_off, give_off, _, _ = solve_and_replay(cells, fc, act, thr, fl & ~change)  # exemption off
        exc = act > thr
        mv.append({"target": target, "flags": int(fl.sum()), "change": int(change.sum()),
                   "moved_to_change_group": recv[change].sum(), "moved_to_change_group_non_exceeders": recv[change & ~exc].sum(),
                   "moved_to_stay_group": recv[fl & ~change].sum(), "moved_to_stay_group_non_exceeders": recv[fl & ~change & ~exc].sum(),
                   **{f"on_{k2}": v for k2, v in outcome(act, thr, recv, give).items()},
                   **{f"off_{k2}": v for k2, v in outcome(act, thr, recv_off, give_off).items()}})
    mvd = pd.DataFrame(mv)
    tot = mvd.sum(numeric_only=True)
    res = pd.DataFrame(rows)
    _write(res, "holiday_groups.csv")
    _write(mvd, "holiday_moves.csv")
    pd.set_option("display.width", 250)
    print(res.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(tot.to_string(float_format=lambda v: f"{v:,.1f}"))


# ============================================================================= (c) percentile
def step_pct():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    d = Test1h(panel, store, cm)
    cells = panel.cells
    rows, flagrows = [], []
    for pct in (85, 90, 95):
        thr = thresholds(panel, P3_END, pct)
        flags = d.F > thr[:, None]
        actual = d.Y > thr[:, None]
        frac = neighbour_fraction(flags)
        flagrows.append({"percentile": pct, **flag_counts(flags.ravel(), actual.ravel())})
        for target in DOWNSTREAM_NONHOLIDAY:
            k = int(np.where(d.targets == pd.Timestamp(target))[0][0])
            fc, act = d.F[:, k], d.Y[:, k]
            fl = fc > thr
            routine = fl & (frac[:, k] >= 0.3)
            recv, give, secs, deficit = solve_and_replay(cells, fc, act, thr, routine)
            covered = np.minimum(recv, deficit)
            rows.append({"percentile": pct, "target": target, "flagged": int(fl.sum()), "routine": int(routine.sum()),
                         "anomalous": int((fl & ~routine).sum()), "actual_exceeding": int((act > thr).sum()),
                         "forecast_deficit": deficit[routine].sum(), "covered": covered[routine].sum(),
                         "fully_resolved": int((routine & (recv >= deficit - 1e-6)).sum()),
                         "solver_seconds": secs, **outcome(act, thr, recv, give)})
    res = pd.DataFrame(rows)
    agg = res.groupby("percentile").sum(numeric_only=True).reset_index()
    agg["coverage"] = agg["covered"] / agg["forecast_deficit"]
    agg["fully_resolved_share"] = agg["fully_resolved"] / agg["routine"]
    agg["routine_share"] = agg["routine"] / agg["flagged"]
    agg["net_reduction_share"] = agg["net_reduction"] / agg["excess_before"]
    agg["gross_relief_share"] = agg["gross_relief"] / agg["excess_before"]
    agg["donor_harm_share"] = agg["donor_harm"] / agg["excess_before"]
    agg["share_moved_to_non_exceeders"] = agg["moved_to_non_exceeders"] / agg["moved"]
    _write(res, "percentile_by_hour.csv")
    _write(agg, "percentile_summary.csv")
    _write(pd.DataFrame(flagrows), "percentile_flags_full_window.csv")
    pd.set_option("display.width", 260)
    print(pd.DataFrame(flagrows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(agg.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


# ============================================================================= (d) hotspot cutoff
def step_hotcut():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    thr = thresholds(panel, P3_END, 90)
    groups = tuple(load_json("final_features.json")["final_groups"])
    size = pd.read_csv(ROOT / "data" / "experiments" / "phase2" / "pre_phase3" / "hotspot_only_final_size.csv")
    sel = size[(size.horizon == 1) & (size.selected == True)].iloc[0]  # noqa: E712
    leaves, trees = int(sel.num_leaves), int(sel.chosen_trees)
    print(f"Frozen size from the 2% final-feature hotspot model at +1h: {leaves} leaves, {trees} trees")
    order = cm.sort_values(ascending=False, kind="stable").index.to_numpy()
    te = origin_index(panel, W1, "test", 1)
    tr = origin_index(panel, W1, "train", 1)
    pooled = load_preds("w1", pooled_variant(), 1, "test").astype(np.float64).reshape(len(panel.cells), len(te))
    rng = np.random.default_rng(BOOT_SEED)
    rows, timing = [], []
    for share in (0.01, 0.02, 0.05):
        n = int(round(len(panel.cells) * share))
        hot = np.sort(order[:n])
        hidx = np.searchsorted(panel.cells, hot)
        X, y, _ = build_rows(store, 1, tr, hidx, cm, groups=groups)
        with Step(f"hotspot-only +1h, top {n} cells") as s:
            model, fit_s = fit_plain(X, y, lgbm_params(leaves, trees))
        Xt, yt, meta = build_rows(store, 1, te, hidx, cm, groups=groups)
        p_hot = predict(model, Xt).astype(np.float64)
        p_pool = pooled[hidx].ravel()
        thr_rows = np.repeat(thr[hidx], len(te))
        act = yt > thr_rows
        dates = pd.DatetimeIndex(meta["target_time"]).date
        _, day_idx = np.unique(dates, return_inverse=True)
        fl = {"pooled": p_pool > thr_rows, "hotspot-only": p_hot > thr_rows}
        b = flag_bootstrap(fl, "pooled", act, day_idx, np.ones(len(yt), bool), rng)["hotspot-only"]
        # MAE bootstrap (paired days)
        days = np.unique(day_idx)
        ae_h = np.bincount(day_idx, np.abs(yt - p_hot))[days]
        ae_p = np.bincount(day_idx, np.abs(yt - p_pool))[days]
        cnt = np.bincount(day_idx)[days]
        draws = rng.integers(0, len(days), size=(BOOT_N, len(days)))
        dm = ae_h[draws].sum(1) / cnt[draws].sum(1) - ae_p[draws].sum(1) / cnt[draws].sum(1)
        lo, hi = np.percentile(dm, BOOT_CI)
        mh, mp = np.abs(yt - p_hot).mean(), np.abs(yt - p_pool).mean()
        rows.append({"cutoff": f"{share:.0%}", "cells": n, "train_rows": len(y), "test_cell_hours": len(yt),
                     "actual_share": act.mean(),
                     "f1_pooled": flag_counts(fl["pooled"], act)["f1"], "f1_hotspot_only": flag_counts(fl["hotspot-only"], act)["f1"],
                     "f1_diff": b[0], "f1_lo": b[1], "f1_hi": b[2], "f1_verdict": b[3],
                     "mae_pooled": mp, "mae_hotspot_only": mh, "mae_diff": mh - mp, "mae_lo": lo, "mae_hi": hi,
                     "mae_verdict": "hotspot-only better" if hi < 0 else "hotspot-only worse" if lo > 0 else "n.s.",
                     "fit_seconds": fit_s, "peak_commit_gb": s.peak_commit})
        del X, Xt, model
        gc.collect()
    res = pd.DataFrame(rows)
    _write(res, "hotspot_cutoff.csv")
    pd.set_option("display.width", 260)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    steps = {"ranges": step_ranges, "rule": step_rule, "holiday": step_holiday, "pct": step_pct, "hotcut": step_hotcut}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
