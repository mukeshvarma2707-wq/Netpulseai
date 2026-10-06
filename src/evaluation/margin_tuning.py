"""
src/evaluation/margin_tuning.py  (Phase 3: flagging margin)

Flag a cell-hour when  forecast > m x threshold, with m chosen per horizon on
the window-1 VALIDATION slice (targets 2013-12-10 .. 12-16; the second and
final use of that week, the first being model size in Phase 2), and report
on the test window (origins 2013-12-17 00:00 .. 2014-01-01 19:00).

Steps:
  flags       thresholds, base rates, precision-recall curves, tuned margins for every
              strategy, test results with paired day-level bootstrap, holiday slices
  downstream  advisory vs margin-adjusted deficit/spare vs "flagged cells cannot donate",
              on a fixed subset of hours, through the existing diagnosis rules and solver

PRE-DECLARED choices (see docs/BALANCEGRID_IMPROVEMENTS.md, Phase 3):
  - Thresholds (Phase 3 definition, used for tuning AND testing): per-cell 90th percentile of
    total_activity over every hour before the validation week (2013-11-01 00:00 .. 12-09 23:00).
  - Margin grid 0.600 .. 1.400 step 0.005; F1 ties -> margin closest to 1.
  - Recall target r: the LARGEST m whose validation recall >= r (highest precision at that recall).
  - S1: one global m maximising validation F1 on all cells.
    S2: separate m for hotspot and typical cells, each maximising its own segment's F1.
  - Material base-rate shift: relative difference > 20% between validation and test.

RUN:
    python src/evaluation/margin_tuning.py flags
    python src/evaluation/margin_tuning.py downstream
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
from src.evaluation.common import HORIZONS, ROOT, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    ALL_HOLIDAYS, BOOT_CI, BOOT_N, BOOT_SEED, INDICATIVE_DAYS, MODEL_DIR, W1, FeatureStore, build_rows,
    eval_thresholds, hotspots, load_preds, origin_index, training_cell_means,
)
from src.evaluation.common import CONGESTION_PERCENTILE  # noqa: E402

OUT = ROOT / "data" / "experiments" / "phase3"
MARGINS = np.round(np.arange(0.600, 1.400 + 1e-9, 0.005), 3)
RECALL_TARGETS = (0.5, 0.6, 0.7)
MATERIAL_SHIFT = 0.20
P3_THRESHOLD_END = "2013-12-10"     # thresholds from activity strictly before the validation week

# Model choices settled before Phase 3 (user decisions after the pre-Phase-3 checks):
#  - pooled model: per pre_phase3/leafrule_decision.json ("final" at 31 leaves or "final_63l")
#  - hotspot model per horizon by validation loss: final-feature model at +1h, old-feature at +2h..+4h
HOTSPOT_VARIANT = {1: "hotspot_only_final", 2: "hotspot_only", 3: "hotspot_only", 4: "hotspot_only"}


def pooled_variant() -> str:
    with open(ROOT / "data" / "experiments" / "phase2" / "pre_phase3" / "leafrule_decision.json") as fh:
        return json.load(fh)["final_variant"]


# Downstream subset (target hours, +1h solver)
DOWNSTREAM_NONHOLIDAY = [f"{d} {h:02d}:00" for d in ("2013-12-18", "2013-12-20", "2013-12-21", "2013-12-29")
                         for h in (3, 8, 12, 15, 18, 21)]
# 12/15/18:00: the last +1h test target is 2014-01-01 20:00, so later slots do not exist on Jan 1.
DOWNSTREAM_HOLIDAY = [f"{d} {h:02d}:00" for d in ("2013-12-25", "2014-01-01") for h in (12, 15, 18)]


# ============================================================================= helpers
def thresholds_before(panel, end: str) -> np.ndarray:
    e = int(panel.hours.get_loc(pd.Timestamp(end)))
    idx = np.arange(0, e)
    q = pd.Series(panel.activity[:, idx].ravel()).groupby(np.repeat(panel.cells, len(idx))).quantile(CONGESTION_PERCENTILE / 100)
    return q.reindex(panel.cells).to_numpy()


def ratio(f: np.ndarray, thr_rows: np.ndarray) -> np.ndarray:
    """r = forecast / threshold, so that 'flag at margin m' == r > m. A zero threshold flags any
    positive forecast at every margin (r = +inf) and never a zero forecast (r = -inf)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = f / thr_rows
    zero = thr_rows <= 0
    r[zero] = np.where(f[zero] > 0, np.inf, -np.inf)
    return r


def pr_curve(r: np.ndarray, actual: np.ndarray, margins=MARGINS) -> pd.DataFrame:
    """Sort-based counts for every margin at once: flagged(m) = #{r > m}, TP(m) = #{r > m & actual}."""
    rs = np.sort(r)
    ra = np.sort(r[actual])
    n, p = len(r), int(actual.sum())
    flagged = n - np.searchsorted(rs, margins, side="right")
    tp = p - np.searchsorted(ra, margins, side="right")
    fp, fn = flagged - tp, p - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, tp / flagged, np.nan)
        recall = tp / p if p else np.full(len(margins), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return pd.DataFrame({"margin": margins, "precision": precision, "recall": recall, "f1": f1,
                         "tp": tp, "fp": fp, "fn": fn, "flagged_share": flagged / n, "actual_share": p / n})


def verify_pr(r, actual, check_margins=(0.6, 0.8, 0.9, 1.0, 1.1, 1.4)):
    cur = pr_curve(r, actual, np.array(check_margins)).set_index("margin")
    for m in check_margins:
        flag = r > m
        tp = int((flag & actual).sum())
        assert tp == cur.loc[m, "tp"] and int(flag.sum()) == cur.loc[m, "tp"] + cur.loc[m, "fp"], m


def f1_best(curve: pd.DataFrame) -> float:
    best = curve["f1"].max()
    cand = curve[np.isclose(curve["f1"], best)]
    return float(cand.iloc[np.argmin(np.abs(cand["margin"] - 1.0))]["margin"])


def margin_for_recall(curve: pd.DataFrame, target: float) -> float | None:
    ok = curve[curve["recall"] >= target]
    return float(ok["margin"].max()) if len(ok) else None


def flag_counts(flags, actual):
    tp = int((flags & actual).sum()); fp = int((flags & ~actual).sum()); fn = int((~flags & actual).sum())
    n = len(flags)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                flagged_share=(tp + fp) / n, actual_share=(tp + fn) / n)


def flag_bootstrap(flag_sets: dict, reference: str, actual, day_index, mask, rng):
    """Paired day-level bootstrap of F1(strategy) - F1(reference) on the rows in `mask`."""
    d = day_index[mask]
    days = np.unique(d)
    nd = day_index.max() + 1
    draws = rng.integers(0, len(days), size=(BOOT_N, len(days)))
    a = actual[mask]

    def parts(fl):
        f = fl[mask]
        return [np.bincount(d, v, minlength=nd)[days] for v in (a & f, ~a & f, a & ~f)]

    def f1(pp, idx=None):
        tp, fp, fn = (x.sum() if idx is None else x[idx].sum(axis=1) for x in pp)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)

    ref = parts(flag_sets[reference])
    f_ref, f_ref_b = f1(ref), f1(ref, draws)
    out = {}
    for name, fl in flag_sets.items():
        if name == reference:
            continue
        pp = parts(fl)
        diff_b = f1(pp, draws) - f_ref_b
        lo, hi = np.nanpercentile(diff_b, BOOT_CI)
        out[name] = (float(f1(pp) - f_ref), float(lo), float(hi),
                     "better" if lo > 0 else "worse" if hi < 0 else "n.s.")
    return out


class Data:
    """Truth, thresholds and every forecaster's predictions for one horizon and split."""

    def __init__(self, panel, store, cm, hot, thr, h: int, split: str):
        t = origin_index(panel, W1, split, h)
        X, y, meta = build_rows(store, h, t, np.arange(len(panel.cells)), cm, groups=("G1",))
        self.y = y
        self.n_t = len(t)
        self.cell = meta["cell"]
        self.target = pd.DatetimeIndex(meta["target_time"])
        self.thr_rows = np.repeat(thr, self.n_t)
        self.actual = y > self.thr_rows
        self.is_hot = np.isin(self.cell, hot)
        self.holiday = np.isin(self.target.date, list(ALL_HOLIDAYS))
        self.weekend = self.target.dayofweek.to_numpy() >= 5
        pos = np.searchsorted(panel.cells, np.sort(hot))
        final = load_preds("w1", pooled_variant(), h, split).astype(np.float64)
        comp = final.copy()
        comp.reshape(len(panel.cells), self.n_t)[pos] = load_preds("w1", HOTSPOT_VARIANT[h], h, split).reshape(len(pos), self.n_t)
        self.fc = {"seasonal-naive": meta["seasonal_naive"], "weekly-naive": X["weekly_naive"].to_numpy(np.float64),
                   "final pooled": final, "pooled+hotspot": comp}
        del X


# ============================================================================= flags
def step_flags():
    OUT.mkdir(parents=True, exist_ok=True)
    timing = {}
    with Step("load panel + features"):
        panel = load_panel()
        store = FeatureStore(panel)
        cm = training_cell_means(panel, W1)
    hot = hotspots(cm)
    thr_p3 = thresholds_before(panel, P3_THRESHOLD_END)
    thr_p2 = eval_thresholds(panel, W1)
    rat = thr_p3 / thr_p2
    thr_info = {"definition": "per-cell 90th percentile of total_activity, 2013-11-01 00:00 .. 2013-12-09 23:00",
                "ratio_p3_over_p2_median": float(np.median(rat)), "ratio_p5": float(np.percentile(rat, 5)),
                "ratio_p95": float(np.percentile(rat, 95)), "ratio_min": float(rat.min()), "ratio_max": float(rat.max()),
                "cells_with_zero_threshold_p3": int((thr_p3 <= 0).sum())}
    print("THRESHOLDS:", thr_info)

    rng = np.random.default_rng(BOOT_SEED)
    curves, choices, tests, boots, base_rates, p2_ref = [], [], [], [], [], []
    for h in HORIZONS:
        val = Data(panel, store, cm, hot, thr_p3, h, "val")
        test = Data(panel, store, cm, hot, thr_p3, h, "test")
        for split, d in (("val", val), ("test", test)):
            for seg, m in (("all", slice(None)), ("hotspot", d.is_hot), ("typical", ~d.is_hot)):
                base_rates.append({"horizon": h, "split": split, "segment": seg, "actual_share": float(d.actual[m].mean())})

        # ---- PR curves on validation (for choosing) and test (for reporting)
        t0 = time.perf_counter()
        margins = {}
        for name, f in val.fc.items():
            for seg, m in (("all", np.ones(len(val.y), bool)), ("hotspot", val.is_hot), ("typical", ~val.is_hot)):
                r = ratio(f[m], val.thr_rows[m])
                if h == 1 and seg == "all":
                    verify_pr(r, val.actual[m])
                c = pr_curve(r, val.actual[m])
                c.insert(0, "segment", seg); c.insert(0, "forecaster", name); c.insert(0, "split", "val"); c.insert(0, "horizon", h)
                curves.append(c)
                margins[(name, seg, "f1")] = f1_best(c)
                for rt in RECALL_TARGETS:
                    margins[(name, seg, f"recall{rt}")] = margin_for_recall(c, rt)
        timing[f"+{h}h tuning (all curves on validation)"] = time.perf_counter() - t0
        for name, f in test.fc.items():
            for seg, m in (("all", np.ones(len(test.y), bool)), ("hotspot", test.is_hot), ("typical", ~test.is_hot)):
                c = pr_curve(ratio(f[m], test.thr_rows[m]), test.actual[m])
                c.insert(0, "segment", seg); c.insert(0, "forecaster", name); c.insert(0, "split", "test"); c.insert(0, "horizon", h)
                curves.append(c)
        for (name, seg, kind), m in margins.items():
            choices.append({"horizon": h, "forecaster": name, "segment": seg, "kind": kind, "margin": m})

        # ---- strategies -> per-row margin arrays on TEST
        def per_row(name, m_all=None, m_hot=None, m_typ=None):
            if m_all is not None:
                return np.full(len(test.y), m_all)
            return np.where(test.is_hot, m_hot, m_typ)

        strategies = {
            "N0 seasonal-naive m=1": ("seasonal-naive", per_row(None, 1.0)),
            "N1 seasonal-naive global m": ("seasonal-naive", per_row(None, margins[("seasonal-naive", "all", "f1")])),
            "W0 weekly-naive m=1": ("weekly-naive", per_row(None, 1.0)),
            "W1 weekly-naive global m": ("weekly-naive", per_row(None, margins[("weekly-naive", "all", "f1")])),
            "P0 final pooled m=1": ("final pooled", per_row(None, 1.0)),
            "P1 final pooled global m": ("final pooled", per_row(None, margins[("final pooled", "all", "f1")])),
            "P2 final pooled hot/typ m": ("final pooled", per_row(None, None, margins[("final pooled", "hotspot", "f1")],
                                                                margins[("final pooled", "typical", "f1")])),
            "H0 pooled+hotspot m=1": ("pooled+hotspot", per_row(None, 1.0)),
            "H1 pooled+hotspot global m": ("pooled+hotspot", per_row(None, margins[("pooled+hotspot", "all", "f1")])),
            "H2 pooled+hotspot hot/typ m": ("pooled+hotspot", per_row(None, None, margins[("pooled+hotspot", "hotspot", "f1")],
                                                                    margins[("pooled+hotspot", "typical", "f1")])),
        }
        for rt in RECALL_TARGETS:
            mr = margins[("final pooled", "all", f"recall{rt}")]
            if mr is not None:
                strategies[f"P recall>={rt} m={mr:.3f}"] = ("final pooled", per_row(None, mr))
        flag_sets = {s: test.fc[f] > m_rows * test.thr_rows for s, (f, m_rows) in strategies.items()}

        dates = test.target.date
        _, day_index = np.unique(dates, return_inverse=True)
        slices = {"all": np.ones(len(test.y), bool), "hotspot": test.is_hot, "typical": ~test.is_hot,
                  "non-holiday": ~test.holiday, "holiday": test.holiday, "weekday": ~test.weekend, "weekend": test.weekend}
        for s, fl in flag_sets.items():
            fname, m_rows = strategies[s]
            for sl, msk in slices.items():
                nd = len(np.unique(dates[msk]))
                tests.append({"horizon": h, "strategy": s, "forecaster": fname,
                              "margin_typical": float(np.unique(m_rows[~test.is_hot])[0]),
                              "margin_hotspot": float(np.unique(m_rows[test.is_hot])[0]),
                              "slice": sl, "days": nd, "indicative": nd < INDICATIVE_DAYS,
                              **flag_counts(fl[msk], test.actual[msk])})
        for ref in ("P0 final pooled m=1", "H0 pooled+hotspot m=1", "N1 seasonal-naive global m"):
            for sl in ("all", "hotspot", "non-holiday"):
                for s, (dif, lo, hi, v) in flag_bootstrap(flag_sets, ref, test.actual, day_index, slices[sl], rng).items():
                    boots.append({"horizon": h, "vs": ref, "strategy": s, "slice": sl, "f1_diff": dif, "lo": lo, "hi": hi, "verdict": v})

        # ---- Phase 2 threshold definition, m = 1, for reference
        thr2_rows = np.repeat(thr_p2, test.n_t)
        act2 = test.y > thr2_rows
        for name in ("final pooled", "pooled+hotspot", "seasonal-naive"):
            for sl, msk in (("all", slice(None)), ("hotspot", test.is_hot)):
                p2_ref.append({"horizon": h, "forecaster": name, "slice": sl, "thresholds": "Phase 2 (to Dec 16)",
                               **flag_counts((test.fc[name] > thr2_rows)[msk], act2[msk])})
        del val, test, flag_sets
        gc.collect()

    pd.concat(curves).to_csv(OUT / "pr_curves.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "margins_chosen.csv", index=False)
    res = pd.DataFrame(tests); res.to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame(boots).to_csv(OUT / "test_bootstrap.csv", index=False)
    br = pd.DataFrame(base_rates).pivot_table(index=["horizon", "segment"], columns="split", values="actual_share").reset_index()
    br["relative_diff"] = (br["test"] - br["val"]) / br["val"]
    br["material"] = br["relative_diff"].abs() > MATERIAL_SHIFT
    br.to_csv(OUT / "base_rates.csv", index=False)
    pd.DataFrame(p2_ref).to_csv(OUT / "phase2_threshold_reference.csv", index=False)
    with open(OUT / "thresholds.json", "w") as fh:
        json.dump(thr_info, fh, indent=2)
    with open(OUT / "timing_tuning.json", "w") as fh:
        json.dump(timing, fh, indent=2)

    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print("\nBASE RATES"); print(br.to_string(index=False, float_format=fmt))
    print("\nMARGINS (validation)"); print(pd.DataFrame(choices).pivot_table(index=["forecaster", "segment", "kind"], columns="horizon", values="margin").to_string(float_format=fmt))
    print("\nTEST, all cells"); print(res[res.slice == "all"].pivot_table(index="strategy", columns="horizon", values="f1", sort=False).to_string(float_format=fmt))


# ============================================================================= downstream
def step_downstream():
    from src.diagnosis.diagnosis_agent import KNOWN_HOLIDAYS, NEIGHBOR_FRACTION_THRESHOLD, get_neighbors
    from src.optimization.solver import solve_hour

    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    thr = thresholds_before(panel, P3_THRESHOLD_END)
    choices = pd.read_csv(OUT / "margins_chosen.csv")
    pick = lambda kind: float(choices[(choices.horizon == 1) & (choices.forecaster == "final pooled") &  # noqa: E731
                                      (choices.segment == "all") & (choices.kind == kind)]["margin"].iloc[0])
    margin_set = {"F1-best": pick("f1"), "recall>=0.6": pick("recall0.6")}
    print("Margins used downstream (+1h, final pooled, from validation):", margin_set)

    t_idx = origin_index(panel, W1, "test", 1)
    _, y, meta = build_rows(store, 1, t_idx, np.arange(len(panel.cells)), cm)
    n_t = len(t_idx)
    f = load_preds("w1", pooled_variant(), 1, "test").astype(np.float64).reshape(len(panel.cells), n_t)
    Y = y.reshape(len(panel.cells), n_t)
    targets = pd.DatetimeIndex(meta["target_time"][:n_t])
    neighbors = {int(c): [n for n in get_neighbors(int(c))] for c in panel.cells}

    def run(target: str, m: float, variant: str):
        k = int(np.where(targets == pd.Timestamp(target))[0][0])
        fc, act = f[:, k], Y[:, k] > thr
        flagged = fc > m * thr
        cells = panel.cells
        flagged_set = set(cells[flagged].tolist())
        holiday = pd.Timestamp(target).date() in KNOWN_HOLIDAYS
        routine = np.zeros(len(cells), bool)
        for i in np.where(flagged)[0]:
            if holiday:
                routine[i] = True
                continue
            nb = neighbors[int(cells[i])]
            routine[i] = sum(n in flagged_set for n in nb) / len(nb) >= NEIGHBOR_FRACTION_THRESHOLD
        if variant in ("V0 current", "A advisory"):
            deficit = np.clip(fc - thr, 0, None); spare = np.clip(thr - fc, 0, None)
        elif variant == "B margin deficit+spare":
            deficit = np.clip(fc - m * thr, 0, None); spare = np.clip(m * thr - fc, 0, None)
        elif variant == "C margin deficit, flagged cannot donate":
            deficit = np.clip(fc - m * thr, 0, None); spare = np.where(flagged, 0.0, np.clip(thr - fc, 0, None))
        else:
            raise ValueError(variant)
        cong = pd.DataFrame({"CellID": cells[routine], "deficit": deficit[routine]})
        all_cells = pd.DataFrame({"CellID": cells, "spare": spare})
        t0 = time.perf_counter()
        sol = solve_hour(cong, all_cells)
        solve_s = time.perf_counter() - t0
        covered = pd.Series(0.0, index=cong["CellID"])
        for (c, n), v in sol.items():
            covered[c] += v
        d = cong.set_index("CellID")["deficit"]
        pos_def = d > 1e-9
        return {"target": target, "holiday": holiday, "margin": m, "variant": variant,
                "flagged": int(flagged.sum()), "flagged_actually_exceeding": int((flagged & act).sum()),
                "actual_exceeding": int(act.sum()),
                "routine": int(routine.sum()), "anomalous": int(flagged.sum() - routine.sum()),
                "routine_zero_deficit": int((~pos_def).sum()),
                "total_deficit": float(d.sum()), "covered": float(covered.sum()),
                "fully_resolved_all": int((covered >= d - 1e-6).sum()),
                "fully_resolved_positive_deficit": int(((covered >= d - 1e-6) & pos_def).sum()),
                "routine_positive_deficit": int(pos_def.sum()),
                "solver_seconds": solve_s}

    rows = []
    for subset, hours in (("non-holiday", DOWNSTREAM_NONHOLIDAY), ("holiday", DOWNSTREAM_HOLIDAY)):
        for target in hours:
            rows.append({**run(target, 1.0, "V0 current"), "subset": subset, "margin_name": "m=1"})
            for mname, m in margin_set.items():
                for variant in ("A advisory", "B margin deficit+spare", "C margin deficit, flagged cannot donate"):
                    rows.append({**run(target, m, variant), "subset": subset, "margin_name": mname})
        print(f"  {subset}: {len(hours)} hours done")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "downstream_by_hour.csv", index=False)
    agg = res.groupby(["subset", "margin_name", "variant"], sort=False).agg(
        hours=("target", "nunique"), margin=("margin", "first"), flagged=("flagged", "sum"),
        flagged_actually_exceeding=("flagged_actually_exceeding", "sum"), actual_exceeding=("actual_exceeding", "sum"),
        routine=("routine", "sum"), anomalous=("anomalous", "sum"), routine_zero_deficit=("routine_zero_deficit", "sum"),
        total_deficit=("total_deficit", "sum"), covered=("covered", "sum"), fully_resolved_all=("fully_resolved_all", "sum"),
        fully_resolved_positive_deficit=("fully_resolved_positive_deficit", "sum"),
        routine_positive_deficit=("routine_positive_deficit", "sum"), solver_seconds=("solver_seconds", "sum")).reset_index()
    agg["coverage"] = agg["covered"] / agg["total_deficit"]
    agg["fully_resolved_share"] = agg["fully_resolved_all"] / agg["routine"]
    agg["fully_resolved_share_positive_deficit"] = agg["fully_resolved_positive_deficit"] / agg["routine_positive_deficit"]
    agg["routine_share"] = agg["routine"] / agg["flagged"]
    agg.to_csv(OUT / "downstream_summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    steps = {"flags": step_flags, "downstream": step_downstream}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
