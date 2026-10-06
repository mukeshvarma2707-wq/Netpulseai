"""
src/evaluation/phase4_followup.py  (fixes after Phase 4 review)

  replayboot  Bootstrap of the outcome replay: paired resampling of HOURS (24 non-holiday
              hours; unit = 1 hour) and, as a sensitivity check, of DAYS (4 days). Differences
              in net reduction and donor harm: B - V0, C - V0, C - B at m = 0.96 and 0.98.
  reconcile   Check (a) on exactly the V0 downstream set (24 hours), with flags, TP and
              precision per class summing to the V0 row; hour-level bootstrap on both sets.
  naive       Model vs seasonal-naive at all horizons: (i) "tuned" (31 leaves, production
              features, no feature selection) and (ii) "final_63l" (23 features; optimistic).
              MAE and F1 at m = 1, plus each with its own margin tuned on Dec 10-16 vs
              seasonal-naive with its own tuned margin. Paired day bootstrap.
  reach2      Perfect-foresight solver with 2-hop neighbours (Chebyshev distance <= 2)
              vs 1-hop, on the 24 non-holiday hours. A copy of solver.solve_hour's LP with the
              neighbour set as a parameter (solver.py itself is unchanged); the copy is first
              checked to reproduce the 1-hop result.

RUN: python src/evaluation/phase4_followup.py {replayboot|reconcile|naive|reach2}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import GRID_SIZE, HORIZONS, ROOT, Step, get_neighbors, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    ALL_HOLIDAYS, BOOT_CI, BOOT_N, BOOT_SEED, W1, FeatureStore, build_rows, hotspots, load_preds,
    origin_index, training_cell_means,
)
from src.evaluation.margin_tuning import (  # noqa: E402
    DOWNSTREAM_NONHOLIDAY, f1_best, flag_bootstrap, flag_counts, pooled_variant, pr_curve, ratio,
)
from src.evaluation.phase4_sensitivity import (  # noqa: E402
    P3_END, Test1h, neighbour_fraction, outcome, thresholds,
)

OUT = ROOT / "data" / "experiments" / "phase4" / "followup"


def _write(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def _ci(draws):
    lo, hi = np.percentile(draws, BOOT_CI)
    return lo, hi, ("positive" if lo > 0 else "negative" if hi < 0 else "n.s.")


# ============================================================================= 1. replay bootstrap
def step_replayboot():
    r = pd.read_csv(ROOT / "data" / "experiments" / "phase3" / "checks" / "replay_by_hour.csv")
    r = r[r.subset == "non-holiday"].copy()
    r["day"] = pd.to_datetime(r["target"]).dt.date
    wide_net = r.pivot(index="target", columns="variant", values="excess_reduction")
    wide_harm = r.pivot(index="target", columns="variant", values="donor_harm")
    day_of = r.drop_duplicates("target").set_index("target")["day"].reindex(wide_net.index)
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for m in (0.96, 0.98):
        V0, B, C = "V0 current", f"B margin deficit+spare m={m}", f"C margin deficit, flagged cannot donate m={m}"
        for metric, wide in (("net reduction", wide_net), ("donor harm", wide_harm)):
            for name, a, b in ((f"B - V0", B, V0), (f"C - V0", C, V0), (f"C - B", C, B)):
                diff = (wide[a] - wide[b]).to_numpy()
                point = diff.sum()
                # unit = hour
                idx = rng.integers(0, len(diff), size=(BOOT_N, len(diff)))
                lo, hi, v = _ci(diff[idx].sum(axis=1))
                # sensitivity: unit = day (4 days, 6 hours each)
                per_day = pd.Series(diff, index=day_of.values).groupby(level=0).sum().to_numpy()
                didx = rng.integers(0, len(per_day), size=(BOOT_N, len(per_day)))
                dlo, dhi, dv = _ci(per_day[didx].sum(axis=1))
                rows.append({"margin": m, "metric": metric, "comparison": name, "total_diff": point,
                             "hour_lo": lo, "hour_hi": hi, "hour_verdict": v,
                             "day_lo": dlo, "day_hi": dhi, "day_verdict": dv, "hours": len(diff), "days": len(per_day)})
    res = pd.DataFrame(rows)
    _write(res, "replay_bootstrap.csv")
    pd.set_option("display.width", 220)
    print(res.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))


# ============================================================================= 2. reconcile (a)
def step_reconcile():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    thr = thresholds(panel, P3_END, 90)
    d = Test1h(panel, store, cm)
    flags = d.F > thr[:, None]
    frac = neighbour_fraction(flags)
    actual = d.Y > thr[:, None]
    rng = np.random.default_rng(BOOT_SEED)
    sets = {"24 downstream non-holiday hours (V0 set)": np.isin(d.targets, pd.DatetimeIndex(DOWNSTREAM_NONHOLIDAY)),
            "all non-holiday test hours (13 days)": ~d.holiday}
    rows, boots = [], []
    for sname, cols in sets.items():
        fl, fr, ac = flags[:, cols], frac[:, cols], actual[:, cols]
        for fth in (0.2, 0.3, 0.4):
            r = fl & (fr >= fth)
            a = fl & ~(fr >= fth)
            for cls, m in (("ROUTINE", r), ("ANOMALOUS", a), ("all flags", fl)):
                rows.append({"set": sname, "hours": int(cols.sum()), "fraction": fth, "class": cls,
                             "flags": int(m.sum()), "true_positives": int((m & ac).sum()),
                             "precision": (m & ac).sum() / m.sum()})
            # hour-level paired bootstrap of precision(ROUTINE) - precision(ANOMALOUS)
            rh, rn = (r & ac).sum(0), r.sum(0)
            ah, an = (a & ac).sum(0), a.sum(0)
            idx = rng.integers(0, cols.sum(), size=(BOOT_N, cols.sum()))
            diff = rh[idx].sum(1) / rn[idx].sum(1) - ah[idx].sum(1) / np.maximum(an[idx].sum(1), 1)
            lo, hi, v = _ci(diff)
            boots.append({"set": sname, "fraction": fth, "diff": rh.sum() / rn.sum() - ah.sum() / an.sum(),
                          "lo": lo, "hi": hi, "verdict": v, "unit": "hour"})
    _write(pd.DataFrame(rows), "reconcile_rule.csv")
    _write(pd.DataFrame(boots), "reconcile_rule_bootstrap.csv")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(boots).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ============================================================================= 3. model vs naive
def step_naive():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    hot = set(hotspots(cm).tolist())
    thr = thresholds(panel, P3_END, 90)
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for h in HORIZONS:
        data = {}
        for split in ("val", "test"):
            t = origin_index(panel, W1, split, h)
            _, y, meta = build_rows(store, h, t, np.arange(len(panel.cells)), cm)
            thr_rows = np.repeat(thr, len(t))
            data[split] = dict(y=y, thr=thr_rows, act=y > thr_rows, naive=meta["seasonal_naive"],
                               hot=np.isin(meta["cell"], list(hot)), dates=pd.DatetimeIndex(meta["target_time"]).date,
                               tuned=load_preds("w1", "tuned", h, split).astype(np.float64),
                               final=load_preds("w1", pooled_variant(), h, split).astype(np.float64))
        v, t = data["val"], data["test"]
        # margins tuned on Dec 10-16 (all cells, F1-best), same rule as Phase 3
        m = {k: f1_best(pr_curve(ratio(v[k], v["thr"]), v["act"])) for k in ("naive", "tuned", "final")}
        _, day_idx = np.unique(t["dates"], return_inverse=True)
        days = np.unique(day_idx)
        for model in ("tuned", "final"):
            label = "(i) tuned: 31 leaves, production features" if model == "tuned" else "(ii) final_63l: 23 features (optimistic)"
            for sl, msk in (("all", np.ones(len(t["y"]), bool)), ("hotspot", t["hot"])):
                # MAE, paired day bootstrap
                ae_m = np.bincount(day_idx[msk], np.abs(t["y"] - t[model])[msk])[days]
                ae_n = np.bincount(day_idx[msk], np.abs(t["y"] - t["naive"])[msk])[days]
                n = np.bincount(day_idx[msk])[days]
                idx = rng.integers(0, len(days), size=(BOOT_N, len(days)))
                dm = ae_m[idx].sum(1) / n[idx].sum(1) - ae_n[idx].sum(1) / n[idx].sum(1)
                lo, hi, _ = _ci(dm)
                mae_m, mae_n = ae_m.sum() / n.sum(), ae_n.sum() / n.sum()
                # F1 at m = 1 and with each forecaster's own tuned margin
                fl = {"model m=1": t[model] > t["thr"], "naive m=1": t["naive"] > t["thr"],
                      "model tuned m": t[model] > m[model] * t["thr"], "naive tuned m": t["naive"] > m["naive"] * t["thr"]}
                b1 = flag_bootstrap({"model m=1": fl["model m=1"], "naive m=1": fl["naive m=1"]}, "naive m=1",
                                    t["act"], day_idx, msk, rng)["model m=1"]
                b2 = flag_bootstrap({"model tuned m": fl["model tuned m"], "naive tuned m": fl["naive tuned m"]}, "naive tuned m",
                                    t["act"], day_idx, msk, rng)["model tuned m"]
                rows.append({"model": label, "horizon": h, "slice": sl,
                             "mae_model": mae_m, "mae_naive": mae_n, "mae_diff": mae_m - mae_n, "mae_lo": lo, "mae_hi": hi,
                             "mae_verdict": "model better" if hi < 0 else "model worse" if lo > 0 else "n.s.",
                             "f1_model_m1": flag_counts(fl["model m=1"][msk], t["act"][msk])["f1"],
                             "f1_naive_m1": flag_counts(fl["naive m=1"][msk], t["act"][msk])["f1"],
                             "f1_diff_m1": b1[0], "f1_m1_lo": b1[1], "f1_m1_hi": b1[2], "f1_m1_verdict": b1[3],
                             "margin_model": m[model], "margin_naive": m["naive"],
                             "f1_model_tuned": flag_counts(fl["model tuned m"][msk], t["act"][msk])["f1"],
                             "f1_naive_tuned": flag_counts(fl["naive tuned m"][msk], t["act"][msk])["f1"],
                             "f1_diff_tuned": b2[0], "f1_tuned_lo": b2[1], "f1_tuned_hi": b2[2], "f1_tuned_verdict": b2[3]})
    res = pd.DataFrame(rows)
    _write(res, "model_vs_naive.csv")
    pd.set_option("display.width", 280)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ============================================================================= 6. 2-hop reach
def neighbours_within(cell_id: int, k: int) -> list:
    x, y = (cell_id - 1) % GRID_SIZE + 1, (cell_id - 1) // GRID_SIZE + 1
    return [(yy - 1) * GRID_SIZE + xx for yy in range(y - k, y + k + 1) for xx in range(x - k, x + k + 1)
            if 1 <= xx <= GRID_SIZE and 1 <= yy <= GRID_SIZE and (xx, yy) != (x, y)]


def solve_hour_k(congested: pd.DataFrame, all_cells: pd.DataFrame, neigh) -> dict:
    """Same LP as src/optimization/solver.solve_hour, with the neighbour function as a parameter."""
    from ortools.linear_solver import pywraplp
    solver = pywraplp.Solver.CreateSolver("GLOP")
    spare = all_cells.set_index("CellID")["spare"].to_dict()
    x, usage = {}, {}
    for c, deficit in zip(congested["CellID"], congested["deficit"]):
        cv = []
        for n in neigh(int(c)):
            if spare.get(n, 0) > 0:
                v = solver.NumVar(0, spare[n], f"x_{c}_{n}")
                x[(c, n)] = v
                cv.append(v)
                usage.setdefault(n, []).append(v)
        if cv:
            solver.Add(sum(cv) <= deficit)
    for n, vl in usage.items():
        solver.Add(sum(vl) <= spare[n])
    if not x:
        return {}
    solver.Maximize(sum(x.values()))
    if solver.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {}
    return {k: v.solution_value() for k, v in x.items() if v.solution_value() > 1e-6}


def step_reach2():
    from src.diagnosis.diagnosis_agent import NEIGHBOR_FRACTION_THRESHOLD
    from src.optimization.solver import solve_hour

    panel = load_panel()
    thr = thresholds(panel, P3_END, 90)
    cells = panel.cells
    n1 = {int(c): get_neighbors(int(c)) for c in cells}
    n2 = {int(c): neighbours_within(int(c), 2) for c in cells}
    rows = []
    for target in DOWNSTREAM_NONHOLIDAY:
        act = panel.activity[:, panel.hours.get_loc(pd.Timestamp(target))]
        flagged = act > thr                                         # perfect foresight: actual as forecast
        frac = neighbour_fraction(flagged[:, None])[:, 0]
        routine = flagged & (frac >= NEIGHBOR_FRACTION_THRESHOLD)   # same diagnosis rule (1-hop neighbours)
        cong = pd.DataFrame({"CellID": cells[routine], "deficit": np.clip(act - thr, 0, None)[routine]})
        allc = pd.DataFrame({"CellID": cells, "spare": np.clip(thr - act, 0, None)})
        res = {"target": target}
        for name, fn in (("1-hop original solver", lambda: solve_hour(cong, allc)),
                         ("1-hop copy", lambda: solve_hour_k(cong, allc, n1.__getitem__)),
                         ("2-hop copy", lambda: solve_hour_k(cong, allc, n2.__getitem__)),
                         ("2-hop copy, all flags routed", lambda: solve_hour_k(
                             pd.DataFrame({"CellID": cells[flagged], "deficit": np.clip(act - thr, 0, None)[flagged]}), allc, n2.__getitem__))):
            t0 = time.perf_counter()
            sol = fn()
            secs = time.perf_counter() - t0
            recv, give = np.zeros(len(cells)), np.zeros(len(cells))
            for (c, n), v in sol.items():
                recv[int(c) - 1] += v
                give[int(n) - 1] += v
            o = outcome(act, thr, recv, give)
            res[f"{name} | net"] = o["net_reduction"]
            res[f"{name} | seconds"] = secs
        res["excess_before"] = o["excess_before"]
        rows.append(res)
    r = pd.DataFrame(rows)
    _write(r, "reach2_by_hour.csv")
    tot = r.sum(numeric_only=True)
    print(tot.to_string(float_format=lambda v: f"{v:,.1f}"))
    for k in [c for c in tot.index if c.endswith("| net")]:
        print(f"{k:45s} {tot[k] / tot['excess_before']:.1%} of actual excess")


if __name__ == "__main__":
    steps = {"replayboot": step_replayboot, "reconcile": step_reconcile, "naive": step_naive, "reach2": step_reach2}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
