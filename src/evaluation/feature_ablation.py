"""
src/evaluation/feature_ablation.py  (Phase 2: feature groups, pooled vs hotspot model, report)

Feature groups are added ONE AT A TIME on top of the production features, with
model size frozen at the tuned size of each window (num_leaves and per-horizon
tree count from tune_trees.py). Full grid.

  G1     weekly lag: lag_168h and the weekly-naive value (activity at t+h-168)
  G2     rolling mean and std over the last 3, 6 and 24 hours (ending at t-1)
  G2MOM  G2 plus momentum (A[t-1]-A[t-2], A[t-1]-A[t-4]); momentum is tested as the
         difference between G2MOM and G2
  G3     neighbour max (8 cells) and the 2-step ring mean (24 cells), both at t-1
  G4     holiday flags (target date, seasonal-naive source date)

PRE-DECLARED RULES (written before any feature result was seen; applied by `decide`
from saved results, never by hand). Window 1, full grid, horizons +1h and +4h,
checks = {MAE all, F1 all, MAE hotspot, F1 hotspot} x {+1h, +4h}, 90% paired
day-level bootstrap:
  Group rule:    keep a group if it is significantly BETTER than "tuned" on at least one
                 check and significantly WORSE on none.
  Momentum rule: drop momentum only if removing it (G2 vs G2MOM) is NOT significantly
                 worse on any check. Momentum only matters if G2 is kept.
Window 2 is used only to check whether the ranking holds, never to select.

Steps:
  fit WINDOW H[,H] [VARIANTS]   fit and save test (and validation, window 1) predictions
  decide                        apply the two rules, write final_features.json
  final                         fit the final feature set at all 4 horizons (window 1)
  report                        all result tables (CSV + printed)

RUN (in order):
    python src/evaluation/feature_ablation.py fit w1 1,4
    python src/evaluation/feature_ablation.py fit w2 1,4
    python src/evaluation/feature_ablation.py decide
    python src/evaluation/feature_ablation.py final
    python src/evaluation/feature_ablation.py report
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import HORIZONS, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    MODEL_DIR, OUT_DIR, WINDOWS, FeatureStore, build_rows, evaluate, eval_thresholds, fit_plain,
    has_preds, hotspots, lgbm_params, load_json, load_preds, origin_index, paired_bootstrap, predict,
    require_headroom, save_json, save_preds, training_cell_means,
)

VARIANT_GROUPS = {"G1": ("G1",), "G2": ("G2",), "G2MOM": ("G2", "MOM"), "G3": ("G3",), "G4": ("G4",)}
RULE_HORIZONS = (1, 4)
CHECKS = [("all", "mae"), ("all", "f1"), ("hotspot", "mae"), ("hotspot", "f1")]


def _sizes(win_name: str) -> dict:
    leaves = load_json("size_choice_leaves.json")["chosen_num_leaves"]
    sizes = pd.read_csv(OUT_DIR / f"size_full_{win_name}.csv")
    return {int(r.horizon): (leaves, int(r.chosen_trees)) for r in sizes.itertuples()}


def _fit_variants(win_name: str, horizons, variants: dict, save_model: bool = False):
    win = WINDOWS[win_name]
    sizes = _sizes(win_name)
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, win)
    cidx = np.arange(len(panel.cells))
    timing = []
    for h in horizons:
        leaves, n = sizes[h]
        for name, groups in variants.items():
            if has_preds(win_name, name, h, "test"):
                continue
            X, y, _ = build_rows(store, h, origin_index(panel, win, "train", h), cidx, cm, groups=groups)
            require_headroom(f"{win_name} {name} +{h}h")
            with Step(f"{win_name} {name} +{h}h ({leaves} leaves, {n} trees, {X.shape[1]} features)") as s:
                model, fit_s = fit_plain(X, y, lgbm_params(leaves, n))
            timing.append({"window": win_name, "variant": name, "horizon": h, "trees": n, "features": X.shape[1],
                           "fit_seconds": fit_s, "peak_ram_gb": s.peak_ws, "peak_commit_gb": s.peak_commit})
            del X
            gc.collect()
            if save_model:
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                model.booster_.save_model(str(MODEL_DIR / f"{win_name}__{name}__{h}h.txt"))
            for split in (("val", "test") if win_name == "w1" else ("test",)):
                Xs, _, _ = build_rows(store, h, origin_index(panel, win, split, h), cidx, cm, groups=groups)
                save_preds(win_name, name, h, split, predict(model, Xs))
                del Xs
            imp = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
            (imp / imp.sum()).sort_values(ascending=False).to_csv(OUT_DIR / f"importance_{win_name}_{name}_{h}h.csv")
            del model
            gc.collect()
            pd.DataFrame(timing).to_csv(OUT_DIR / f"timing_fit_{win_name}_{'_'.join(map(str, horizons))}_{len(timing)}.csv", index=False)


def step_fit(win_name: str, horizons: str, variants: str | None = None):
    hs = [int(x) for x in horizons.split(",")]
    chosen = VARIANT_GROUPS if variants is None else {v: VARIANT_GROUPS[v] for v in variants.split(",")}
    _fit_variants(win_name, hs, chosen)


# ============================================================================ evaluation helpers
class _Ctx:
    """Evaluation context for one window: truth, thresholds and metadata per horizon (test split)."""

    def __init__(self, win_name: str):
        self.win = WINDOWS[win_name]
        self.panel = load_panel()
        self.store = FeatureStore(self.panel)
        self.cm = training_cell_means(self.panel, self.win)
        self.hot = set(hotspots(self.cm).tolist())
        self.thr = eval_thresholds(self.panel, self.win)

    def truth(self, h: int):
        te = origin_index(self.panel, self.win, "test", h)
        X, y, meta = build_rows(self.store, h, te, np.arange(len(self.panel.cells)), self.cm, groups=("G1",))
        meta["weekly_naive"] = X["weekly_naive"].to_numpy(np.float64)   # activity at t+h-168
        del X
        thr_rows = np.repeat(self.thr, len(te))
        return y, meta, thr_rows, len(te)

    def composite_hotspot(self, h: int, n_t: int) -> np.ndarray:
        """Pooled tuned predictions, with hotspot rows replaced by the hotspot-only model."""
        comp = load_preds(self.win.name, "tuned", h, "test").copy()
        hot_sorted = np.sort(np.array(list(self.hot)))
        hot_preds = load_preds(self.win.name, "hotspot_only", h, "test").reshape(len(hot_sorted), n_t)
        pos = np.searchsorted(self.panel.cells, hot_sorted)
        comp.reshape(len(self.panel.cells), n_t)[pos] = hot_preds
        return comp


def _bootstrap_variants(ctx: _Ctx, h: int, names, reference="tuned", slices=("all", "hotspot", "non-holiday")):
    y, meta, thr_rows, _ = ctx.truth(h)
    fc = {n: load_preds(ctx.win.name, n, h, "test") for n in names if has_preds(ctx.win.name, n, h)}
    res = paired_bootstrap(fc, reference, y, thr_rows, meta, ctx.hot, slices=slices)
    res.insert(0, "horizon", h)
    res.insert(0, "window", ctx.win.name)
    return res


# ============================================================================ decide
def step_decide():
    ctx = _Ctx("w1")
    rows = []
    for h in RULE_HORIZONS:
        rows.append(_bootstrap_variants(ctx, h, ["tuned", "G1", "G2", "G3", "G4", "G2MOM"], slices=("all", "hotspot")))
        mom = _bootstrap_variants(ctx, h, ["G2MOM", "G2"], reference="G2MOM", slices=("all", "hotspot"))
        mom["forecaster"] = "G2 (momentum removed)"
        rows.append(mom)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT_DIR / "decide_bootstrap_w1.csv", index=False)

    def verdicts(df, name):
        out = {}
        for h in RULE_HORIZONS:
            for sl, metric in CHECKS:
                r = df[(df.forecaster == name) & (df.horizon == h) & (df.slice == sl)].iloc[0]
                out[f"+{h}h {sl} {metric}"] = r[f"{metric}_verdict"]
        return out

    groups = {}
    for g in ("G1", "G2", "G3", "G4"):
        v = verdicts(res[res.vs == "tuned"], g)
        groups[g] = {"verdicts": v, "keep": any(x == "better" for x in v.values()) and not any(x == "worse" for x in v.values())}
    mom_v = verdicts(res[res.vs == "G2MOM"], "G2 (momentum removed)")
    drop_momentum = not any(x == "worse" for x in mom_v.values())
    final = [g for g, d in groups.items() if d["keep"]]
    if "G2" in final and not drop_momentum:
        final.append("MOM")
    decision = {"groups": groups, "momentum_removed_verdicts": mom_v, "drop_momentum": drop_momentum,
                "final_groups": final, "rule_source": "window 1, full grid, +1h and +4h, 90% paired day bootstrap"}
    save_json("final_features.json", decision)
    print("\nRULE OUTCOMES (window 1, full grid)")
    for g, d in groups.items():
        print(f"  {g}: keep={d['keep']}  {d['verdicts']}")
    print(f"  momentum: drop={drop_momentum}  (removing it: {mom_v})")
    print(f"  FINAL GROUPS: production + {final}")


def step_final():
    final = tuple(load_json("final_features.json")["final_groups"])
    print(f"Final feature set: production + {final}")
    if not final:
        print("No group kept: the final model is the tuned production-feature model (variant 'tuned').")
        return
    _fit_variants("w1", HORIZONS, {"final": final}, save_model=True)


# ============================================================================ report
def step_report():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    all_metrics, all_boot = [], []
    for win_name in ("w1", "w2"):
        ctx = _Ctx(win_name)
        for h in HORIZONS:
            if not has_preds(win_name, "tuned", h):
                continue
            y, meta, thr_rows, n_t = ctx.truth(h)
            fc = {"seasonal-naive": meta["seasonal_naive"], "weekly-naive": meta["weekly_naive"]}
            for n in ("prod60", "tuned", "G1", "G2", "G2MOM", "G3", "G4", "final"):
                if has_preds(win_name, n, h):
                    fc[n] = load_preds(win_name, n, h, "test")
            if win_name == "w1" and has_preds("w1", "hotspot_only", h):
                fc["pooled+hotspot"] = ctx.composite_hotspot(h, n_t)
            m = evaluate(fc, y, thr_rows, meta, ctx.hot)
            m.insert(0, "horizon", h)
            m.insert(0, "window", win_name)
            all_metrics.append(m)
            b = paired_bootstrap(fc, "tuned", y, thr_rows, meta, ctx.hot,
                                 slices=("all", "hotspot", "non-holiday", "weekday"))
            b.insert(0, "horizon", h)
            b.insert(0, "window", win_name)
            all_boot.append(b)
            if "prod60" in fc:
                b2 = paired_bootstrap({k: fc[k] for k in ("prod60", "tuned", "seasonal-naive")}, "prod60", y, thr_rows,
                                      meta, ctx.hot, slices=("all", "hotspot", "non-holiday"))
                b2.insert(0, "horizon", h)
                b2.insert(0, "window", win_name)
                all_boot.append(b2)
            if "final" in fc:
                b3 = paired_bootstrap({k: fc[k] for k in ("seasonal-naive", "final")}, "seasonal-naive", y, thr_rows,
                                      meta, ctx.hot, slices=("all", "hotspot", "non-holiday"))
                b3.insert(0, "horizon", h)
                b3.insert(0, "window", win_name)
                all_boot.append(b3)
        del ctx
        gc.collect()
    metrics = pd.concat(all_metrics, ignore_index=True)
    boot = pd.concat(all_boot, ignore_index=True)
    metrics.to_csv(OUT_DIR / "report_metrics.csv", index=False)
    boot.to_csv(OUT_DIR / "report_bootstrap.csv", index=False)

    print("\nMAE / F1 (all cells and hotspots), test window")
    for (w, sl) in (("w1", "all"), ("w1", "hotspot"), ("w2", "all"), ("w2", "hotspot")):
        sub = metrics[(metrics.window == w) & (metrics.slice == sl)]
        if len(sub):
            print(f"\n{w} {sl}: MAE")
            print(sub.pivot_table(index="forecaster", columns="horizon", values="mae", sort=False).to_string(float_format=fmt))
            print(f"{w} {sl}: F1")
            print(sub.pivot_table(index="forecaster", columns="horizon", values="f1", sort=False).to_string(float_format=fmt))
    print("\nBootstrap (forecaster minus reference)")
    print(boot[["window", "horizon", "slice", "forecaster", "vs", "days", "mae_diff", "mae_lo", "mae_hi", "mae_verdict",
                "f1_diff", "f1_lo", "f1_hi", "f1_verdict"]].to_string(index=False, float_format=fmt))

    # Feature ranking per window: MAE improvement over "tuned" on all cells at +1h and +4h
    rank = []
    for w in ("w1", "w2"):
        for h in RULE_HORIZONS:
            sub = metrics[(metrics.window == w) & (metrics.slice == "all") & (metrics.horizon == h)].set_index("forecaster")
            if "tuned" not in sub.index:
                continue
            for g in ("G1", "G2", "G2MOM", "G3", "G4"):
                if g in sub.index:
                    rank.append({"window": w, "horizon": h, "variant": g,
                                 "mae_gain": sub.loc["tuned", "mae"] - sub.loc[g, "mae"],
                                 "f1_gain": sub.loc[g, "f1"] - sub.loc["tuned", "f1"]})
    rank = pd.DataFrame(rank)
    if len(rank):
        rank["mae_rank"] = rank.groupby(["window", "horizon"])["mae_gain"].rank(ascending=False).astype(int)
        rank["f1_rank"] = rank.groupby(["window", "horizon"])["f1_gain"].rank(ascending=False).astype(int)
        rank.to_csv(OUT_DIR / "report_feature_ranking.csv", index=False)
        print("\nFEATURE RANKING (gain over tuned; rank 1 = best)")
        print(rank.pivot_table(index="variant", columns=["window", "horizon"], values=["mae_gain", "mae_rank", "f1_rank"]).to_string(float_format=fmt))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2 or sys.argv[1] not in ("fit", "decide", "final", "report"):
        raise SystemExit("usage: feature_ablation.py fit WINDOW H[,H] [VARIANTS] | decide | final | report")
    cmd = sys.argv[1]
    with Step(f"TOTAL {' '.join(sys.argv[1:])}"):
        if cmd == "fit":
            step_fit(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
        else:
            {"decide": step_decide, "final": step_final, "report": step_report}[cmd]()
    pd.DataFrame(Step.log).to_csv(OUT_DIR / f"timing_ablation_{'_'.join(sys.argv[1:])}.csv".replace(",", "-"), index=False)
