"""
src/evaluation/pre_phase3_checks.py

Three bounded full-grid checks before Phase 3. Choices are judged on the
window-1 validation slice (targets 2013-12-10 to 12-16) ONLY; test numbers
are reported for information.

  leaves1   +1h, final features: 31 and 63 leaves, validation curve up to 3,000 trees,
            0.5% rule on each. Does the sample-based leaf choice transfer to the full grid?
  trees4    +4h, final features, 31 leaves: validation curve and 0.5% rule. Is the tree count
            chosen for the old features (2,695) within 15% of the new choice?
  hotspot   Hotspot-only model refit with the final features (same per-horizon procedure as
            tune_trees.py hotspot), compared with the old-feature hotspot-only model on hotspot
            MAE and F1 with the paired day bootstrap (Phase 2 test thresholds).

PRE-DECLARED criteria for "a better configuration" (stop and ask if met):
  leaves1  63 leaves is better if its best validation loss is MORE THAN 0.5% lower than 31's
           (the tolerance the size rule itself treats as equivalent).
  trees    the inherited tree count is out if |inherited - new| / new > 15%.

RUN:
    python src/evaluation/pre_phase3_checks.py leaves1
    python src/evaluation/pre_phase3_checks.py trees4
    python src/evaluation/pre_phase3_checks.py hotspot
    python src/evaluation/pre_phase3_checks.py leafrule   (after the user's decision on check 1)
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
    LEAF_GRID, MODEL_DIR, OUT_DIR, TREE_CAP, W1, FeatureStore, build_rows, choose_trees, eval_thresholds,
    fit_with_curve, has_preds, hotspots, lgbm_params, load_json, load_preds, origin_index, paired_bootstrap,
    predict, require_headroom, save_json, save_preds, training_cell_means,
)

CHECK_DIR = OUT_DIR / "pre_phase3"
INHERITED_TREES = {1: 2690, 2: 2749, 3: 2745, 4: 2695}
LEAF_TOLERANCE = 0.005
TREE_DRIFT = 0.15


def _final_groups():
    return tuple(load_json("final_features.json")["final_groups"])


def _setup():
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, W1)
    return panel, store, cm, hotspots(cm)


def _curve_fit(h: int, leaves: int, tag: str, save_variant: str | None = None):
    out = CHECK_DIR / f"{tag}.json"
    if out.exists():
        return load_json(f"pre_phase3/{tag}.json")
    panel, store, cm, _ = _setup()
    groups = _final_groups()
    cidx = np.arange(len(panel.cells))
    X, y, _ = build_rows(store, h, origin_index(panel, W1, "train", h), cidx, cm, groups=groups)
    Xv, yv, _ = build_rows(store, h, origin_index(panel, W1, "val", h), cidx, cm, groups=groups)
    require_headroom(f"{tag} fit")
    with Step(f"{tag}: +{h}h final features, {leaves} leaves, up to {TREE_CAP} trees") as s:
        model, curve, fit_s = fit_with_curve(X, y, Xv, yv, lgbm_params(leaves, TREE_CAP))
    np.save(CHECK_DIR / f"{tag}_curve.npy", curve)
    choice = choose_trees(curve)
    inherited = INHERITED_TREES[h]
    res = {"horizon": h, "num_leaves": leaves, "features": X.shape[1], **choice,
           "inherited_trees": inherited,
           "inherited_drift": abs(inherited - choice["chosen_trees"]) / choice["chosen_trees"],
           "val_loss_at_inherited": float(curve[inherited - 1]),
           "fit_seconds_3000": fit_s, "peak_ram_gb": s.peak_ws, "peak_commit_gb": s.peak_commit}
    # Test numbers, for information only (never used for a choice).
    Xt, yt, _ = build_rows(store, h, origin_index(panel, W1, "test", h), cidx, cm, groups=groups)
    pt = predict(model, Xt, choice["chosen_trees"])
    res["test_mae_info"] = float(np.mean(np.abs(yt - pt)))
    if save_variant:
        # Keep this fit as a candidate final model: predictions at the 0.5%-rule tree count.
        n = choice["chosen_trees"]
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(MODEL_DIR / f"w1__{save_variant}__{h}h.txt"), num_iteration=n)
        save_preds("w1", save_variant, h, "val", predict(model, Xv, n))
        save_preds("w1", save_variant, h, "test", pt)
        res["saved_variant"] = save_variant
    save_json(f"pre_phase3/{tag}.json", res)
    print(res)
    return res


def step_leaves1():
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    r31 = _curve_fit(1, 31, "leaves1_31")
    r63 = _curve_fit(1, 63, "leaves1_63")
    gain = 1 - r63["best_loss"] / r31["best_loss"]
    verdict = {"best_loss_31": r31["best_loss"], "best_loss_63": r63["best_loss"],
               "relative_improvement_63_vs_31": gain, "better_config_found": gain > LEAF_TOLERANCE,
               "trees_31": r31["chosen_trees"], "trees_63": r63["chosen_trees"],
               "inherited_drift_31": r31["inherited_drift"]}
    save_json("pre_phase3/leaves1_verdict.json", verdict)
    print("\nCHECK 1 VERDICT:", verdict)


def step_trees4():
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    r = _curve_fit(4, 31, "trees4_31")
    verdict = {"chosen_trees_new": r["chosen_trees"], "inherited": INHERITED_TREES[4],
               "drift": r["inherited_drift"], "better_config_found": r["inherited_drift"] > TREE_DRIFT}
    save_json("pre_phase3/trees4_verdict.json", verdict)
    print("\nCHECK 2 VERDICT:", verdict)


def step_hotspot():
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    panel, store, cm, hot = _setup()
    groups = _final_groups()
    hidx = np.searchsorted(panel.cells, np.sort(hot))
    res_file = CHECK_DIR / "hotspot_only_final_size.csv"
    rows = pd.read_csv(res_file).to_dict("records") if res_file.exists() else []
    for h in HORIZONS:
        if has_preds("w1", "hotspot_only_final", h, "test"):
            continue
        tr, va, te = (origin_index(panel, W1, s, h) for s in ("train", "val", "test"))
        X, y, _ = build_rows(store, h, tr, hidx, cm, groups=groups)
        Xv, yv, _ = build_rows(store, h, va, hidx, cm, groups=groups)
        best = None
        for leaves in LEAF_GRID:
            with Step(f"hotspot-only final +{h}h {leaves} leaves") as s:
                model, curve, fit_s = fit_with_curve(X, y, Xv, yv, lgbm_params(leaves, TREE_CAP))
            choice = choose_trees(curve)
            rows.append({"horizon": h, "num_leaves": leaves, **choice, "fit_seconds_3000": fit_s,
                         "peak_commit_gb": s.peak_commit, "selected": False})
            if best is None or choice["best_loss"] < best[1]["best_loss"]:
                best = (leaves, choice, model)
        leaves, choice, model = best
        n = choice["chosen_trees"]
        for r in rows:
            if r["horizon"] == h and r["num_leaves"] == leaves:
                r["selected"] = True
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(MODEL_DIR / f"w1__hotspot_only_final__{h}h.txt"), num_iteration=n)
        Xt, _, _ = build_rows(store, h, te, hidx, cm, groups=groups)
        with Step(f"hotspot-only final +{h}h predict"):
            save_preds("w1", "hotspot_only_final", h, "val", predict(model, Xv, n))
            save_preds("w1", "hotspot_only_final", h, "test", predict(model, Xt, n))
        pd.DataFrame(rows).to_csv(res_file, index=False)
        print(f"  +{h}h hotspot-only (final features): {leaves} leaves, {n} trees")
        del X, Xv, Xt, model
        gc.collect()

    # Compare against the old-feature hotspot-only model on hotspot rows (Phase 2 thresholds).
    thr = eval_thresholds(panel, W1)
    hot_set = set(hot.tolist())
    out = []
    for h in HORIZONS:
        te = origin_index(panel, W1, "test", h)
        _, y, meta = build_rows(store, h, te, np.arange(len(panel.cells)), cm)
        n_t = len(te)
        thr_rows = np.repeat(thr, n_t)
        base = load_preds("w1", "final", h, "test")
        pos = np.searchsorted(panel.cells, np.sort(hot))
        fc = {}
        for name, var in (("hotspot-only (old features)", "hotspot_only"), ("hotspot-only (final features)", "hotspot_only_final")):
            comp = base.copy()
            comp.reshape(len(panel.cells), n_t)[pos] = load_preds("w1", var, h, "test").reshape(len(pos), n_t)
            fc[name] = comp
        fc["final pooled"] = base
        b = paired_bootstrap(fc, "hotspot-only (old features)", y, thr_rows, meta, hot_set, slices=("hotspot", "all"))
        b.insert(0, "horizon", h)
        out.append(b)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(CHECK_DIR / "hotspot_only_final_vs_old.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def step_leafrule():
    """Re-applies the Phase 2 leaf rule on the FULL grid with the final features (user decision):
    one num_leaves for all horizons = lowest mean over horizons of (best validation loss /
    best of the two candidates at that horizon). Candidates: 31 and 63 only (15 and 127 are not
    re-tested on the full grid). If 63 wins, the 63-leaf fits become the final model
    ("final_63l", tree counts re-derived by the 0.5% rule); if 31 wins, the existing "final"
    model is kept (its inherited tree counts were within 15%: checks 1 and 2)."""
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    have = {(1, 31): "leaves1_31", (4, 31): "trees4_31", (1, 63): "leaves1_63"}
    res = {}
    for h in HORIZONS:
        for leaves in (31, 63):
            tag = have.get((h, leaves), f"leafrule_{h}h_{leaves}")
            res[(h, leaves)] = _curve_fit(h, leaves, tag, save_variant="final_63l" if leaves == 63 and (h, leaves) not in have else None)
    rows = []
    for h in HORIZONS:
        best = min(res[(h, 31)]["best_loss"], res[(h, 63)]["best_loss"])
        for leaves in (31, 63):
            r = res[(h, leaves)]
            rows.append({"horizon": h, "num_leaves": leaves, "best_loss": r["best_loss"], "rel_to_best": r["best_loss"] / best,
                         "chosen_trees": r["chosen_trees"], "best_trees": r["best_trees"], "cap_reached": r["cap_reached"],
                         "fit_seconds_3000": r["fit_seconds_3000"], "peak_commit_gb": r["peak_commit_gb"],
                         "test_mae_info": r["test_mae_info"]})
    table = pd.DataFrame(rows)
    mean_rel = table.groupby("num_leaves")["rel_to_best"].mean()
    chosen = int(mean_rel.idxmin())
    final_variant = "final" if chosen == 31 else "final_63l"
    if chosen == 63:
        # The +1h 63-leaf check fit did not keep its model; refit it once to save predictions.
        _curve_fit(1, 63, "leafrule_1h_63", save_variant="final_63l")
    table.to_csv(CHECK_DIR / "leafrule_full_grid.csv", index=False)
    decision = {"chosen_num_leaves": chosen, "mean_rel_loss": mean_rel.to_dict(), "final_variant": final_variant,
                "trees": {int(h): int(table[(table.horizon == h) & (table.num_leaves == chosen)]["chosen_trees"].iloc[0])
                          for h in HORIZONS} if chosen == 63 else INHERITED_TREES,
                "note": "full grid, final features; candidates 31 and 63 only"}
    save_json("pre_phase3/leafrule_decision.json", decision)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    print("LEAF RULE DECISION:", decision)


if __name__ == "__main__":
    steps = {"leaves1": step_leaves1, "trees4": step_trees4, "hotspot": step_hotspot, "leafrule": step_leafrule}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(CHECK_DIR / f"timing_{sys.argv[1]}.csv", index=False)
