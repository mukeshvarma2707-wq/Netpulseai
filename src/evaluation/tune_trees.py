"""
src/evaluation/tune_trees.py  (Phase 2: model size)

Chooses LightGBM model size on the VALIDATION slice only (window 1: targets
2013-12-10 to 2013-12-16; training targets before 2013-12-10), then keeps the
selection fit itself as the final model, truncated at the chosen tree count.
Nothing is refit on the validation days, so they stay unseen for Phase 3.

Steps (resumable: a step skips work whose outputs already exist):

  leaves    Stratified 1,000-cell sample (20 hotspots = 2%): for each horizon and
            num_leaves in {15, 31, 63, 127}, fit up to 3,000 trees at learning rate 0.1
            and record the validation curve. Choose ONE num_leaves for all horizons:
            the value with the lowest mean (over horizons) of its best validation loss
            relative to the best leaf setting at that horizon.            [SAMPLE]
  full      Full grid, chosen num_leaves, 3,000-tree cap, per horizon: tree count =
            smallest count within 0.5% of the best validation loss. Saves validation and
            test predictions of the truncated model as variant "tuned".   [FULL GRID]
  hotspot   Hotspot-only model on the 200 hotspot cells (training rows, validation rows
            of those cells only): its own num_leaves (same grid) and tree count by the same
            rule. Saves variant "hotspot_only" (hotspot rows) and the composite
            "pooled+hotspot" (tuned pooled for typical cells, hotspot model for hotspots).
  w2        Window 2 (train targets < Nov 18, validation Nov 18-24, test origins Nov 25 -
            Dec 7 19:00): re-derives the tree count per horizon on Nov 18-24 with the
            window-1 num_leaves, full grid; saves variant "tuned" for w2.
  production  60-tree production settings (lr 0.05, 15 leaves) fitted on each window's
            training rows, for reference (variant "prod60").

RUN:
    python src/evaluation/tune_trees.py leaves
    python src/evaluation/tune_trees.py full
    python src/evaluation/tune_trees.py hotspot
    python src/evaluation/tune_trees.py w2
    python src/evaluation/tune_trees.py production
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import HORIZONS, PRODUCTION_PARAMS, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import (  # noqa: E402
    LEAF_GRID, MODEL_DIR, N_JOBS, OUT_DIR, TREE_CAP, W1, W2, FeatureStore, build_rows, choose_trees,
    fit_plain, fit_with_curve, has_preds, hotspots, lgbm_params, load_json, origin_index, predict,
    require_headroom, save_json, save_preds, stratified_sample, training_cell_means,
)


def _setup(win=W1):
    panel = load_panel()
    store = FeatureStore(panel)
    cm = training_cell_means(panel, win)
    hot = hotspots(cm)
    return panel, store, cm, hot


# ============================================================================ leaves (sample)
def step_leaves():
    out_file = OUT_DIR / "size_leaf_search_sample.csv"
    done = pd.read_csv(out_file) if out_file.exists() else pd.DataFrame()
    panel, store, cm, hot = _setup()
    sample = stratified_sample(panel, hot)
    cidx = np.searchsorted(panel.cells, sample)
    save_json("sample_cells_w1.json", {"cells": sample.tolist(), "hotspots_in_sample": int(np.isin(sample, hot).sum())})
    rows = done.to_dict("records")
    for h in HORIZONS:
        X, y, _ = build_rows(store, h, origin_index(panel, W1, "train", h), cidx, cm)
        Xv, yv, _ = build_rows(store, h, origin_index(panel, W1, "val", h), cidx, cm)
        for leaves in LEAF_GRID:
            if any((r["horizon"] == h) and (r["num_leaves"] == leaves) for r in rows):
                continue
            with Step(f"sample +{h}h {leaves} leaves, {TREE_CAP} trees") as s:
                _, curve, fit_s = fit_with_curve(X, y, Xv, yv, lgbm_params(leaves, TREE_CAP))
            np.save(OUT_DIR / f"curve_sample_w1_{h}h_{leaves}l.npy", curve)
            rows.append({"horizon": h, "num_leaves": leaves, **choose_trees(curve), "fit_seconds": fit_s,
                         "peak_commit_gb": s.peak_commit})
            pd.DataFrame(rows).to_csv(out_file, index=False)
        del X, Xv
        gc.collect()

    res = pd.DataFrame(rows)
    res["rel_to_best_leaf"] = res["best_loss"] / res.groupby("horizon")["best_loss"].transform("min")
    summary = res.groupby("num_leaves")["rel_to_best_leaf"].mean().sort_values()
    chosen = int(summary.index[0])
    res.to_csv(out_file, index=False)
    save_json("size_choice_leaves.json", {"chosen_num_leaves": chosen, "mean_rel_loss_by_leaves": summary.to_dict(),
                                          "source": "stratified 1,000-cell sample, window 1 validation"})
    pd.set_option("display.width", 220)
    print("\nLEAF SEARCH (sample, window 1 validation)")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nmean best-loss relative to the best leaf setting per horizon:")
    print(summary.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\nCHOSEN num_leaves = {chosen}")


# ============================================================================ full-grid size
def _select_full(win, leaves: int, tag: str, cells_idx=None, variant="tuned", hot_set=None):
    """Full-grid (or given cells) size selection per horizon; saves preds and the truncated model."""
    panel, store, cm, hot = _setup(win)
    cidx = np.arange(len(panel.cells)) if cells_idx is None else cells_idx(panel, hot)
    rows = []
    res_file = OUT_DIR / f"size_{tag}.csv"
    if res_file.exists():
        rows = pd.read_csv(res_file).to_dict("records")
    for h in HORIZONS:
        if any(r["horizon"] == h for r in rows) and has_preds(win.name, variant, h, "test"):
            continue
        tr, va, te = (origin_index(panel, win, s, h) for s in ("train", "val", "test"))
        X, y, _ = build_rows(store, h, tr, cidx, cm)
        Xv, yv, _ = build_rows(store, h, va, cidx, cm)
        require_headroom(f"{tag} +{h}h fit")
        with Step(f"{tag} +{h}h {leaves} leaves up to {TREE_CAP} trees") as s:
            model, curve, fit_s = fit_with_curve(X, y, Xv, yv, lgbm_params(leaves, TREE_CAP))
        del X
        gc.collect()
        np.save(OUT_DIR / f"curve_{tag}_{h}h.npy", curve)
        choice = choose_trees(curve)
        n = choice["chosen_trees"]
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(MODEL_DIR / f"{win.name}__{variant}__{h}h.txt"), num_iteration=n)
        save_preds(win.name, variant, h, "val", predict(model, Xv, n))
        Xt, _, _ = build_rows(store, h, te, cidx, cm)
        save_preds(win.name, variant, h, "test", predict(model, Xt, n))
        rows.append({"horizon": h, "num_leaves": leaves, **choice, "fit_seconds_3000": fit_s,
                     "est_fit_seconds_chosen": fit_s * n / len(curve), "peak_ram_gb": s.peak_ws,
                     "peak_commit_gb": s.peak_commit, "train_rows": len(y), "val_rows": len(yv)})
        pd.DataFrame(rows).to_csv(res_file, index=False)
        print(f"  +{h}h: best {choice['best_trees']} trees (loss {choice['best_loss']:.5f}), "
              f"chosen {n} (loss {choice['chosen_loss']:.5f}), cap reached: {choice['cap_reached']}")
        del model, Xv, Xt
        gc.collect()
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}"))


def step_full():
    leaves = load_json("size_choice_leaves.json")["chosen_num_leaves"]
    _select_full(W1, leaves, "full_w1")


def step_w2():
    leaves = load_json("size_choice_leaves.json")["chosen_num_leaves"]
    print(f"Window 2 uses num_leaves = {leaves} chosen on window 1 (look-ahead in this one hyperparameter).")
    _select_full(W2, leaves, "full_w2")


# ============================================================================ hotspot-only model
def step_hotspot():
    panel, store, cm, hot = _setup(W1)
    hidx = np.searchsorted(panel.cells, np.sort(hot))
    res_file = OUT_DIR / "size_hotspot_only_w1.csv"
    rows = pd.read_csv(res_file).to_dict("records") if res_file.exists() else []
    for h in HORIZONS:
        if has_preds("w1", "hotspot_only", h, "test"):
            continue
        tr, va, te = (origin_index(panel, W1, s, h) for s in ("train", "val", "test"))
        X, y, _ = build_rows(store, h, tr, hidx, cm)
        Xv, yv, _ = build_rows(store, h, va, hidx, cm)
        best = None
        for leaves in LEAF_GRID:
            with Step(f"hotspot-only +{h}h {leaves} leaves") as s:
                model, curve, fit_s = fit_with_curve(X, y, Xv, yv, lgbm_params(leaves, TREE_CAP))
            choice = choose_trees(curve)
            rows.append({"horizon": h, "num_leaves": leaves, **choice, "fit_seconds_3000": fit_s, "peak_commit_gb": s.peak_commit})
            if best is None or choice["best_loss"] < best[1]["best_loss"]:
                best = (leaves, choice, model)
        # Per horizon: the leaf setting with the lowest best validation loss (hotspot rows are few,
        # so fits are cheap and a per-horizon choice costs little).
        leaves, choice, model = best
        n = choice["chosen_trees"]
        for r in rows:
            if r["horizon"] == h:
                r["selected"] = r["num_leaves"] == leaves
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(MODEL_DIR / f"w1__hotspot_only__{h}h.txt"), num_iteration=n)
        Xt, _, _ = build_rows(store, h, te, hidx, cm)
        save_preds("w1", "hotspot_only", h, "val", predict(model, Xv, n))
        save_preds("w1", "hotspot_only", h, "test", predict(model, Xt, n))
        pd.DataFrame(rows).to_csv(res_file, index=False)
        print(f"  +{h}h hotspot-only: {leaves} leaves, {n} trees (best {choice['best_trees']})")
        del X, Xv, Xt, model
        gc.collect()
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}"))


# ============================================================================ production reference
def step_production():
    for win in (W1, W2):
        panel, store, cm, hot = _setup(win)
        cidx = np.arange(len(panel.cells))
        for h in HORIZONS:
            if has_preds(win.name, "prod60", h, "test"):
                continue
            X, y, _ = build_rows(store, h, origin_index(panel, win, "train", h), cidx, cm)
            require_headroom(f"prod60 {win.name} +{h}h")
            with Step(f"prod60 {win.name} +{h}h"):
                model, fit_s = fit_plain(X, y, {**PRODUCTION_PARAMS, "n_jobs": N_JOBS})
            del X
            for split in ("val", "test"):
                Xs, _, _ = build_rows(store, h, origin_index(panel, win, split, h), cidx, cm)
                save_preds(win.name, "prod60", h, split, predict(model, Xs))
                del Xs
            gc.collect()
        del panel, store
        gc.collect()


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = {"leaves": step_leaves, "full": step_full, "hotspot": step_hotspot, "w2": step_w2,
             "production": step_production}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT_DIR / f"timing_tune_{sys.argv[1]}.csv", index=False)
