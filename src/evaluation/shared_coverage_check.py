"""
src/evaluation/shared_coverage_check.py  (shared base-station coverage between grid squares)

Barlacchi et al. 2015: a square's value is the sum over base-station coverage areas of
(records in the area x share of the area overlapping the square). Equal-size squares lying
fully inside the same coverage area must therefore have identical series in every channel.

  stage1   Diagnose (no solver). Twin / proportional / ordinary 8-neighbour pairs, twin groups,
           activity share, known-area pictures, and the materiality decision.
  stage2   (not yet implemented; awaits approval) solver and diagnosis rule without such pairs.

Milan only. Input (read-only): data/raw/cdr_activity_aggregated.parquet.
Outputs: data/experiments/shared_coverage/.

DECISIONS AND CONSTANTS (declared before results):
  - Pairs: each unordered 8-neighbour pair once, on the 100 x 100 grid (CellID = row*100 + col + 1,
    as diagnosis_agent.get_neighbors): offsets (0,+1), (+1,0), (+1,+1), (+1,-1) -> 39,402 pairs.
    "Directed neighbour relations" (each cell's up-to-8 neighbours) = 2 x pairs = 78,804.
  - Twin at tolerance tol: for all 5 channels and all 1,488 hours, |a - b| <= tol * max(|a|, |b|)
    (two zeros match). Primary tol = 1e-9; sensitivity 1e-6 and 1e-4.
  - Near-twin (reported, not used for decisions): all channels match at tol 1e-9 in >= 99% of hours.
    The loader fills missing (cell, hour) rows with 0 (2,515 cell-hours), which can break an
    otherwise exact twin at a few hours.
  - Exclusion from the twin / proportional tests: a cell is excluded if its total activity has zero
    variance over the 1,488 hours, or its mean total activity is below 1.0 unit per hour
    (near-zero; two near-empty series match trivially). A pair is excluded if either cell is.
  - Proportional: not a twin, and the hourly ratio of totals A_i(t) / A_j(t), over hours where both
    are > 0, has coefficient of variation (std / mean) < 1e-6, with no hour where exactly one of the
    two is 0. Also reported: whether each channel's ratio is constant too.
  - Ordinary pairs: Pearson correlation of the two hourly total-activity series.
  - Twin groups: connected components of the twin graph (tol 1e-9).
  - Materiality (ARBITRARY threshold, fixed before Stage 2): sharing is "material" if more than 10%
    of neighbour pairs, or more than 10% of total activity (carried by cells in a twin group of size
    >= 2 or in a proportional pair), involve twin or proportional pairs.

RUN: python src/evaluation/shared_coverage_check.py stage1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import ROOT, Step  # noqa: E402

OUT = ROOT / "data" / "experiments" / "shared_coverage"
SRC = ROOT / "data" / "raw" / "cdr_activity_aggregated.parquet"
CHANNELS = ["smsin", "smsout", "callin", "callout", "internet"]
G = 100
OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]
TOLS = (1e-9, 1e-6, 1e-4)
NEAR_TWIN_SHARE = 0.99
MIN_MEAN_ACTIVITY = 1.0
PROP_CV = 1e-6
MATERIAL = 0.10
KNOWN = {"Duomo": 5060, "Bocconi": 4259, "Navigli": 4456}


def _w(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def load():
    df = pd.read_parquet(SRC, columns=["CellID", "datetime"] + CHANNELS)
    df = df.sort_values(["CellID", "datetime"], kind="stable").reset_index(drop=True)
    cells = np.sort(df["CellID"].unique())
    hours = pd.DatetimeIndex(np.sort(df["datetime"].unique()))
    assert len(cells) == G * G and (cells == np.arange(1, G * G + 1)).all(), "expected cells 1..10000"
    assert len(df) == len(cells) * len(hours)
    assert (df["CellID"].to_numpy().reshape(len(cells), len(hours))[:, 0] == cells).all()
    X = {c: df[c].to_numpy(np.float64).reshape(len(cells), len(hours)) for c in CHANNELS}
    del df
    return cells, hours, X


def pairs_index():
    """Unordered 8-neighbour pairs as (i, j) cell positions (0-based, row-major)."""
    rows = []
    for dr, dc in OFFSETS:
        r = np.arange(G)[:, None]
        c = np.arange(G)[None, :]
        r2, c2 = r + dr, c + dc
        ok = (r2 >= 0) & (r2 < G) & (c2 >= 0) & (c2 < G)
        rr, cc = np.broadcast_to(r, ok.shape)[ok], np.broadcast_to(c, ok.shape)[ok]
        rows.append(np.stack([rr * G + cc, (rr + dr) * G + (cc + dc)], 1))
    return np.concatenate(rows)


def match_hours(X, i, j, tol):
    """Per pair: number of hours where all 5 channels match within relative tolerance."""
    ok = None
    for c in CHANNELS:
        a, b = X[c][i], X[c][j]
        m = np.abs(a - b) <= tol * np.maximum(np.abs(a), np.abs(b))
        ok = m if ok is None else (ok & m)
    return ok.sum(axis=1)


def union_find(n, edges):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(x) for x in range(n)])


def step_stage1():
    with Step("load cdr_activity_aggregated (read-only)"):
        cells, hours, X = load()
    T = sum(X[c] for c in CHANNELS)
    H = len(hours)
    mean = T.mean(axis=1)
    var = T.var(axis=1)
    excluded = (var == 0) | (mean < MIN_MEAN_ACTIVITY)
    P = pairs_index()
    assert len(P) == 39_402
    i, j = P[:, 0], P[:, 1]
    pair_ok = ~excluded[i] & ~excluded[j]

    with Step("twin / proportional / correlation per pair"):
        n_match = {}
        CH = 4000
        for tol in TOLS:
            n_match[tol] = np.concatenate([match_hours(X, i[k:k + CH], j[k:k + CH], tol) for k in range(0, len(P), CH)])
        twin = {tol: pair_ok & (n_match[tol] == H) for tol in TOLS}
        near = pair_ok & (n_match[1e-9] >= NEAR_TWIN_SHARE * H) & ~twin[1e-9]
        # proportional on totals
        cv = np.full(len(P), np.nan)
        cv_ch_max = np.full(len(P), np.nan)
        corr = np.full(len(P), np.nan)
        for k in range(0, len(P), CH):
            a, b = T[i[k:k + CH]], T[j[k:k + CH]]
            both = (a > 0) & (b > 0)
            one = (a > 0) ^ (b > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(both, a / np.where(b > 0, b, 1), np.nan)
                m = np.nanmean(r, axis=1)
                s = np.nanstd(r, axis=1)
                c_ = s / m
            c_[one.any(axis=1)] = np.inf
            cv[k:k + CH] = c_
            am, bm = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr[k:k + CH] = (am * bm).sum(1) / np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))
        prop = pair_ok & ~twin[1e-9] & (cv < PROP_CV)
        if prop.any():                                   # per-channel ratio constancy for proportional pairs
            for k in np.where(prop)[0]:
                worst = 0.0
                for c in CHANNELS:
                    a, b = X[c][i[k]], X[c][j[k]]
                    both = (a > 0) & (b > 0)
                    if both.sum() > 1:
                        rr = a[both] / b[both]
                        worst = max(worst, rr.std() / rr.mean())
                cv_ch_max[k] = worst

    pairs = pd.DataFrame({"cell_a": cells[i], "cell_b": cells[j], "excluded": ~pair_ok,
                          "match_hours_1e-9": n_match[1e-9], "match_hours_1e-6": n_match[1e-6], "match_hours_1e-4": n_match[1e-4],
                          "twin_1e-9": twin[1e-9], "twin_1e-6": twin[1e-6], "twin_1e-4": twin[1e-4],
                          "near_twin_99pct": near, "ratio_cv_total": cv, "proportional": prop,
                          "ratio_cv_max_channel": cv_ch_max, "corr_total": corr})
    _w(pairs, "pairs.csv")

    # twin groups (tol 1e-9)
    root = union_find(len(cells), P[twin[1e-9]])
    gid, gsize = np.unique(root, return_counts=True)
    size_of = dict(zip(gid, gsize))
    cell_group_size = np.array([size_of[r] for r in root])
    in_twin = cell_group_size >= 2
    in_prop = np.zeros(len(cells), bool)
    in_prop[i[prop]] = True
    in_prop[j[prop]] = True
    total_act = T.sum()
    groups = pd.Series(gsize[gsize >= 2]).value_counts().sort_index()
    cells_df = pd.DataFrame({"CellID": cells, "excluded": excluded, "mean_activity": mean,
                             "twin_group_root": cells[root], "twin_group_size": cell_group_size, "in_proportional_pair": in_prop})
    _w(cells_df, "cells.csv")

    n_pairs, n_valid = len(P), int(pair_ok.sum())
    summary = {
        "pairs_total_unordered": n_pairs, "directed_neighbour_relations": 2 * n_pairs,
        "excluded_cells": int(excluded.sum()), "excluded_rule": f"zero variance or mean total < {MIN_MEAN_ACTIVITY}/h",
        "pairs_tested": n_valid,
        "twins_1e-9": int(twin[1e-9].sum()), "twins_1e-6": int(twin[1e-6].sum()), "twins_1e-4": int(twin[1e-4].sum()),
        "near_twins_99pct_not_exact": int(near.sum()),
        "proportional_not_twin": int(prop.sum()),
        "share_pairs_twin_or_prop": float((twin[1e-9] | prop).sum() / n_pairs),
        "twin_groups_size_ge2": int((gsize >= 2).sum()),
        "group_size_distribution": {int(k): int(v) for k, v in groups.items()},
        "cells_in_twin_group": int(in_twin.sum()), "share_cells_in_twin_group": float(in_twin.mean()),
        "share_activity_in_twin_group_cells": float(T[in_twin].sum() / total_act),
        "cells_in_twin_or_prop": int((in_twin | in_prop).sum()),
        "share_activity_twin_or_prop_cells": float(T[in_twin | in_prop].sum() / total_act),
    }
    summary["material"] = bool(summary["share_pairs_twin_or_prop"] > MATERIAL or summary["share_activity_twin_or_prop_cells"] > MATERIAL)

    # correlation distribution: ordinary pairs (tested, not twin, not proportional)
    ordinary = pair_ok & ~twin[1e-9] & ~prop
    q = np.nanpercentile(corr[ordinary], [1, 5, 10, 25, 50, 75, 90, 95, 99])
    corr_tab = pd.DataFrame({"percentile": [1, 5, 10, 25, 50, 75, 90, 95, 99], "corr_total": q})
    corr_tab = pd.concat([corr_tab, pd.DataFrame({"percentile": ["share > 0.99", "share > 0.999", "share > 0.9999"],
                                                  "corr_total": [float(np.mean(corr[ordinary] > t)) for t in (0.99, 0.999, 0.9999)]})])
    _w(corr_tab, "ordinary_corr.csv")
    # match-hour distribution among tested non-twin pairs (how close do pairs come?)
    mh = n_match[1e-9][pair_ok & ~twin[1e-9]] / H
    mh_tab = pd.DataFrame({"share_of_hours_matching_1e-9": ["0", "(0, 0.5)", "[0.5, 0.9)", "[0.9, 0.99)", "[0.99, 1)"],
                           "pairs": [int((mh == 0).sum()), int(((mh > 0) & (mh < .5)).sum()), int(((mh >= .5) & (mh < .9)).sum()),
                                     int(((mh >= .9) & (mh < .99)).sum()), int(((mh >= .99) & (mh < 1)).sum())]})
    _w(mh_tab, "match_hours_distribution.csv")

    # known areas: 5x5 pictures (group size label; 'C' centre; '=' twin with centre; '~' proportional with centre)
    tw = set(map(tuple, np.sort(P[twin[1e-9]], axis=1).tolist()))
    pp = set(map(tuple, np.sort(P[prop], axis=1).tolist()))
    pics = {}
    for name, cid in KNOWN.items():
        pos = cid - 1
        r0, c0 = divmod(pos, G)
        nb = [(r0 + dr) * G + (c0 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
              if (dr or dc) and 0 <= r0 + dr < G and 0 <= c0 + dc < G]
        n_tw = sum(tuple(sorted((pos, n))) in tw for n in nb)
        n_pp = sum(tuple(sorted((pos, n))) in pp for n in nb)
        lines = []
        for dr in range(-2, 3):
            row = []
            for dc in range(-2, 3):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < G and 0 <= c < G):
                    row.append("  ")
                    continue
                p = r * G + c
                key = tuple(sorted((pos, p)))
                mark = "C" if p == pos else "=" if key in tw else "~" if key in pp else "."
                row.append(f"{mark}{min(cell_group_size[p], 9) if cell_group_size[p] > 1 else ' '}")
            lines.append(" ".join(row))
        pics[name] = {"cell": cid, "neighbours": len(nb), "twin_neighbours": n_tw, "proportional_neighbours": n_pp,
                      "group_size": int(cell_group_size[pos]), "picture": lines,
                      "neighbour_corr_range": [float(np.nanmin(corr[(i == pos) | (j == pos)])), float(np.nanmax(corr[(i == pos) | (j == pos)]))]}
    summary["known_areas"] = pics
    with open(OUT / "stage1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in summary.items() if k != "known_areas"}, indent=1))
    print(corr_tab.to_string(index=False))
    print(mh_tab.to_string(index=False))
    for name, p in pics.items():
        print(f"\n{name} (cell {p['cell']}): twin neighbours {p['twin_neighbours']}/{p['neighbours']}, proportional {p['proportional_neighbours']}, "
              f"group size {p['group_size']}, neighbour corr {p['neighbour_corr_range'][0]:.4f}..{p['neighbour_corr_range'][1]:.4f}")
        print("\n".join("   " + ln for ln in p["picture"]))


def step_stage1map():
    """From stage1 outputs only: where twins sit (distance from Duomo, activity rank) and a coarse map;
    and how common near-perfect correlation is among ordinary pairs, by activity level."""
    cells = pd.read_csv(OUT / "cells.csv")
    pairs = pd.read_csv(OUT / "pairs.csv")
    pos = cells.CellID.to_numpy() - 1
    r, c = pos // G, pos % G
    rd, cd = divmod(KNOWN["Duomo"] - 1, G)
    dist = np.hypot(r - rd, c - cd)
    twin = cells.twin_group_size.to_numpy() >= 2
    act_rank = cells.mean_activity.rank(pct=True).to_numpy()
    bands = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 80)]
    tab = pd.DataFrame([{"distance_from_duomo_squares": f"{a}-{b}", "cells": int(((dist >= a) & (dist < b)).sum()),
                         "share_in_twin_group": float(twin[(dist >= a) & (dist < b)].mean()),
                         "median_mean_activity": float(np.median(cells.mean_activity[(dist >= a) & (dist < b)]))} for a, b in bands])
    act = pd.DataFrame([{"activity_decile": d + 1, "share_in_twin_group": float(twin[(act_rank > d / 10) & (act_rank <= (d + 1) / 10)].mean())}
                        for d in range(10)])
    _w(tab, "twins_by_distance.csv")
    _w(act, "twins_by_activity_decile.csv")
    # coarse map: 20 x 20 blocks of 5 x 5 squares; digit = number of twin cells in block (0-9, '#' = 10+)
    lines = []
    for br in range(G // 5 - 1, -1, -1):          # highest grid row at top (compass orientation not verified)
        row = ""
        for bc in range(G // 5):
            m = (r // 5 == br) & (c // 5 == bc)
            n = int(twin[m].sum())
            row += "D" if (rd // 5 == br and cd // 5 == bc) else ("." if n == 0 else (str(n) if n < 10 else "#"))
        lines.append(row)
    # near-perfect ordinary pairs by activity of the busier cell
    ordn = pairs[~pairs.excluded & ~pairs["twin_1e-9"] & ~pairs.proportional].copy()
    ma = cells.set_index("CellID").mean_activity
    ordn["min_activity_rank"] = np.minimum(ma.rank(pct=True).reindex(ordn.cell_a).to_numpy(), ma.rank(pct=True).reindex(ordn.cell_b).to_numpy())
    ordn["dec"] = np.ceil(ordn.min_activity_rank * 10).clip(1, 10).astype(int)
    hc = ordn.groupby("dec").agg(pairs=("corr_total", "size"), share_corr_gt_0999=("corr_total", lambda s: float((s > 0.999).mean())),
                                 share_corr_gt_09999=("corr_total", lambda s: float((s > 0.9999).mean()))).reset_index()
    _w(hc, "ordinary_corr_by_activity_decile.csv")
    (OUT / "twin_map.txt").write_text("\n".join(lines))
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(act.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(hc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("twin cells per 5x5 block (highest grid row at top; orientation vs north not verified; D = Duomo block):")
    print("\n".join(lines))


# ============================================================================= STAGE 2
# DECISIONS (declared before Stage 2 results):
#  - Setup as Phase 3 / 4: final pooled model ("final_63l") +1h test predictions, Phase 3 thresholds
#    (Nov 1 - Dec 9), the 24 non-holiday + 6 holiday downstream hours, unchanged diagnosis rule for
#    the solver runs (so only the transfer restriction changes), outcome replay under the
#    one-for-one activity-unit assumption.
#  - Restrictions (unordered pairs, forbidden in both directions):
#      S        strict: exact twins (tol 1e-9) + proportional + the 9 near-twins (treated as twins).
#      S-noNT   strict without near-twins (reported for comparison only).
#      L9999    S + ordinary pairs with total-activity correlation >= 0.9999 (loose proxy, upper bracket).
#      L999     S + ordinary pairs with correlation >= 0.999 (looser bracket).
#    The proxy also removes genuinely similar neighbours, so L is over-restrictive, not an estimate.
#  - Twin groups: connected components of exact twins + near-twins. Proportional pairs are forbidden
#    as pairs but not merged into groups.
#  - 2-hop ceiling: additionally forbid every pair inside the same twin group (within distance 2),
#    and for L brackets also distance-2 pairs whose correlation passes the proxy (correlations for
#    distance-2 pairs computed the same way, from the same total-activity panel).
#  - De-duplicated neighbour fraction:
#      D1 (primary): neighbours in the cell's OWN twin group are excluded (they are the same
#         measurement as the cell); each other twin group among the neighbours counts as ONE unit,
#         flagged if any member is flagged; every other neighbour is its own unit.
#         Denominator = number of units; a cell with no units gets fraction 0 (ANOMALOUS).
#      D2 (sensitivity): as D1, but the cell's own-group neighbours also count as one unit.
#  - Bootstrap: hour as the unit, 5,000 resamples, seed 0, 90% percentile intervals.
STAGE2_TH = {"L9999": 0.9999, "L999": 0.999}


def _groups_stage2(pairs):
    tw = pairs["twin_1e-9"] | pairs["near_twin_99pct"]
    edges = np.stack([pairs.cell_a[tw].to_numpy() - 1, pairs.cell_b[tw].to_numpy() - 1], 1)
    root = union_find(G * G, edges)
    _, inv, cnt = np.unique(root, return_inverse=True, return_counts=True)
    size = cnt[inv]
    return root, size


def _corr_within2(A):
    """Pearson correlation of hourly total activity for every pair at Chebyshev distance 1 or 2."""
    Z = A - A.mean(1, keepdims=True)
    Z /= np.sqrt((Z ** 2).sum(1, keepdims=True))
    out = {}
    for dr in range(0, 3):
        for dc in range(-2, 3):
            if (dr, dc) <= (0, 0):
                continue
            for r in range(G):
                r2 = r + dr
                if r2 >= G:
                    continue
                cs = np.arange(max(0, -dc), min(G, G - dc))
                a, b = r * G + cs, r2 * G + cs + dc
                v = (Z[a] * Z[b]).sum(1)
                for x, y, cv in zip(a + 1, b + 1, v):
                    out[(int(min(x, y)), int(max(x, y)))] = float(cv)
    return out


def _dedup_fraction(flags, root, size, own_unit: bool):
    """flags (10000, T). Returns de-duplicated neighbour fraction (10000, T)."""
    from src.diagnosis.diagnosis_agent import get_neighbors
    N = G * G
    M = np.full((N, 8, 8), N, dtype=np.int32)          # member positions; N = always-False pad row
    n_units = np.zeros(N, np.int16)
    for i in range(N):
        own = root[i] if size[i] >= 2 else -1
        units = {}
        for nb in get_neighbors(i + 1):
            p = nb - 1
            if size[p] >= 2 and root[p] == own:
                if not own_unit:
                    continue
                key = ("own", own)
            elif size[p] >= 2:
                key = ("grp", root[p])
            else:
                key = ("cell", p)
            units.setdefault(key, []).append(p)
        for u, (k, mem) in enumerate(units.items()):
            M[i, u, :len(mem)] = mem
        n_units[i] = len(units)
    Fx = np.vstack([flags, np.zeros((1, flags.shape[1]), bool)])
    cnt = np.zeros(flags.shape, np.int16)
    for u in range(8):
        cnt += Fx[M[:, u, :]].any(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_units[:, None] > 0, cnt / np.maximum(n_units, 1)[:, None], 0.0)
    return frac, n_units


def _prec_boot(flags, frac, actual, cols, fth, rng):
    fl, fr, ac = flags[:, cols], frac[:, cols], actual[:, cols]
    r = fl & (fr >= fth)
    a = fl & ~(fr >= fth)
    rh, rn, ah, an = (r & ac).sum(0), r.sum(0), (a & ac).sum(0), a.sum(0)
    idx = rng.integers(0, cols.sum(), size=(5000, cols.sum()))
    diff = rh[idx].sum(1) / rn[idx].sum(1) - ah[idx].sum(1) / np.maximum(an[idx].sum(1), 1)
    lo, hi = np.percentile(diff, (5, 95))
    return {"routine_flags": int(rn.sum()), "anomalous_flags": int(an.sum()),
            "precision_routine": rh.sum() / rn.sum(), "precision_anomalous": ah.sum() / an.sum(),
            "diff": rh.sum() / rn.sum() - ah.sum() / an.sum(), "lo": lo, "hi": hi,
            "verdict": "sig" if lo > 0 or hi < 0 else "n.s."}, r


def step_stage2():
    from src.diagnosis.diagnosis_agent import KNOWN_HOLIDAYS, NEIGHBOR_FRACTION_THRESHOLD, get_neighbors
    from src.evaluation.common import load_panel
    from src.evaluation.margin_tuning import DOWNSTREAM_HOLIDAY, DOWNSTREAM_NONHOLIDAY
    from src.evaluation.phase2_common import W1, FeatureStore, training_cell_means
    from src.evaluation.phase4_followup import neighbours_within, solve_hour_k
    from src.evaluation.phase4_sensitivity import P3_END, Test1h, neighbour_fraction, outcome, thresholds
    from src.optimization.solver import solve_hour

    pairs = pd.read_csv(OUT / "pairs.csv")
    root, size = _groups_stage2(pairs)
    with Step("load panel + test predictions"):
        panel = load_panel()
        store = FeatureStore(panel)
        cm = training_cell_means(panel, W1)
        thr = thresholds(panel, P3_END, 90)
        d = Test1h(panel, store, cm)
        del store
    cells = panel.cells
    assert (cells == np.arange(1, G * G + 1)).all()

    # ---------------- forbidden pair sets
    with Step("correlations within distance 2"):
        corr2 = _corr_within2(panel.activity)
    chk = pairs.sample(2000, random_state=0)
    max_corr_gap = max(abs(corr2[(int(a), int(b))] - c) for a, b, c in zip(chk.cell_a, chk.cell_b, chk.corr_total))
    key = lambda a, b: (min(a, b), max(a, b))  # noqa: E731
    strict_nont = {key(a, b) for a, b, t, p in zip(pairs.cell_a, pairs.cell_b, pairs["twin_1e-9"], pairs.proportional) if t or p}
    near = {key(a, b) for a, b, n in zip(pairs.cell_a, pairs.cell_b, pairs.near_twin_99pct) if n}
    S = strict_nont | near
    one_hop = {key(a, b) for a, b in zip(pairs.cell_a, pairs.cell_b)}
    sets1 = {"none": set(), "S-noNT": strict_nont, "S": S}
    for name, th in STAGE2_TH.items():
        sets1[name] = S | {k for k in one_hop if corr2[k] >= th}
    same_group = set()
    for c in range(1, G * G + 1):
        if size[c - 1] >= 2:
            for n in neighbours_within(c, 2):
                if size[n - 1] >= 2 and root[n - 1] == root[c - 1]:
                    same_group.add(key(c, n))
    sets2 = {name: (s | same_group if name not in ("none", "S-noNT") else s) for name, s in sets1.items()}
    for name, th in STAGE2_TH.items():
        sets2[name] = sets2[name] | {k for k, v in corr2.items() if v >= th}
    counts = pd.DataFrame([{"restriction": n, "forbidden_1hop_pairs": len(sets1[n]), "share_of_39402": len(sets1[n]) / 39402,
                            "forbidden_pairs_within_2": len(sets2[n]), "share_of_within2_pairs": len(sets2[n]) / len(corr2)}
                           for n in sets1])
    counts["near_twin_pairs"] = len(near)
    counts["same_group_pairs_within2"] = len(same_group)
    counts["max_corr_gap_vs_stage1"] = max_corr_gap
    _w(counts, "stage2_forbidden_counts.csv")
    print(counts.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("near-twin pairs:", sorted(near))

    n1 = {c: get_neighbors(c) for c in range(1, G * G + 1)}
    n2 = {c: neighbours_within(c, 2) for c in range(1, G * G + 1)}
    nb_allowed = {}
    for name in sets1:
        nb_allowed[(name, 1)] = {c: [n for n in n1[c] if key(c, n) not in sets1[name]] for c in n1}
        nb_allowed[(name, 2)] = {c: [n for n in n2[c] if key(c, n) not in sets2[name]] for c in n2}

    # ---------------- solver
    def run(fc, act, hol, m, variant, hop, restr):
        flagged = fc > m * thr
        frac = neighbour_fraction(flagged[:, None])[:, 0]
        routine = flagged & (hol | (frac >= NEIGHBOR_FRACTION_THRESHOLD))
        deficit = np.clip(fc - m * thr, 0, None)
        spare = np.clip(m * thr - fc, 0, None) if variant == "B" else np.clip(thr - fc, 0, None)
        cong = pd.DataFrame({"CellID": cells[routine], "deficit": deficit[routine]})
        allc = pd.DataFrame({"CellID": cells, "spare": spare})
        if restr == "none" and hop == 1:
            sol = solve_hour(cong, allc)                       # the unchanged solver
        else:
            sol = solve_hour_k(cong, allc, nb_allowed[(restr, hop)].__getitem__)
        return sol

    rows, mv_rows = [], []
    t_idx = {pd.Timestamp(t): int(np.where(d.targets == pd.Timestamp(t))[0][0]) for t in DOWNSTREAM_NONHOLIDAY + DOWNSTREAM_HOLIDAY}
    specs = [("V0", 1, "V0"), ("B m=0.96", 1, "B"), ("PF 1-hop", 1, "PF"), ("PF 2-hop", 2, "PF")]
    with Step("solver runs"):
        for subset, hours in (("non-holiday", DOWNSTREAM_NONHOLIDAY), ("holiday", DOWNSTREAM_HOLIDAY)):
            for target in hours:
                ts = pd.Timestamp(target)
                k = t_idx[ts]
                act = panel.activity[:, panel.hours.get_loc(ts)]
                hol = ts.date() in KNOWN_HOLIDAYS
                for vname, hop, kind in specs:
                    fc = act if kind == "PF" else d.F[:, k]
                    m = 0.96 if kind == "B" else 1.0
                    base = None
                    for restr in sets1:
                        sol = run(fc, act, hol, m, "B" if kind == "B" else "V0", hop, restr)
                        if restr == "none":
                            base = sol
                        recv, give = np.zeros(len(cells)), np.zeros(len(cells))
                        for (c, n), v in sol.items():
                            recv[int(c) - 1] += v
                            give[int(n) - 1] += v
                        o = outcome(act, thr, recv, give)
                        rows.append({"subset": subset, "target": target, "variant": vname, "restriction": restr, **o})
                    fset = sets1 if hop == 1 else sets2
                    for restr in sets1:
                        if restr == "none":
                            continue
                        tot = sum(base.values())
                        bad = sum(v for (c, n), v in base.items() if key(int(c), int(n)) in fset[restr])
                        nbad = sum(1 for (c, n) in base if key(int(c), int(n)) in fset[restr])
                        mv_rows.append({"subset": subset, "target": target, "variant": vname, "restriction": restr,
                                        "moved": tot, "moved_forbidden": bad, "moves": len(base), "moves_forbidden": nbad})
            print(f"  {subset} done")
    res = pd.DataFrame(rows)
    _w(res, "stage2_solver_by_hour.csv")
    mvs = pd.DataFrame(mv_rows)
    _w(mvs, "stage2_moves_forbidden_by_hour.csv")
    agg = res.groupby(["subset", "variant", "restriction"], sort=False)[["excess_before", "net_reduction", "gross_relief", "donor_harm", "moved"]].sum().reset_index()
    agg["net_share"] = agg.net_reduction / agg.excess_before
    mvagg = mvs.groupby(["subset", "variant", "restriction"], sort=False)[["moved", "moved_forbidden", "moves", "moves_forbidden"]].sum().reset_index()
    mvagg["share_amount"] = mvagg.moved_forbidden / mvagg.moved
    mvagg["share_moves"] = mvagg.moves_forbidden / mvagg.moves
    # hour bootstrap of (restricted - unrestricted) net reduction, non-holiday
    rng = np.random.default_rng(0)
    boot = []
    nh = res[res.subset == "non-holiday"]
    for vname, *_ in specs:
        base = nh[(nh.variant == vname) & (nh.restriction == "none")].set_index("target").net_reduction
        for restr in sets1:
            if restr == "none":
                continue
            diff = (nh[(nh.variant == vname) & (nh.restriction == restr)].set_index("target").net_reduction - base).to_numpy()
            idx = rng.integers(0, len(diff), size=(5000, len(diff)))
            lo, hi = np.percentile(diff[idx].sum(1), (5, 95))
            boot.append({"variant": vname, "restriction": restr, "net_diff": diff.sum(), "lo": lo, "hi": hi,
                         "verdict": "sig" if lo > 0 or hi < 0 else "n.s."})
    _w(agg, "stage2_solver_summary.csv")
    _w(mvagg, "stage2_moves_forbidden_summary.csv")
    _w(pd.DataFrame(boot), "stage2_solver_bootstrap.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(mvagg.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(pd.DataFrame(boot).to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    # ---------------- diagnosis rule
    with Step("diagnosis rule"):
        flags = d.F > thr[:, None]
        actual = d.Y > thr[:, None]
        excess = np.clip(d.Y - thr[:, None], 0, None)
        in_group = (size >= 2)[:, None]
        sets = {"24 downstream hours": np.isin(d.targets, pd.DatetimeIndex(DOWNSTREAM_NONHOLIDAY)), "311 non-holiday hours": ~d.holiday}
        share_rows = []
        for sname, cols in sets.items():
            f, a, e = flags[:, cols], actual[:, cols], excess[:, cols]
            g = np.broadcast_to(in_group, f.shape)
            share_rows.append({"set": sname, "hours": int(cols.sum()),
                               "share_flags_in_twin_cells": f[g].sum() / f.sum(),
                               "share_true_positives_in_twin_cells": (f & a)[g].sum() / (f & a).sum(),
                               "share_exceedances_in_twin_cells": a[g].sum() / a.sum(),
                               "share_actual_excess_units_in_twin_cells": e[g].sum() / e.sum(),
                               "share_cell_hours_in_twin_cells": float(in_group.mean()),
                               "precision_twin_cells": (f & a)[g].sum() / f[g].sum(),
                               "precision_other_cells": (f & a)[~g].sum() / f[~g].sum()})
        _w(pd.DataFrame(share_rows), "stage2_twin_shares.csv")
        print(pd.DataFrame(share_rows).T.to_string())
        rng = np.random.default_rng(0)
        frac0 = neighbour_fraction(flags)
        defs = {"original": (frac0, None)}
        for nm, own in (("D1 dedup, own group excluded", False), ("D2 dedup, own group = 1 unit", True)):
            fr, nu = _dedup_fraction(flags, root, size, own)
            defs[nm] = (fr, nu)
        prow = []
        for sname, cols in sets.items():
            for fth in (0.2, 0.3, 0.4):
                r0 = None
                for nm, (fr, nu) in defs.items():
                    st, rmask = _prec_boot(flags, fr, actual, cols, fth, rng)
                    if nm == "original":
                        r0 = rmask
                    changed = int((rmask != r0)[flags[:, cols]].sum())
                    prow.append({"set": sname, "fraction": fth, "definition": nm, **st, "flags_changing_class_vs_original": changed,
                                 "flags_with_zero_units": int((flags[:, cols] & (nu[:, None] == 0 if nu is not None else False)).sum()) if nu is not None else 0})
        pr = pd.DataFrame(prow)
        _w(pr, "stage2_diagnosis_precision.csv")
        print(pr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    steps = {"stage1": step_stage1, "stage1map": step_stage1map, "stage2": step_stage2}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        raise SystemExit(f"usage: python {Path(__file__).name} {{{'|'.join(steps)}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    with Step(f"TOTAL {sys.argv[1]}"):
        steps[sys.argv[1]]()
    pd.DataFrame(Step.log).to_csv(OUT / f"timing_{sys.argv[1]}.csv", index=False)
