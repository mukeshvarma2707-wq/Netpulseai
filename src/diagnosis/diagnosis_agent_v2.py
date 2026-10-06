"""
src/diagnosis/diagnosis_agent_v2.py  (promotion v2, stage 1)

Vectorised version of diagnosis_agent.py's rules, unchanged:
  - a (cell, target hour) is flagged when forecast > congestion_threshold (m = 1);
  - target date a known holiday            -> ROUTINE ("target date is a known holiday");
  - >= 30% of the cell's grid neighbours flagged for the same target hour (same horizon)
                                            -> ROUTINE ("k/n neighbors also at risk");
  - else, forecast source date (target - 24h) a known holiday
                                            -> ANOMALOUS ("isolated, AND forecast is based on a holiday ...");
  - else                                    -> ANOMALOUS ("isolated -- only k/n neighbors at risk").
The holiday list, the 30% fraction, the grid and the neighbour definition are imported from
diagnosis_agent.py, so the two cannot drift apart.

Also writes WATCH flags (D4): typical cells only (hotspots stay at m = 1), when
m x threshold < forecast <= threshold, with m = 0.955 / 0.95 / 0.95 / 0.94 for +1h .. +4h.
Watch flags are a separate status: they never enter the diagnosis or the solver.

INPUT : data/processed/v2/cell_forecasts_v2.parquet, data/processed/v2/thresholds_v2.parquet
OUTPUT: data/processed/v2/diagnosis_results_v2.parquet  (flagged cases, all horizons)
        data/processed/v2/watch_flags_v2.parquet         (watch flags, all horizons)
RUN   : python src/diagnosis/diagnosis_agent_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.diagnosis.diagnosis_agent import GRID_SIZE, KNOWN_HOLIDAYS, NEIGHBOR_FRACTION_THRESHOLD  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V2_DIR = ROOT / "data" / "processed" / "v2"
WATCH_MARGINS = {1: 0.955, 2: 0.95, 3: 0.95, 4: 0.94}
N_CELLS = GRID_SIZE * GRID_SIZE


def neighbour_counts(flags: np.ndarray):
    """flags: (N_CELLS, T) bool for cells 1..N_CELLS. Returns (flagged-neighbour count per cell and
    column, number of existing grid neighbours per cell), the 8-neighbourhood of get_neighbors()."""
    T = flags.shape[1]
    g = np.pad(flags.reshape(GRID_SIZE, GRID_SIZE, T).astype(np.int16), ((1, 1), (1, 1), (0, 0)))
    ones = np.pad(np.ones((GRID_SIZE, GRID_SIZE), np.int16), 1)
    cnt = np.zeros((GRID_SIZE, GRID_SIZE, T), np.int16)
    nb = np.zeros((GRID_SIZE, GRID_SIZE), np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                cnt += g[1 + dy:1 + dy + GRID_SIZE, 1 + dx:1 + dx + GRID_SIZE]
                nb += ones[1 + dy:1 + dy + GRID_SIZE, 1 + dx:1 + dx + GRID_SIZE]
    return cnt.reshape(N_CELLS, T), nb.reshape(N_CELLS)


def diagnose(fc: pd.DataFrame, thr: pd.Series):
    """fc: forecast rows of ONE horizon. thr: congestion threshold indexed by CellID.
    Returns (flagged cases with classification, watch flags)."""
    h = int(fc["horizon"].iloc[0])
    th = thr.reindex(fc["CellID"]).to_numpy()
    f = fc["forecast"].to_numpy()
    flagged = f > th
    # one (cell, target) per horizon; the neighbour rule works per target hour
    targets, tcode = np.unique(fc["target_datetime"].to_numpy(), return_inverse=True)
    ci = fc["CellID"].to_numpy() - 1
    F = np.zeros((N_CELLS, len(targets)), bool)
    F[ci[flagged], tcode[flagged]] = True
    cnt, nb = neighbour_counts(F)

    out = fc.loc[flagged].copy()
    k = cnt[ci[flagged], tcode[flagged]].astype(np.int64)
    n = nb[ci[flagged]].astype(np.int64)
    frac = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    tdt = pd.DatetimeIndex(out["target_datetime"])
    tgt_hol = np.isin(tdt.date, list(KNOWN_HOLIDAYS))
    src_hol = np.isin((tdt - pd.Timedelta(hours=24)).date, list(KNOWN_HOLIDAYS))
    nbr_ok = frac >= NEIGHBOR_FRACTION_THRESHOLD
    routine = tgt_hol | nbr_ok
    ks, ns = k.astype(str), n.astype(str)
    reason = np.where(tgt_hol, "target date is a known holiday",
             np.where(nbr_ok, np.char.add(np.char.add(np.char.add(ks, "/"), ns), " neighbors also at risk"),
             np.where(src_hol, "isolated, AND forecast is based on a holiday -- basis may be unreliable",
                      np.char.add(np.char.add(np.char.add(np.char.add("isolated -- only ", ks), "/"), ns), " neighbors at risk"))))
    out["congestion_threshold"] = th[flagged]
    out["deficit"] = f[flagged] - th[flagged]
    out["n_neighbors_at_risk"] = k
    out["n_neighbors"] = n
    out["neighbor_fraction"] = frac
    out["classification"] = np.where(routine, "ROUTINE", "ANOMALOUS")
    out["reason"] = reason

    m = WATCH_MARGINS[h]
    typical = (fc["segment"] == "typical").to_numpy()
    w = typical & (f > m * th) & (f <= th)
    watch = fc.loc[w].copy()
    watch["congestion_threshold"] = th[w]
    watch["margin"] = m
    return out, watch


def run(fc_all: pd.DataFrame, thr: pd.Series):
    diag, watch = [], []
    for h, fc in fc_all.groupby("horizon", sort=True, observed=True):
        d, w = diagnose(fc.reset_index(drop=True), thr)
        diag.append(d)
        watch.append(w)
    return pd.concat(diag, ignore_index=True), pd.concat(watch, ignore_index=True)


if __name__ == "__main__":
    from src.evaluation.common import Step
    with Step("diagnosis_agent_v2"):
        fc_all = pd.read_parquet(V2_DIR / "cell_forecasts_v2.parquet")
        thr = pd.read_parquet(V2_DIR / "thresholds_v2.parquet").set_index("CellID")["congestion_threshold"]
        assert (np.sort(fc_all["CellID"].unique()) == np.arange(1, N_CELLS + 1)).all()
        diag, watch = run(fc_all, thr)
        diag.to_parquet(V2_DIR / "diagnosis_results_v2.parquet", index=False)
        watch.to_parquet(V2_DIR / "watch_flags_v2.parquet", index=False)
    print(diag.groupby(["horizon", "forecast_kind", "classification"], observed=True).size().unstack().to_string())
    print(watch.groupby(["horizon", "forecast_kind"], observed=True).size().to_string())
    pd.DataFrame(Step.log).to_csv(V2_DIR / "timing_diagnosis_agent_v2.csv", index=False)
