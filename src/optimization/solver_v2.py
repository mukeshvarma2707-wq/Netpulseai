"""
src/optimization/solver_v2.py  (promotion v2, stage 1; solver variant V0)

Runs the UNCHANGED per-hour LP (solve_hour, imported from solver.py) on the v2 inputs, +1h:
  - congested cells = ROUTINE cases of diagnosis_results_v2 at +1h, deficit = forecast - threshold;
  - every cell with a +1h forecast for that target hour offers spare = max(threshold - forecast, 0).
Watch flags are not used (D5: V0 for moves). The per-case coverage loop of solver.py is replaced
by a grouped sum; the LP itself is not touched.

INPUT : data/processed/v2/{cell_forecasts_v2, thresholds_v2, diagnosis_results_v2}.parquet
OUTPUT: data/processed/v2/reallocation_results_v2.parquet   (one row per move)
        data/processed/v2/reallocation_coverage_v2.parquet  (one row per ROUTINE case)
RUN   : python src/optimization/solver_v2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.optimization.solver import solve_hour  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V2_DIR = ROOT / "data" / "processed" / "v2"
HORIZON = 1


def inputs_for_hour(fc_hour: pd.DataFrame, thr: pd.Series, routine_hour: pd.DataFrame):
    th = thr.reindex(fc_hour["CellID"]).to_numpy()
    all_cells = pd.DataFrame({"CellID": fc_hour["CellID"].to_numpy(),
                              "spare": np.clip(th - fc_hour["forecast"].to_numpy(), 0, None)})
    congested = pd.DataFrame({"CellID": routine_hour["CellID"].to_numpy(), "deficit": routine_hour["deficit"].to_numpy()})
    return congested, all_cells


def run(fc: pd.DataFrame, thr: pd.Series, diag: pd.DataFrame, hours=None):
    fc = fc[fc["horizon"] == HORIZON]
    routine = diag[(diag["horizon"] == HORIZON) & (diag["classification"] == "ROUTINE")]
    fc_by_hour = dict(tuple(fc.groupby("target_datetime", sort=True)))
    rt_by_hour = dict(tuple(routine.groupby("target_datetime", sort=True)))
    kind_of = fc.drop_duplicates("target_datetime").set_index("target_datetime")["forecast_kind"]
    moves, solve_s = [], 0.0
    for hour in (sorted(rt_by_hour) if hours is None else hours):
        if hour not in rt_by_hour:
            continue
        congested, all_cells = inputs_for_hour(fc_by_hour[hour], thr, rt_by_hour[hour])
        t0 = time.perf_counter()
        sol = solve_hour(congested, all_cells)
        solve_s += time.perf_counter() - t0
        for (c, n), v in sol.items():
            moves.append((hour, int(c), int(n), float(v)))
    mv = pd.DataFrame(moves, columns=["target_datetime", "congested_cell", "source_cell", "amount_moved"])
    mv["forecast_kind"] = mv["target_datetime"].map(kind_of).astype(str)
    cov = routine[["CellID", "target_datetime", "deficit", "forecast_kind"]].copy()
    if hours is not None:
        cov = cov[cov["target_datetime"].isin(hours)]
    got = mv.groupby(["congested_cell", "target_datetime"])["amount_moved"].sum()
    cov["covered"] = got.reindex(pd.MultiIndex.from_arrays([cov["CellID"], cov["target_datetime"]])).fillna(0.0).to_numpy()
    cov["fully_resolved"] = cov["covered"] >= cov["deficit"] - 1e-6
    return mv, cov.reset_index(drop=True), solve_s


if __name__ == "__main__":
    from src.evaluation.common import Step
    with Step("solver_v2 (V0, +1h)"):
        fc = pd.read_parquet(V2_DIR / "cell_forecasts_v2.parquet", filters=[("horizon", "==", HORIZON)])
        thr = pd.read_parquet(V2_DIR / "thresholds_v2.parquet").set_index("CellID")["congestion_threshold"]
        diag = pd.read_parquet(V2_DIR / "diagnosis_results_v2.parquet", filters=[("horizon", "==", HORIZON)])
        mv, cov, solve_s = run(fc, thr, diag)
        mv.to_parquet(V2_DIR / "reallocation_results_v2.parquet", index=False)
        cov.to_parquet(V2_DIR / "reallocation_coverage_v2.parquet", index=False)
    print(f"hours solved: {cov['target_datetime'].nunique()}, LP time {solve_s:.1f} s")
    print(cov.groupby("forecast_kind", observed=True).agg(cases=("deficit", "size"), deficit=("deficit", "sum"),
                                                          covered=("covered", "sum"), fully=("fully_resolved", "mean")).to_string())
    print(f"moves: {len(mv):,}")
    pd.DataFrame(Step.log).to_csv(V2_DIR / "timing_solver_v2.csv", index=False)
