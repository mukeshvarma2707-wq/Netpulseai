"""
src/optimization/solver.py

Computes capacity reallocation for ROUTINE congested cells using a
constrained linear program (Google OR-Tools) — never the LLM. This is
BalanceGrid's central design principle: the solver guarantees every
recommendation is actually feasible; an LLM asked to invent numbers has no
such guarantee.

DESIGN DECISION (flagging explicitly): solving each congested cell's
reallocation INDEPENDENTLY would risk double-allocating the same neighbor's
spare capacity to two different congested cells that happen to share that
neighbor. The correct fix, used here: solve ONE JOINT LP PER HOUR, covering
every congested cell and its real neighbors together, so a shared neighbor's
spare capacity can never be promised to more than one cell at once.

WHAT "CAPACITY" MEANS HERE (since the dataset has no true capacity ceiling —
recall the k-constant issue: activity values are scaled by an undisclosed
constant, so there's no real-world capacity number to work against):
  - deficit of a congested cell c  = forecast_c - threshold_c  (how far over
    its own normal ceiling it's predicted to go)
  - spare capacity of a neighbor n = max(0, threshold_n - forecast_n)  (how
    much n could absorb before IT would also cross its own threshold)
This is a self-consistent, defensible definition using only data we
actually have, not an invented number.

THE LP, PER HOUR:
    Variables:      x[c, n] >= 0  for every congested cell c and each of its
                     real neighbors n
    Constraint 1:    sum_n x[c, n] <= deficit_c            (per congested cell)
    Constraint 2:    sum_c x[c, n] <= spare_n              (per neighbor,
                     aggregated across every congested cell that might draw
                     from it — this is what prevents double-allocation)
    Objective:       maximize total deficit covered across the whole city
                     for that hour

REQUIREMENTS:
    pip install pandas numpy ortools pyarrow

INPUT:
    data/raw/cdr_with_congestion_flags.parquet (or .csv)

RUN:
    python src/optimization/solver.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ACTIVITY_COLS = ["smsin", "smsout", "callin", "callout", "internet"]
GRID_SIZE = 100
NEIGHBOR_FRACTION_THRESHOLD = 0.3
KNOWN_HOLIDAYS = {pd.Timestamp("2013-11-01").date()}
HORIZON = 1  # start with +1h; the same logic extends to other horizons


# ---- Shared helpers (kept in sync with diagnosis_agent.py; duplicated
# deliberately here for robustness rather than a cross-folder import) ----

def cell_id_to_xy(cell_id: int) -> tuple:
    y, x = divmod(cell_id - 1, GRID_SIZE)
    return x + 1, y + 1


def xy_to_cell_id(x: int, y: int) -> int:
    return (y - 1) * GRID_SIZE + x


def get_neighbors(cell_id: int) -> list:
    x, y = cell_id_to_xy(cell_id)
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
                neighbors.append(xy_to_cell_id(nx, ny))
    return neighbors


def load_data() -> pd.DataFrame:
    parquet_path = DATA_DIR / "cdr_with_congestion_flags.parquet"
    csv_path = DATA_DIR / "cdr_with_congestion_flags.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Couldn't find cdr_with_congestion_flags file in {DATA_DIR}.")
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
    present_cols = [c for c in ACTIVITY_COLS if c in df.columns]
    if "total_activity" not in df.columns:
        df["total_activity"] = df[present_cols].sum(axis=1)
    return df


def compute_forecast_and_risk(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values(["CellID", "datetime"]).reset_index(drop=True)
    grouped = df.groupby("CellID")["total_activity"]
    df["forecast"] = grouped.shift(24 - horizon)
    df["target_datetime"] = df["datetime"] + pd.to_timedelta(horizon, unit="h")
    df["deficit"] = (df["forecast"] - df["congestion_threshold"]).clip(lower=0)
    df["spare"] = (df["congestion_threshold"] - df["forecast"]).clip(lower=0)
    df = df.dropna(subset=["forecast"])
    return df


def classify_routine(df: pd.DataFrame) -> pd.DataFrame:
    """Reruns the diagnosis classification (kept in sync with diagnosis_agent.py)
    so the solver only acts on ROUTINE cases, exactly as designed."""
    is_risk = df["deficit"] > 0
    flagged = df[is_risk].copy()
    at_risk_by_hour = flagged.groupby(
        flagged["target_datetime"].values.astype("datetime64[h]")
    )["CellID"].apply(set).to_dict()

    def classify(row):
        target_date = row["target_datetime"].date()
        source_date = (row["target_datetime"] - pd.to_timedelta(24, unit="h")).date()
        if target_date in KNOWN_HOLIDAYS:
            return True
        hour_key = np.datetime64(row["target_datetime"], "h")
        at_risk_cells = at_risk_by_hour.get(hour_key, set())
        neighbors = get_neighbors(row["CellID"])
        frac = sum(1 for n in neighbors if n in at_risk_cells) / len(neighbors) if neighbors else 0
        if frac >= NEIGHBOR_FRACTION_THRESHOLD:
            return True
        if source_date in KNOWN_HOLIDAYS:
            return False  # holiday-based forecast basis -> treated as anomalous, not auto-handled
        return False

    flagged["is_routine"] = flagged.apply(classify, axis=1)
    return flagged


def solve_hour(congested_cells: pd.DataFrame, all_cells_this_hour: pd.DataFrame):
    """One joint LP for a single hour: which congested cells draw how much
    from which real neighbors, without double-allocating any neighbor's
    spare capacity."""
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("Could not create OR-Tools GLOP solver.")

    spare_lookup = all_cells_this_hour.set_index("CellID")["spare"].to_dict()

    x_vars = {}  # (congested_cell, neighbor) -> LP variable
    neighbor_usage = {}  # neighbor -> list of variables drawing from it

    for _, row in congested_cells.iterrows():
        c = row["CellID"]
        deficit = row["deficit"]
        neighbors = [n for n in get_neighbors(c) if n in spare_lookup and spare_lookup[n] > 0]
        cell_vars = []
        for n in neighbors:
            var = solver.NumVar(0, spare_lookup[n], f"x_{c}_{n}")
            x_vars[(c, n)] = var
            cell_vars.append(var)
            neighbor_usage.setdefault(n, []).append(var)
        if cell_vars:
            solver.Add(sum(cell_vars) <= deficit)

    # Constraint 2: a neighbor's spare capacity can't be over-allocated across
    # multiple congested cells drawing from it simultaneously
    for n, var_list in neighbor_usage.items():
        solver.Add(sum(var_list) <= spare_lookup[n])

    if not x_vars:
        return {}

    solver.Maximize(sum(x_vars.values()))
    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {}

    return {k: v.solution_value() for k, v in x_vars.items() if v.solution_value() > 1e-6}


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")

    df = compute_forecast_and_risk(df, HORIZON)
    routine_flagged = classify_routine(df)
    routine_cells = routine_flagged[routine_flagged["is_routine"]]
    print(f"\n{len(routine_cells):,} ROUTINE congested (cell, hour) combinations to solve for.")

    unique_hours = sorted(routine_cells["target_datetime"].unique())
    print(f"Spanning {len(unique_hours)} distinct target hours.")

    all_solutions = {}
    total_deficit_covered = 0.0
    total_deficit_needed = 0.0
    n_fully_resolved = 0

    for hour in unique_hours:
        congested_this_hour = routine_cells[routine_cells["target_datetime"] == hour]
        all_cells_this_hour = df[df["target_datetime"] == hour]
        solution = solve_hour(congested_this_hour, all_cells_this_hour)
        all_solutions[hour] = solution

        for _, row in congested_this_hour.iterrows():
            c = row["CellID"]
            covered = sum(v for (cc, n), v in solution.items() if cc == c)
            total_deficit_covered += covered
            total_deficit_needed += row["deficit"]
            if covered >= row["deficit"] - 1e-6:
                n_fully_resolved += 1

    print(f"\n--- RESULTS ---")
    print(f"Total deficit needed across all ROUTINE cases: {total_deficit_needed:,.1f}")
    print(f"Total deficit covered by reallocation:         {total_deficit_covered:,.1f}")
    if total_deficit_needed > 0:
        print(f"Overall coverage rate: {total_deficit_covered/total_deficit_needed:.1%}")
    print(f"Fully resolved cases: {n_fully_resolved:,} / {len(routine_cells):,} "
          f"({n_fully_resolved/len(routine_cells):.1%})" if len(routine_cells) else "")

    print("\n--- Sample solved hour, showing actual reallocation recommendations ---")
    for hour in unique_hours[:1]:
        sol = all_solutions[hour]
        print(f"Hour: {hour}")
        for (c, n), amount in list(sol.items())[:10]:
            print(f"  Move {amount:.2f} units from Cell {n} -> Cell {c}")
        if not sol:
            print("  (no reallocation needed or possible this hour)")

    # Persist ALL reallocation results (not just the printed sample) so the
    # report-generation agent can consume them.
    records = []
    for hour, sol in all_solutions.items():
        for (c, n), amount in sol.items():
            records.append({"target_datetime": hour, "congested_cell": c, "source_cell": n, "amount_moved": amount})
    solutions_df = pd.DataFrame(records)
    out_path = DATA_DIR / "reallocation_results.parquet"
    try:
        solutions_df.to_parquet(out_path, index=False)
        print(f"\nSaved {len(solutions_df):,} reallocation records to {out_path}")
    except ImportError:
        csv_out = DATA_DIR / "reallocation_results.csv"
        solutions_df.to_csv(csv_out, index=False)
        print(f"\n'pyarrow' not installed — saved as CSV instead: {csv_out}")

    # Also persist per-case coverage summary (deficit vs. covered), needed by
    # the report agent to state HOW MUCH of each case was actually resolved.
    coverage_records = []
    for _, row in routine_cells.iterrows():
        c, hour = row["CellID"], row["target_datetime"]
        covered = sum(v for (cc, n), v in all_solutions.get(hour, {}).items() if cc == c)
        coverage_records.append({
            "CellID": c, "target_datetime": hour, "deficit": row["deficit"],
            "covered": covered, "fully_resolved": covered >= row["deficit"] - 1e-6,
        })
    coverage_df = pd.DataFrame(coverage_records)
    cov_out = DATA_DIR / "reallocation_coverage.parquet"
    try:
        coverage_df.to_parquet(cov_out, index=False)
        print(f"Saved {len(coverage_df):,} coverage records to {cov_out}")
    except ImportError:
        coverage_df.to_csv(DATA_DIR / "reallocation_coverage.csv", index=False)
        print(f"'pyarrow' not installed — saved coverage as CSV instead.")

    print("\nDone. Paste the printed output back so we can validate the solver's behavior together.")