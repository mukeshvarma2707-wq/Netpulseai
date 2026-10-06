"""
src/optimization/solver.py

Computes capacity reallocation for ROUTINE congested cells using a
constrained linear program (Google OR-Tools) -- never the LLM.

UPDATED: now loads the FORECAST from generate_forecasts.py's saved
LightGBM predictions, instead of computing an internal seasonal-naive
forecast -- same reasoning as diagnosis_agent.py's update. The solver's
actual optimization logic (joint per-hour LP, no double-allocation of a
shared neighbor's spare capacity) is UNCHANGED.

REQUIREMENTS:
    pip install pandas numpy ortools pyarrow

INPUT:
    data/raw/cdr_with_congestion_flags.parquet (or .csv)
    data/raw/cell_forecasts.parquet (or .csv) -- from generate_forecasts.py

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
KNOWN_HOLIDAYS = {
    pd.Timestamp("2013-11-01").date(),
    pd.Timestamp("2013-12-08").date(),
    pd.Timestamp("2013-12-25").date(),
    pd.Timestamp("2013-12-26").date(),
    pd.Timestamp("2014-01-01").date(),
}
HORIZON = 1


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


def load_forecasts() -> pd.DataFrame:
    parquet_path = DATA_DIR / "cell_forecasts.parquet"
    csv_path = DATA_DIR / "cell_forecasts.csv"
    if parquet_path.exists():
        forecasts = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        forecasts = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Couldn't find cell_forecasts file in {DATA_DIR}. Run generate_forecasts.py first.")
    dt_cols = ["datetime"] + [c for c in forecasts.columns if c.startswith("target_datetime_")]
    for col in dt_cols:
        if not pd.api.types.is_datetime64_any_dtype(forecasts[col]):
            forecasts[col] = pd.to_datetime(forecasts[col])
    return forecasts


def attach_forecast_and_risk(df: pd.DataFrame, forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    cols_needed = ["CellID", "datetime", f"forecast_{horizon}h", f"target_datetime_{horizon}h"]
    df = df.merge(forecasts[cols_needed], on=["CellID", "datetime"], how="left")
    df = df.rename(columns={f"forecast_{horizon}h": "forecast", f"target_datetime_{horizon}h": "target_datetime"})
    df["deficit"] = (df["forecast"] - df["congestion_threshold"]).clip(lower=0)
    df["spare"] = (df["congestion_threshold"] - df["forecast"]).clip(lower=0)
    df = df.dropna(subset=["forecast"])
    return df


def classify_routine(df: pd.DataFrame) -> pd.DataFrame:
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
            return False
        return False

    flagged["is_routine"] = flagged.apply(classify, axis=1)
    return flagged


def solve_hour(congested_cells: pd.DataFrame, all_cells_this_hour: pd.DataFrame):
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("Could not create OR-Tools GLOP solver.")

    spare_lookup = all_cells_this_hour.set_index("CellID")["spare"].to_dict()

    x_vars = {}
    neighbor_usage = {}

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
    forecasts = load_forecasts()
    print(f"Loaded {len(df):,} rows across {df['CellID'].nunique()} cells.")
    print(f"Loaded {len(forecasts):,} pre-computed LightGBM forecast rows.")

    df = attach_forecast_and_risk(df, forecasts, HORIZON)
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
    if len(routine_cells):
        print(f"Fully resolved cases: {n_fully_resolved:,} / {len(routine_cells):,} "
              f"({n_fully_resolved/len(routine_cells):.1%})")

    print("\n--- Sample solved hour, showing actual reallocation recommendations ---")
    for hour in unique_hours[:1]:
        sol = all_solutions[hour]
        print(f"Hour: {hour}")
        for (c, n), amount in list(sol.items())[:10]:
            print(f"  Move {amount:.2f} units from Cell {n} -> Cell {c}")
        if not sol:
            print("  (no reallocation needed or possible this hour)")

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
        solutions_df.to_csv(DATA_DIR / "reallocation_results.csv", index=False)

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

    print("\nDone. Paste the printed output back so we can validate the solver's behavior together.")