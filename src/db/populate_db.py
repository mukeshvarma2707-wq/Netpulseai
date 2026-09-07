"""
src/db/populate_db.py

Loads the diagnosis + solver pipeline outputs into the database:
    diagnosis_results_1h.parquet    -> DiagnosedCase (every flagged case, both ROUTINE and ANOMALOUS)
    reallocation_coverage.parquet   -> merged into DiagnosedCase (deficit/covered/fully_resolved, ROUTINE cases only)
    reallocation_results.parquet    -> ReallocationSource (one row per neighbor that contributed capacity)

Safe to re-run: clears and reloads everything each time (cases, sources,
AND reports), since all of this is derived from the pipeline's output
files, not from anything generated independently in the app itself.

REQUIREMENTS:
    pip install sqlalchemy pandas pyarrow

RUN:
    python src/db/populate_db.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.database import init_db, SessionLocal
from src.db.models import DiagnosedCase, ReallocationSource, Report

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_parquet_or_csv(name: str) -> pd.DataFrame:
    p_path, c_path = DATA_DIR / f"{name}.parquet", DATA_DIR / f"{name}.csv"
    if p_path.exists():
        return pd.read_parquet(p_path)
    if c_path.exists():
        return pd.read_csv(c_path)
    raise FileNotFoundError(f"Couldn't find {name}.parquet or .csv in {DATA_DIR}.")


if __name__ == "__main__":
    init_db()

    diagnosis = load_parquet_or_csv("diagnosis_results_1h")
    print(f"Loaded {len(diagnosis):,} diagnosed cases.")

    coverage = None
    try:
        coverage = load_parquet_or_csv("reallocation_coverage")
        print(f"Loaded {len(coverage):,} coverage records.")
    except FileNotFoundError:
        print("No reallocation_coverage file found - proceeding without deficit/covered data (run solver.py to get this).")

    reallocation = None
    try:
        reallocation = load_parquet_or_csv("reallocation_results")
        print(f"Loaded {len(reallocation):,} reallocation records.")
    except FileNotFoundError:
        print("No reallocation_results file found - proceeding without source-cell data.")

    diagnosis["target_datetime_1h"] = pd.to_datetime(diagnosis["target_datetime_1h"]).astype(str)
    if coverage is not None:
        coverage["target_datetime"] = pd.to_datetime(coverage["target_datetime"]).astype(str)
    if reallocation is not None:
        reallocation["target_datetime"] = pd.to_datetime(reallocation["target_datetime"]).astype(str)

    session = SessionLocal()
    try:
        session.query(Report).delete()
        session.query(ReallocationSource).delete()
        deleted = session.query(DiagnosedCase).delete()
        print(f"Cleared {deleted:,} existing cases (and their sources/reports).")

        coverage_lookup = {}
        if coverage is not None:
            for _, row in coverage.iterrows():
                coverage_lookup[(int(row["CellID"]), row["target_datetime"])] = (
                    float(row["deficit"]), float(row["covered"]), bool(row["fully_resolved"])
                )

        case_id_lookup = {}
        cases_to_insert = []
        for _, row in diagnosis.iterrows():
            key = (int(row["CellID"]), row["target_datetime_1h"])
            cov = coverage_lookup.get(key)
            case = DiagnosedCase(
                cell_id=int(row["CellID"]),
                target_datetime=row["target_datetime_1h"],
                naive_forecast=float(row.get("naive_forecast_1h", 0) or 0),
                congestion_threshold=float(row.get("congestion_threshold", 0) or 0),
                classification=row["classification"],
                reason=row["reason"],
                deficit=cov[0] if cov else None,
                covered=cov[1] if cov else None,
                fully_resolved=cov[2] if cov else None,
            )
            cases_to_insert.append(case)

        session.add_all(cases_to_insert)
        session.flush()

        for case in cases_to_insert:
            case_id_lookup[(case.cell_id, case.target_datetime)] = case.id

        n_sources = 0
        if reallocation is not None:
            sources_to_insert = []
            for _, row in reallocation.iterrows():
                key = (int(row["congested_cell"]), row["target_datetime"])
                case_id = case_id_lookup.get(key)
                if case_id is None:
                    continue
                sources_to_insert.append(ReallocationSource(
                    case_id=case_id,
                    source_cell_id=int(row["source_cell"]),
                    amount_moved=float(row["amount_moved"]),
                ))
            session.add_all(sources_to_insert)
            n_sources = len(sources_to_insert)

        session.commit()
        print(f"Inserted {len(cases_to_insert):,} cases and {n_sources:,} reallocation source records.")
    finally:
        session.close()

    print("\nDone. Database is ready at data/raw/balancegrid.db")