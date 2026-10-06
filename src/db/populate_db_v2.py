"""
src/db/populate_db_v2.py  (promotion v2, stage 2)

Loads the stage-1 v2 outputs into the v2 database (data/processed/v2/balancegrid_v2.db) ONLY:
    diagnosis_results_v2.parquet (+1h)    -> DiagnosedCase (ROUTINE and ANOMALOUS)
    reallocation_coverage_v2.parquet      -> deficit / covered / fully_resolved of ROUTINE cases
    reallocation_results_v2.parquet       -> ReallocationSource
    watch_flags_v2.parquet (all horizons) -> WatchFlag (advisory; never a case, never resolved)
Clears and reloads the v2 tables each time (including v2 reports), like populate_db.py does for v1.
The v1 database is never opened: only database_v2 / models_v2 are imported.

Refuses to run when commit headroom is below 6 GB.

RUN: python src/db/populate_db_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.database_v2 import DB_PATH, SessionLocal, init_db  # noqa: E402
from src.db.models_v2 import DiagnosedCase, ReallocationSource, Report, WatchFlag  # noqa: E402
from src.evaluation.common import Step  # noqa: E402
from src.evaluation.phase2_common import require_headroom  # noqa: E402

V2_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "v2"
CASE_HORIZON = 1
CHUNK = 50_000


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d %H:%M:%S")


def _insert(session, table, records):
    for i in range(0, len(records), CHUNK):
        session.execute(table.__table__.insert(), records[i:i + CHUNK])


if __name__ == "__main__":
    require_headroom("populate_db_v2")
    counts = {}
    with Step("read v2 parquet files"):
        diag = pd.read_parquet(V2_DIR / "diagnosis_results_v2.parquet", filters=[("horizon", "==", CASE_HORIZON)])
        cov = pd.read_parquet(V2_DIR / "reallocation_coverage_v2.parquet")
        mv = pd.read_parquet(V2_DIR / "reallocation_results_v2.parquet")
        watch = pd.read_parquet(V2_DIR / "watch_flags_v2.parquet")
    with Step("build records"):
        diag["target_datetime"] = _ts(diag["target_datetime"])
        cov["target_datetime"] = _ts(cov["target_datetime"])
        mv["target_datetime"] = _ts(mv["target_datetime"])
        watch["target_datetime"] = _ts(watch["target_datetime"])
        d = diag.merge(cov[["CellID", "target_datetime", "deficit", "covered", "fully_resolved"]].rename(
            columns={"deficit": "cov_deficit"}), on=["CellID", "target_datetime"], how="left")
        assert len(d) == len(diag)
        routine = d["classification"] == "ROUTINE"
        assert d.loc[routine, "covered"].notna().all() and d.loc[~routine, "covered"].isna().all()
        d["id"] = np.arange(1, len(d) + 1)
        cases = [{"id": int(r.id), "cell_id": int(r.CellID), "target_datetime": r.target_datetime,
                  "naive_forecast": float(r.forecast), "congestion_threshold": float(r.congestion_threshold),
                  "classification": r.classification, "reason": r.reason,
                  "deficit": float(r.cov_deficit) if r.classification == "ROUTINE" else None,
                  "covered": float(r.covered) if r.classification == "ROUTINE" else None,
                  "fully_resolved": bool(r.fully_resolved) if r.classification == "ROUTINE" else None,
                  "horizon": CASE_HORIZON, "forecast_kind": str(r.forecast_kind),
                  "model_version": str(r.model_version), "segment": str(r.segment)} for r in d.itertuples()]
        idmap = d.set_index(["CellID", "target_datetime"])["id"]
        mv["case_id"] = idmap.reindex(pd.MultiIndex.from_arrays([mv["congested_cell"], mv["target_datetime"]])).to_numpy()
        assert mv["case_id"].notna().all(), "a move without a ROUTINE case"
        sources = [{"case_id": int(c), "source_cell_id": int(s), "amount_moved": float(a)}
                   for c, s, a in zip(mv.case_id, mv.source_cell, mv.amount_moved)]
        watches = [{"cell_id": int(r.CellID), "target_datetime": r.target_datetime, "horizon": int(r.horizon),
                    "forecast": float(r.forecast), "threshold": float(r.congestion_threshold), "margin": float(r.margin),
                    "model_version": str(r.model_version), "forecast_kind": str(r.forecast_kind)} for r in watch.itertuples()]
    with Step("write balancegrid_v2.db"):
        init_db()
        session = SessionLocal()
        try:
            for t in (Report, ReallocationSource, DiagnosedCase, WatchFlag):
                counts[f"cleared_{t.__tablename__}"] = session.query(t).delete()
            _insert(session, DiagnosedCase, cases)
            _insert(session, ReallocationSource, sources)
            _insert(session, WatchFlag, watches)
            session.commit()
            for t in (DiagnosedCase, ReallocationSource, WatchFlag, Report):
                counts[t.__tablename__] = session.query(t).count()
        finally:
            session.close()
    counts["routine_cases"] = int(routine.sum())
    counts["anomalous_cases"] = int((~routine).sum())
    print(counts)
    print(f"Database ready at {DB_PATH}")
    pd.DataFrame(Step.log).to_csv(V2_DIR / "timing_populate_db_v2.csv", index=False)
    pd.Series(counts).to_csv(V2_DIR / "populate_db_v2_counts.csv", header=["rows"])
