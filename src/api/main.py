"""
src/api/main.py

FastAPI backend for BalanceGrid. Exposes the diagnosed cases + their
reallocation sources, and lets you generate a plain-language report for
any case on demand.

ENDPOINTS:
    GET  /cases                       - list diagnosed cases (filter by classification, hour or day)
    GET  /hours                       - every forecast hour with case counts
    GET  /cases/{case_id}              - full detail for one case, including reallocation sources
    POST /cases/{case_id}/report       - generate (or regenerate) a plain-language report for this case
    GET  /reports                      - list all generated reports

REQUIREMENTS:
    pip install fastapi uvicorn sqlalchemy

RUN:
    uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs

V2 DATABASE (promotion stage 2): with the environment variable BALANCEGRID_DB=v2 the API reads
data/processed/v2/balancegrid_v2.db through src/db/database_v2.py and models_v2.py instead.
Only then: every case / hour / report carries forecast_kind ("validation" = Dec 10-16, used for
model size and margin selection; "test" = Dec 17 on), /hours adds a "watch" count (+1h), and
GET /watch lists ADVISORY watch flags with their measured precision. Watch flags live in their
own table: never counted as resolved, never in /cases, never sent to the solver. Without the
variable, behaviour is exactly as before (v1 database, no /watch route, no extra fields).
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import case as sql_case, func
from sqlalchemy.orm import Session, selectinload

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
V2 = os.environ.get("BALANCEGRID_DB", "").strip().lower() == "v2"
if V2:
    from src.db.database_v2 import META_PATH, init_db, get_session
    from src.db.models_v2 import DiagnosedCase, ReallocationSource, Report, WatchFlag
    V2_META = json.loads(META_PATH.read_text())
else:
    from src.db.database import init_db, get_session
    from src.db.models import DiagnosedCase, ReallocationSource, Report

app = FastAPI(title="BalanceGrid API (v2 database)", version="2.0") if V2 else FastAPI(title="BalanceGrid API", version="1.0")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/cases")
def list_cases(
    classification: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    target_datetime: Optional[str] = None,
    date: Optional[str] = None,
    include_sources: bool = False,
    session: Session = Depends(get_session),
):
    """target_datetime filters to one exact hour ("2013-11-02 08:00:00");
    date filters to one day ("2013-11-02"). include_sources adds each case's
    reallocation sources, so the UI doesn't need a request per case."""
    query = session.query(DiagnosedCase).options(selectinload(DiagnosedCase.report))
    if include_sources:
        query = query.options(selectinload(DiagnosedCase.sources))
    if classification:
        query = query.filter(DiagnosedCase.classification == classification.upper())
    if target_datetime:
        query = query.filter(DiagnosedCase.target_datetime == target_datetime)
    if date:
        query = query.filter(DiagnosedCase.target_datetime.between(f"{date} 00:00:00", f"{date} 23:59:59"))
    cases = (
        query.order_by(DiagnosedCase.target_datetime.asc(), DiagnosedCase.id.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    result = []
    for c in cases:
        row = {
            "id": c.id,
            "cell_id": c.cell_id,
            "target_datetime": c.target_datetime,
            "classification": c.classification,
            "naive_forecast": c.naive_forecast,
            "congestion_threshold": c.congestion_threshold,
            "deficit": c.deficit,
            "covered": c.covered,
            "fully_resolved": c.fully_resolved,
            "has_report": c.report is not None,
        }
        if include_sources:
            row["sources"] = [
                {"source_cell_id": s.source_cell_id, "amount_moved": s.amount_moved}
                for s in c.sources
            ]
        if V2:
            row.update(_v2_case_fields(c))
        result.append(row)
    return result


@app.get("/hours")
def list_hours(session: Session = Depends(get_session)):
    """Every forecast hour that has diagnosed cases, with counts per class."""
    rows = (
        session.query(
            DiagnosedCase.target_datetime,
            func.count(),
            func.sum(sql_case((DiagnosedCase.classification == "ANOMALOUS", 1), else_=0)),
        )
        .group_by(DiagnosedCase.target_datetime)
        .order_by(DiagnosedCase.target_datetime.asc())
        .all()
    )
    if V2:
        return _v2_hours(session, rows)
    return [
        {"target_datetime": t, "total": total, "anomalous": int(anom or 0), "routine": total - int(anom or 0)}
        for t, total, anom in rows
    ]


@app.get("/cases/{case_id}")
def get_case(case_id: int, session: Session = Depends(get_session)):
    case = session.query(DiagnosedCase).filter(DiagnosedCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"No case with id {case_id}")

    detail = {
        "id": case.id,
        "cell_id": case.cell_id,
        "target_datetime": case.target_datetime,
        "classification": case.classification,
        "reason": case.reason,
        "naive_forecast": case.naive_forecast,
        "congestion_threshold": case.congestion_threshold,
        "deficit": case.deficit,
        "covered": case.covered,
        "fully_resolved": case.fully_resolved,
        "sources": [
            {"source_cell_id": s.source_cell_id, "amount_moved": s.amount_moved}
            for s in case.sources
        ],
        "report": {
            "report_text": case.report.report_text,
            "generated_at": str(case.report.generated_at),
        } if case.report else None,
    }
    if V2:
        detail.update(_v2_case_fields(case))
        detail["forecast_kind_label"] = V2_META["forecast_kind_label"].get(case.forecast_kind)
    return detail


@app.post("/cases/{case_id}/report")
def generate_report(case_id: int, session: Session = Depends(get_session)):
    """Generates a plain-language report for this case, using the same
    logic as report_agent.py (mock template, or a real LLM if credentials
    are configured)."""
    case = session.query(DiagnosedCase).filter(DiagnosedCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"No case with id {case_id}")

    agents_dir = Path(__file__).resolve().parents[2] / "src" / "agents"
    sys.path.insert(0, str(agents_dir))
    import report_agent as ra

    state = {
        "CellID": case.cell_id,
        "target_datetime": case.target_datetime,
        "classification": case.classification,
        "reason": case.reason,
        "deficit": case.deficit,
        "covered": case.covered or 0,
        "reallocation_sources": [(s.source_cell_id, s.amount_moved) for s in case.sources],
    }

    if case.classification == "ROUTINE":
        result_state = ra.generate_routine_report(state)
    else:
        result_state = ra.generate_anomalous_report(state)

    report_text = result_state["report"]

    if case.report:
        case.report.report_text = report_text
    else:
        session.add(Report(case_id=case.id, report_text=report_text))
    session.commit()

    return {"status": "generated", "report_text": report_text}


@app.get("/reports")
def list_reports(session: Session = Depends(get_session)):
    reports = session.query(Report).all()
    return [
        {
            "case_id": r.case_id,
            "cell_id": r.case.cell_id,
            "target_datetime": r.case.target_datetime,
            "classification": r.case.classification,
            "generated_at": str(r.generated_at),
            **({"forecast_kind": r.case.forecast_kind} if V2 else {}),
        }
        for r in reports
    ]


# ============================================================================= v2 only
def _v2_case_fields(c) -> dict:
    return {"forecast_kind": c.forecast_kind, "horizon": c.horizon, "model_version": c.model_version, "segment": c.segment}


def _v2_watch_precision(horizon: int, kind: str):
    return V2_META["watch_precision"].get(f"{horizon}h_{kind}")


def _v2_hours(session, case_rows):
    """/hours in v2: the v1 fields plus forecast_kind and the +1h watch count. Hours that have
    watch flags but no case are included with zero case counts."""
    kinds = dict(session.query(DiagnosedCase.target_datetime, DiagnosedCase.forecast_kind)
                 .group_by(DiagnosedCase.target_datetime).all())
    watch = {t: (n, k) for t, n, k in session.query(WatchFlag.target_datetime, func.count(), WatchFlag.forecast_kind)
             .filter(WatchFlag.horizon == 1).group_by(WatchFlag.target_datetime).all()}
    out = {t: {"target_datetime": t, "total": total, "anomalous": int(anom or 0), "routine": total - int(anom or 0),
               "watch": watch.get(t, (0, None))[0], "forecast_kind": kinds.get(t)}
           for t, total, anom in case_rows}
    for t, (n, k) in watch.items():
        if t not in out:
            out[t] = {"target_datetime": t, "total": 0, "anomalous": 0, "routine": 0, "watch": n, "forecast_kind": k}
    return [out[t] for t in sorted(out)]


if V2:
    @app.get("/watch")
    def list_watch(
        target_datetime: Optional[str] = None,
        date: Optional[str] = None,
        horizon: int = 1,
        forecast_kind: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        session: Session = Depends(get_session),
    ):
        """ADVISORY watch flags (typical cells, margin x threshold < forecast <= threshold).
        Not cases: never counted as resolved, never in /cases, never sent to the solver."""
        if horizon not in (1, 2, 3, 4):
            raise HTTPException(status_code=422, detail="horizon must be 1, 2, 3 or 4")
        query = session.query(WatchFlag).filter(WatchFlag.horizon == horizon)
        if target_datetime:
            query = query.filter(WatchFlag.target_datetime == target_datetime)
        if date:
            query = query.filter(WatchFlag.target_datetime.between(f"{date} 00:00:00", f"{date} 23:59:59"))
        if forecast_kind:
            query = query.filter(WatchFlag.forecast_kind == forecast_kind.lower())
        total = query.count()
        items = query.order_by(WatchFlag.target_datetime.asc(), WatchFlag.cell_id.asc()).offset(offset).limit(limit).all()
        return {
            "status": "advisory",
            "note": V2_META["watch_status"],
            "definition": V2_META["watch_definition"],
            "horizon": horizon,
            "margin": V2_META["watch_margins"][str(horizon)],
            "watch_precision": {k: _v2_watch_precision(horizon, k) for k in ("validation", "test")},
            "watch_precision_note": V2_META["watch_precision_note"],
            "forecast_kind_label": V2_META["forecast_kind_label"],
            "total": total,
            "items": [
                {"id": w.id, "cell_id": w.cell_id, "target_datetime": w.target_datetime, "horizon": w.horizon,
                 "forecast": w.forecast, "threshold": w.threshold, "margin": w.margin, "model_version": w.model_version,
                 "forecast_kind": w.forecast_kind, "advisory": True,
                 "watch_precision": _v2_watch_precision(w.horizon, w.forecast_kind)}
                for w in items
            ],
        }