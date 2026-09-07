"""
src/api/main.py

FastAPI backend for BalanceGrid. Exposes the diagnosed cases + their
reallocation sources, and lets you generate a plain-language report for
any case on demand.

ENDPOINTS:
    GET  /cases                       - list diagnosed cases (filterable by classification)
    GET  /cases/{case_id}              - full detail for one case, including reallocation sources
    POST /cases/{case_id}/report       - generate (or regenerate) a plain-language report for this case
    GET  /reports                      - list all generated reports

REQUIREMENTS:
    pip install fastapi uvicorn sqlalchemy

RUN:
    uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.database import init_db, get_session
from src.db.models import DiagnosedCase, ReallocationSource, Report

app = FastAPI(title="BalanceGrid API", version="1.0")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/cases")
def list_cases(
    classification: Optional[str] = None,
    limit: int = 20,
    session: Session = Depends(get_session),
):
    query = session.query(DiagnosedCase)
    if classification:
        query = query.filter(DiagnosedCase.classification == classification.upper())
    cases = query.order_by(DiagnosedCase.target_datetime.asc()).limit(limit).all()
    return [
        {
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
        for c in cases
    ]


@app.get("/cases/{case_id}")
def get_case(case_id: int, session: Session = Depends(get_session)):
    case = session.query(DiagnosedCase).filter(DiagnosedCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"No case with id {case_id}")

    return {
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
        }
        for r in reports
    ]