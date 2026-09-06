"""FastAPI service for BalanceGrid."""
from fastapi import FastAPI
from pydantic import BaseModel

from src.diagnosis.diagnosis_agent import diagnose

app = FastAPI(title="BalanceGrid API", version="0.1.0")


class DiagnosisRequest(BaseModel):
    forecast: float | None = None
    actual: float | None = None
    threshold: float = 0.0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/diagnose")
def run_diagnosis(request: DiagnosisRequest) -> dict:
    return diagnose(request.model_dump())
