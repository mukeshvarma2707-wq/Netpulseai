from src.diagnosis.diagnosis_agent import diagnose
from src.optimization.solver import solve_dispatch


def test_diagnosis_baseline() -> None:
    assert diagnose({"forecast": 10, "actual": 10, "threshold": 1})["diagnosis"]["status"] == "ok"


def test_dispatch_respects_supply() -> None:
    assert solve_dispatch([10, 5], [8, 7])["shortfall"] == [2, 0]
