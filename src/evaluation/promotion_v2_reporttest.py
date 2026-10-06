"""
src/evaluation/promotion_v2_reporttest.py  (promotion v2, stage 2: POST /report in v2 mode)

Runs POST /cases/{id}/report in v2 mode against a COPY of balancegrid_v2.db, never the original
and never the v1 database. Uses FastAPI's TestClient in-process with get_session overridden to the
copy; the client is not entered as a context manager, so the startup hook (init_db on the original
v2 file) does not run. LLM credentials are removed from the environment, so report_agent uses its
offline template and nothing is sent to an external service. SHA-256 and mtime of the v1 database
and the original v2 database are recorded before and after.

RUN: python src/evaluation/promotion_v2_reporttest.py
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "experiments" / "promotion_v2" / "api"
V1_DB = ROOT / "data" / "raw" / "balancegrid.db"
V2_DB = ROOT / "data" / "processed" / "v2" / "balancegrid_v2.db"
COPY = OUT / "balancegrid_v2_COPY_for_report_test.db"


def state(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "mtime_ns": p.stat().st_mtime_ns}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    before = {"v1": state(V1_DB), "v2_original": state(V2_DB)}
    shutil.copy2(V2_DB, COPY)
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "OPENAI_API_KEY"):
        os.environ.pop(k, None)
    os.environ["BALANCEGRID_DB"] = "v2"
    sys.path.insert(0, str(ROOT))

    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import src.api.main as api

    assert api.V2
    engine = create_engine(f"sqlite:///{COPY}", connect_args={"check_same_thread": False})
    Copy = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def copy_session():
        s = Copy()
        try:
            yield s
        finally:
            s.close()

    api.app.dependency_overrides[api.get_session] = copy_session
    client = TestClient(api.app)            # not a context manager: no startup hook
    routine = client.get("/cases", params={"classification": "ROUTINE", "date": "2013-12-18", "limit": 1}).json()[0]
    anomalous = client.get("/cases", params={"classification": "ANOMALOUS", "date": "2013-12-12", "limit": 1}).json()[0]
    res = {"cases": {"routine": routine["id"], "anomalous": anomalous["id"]}}
    for label, cid in res["cases"].items():
        r1 = client.post(f"/cases/{cid}/report")
        r2 = client.post(f"/cases/{cid}/report")          # regenerate: must update, not duplicate
        d = client.get(f"/cases/{cid}").json()
        res[label] = {"post_status": r1.status_code, "repost_status": r2.status_code,
                      "report_excerpt": r1.json().get("report_text", "")[:240],
                      "detail_has_report": d["report"] is not None, "detail_forecast_kind": d.get("forecast_kind")}
    res["post_missing_case_status"] = client.post("/cases/999999999/report").status_code
    reports = client.get("/reports").json()
    res["reports_listed"] = len(reports)
    res["reports_have_forecast_kind"] = all("forecast_kind" in r for r in reports)
    with Copy() as s:
        from sqlalchemy import text
        res["copy_report_rows"] = s.execute(text("select count(*) from reports")).scalar()
    engine.dispose()
    after = {"v1": state(V1_DB), "v2_original": state(V2_DB)}
    res["v1_db_unchanged"] = before["v1"] == after["v1"]
    res["v2_original_unchanged"] = before["v2_original"] == after["v2_original"]
    res["v2_original_sha256"] = after["v2_original"]["sha256"]
    res["v1_sha256"] = after["v1"]["sha256"]
    json.dump(res, open(OUT / "report_test_v2_copy.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
