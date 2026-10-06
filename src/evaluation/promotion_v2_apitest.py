"""
src/evaluation/promotion_v2_apitest.py  (promotion v2, stage 2: API safety test)

Starts its own uvicorn server on a spare port (default mode or BALANCEGRID_DB=v2), sends a fixed
set of GET requests, saves the responses, stops the server it started, and records SHA-256 and
mtime of the v1 database (data/raw/balancegrid.db) before and after. Read-only requests only:
POST /cases/{id}/report writes to the database by design and is not sent.

RUN:
    python src/evaluation/promotion_v2_apitest.py capture <label> [v1|v2]
    python src/evaluation/promotion_v2_apitest.py compare <label_a> <label_b>
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "experiments" / "promotion_v2" / "api"
V1_DB = ROOT / "data" / "raw" / "balancegrid.db"
PORT = 8765
PY = ROOT / "venv" / "Scripts" / "python.exe"

V1_REQUESTS = [
    "/hours",
    "/cases",
    "/cases?classification=ANOMALOUS&limit=50",
    "/cases?classification=routine&limit=30&offset=100",
    "/cases?date=2013-12-25&limit=100&include_sources=true",
    "/cases?target_datetime=2013-11-02 08:00:00&include_sources=true&limit=100",
    "/cases/1",
    "/cases/12345",
    "/cases/999999999",
    "/reports",
    "/watch",
    "/openapi.json",
]
V2_REQUESTS = [
    "/hours",
    "/cases?limit=5",
    "/cases?date=2013-12-12&limit=5&include_sources=true",
    "/cases/1",
    "/reports",
    "/watch?target_datetime=2013-12-18 12:00:00&limit=5",
    "/watch?date=2013-12-12&horizon=4&limit=3",
    "/watch?limit=2",
]


def db_state(p: Path) -> dict:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {"sha256": h.hexdigest(), "mtime_ns": st.st_mtime_ns, "bytes": st.st_size}


def get(path: str):
    url = f"http://127.0.0.1:{PORT}" + urllib.parse.quote(path, safe="/?=&")
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "null")


def capture(label: str, mode: str):
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("BALANCEGRID_DB", None)
    if mode == "v2":
        env["BALANCEGRID_DB"] = "v2"
    before = db_state(V1_DB)
    proc = subprocess.Popen([str(PY), "-m", "uvicorn", "src.api.main:app", "--port", str(PORT)], cwd=ROOT, env=env,
                            stdout=open(OUT / f"{label}_server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/openapi.json", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")
        res, timing = {}, {}
        for path in (V2_REQUESTS if mode == "v2" else V1_REQUESTS):
            t0 = time.perf_counter()
            res[path] = get(path)
            timing[path] = round(time.perf_counter() - t0, 3)
    finally:
        proc.terminate()                 # the server this script started, nothing else
        proc.wait(timeout=30)
    after = db_state(V1_DB)
    json.dump({"mode": mode, "responses": res}, open(OUT / f"{label}_responses.json", "w"), indent=1, default=str)
    summary = {"label": label, "mode": mode, "v1_db_before": before, "v1_db_after": after,
               "v1_db_unchanged": before == after, "timing_s": timing,
               "status": {p: r[0] for p, r in res.items()}}
    json.dump(summary, open(OUT / f"{label}_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def compare(a: str, b: str):
    ra = json.load(open(OUT / f"{a}_responses.json"))["responses"]
    rb = json.load(open(OUT / f"{b}_responses.json"))["responses"]
    sa = json.load(open(OUT / f"{a}_summary.json"))
    sb = json.load(open(OUT / f"{b}_summary.json"))
    rows = []
    for p in ra:
        same = ra[p] == rb.get(p)
        rows.append({"request": p, "status": ra[p][0], "identical": same})
        print(f"{'IDENTICAL' if same else 'DIFFERENT':9s}  {ra[p][0]}  {p}")
    print("v1 db identical across all captures:",
          sa["v1_db_before"] == sa["v1_db_after"] == sb["v1_db_before"] == sb["v1_db_after"])
    json.dump(rows, open(OUT / f"compare_{a}_vs_{b}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "v1")
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
