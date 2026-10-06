"""
src/evaluation/promotion_v2_dashtest.py  (promotion v2, stage 2: headless dashboard check)

Starts the API on port 8000 (the dashboard's hard-coded address) in the given mode, runs
src/dashboard/app.py headless with streamlit.testing.AppTest, reports exceptions and the
subheaders / table columns shown, then stops the API server it started.

RUN: python src/evaluation/promotion_v2_dashtest.py {v1|v2}
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "experiments" / "promotion_v2" / "api"

mode = sys.argv[1]
env = dict(os.environ)
env.pop("BALANCEGRID_DB", None)
if mode == "v2":
    env["BALANCEGRID_DB"] = "v2"
proc = subprocess.Popen([str(ROOT / "venv" / "Scripts" / "python.exe"), "-m", "uvicorn", "src.api.main:app", "--port", "8000"],
                        cwd=ROOT, env=env, stdout=open(OUT / f"dash_{mode}_server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/openapi.json", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "src" / "dashboard" / "app.py"), default_timeout=120).run()
    res = {"mode": mode, "exceptions": [str(e.value) for e in at.exception],
           "subheaders": [s.value for s in at.subheader],
           "dataframe_columns": [list(d.value.columns) for d in at.dataframe],
           "captions": [c.value for c in at.caption]}
finally:
    proc.terminate()                 # the server this script started, nothing else
    proc.wait(timeout=30)
json.dump(res, open(OUT / f"dash_{mode}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
