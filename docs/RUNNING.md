# Running BalanceGrid: v1 and v2 modes

All commands are for **Windows PowerShell**, run from the project root
(`D:\Balance_grid`), using the project's virtual environment
(`venv\Scripts\python.exe`). Do not use a system Python: the study's code
expects the venv's library versions.

```powershell
cd D:\Balance_grid
.\venv\Scripts\Activate.ps1          # optional; otherwise call venv\Scripts\python.exe directly
```

## The two modes

| Mode | Selected by | Database | What it serves |
|---|---|---|---|
| **v1 (default)** | no `BALANCEGRID_DB` variable | `data\raw\balancegrid.db` | The original pipeline output, exactly as before. No `/watch` route, no extra fields. |
| **v2** | `$env:BALANCEGRID_DB = "v2"` | `data\processed\v2\balancegrid_v2.db` | Out-of-sample forecasts for Dec 10 - Jan 1. Every response carries `forecast_kind`; `/hours` adds a `watch` count; `GET /watch` lists the advisory watch flags with their measured precision. |

The variable is read **once, when the API starts**. To switch modes, stop
the API (Ctrl+C), change the variable, and start it again. The dashboard
needs no variable: it shows the v2 extras only when the API serves them.

The dashboard calls the API at `http://127.0.0.1:8000`, so run the API on
port 8000.

## v1 mode (default)

**Terminal 1, the API:**

```powershell
cd D:\Balance_grid
Remove-Item Env:BALANCEGRID_DB -ErrorAction SilentlyContinue   # make sure v2 is not set
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
```

**Terminal 2, the dashboard:**

```powershell
cd D:\Balance_grid
.\venv\Scripts\python.exe -m streamlit run src\dashboard\app.py
```

API docs: http://127.0.0.1:8000/docs. The dashboard opens at
http://localhost:8501.

## v2 mode

**Build first (once):** `data\processed\v2\balancegrid_v2.db` and
`v2_metadata.json` must exist. See "Rebuilding the v2 outputs" below.

**Terminal 1, the API:**

```powershell
cd D:\Balance_grid
$env:BALANCEGRID_DB = "v2"
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
```

**Terminal 2, the dashboard** (same command as v1; no variable needed):

```powershell
cd D:\Balance_grid
.\venv\Scripts\python.exe -m streamlit run src\dashboard\app.py
```

**Example v2 requests:**

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/hours" | Select-Object -First 3
Invoke-RestMethod "http://127.0.0.1:8000/watch?date=2013-12-18&horizon=1&limit=5"
Invoke-RestMethod "http://127.0.0.1:8000/cases?date=2013-12-12&limit=5"
```

**Back to v1:** stop the API, then:

```powershell
Remove-Item Env:BALANCEGRID_DB
```

`$env:` only lasts for the current PowerShell window, so a new window starts
in v1 mode.

## Rebuilding the v2 outputs

No model is fitted. The v2 forecasts are assembled from the saved
evaluation predictions in `data\experiments\phase2\preds\`
(`w1__final_63l__*` and `w1__hotspot_only_final__1h__*`). The scripts read
`data\raw\cdr_with_congestion_flags.parquet` (read only) and write only to
`data\processed\v2\`.

**Run in this order** (measured on this laptop):

| # | Command | Output | Time | Peak commit |
|---|---|---|---|---|
| 1 | `.\venv\Scripts\python.exe src\forecasting\thresholds_v2.py` | `thresholds_v2.parquet` | 12 s | 2.1 GB |
| 2 | `.\venv\Scripts\python.exe src\forecasting\generate_forecasts_v2.py` | `cell_forecasts_v2.parquet` (21.9M rows) | 31 s | **4.5 GB** |
| 3 | `.\venv\Scripts\python.exe src\diagnosis\diagnosis_agent_v2.py` | `diagnosis_results_v2.parquet`, `watch_flags_v2.parquet` | 9 s | 3.6 GB |
| 4 | `.\venv\Scripts\python.exe src\optimization\solver_v2.py` | `reallocation_results_v2.parquet`, `reallocation_coverage_v2.parquet` | 24 s | 1.8 GB |
| 5 | `.\venv\Scripts\python.exe src\evaluation\promotion_v2_followup.py watchprec` | `watch_precision_v2.csv`, **`v2_metadata.json`** (the v2 API needs it) | 12 s | 1.9 GB |
| 6 | `.\venv\Scripts\python.exe src\db\populate_db_v2.py` | `balancegrid_v2.db` (271,504 cases, 195,576 sources, 920,357 watch flags) | 33 s | 2.0 GB |
| (optional) | `.\venv\Scripts\python.exe src\evaluation\promotion_v2_parity.py` | parity tests and SHA-256 in `data\experiments\promotion_v2\` | 265 s | 3.3 GB |

The whole build (steps 1-6) takes about 2 minutes. Step 6 clears and
reloads the v2 tables, including any reports generated in v2 mode.

## Memory

- **Check headroom first.** Check the commit headroom before a heavy step:

  ```powershell
  .\venv\Scripts\python.exe -c "from src.evaluation.phase2_common import commit_headroom_gb as c; print(round(c(), 1), 'GB headroom')"
  ```

- **Model fits** (any refit of the evaluation models; not needed for v2) peak
  at about 4-6.3 GB of commit. **Do not start a fit with less than about 6 GB
  of headroom.** The fitting scripts check this and stop.
- **The v2 build peaks near 4.5 GB** (step 2, assembling the forecasts).
  `populate_db_v2.py` refuses to run below 6 GB of headroom. Close heavy
  applications if the check fails.
- **Avoid parallel runs:** do not run two heavy steps at once. Each loads the
  Milan panel (about 2-3 GB).

## What not to do

- Do not write anything to `data\raw` or run the v1 pipeline scripts
  (`populate_db.py`, `diagnosis_agent.py`, `solver.py`) unless you intend to
  rebuild v1. They overwrite v1 files there.
- Do not point v2 at the v1 database: `database_v2.py` refuses any path equal
  to `data\raw\balancegrid.db` or under `data\raw`.
- **`POST /cases/{id}/report` writes to whichever database the API is
  using.** With an LLM key in the environment (`OPENAI_API_KEY`, or
  `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY`), it sends case data to that
  service. Without one, it uses an offline template.

## Reproducing the evaluation

**The evaluation outputs are not in git.**
- **What is missing:** everything under `data\experiments\` (saved
  predictions, models, CSV results, logs; about 4.3 GB), as well as
  `data\raw\`, `data\processed\` and `data\trentino\`, is ignored by
  `.gitignore`.
- **To regenerate it:** re-run the evaluation scripts in phase order,
  because each phase reads the saved outputs of the earlier ones. The v2
  build (above) needs the Phase 2 and pre-Phase-3 predictions in
  `data\experiments\phase2\preds\`.

### Data downloads (Telecom Italia Big Data Challenge, Harvard Dataverse, ODbL)

| Data | Dataverse title | DOI | Put it in |
|---|---|---|---|
| Milan telecom activity | Telecommunications - SMS, Call, Internet - MI | [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV) | `data\raw\` (`sms-call-internet-mi-*.txt`, 62 daily files) |
| Trentino telecom activity | Telecommunications - SMS, Call, Internet - TN | [10.7910/DVN/QLCABU](https://doi.org/10.7910/DVN/QLCABU) | `data\trentino\` (`sms-call-internet-tn-*.txt`) |
| Trentino grid | Trentino Grid | [10.7910/DVN/FZRVSX](https://doi.org/10.7910/DVN/FZRVSX) | `data\trentino\trentino-grid.geojson` |
| Milan grid (optional; the Milan code uses the 100 × 100 ID layout) | Milano Grid | [10.7910/DVN/QJWLFU](https://doi.org/10.7910/DVN/QJWLFU) | – |

The titles were checked against the Dataverse API. The dataset paper is
Barlacchi et al., *Scientific Data* 2, 150055 (2015).

### Order, time and peak memory (measured on this laptop; from docs\BALANCEGRID_IMPROVEMENTS.md)

**Before the phases:** build the v1 inputs with the v1 pipeline. The
loader (`src\ingestion\cdr_loader.py`), then
`src\features\congestion_threshold.py`, produce
`data\raw\cdr_with_congestion_flags.parquet`. The optional
`src\features\validate_known_cells.py` runs the paper's example-cell
checks.

**Check headroom before every fitting phase.** Do not start a fit with
less than about 6 GB.

| Phase | Commands (`.\venv\Scripts\python.exe src\evaluation\...`) | Time | Peak commit |
|---|---|---|---|
| 1 Diagnosis | `flag_quality.py replicate`, `flags`, `bootstrap`, `insample`, `channels`, `holiday`; `capacity_check.py` | about 8 min | 5.6 GB |
| 2 Model size and features | `tune_trees.py leaves`, `full`, `hotspot`, `production`, `w2`; `feature_ablation.py fit w1 1,4`, `fit w2 1,4`, `decide`, `final`, `report` | about 5-6 h (about 30 full-grid fits of 3-16 min each) | 5.7 GB |
| Pre-Phase-3 checks | `pre_phase3_checks.py leaves1`, `trees4`, `hotspot`, `leafrule` | about 1.5 h (four 63-leaf fits of 13-22 min) | **6.3 GB** |
| 3 Margins | `margin_tuning.py flags`, `downstream`; `phase3_checks.py rates`, `recount`, `h3`, `replay`, `w2fit`, `w2` | about 30 min (`w2fit`: two fits of about 7 min) | 5.1 GB |
| 4 Sensitivity | `phase4_sensitivity.py ranges`, `rule`, `holiday`, `pct`, `hotcut`; `phase4_followup.py replayboot`, `reconcile`, `naive`, `reach2` | about 5 min (under 1 min per step) | 3.4 GB |
| 5 Trentino prep | Trentino loader copy, inspection, grid and threshold checks (see the gap below) | about 17 min (scan 12.9 min, loader 3 min) | 3.4 GB |
| 5 Transfer | `phase5_transfer.py fit`, `eval` | about 55 min (fits about 40 min, eval 16 min) | 4.5 GB |
| 5 Follow-ups | `phase5_followup.py rates`, `train`, `fit2b`, `cmp2b`, `margins`; `phase5_review.py bounds`, `marginspre`, `thrsens`, `dec31`; `promotion_v2_followup.py fairmargin`, `tnthr` | about 25 min (`fit2b` 20 min) | 4.5 GB |

The longest phase is Phase 2. Every Phase 2 and pre-Phase-3 fit uses about
4-6.3 GB of commit, so close heavy applications first. The fitting scripts
stop on their own if headroom is below 6 GB.

**Gap: some runner and Phase 5 prep scripts are not in git.** They were
written under `data\experiments\`, which is ignored:
- the Phase 2 / 3 runners (`phase2\run_phase2.sh`,
  `phase2\pre_phase3\run_checks.sh`, `phase3\run_phase3.sh`,
  `phase3\checks\run_w2.sh`);
- the Part A parity copies (`parity\scripts\`);
- the Phase 5 data preparation (`phase5\scripts\cdr_loader_tn.py`,
  `inspect_trentino.py`, `grid_checks.py`, `threshold_checks.py`,
  `loader_parity.py`).

To make the Trentino part reproducible from git, copy these into the
repository (for example under `src\evaluation\phase5_prep\`) before
committing. The table above lists the evaluation commands in the order the
runners used.
