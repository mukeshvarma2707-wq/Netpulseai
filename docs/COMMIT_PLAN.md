# Commit plan

Seven commits (A-G) that put the current working tree under version
control in logical groups. `commit_plan.ps1` in the project root contains
exactly these commands. **Nothing has been staged or committed.** Review
this file, then run the script yourself from the repository root (the folder
that contains `.git`).

**How to run it.** From any folder inside the repository:

```powershell
$root = git rev-parse --show-toplevel      # check: prints the folder that contains .git
if ($LASTEXITCODE -ne 0) { throw "Not inside a git repository" }
cd $root
Test-Path .git                             # check: must print True
powershell -ExecutionPolicy Bypass -File .\commit_plan.ps1
```

**How the script behaves:**
- It stops at the first failing git command.
- It refuses to start if anything is already staged, or if a listed path is
  missing.
- It commits on the current branch (`main`). To use a separate branch,
  uncomment the `git switch -c` line near the top of the script.

**Ignored, so never staged:**
- `data/raw/`, `data/processed/` (including `data/processed/v2/` and
  `balancegrid_v2.db`), `.env`, `venv/`;
- `data/trentino/` (11 GB), `data/experiments/` (4.3 GB) and
  `backup_initial/`, newly added to `.gitignore`;
- `frontend/node_modules/` and `frontend/dist/` (via `frontend/.gitignore`).

**Pre-commit checks** on all 70 candidate files, run read-only:
- **Size:** no file is over 5 MB. The largest is
  `docs/BALANCEGRID_IMPROVEMENTS.md` at 192 KB, then
  `frontend/package-lock.json` at 111 KB.
- **Secrets:** no API keys, tokens or passwords. The pattern hits are only
  environment-variable *names* (`OPENAI_API_KEY`, `AZURE_OPENAI_KEY`) in
  `docs/RUNNING.md` and `src/evaluation/promotion_v2_reporttest.py`, and
  the npm package `js-tokens` in `frontend/package-lock.json`.
- **`.env.example` files:**
  - `.env.example` (already tracked) has only `APP_NAME`, `LOG_LEVEL` and
    `DATA_ROOT`.
  - `frontend/.env.example` has only `VITE_API_TARGET` (a local URL) and
    `VITE_CASE_LIMIT`.
- **Frontend:** no `node_modules/`, `dist/` or build output among the
  candidates. `package-lock.json` is included, as it should be for
  reproducible installs.

**Attribution:**
- Commits B-G end with a `Co-Authored-By` trailer for the AI-assisted work.
- Commit A has none, because it records changes made before this study.

Edit or remove the trailers in the script if you prefer.

---

## A. Changes made before this study

These files were already modified or untracked when the study began.
Review the diffs before committing.

- `src/diagnosis/diagnosis_agent.py`
- `src/optimization/solver.py`
- `src/ingestion/cdr_loader.py`
- `src/forecasting/lightgbm_model.py`
- `src/forecasting/generate_forecasts.py`

Message: `Pipeline: in-sample v1 forecasts from an under-sized 60-tree LightGBM model feed diagnosis and solver (pre-study changes)`

The body says that the v1 forecast file (`cell_forecasts.parquet`,
written by `generate_forecasts.py`) is **in-sample**: the under-sized
60-tree model is fitted on every row that has a target and then predicts
those same rows. It is **not a measure of prediction quality**;
out-of-sample results are in `docs/BALANCEGRID_IMPROVEMENTS.md`.

## B. Backend fixes from the UI work

- `src/db/database.py` (`init_db` creates indexes added after first build)
- `src/db/models.py` (composite index for `/hours`)
- `src/db/populate_db.py` (`forecast_1h` column fallback)

`src/api/main.py` is **not** here: it also carries the stage-2 v2 changes,
so it is committed once, in F.

Message: `DB: composite index for /hours and forecast_1h fallback in populate_db`

## C. React command-center UI

All 31 files under `frontend/`:
- `.env.example`, `.gitignore`, `README.md`, `index.html`;
- `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`;
- `src/` (`App.tsx`, `main.tsx`, `index.css`, `lib/`, `scene/`, `ui/`).

Message: `Frontend: React/R3F 3D command-center UI with dashboard view`

## D. Evaluation scripts, Phases 1-5

`src/evaluation/`:
- `common.py`, `phase2_common.py`;
- `flag_quality.py`, `capacity_check.py`;
- `tune_trees.py`, `feature_ablation.py`, `pre_phase3_checks.py`;
- `margin_tuning.py`, `phase3_checks.py`;
- `phase4_sensitivity.py`, `phase4_followup.py`;
- `phase5_transfer.py`, `phase5_followup.py`, `phase5_review.py`.

Message: `Evaluation: phased study scripts (diagnosis, model size, margins, sensitivity, Trentino transfer)`

## E. v2 pipeline (promotion stage 1)

- `src/forecasting/thresholds_v2.py`
- `src/forecasting/generate_forecasts_v2.py`
- `src/diagnosis/diagnosis_agent_v2.py`
- `src/optimization/solver_v2.py`
- `src/evaluation/promotion_v2_parity.py`
- `src/evaluation/promotion_v2_followup.py`

Message: `v2 pipeline: out-of-sample forecasts, vectorised diagnosis, watch flags, V0 solver, parity tests`

## F. v2 database, API and dashboard (promotion stage 2)

- `src/db/models_v2.py`
- `src/db/database_v2.py`
- `src/db/populate_db_v2.py`
- `src/api/main.py`: **includes the earlier B-group API changes** (`/hours`
  endpoint, date filter, `include_sources`) as well as the v2 additions
  behind `BALANCEGRID_DB=v2`
- `src/dashboard/app.py`
- `src/evaluation/promotion_v2_apitest.py`
- `src/evaluation/promotion_v2_dashtest.py`
- `src/evaluation/promotion_v2_reporttest.py`

Message: `v2 DB + API (/watch, forecast_kind) behind BALANCEGRID_DB=v2; main.py also carries the earlier /hours, date-filter and include_sources changes`

## G. Docs and ignore rules

- `docs/BALANCEGRID_IMPROVEMENTS.md`
- `docs/RUNNING.md`
- `docs/COMMIT_PLAN.md`
- `.gitignore` (adds `data/trentino/`, `data/experiments/`, `backup_initial/`)

Message: `Docs: evaluation report, running guide, commit plan; ignore study data and backups`

---

## Not in any group

- `commit_plan.ps1`: the runner itself. It is left untracked; delete it
  after use, or commit it separately if you want to keep it.
