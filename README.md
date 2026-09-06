# BalanceGrid

BalanceGrid is a starter platform for Milano CDR demand ingestion, short-term forecasting, anomaly diagnosis, and grid dispatch optimization.

## Project layout

- `data/raw/`: source CDR and event/holiday data
- `data/processed/`: cleaned datasets and model-ready outputs
- `notebooks/`: exploration and experiments
- `src/`: ingestion, feature engineering, forecasting, diagnosis, optimization, API, and dashboard code
- `tests/`: automated checks
- `deployment/`: container and Azure deployment assets

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
pytest
```

Run the API:

```powershell
uvicorn src.api.main:app --reload
```

Run the dashboard:

```powershell
streamlit run src/dashboard/app.py
```

Copy `.env.example` to `.env` and add environment-specific settings as the project grows. Raw datasets are intentionally ignored by Git; keep only reproducible processing code and small fixtures in version control.
