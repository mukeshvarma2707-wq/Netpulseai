# BalanceGrid Command Center (frontend)

3D network-operations view over the BalanceGrid FastAPI backend: every flagged
cell is a pillar at its real Milan grid position, and ROUTINE cases show the
solver's reallocation as capacity flowing in from real neighbouring cells.

React + TypeScript + Vite, Tailwind v4, react-three-fiber / drei /
postprocessing, Framer Motion.

## Run

```bash
# 1. backend (repo root)
python src/db/populate_db.py          # once, loads pipeline outputs into SQLite
uvicorn src.api.main:app --reload     # http://127.0.0.1:8000

# 2. frontend
cd frontend
npm install
npm run dev                           # http://localhost:5173
```

The backend has no CORS middleware, so in dev the UI calls `/api/*` and Vite
proxies it to `VITE_API_TARGET`. Copy `.env.example` to `.env` to change:

| Variable            | Default                  | Purpose                                                |
| ------------------- | ------------------------ | ------------------------------------------------------ |
| `VITE_API_TARGET`   | `http://127.0.0.1:8000`  | Where the dev proxy forwards `/api`                    |
| `VITE_API_BASE_URL` | `/api`                   | Call the API directly instead (needs CORS on backend)  |
| `VITE_CASE_LIMIT`   | `10000`                  | Max cases fetched for one hour (busiest hour has ~4k)  |

## How the data maps to the scene

- **Position**: `x = ((cell_id - 1) mod 100) + 1`, `y = floor((cell_id - 1) / 100) + 1`,
  one world unit per cell; grid y points north (away from the default camera).
- **Pillar height**: `naive_forecast - congestion_threshold`, sqrt-scaled and
  normalised to the 95th percentile of loaded cases.
- **Colour**: amber = ROUTINE, red = ANOMALOUS (used for nothing else), green =
  fully resolved (settles from amber once the flows arrive).
- **Beams**: one arc per `sources[]` entry, from source cell to congested cell;
  particle count and size scale with `amount_moved`.
- **Time**: `GET /hours` lists every forecast hour with counts; the date
  picker and 24-hour strip choose one, and the map loads just that hour via
  `GET /cases?target_datetime=...&include_sources=true`.
- **Sources** come inline from `include_sources`. Against an older backend
  without it, the UI falls back to `GET /cases/{id}` per ROUTINE case.

## Views

- **Command Center**: the 3D map, date/hour navigation, case detail panel.
- **Dashboard**: table view (classification, date, hour and row-count
  filters, sortable columns) with case detail and report below.
- **Incident Log**: every generated report; any row can be opened on the map.

## Layout

```
src/
  lib/     api client, grid maths, colours, data hooks
  scene/   three.js: BaseGrid (10k instanced tiles), Towers (instanced
           pillars), Beams (shader particles), CameraRig (fly-to)
  ui/      HUD overlays, case detail panel, incident log, status screens
```
