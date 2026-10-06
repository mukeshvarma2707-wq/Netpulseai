"""
src/forecasting/generate_forecasts_v2.py  (promotion v2, stage 1; option F1, refit policy R1)

Assembles OUT-OF-SAMPLE forecasts from the saved evaluation predictions. No model is fitted.
  - Pooled model "final_63l" (23 features, 63 leaves, 2,153 / 2,154 / 2,193 / 2,181 trees,
    trained on targets before 2013-12-10), +1h .. +4h.                      (D1a, D2, R1)
  - +1h only: the 200 hotspot cells use "hotspot_only_final" (31 leaves, 568 trees).   (D3)
  - forecast_kind = "validation" for targets Dec 10-16 (used to choose model size and margins),
    "test" for targets from Dec 17 on (origins Dec 17 00:00 .. Jan 1 19:00).           (F1)
Origins between the two splits have no forecast (for example the +1h target Dec 17 00:00), as in
the evaluation.

INPUT : data/experiments/phase2/preds/w1__{final_63l,hotspot_only_final}__{h}h__{val,test}.npy,
        data/processed/v2/thresholds_v2.parquet (hotspot set)
OUTPUT: data/processed/v2/cell_forecasts_v2.parquet (long: one row per cell, origin, horizon)
RUN   : python src/forecasting/generate_forecasts_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import HORIZONS, ROOT, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import W1, load_preds, origin_index  # noqa: E402

V2_DIR = ROOT / "data" / "processed" / "v2"
POOLED = "final_63l"
HOTSPOT_MODEL = {1: "hotspot_only_final"}     # D3: hotspot model at +1h only
KIND = {"val": "validation", "test": "test"}


def assemble(panel, hot_cells: np.ndarray) -> pd.DataFrame:
    cells = panel.cells
    pos = np.searchsorted(cells, np.sort(hot_cells))      # same row layout as the evaluation
    is_hot_cell = np.zeros(len(cells), bool)
    is_hot_cell[pos] = True
    parts = []
    for h in HORIZONS:
        for split in ("val", "test"):
            t = origin_index(panel, W1, split, h)
            n_t = len(t)
            f = load_preds("w1", POOLED, h, split).astype(np.float64).reshape(len(cells), n_t)
            model = np.full((len(cells), n_t), POOLED, dtype=object)
            if h in HOTSPOT_MODEL:
                f[pos] = load_preds("w1", HOTSPOT_MODEL[h], h, split).astype(np.float64).reshape(len(pos), n_t)
                model[pos] = HOTSPOT_MODEL[h]
            origin = panel.hours[t]
            parts.append(pd.DataFrame({
                "CellID": np.repeat(cells, n_t).astype(np.int64),
                "origin_datetime": np.tile(origin.to_numpy(), len(cells)),
                "horizon": np.int8(h),
                "target_datetime": np.tile((origin + pd.Timedelta(hours=h)).to_numpy(), len(cells)),
                "forecast": f.ravel(),
                "segment": np.repeat(np.where(is_hot_cell, "hotspot", "typical"), n_t),
                "model_version": model.ravel(),
                "forecast_kind": KIND[split],
            }))
    df = pd.concat(parts, ignore_index=True)
    for c in ("segment", "model_version", "forecast_kind"):
        df[c] = df[c].astype("category")
    df["origin_datetime"] = df["origin_datetime"].astype("datetime64[ns]")
    df["target_datetime"] = df["target_datetime"].astype("datetime64[ns]")
    # tag check: validation targets fall on Dec 10-16, test targets on Dec 17 or later
    val = df["forecast_kind"] == "validation"
    assert (df.loc[val, "target_datetime"] < pd.Timestamp("2013-12-17")).all()
    assert (df.loc[val, "target_datetime"] >= pd.Timestamp("2013-12-10")).all()
    assert (df.loc[~val, "target_datetime"] >= pd.Timestamp("2013-12-17")).all()
    return df


if __name__ == "__main__":
    V2_DIR.mkdir(parents=True, exist_ok=True)
    with Step("generate_forecasts_v2"):
        panel = load_panel()
        thr = pd.read_parquet(V2_DIR / "thresholds_v2.parquet")
        df = assemble(panel, thr.loc[thr.is_hotspot, "CellID"].to_numpy())
        df.to_parquet(V2_DIR / "cell_forecasts_v2.parquet", index=False)
    print(df.groupby(["horizon", "forecast_kind"], observed=True).agg(
        rows=("forecast", "size"), first_target=("target_datetime", "min"), last_target=("target_datetime", "max")).to_string())
    print(df.groupby(["horizon", "model_version"], observed=True).size().to_string())
    pd.DataFrame(Step.log).to_csv(V2_DIR / "timing_generate_forecasts_v2.csv", index=False)
