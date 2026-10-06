"""
src/forecasting/thresholds_v2.py  (promotion v2, stage 1)

Per-cell congestion thresholds for the v2 path: the 90th percentile of total_activity over every
hour BEFORE the training cutoff (2013-11-01 00:00 .. 2013-12-09 23:00), the Phase 3 definition.
Unlike v1 (cdr_with_congestion_flags, thresholds from all 62 days), no forecast period is used.

Also records the hotspot set (top 2% = 200 cells by mean activity over the training target hours,
Nov 2 .. Dec 9), which decides where the +1h hotspot model and the typical-cell watch margin apply.

INPUT : data/raw/cdr_with_congestion_flags.parquet (read only)
OUTPUT: data/processed/v2/thresholds_v2.parquet
RUN   : python src/forecasting/thresholds_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import ROOT, Step, load_panel  # noqa: E402
from src.evaluation.phase2_common import W1, hotspots, training_cell_means  # noqa: E402
from src.evaluation.margin_tuning import thresholds_before  # noqa: E402

V2_DIR = ROOT / "data" / "processed" / "v2"
THRESHOLD_START = "2013-11-01 00:00"
THRESHOLD_END = "2013-12-10"          # exclusive: last hour used is 2013-12-09 23:00
PERCENTILE = 90


def build(panel) -> pd.DataFrame:
    thr = thresholds_before(panel, THRESHOLD_END)
    cm = training_cell_means(panel, W1)
    hot = set(hotspots(cm).tolist())
    return pd.DataFrame({"CellID": panel.cells.astype(np.int64), "congestion_threshold": thr,
                         "training_mean": cm.reindex(panel.cells).to_numpy(),
                         "is_hotspot": np.isin(panel.cells, list(hot)),
                         "percentile": PERCENTILE, "threshold_start": THRESHOLD_START,
                         "threshold_end_exclusive": THRESHOLD_END})


if __name__ == "__main__":
    V2_DIR.mkdir(parents=True, exist_ok=True)
    with Step("thresholds_v2"):
        panel = load_panel()
        df = build(panel)
        df.to_parquet(V2_DIR / "thresholds_v2.parquet", index=False)
    print(f"{len(df):,} cells, {int(df.is_hotspot.sum())} hotspots, zero thresholds: {int((df.congestion_threshold <= 0).sum())}")
    pd.DataFrame(Step.log).to_csv(V2_DIR / "timing_thresholds_v2.csv", index=False)
