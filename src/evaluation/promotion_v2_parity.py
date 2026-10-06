"""
src/evaluation/promotion_v2_parity.py  (promotion v2, stage 1: parity tests)

  (i)   diagnosis_agent_v2 vs the existing diagnosis_agent.diagnose() on the same forecast input:
        flagged set, classification and reason, for the 30 replay hours at +1h .. +4h, and for every
        +1h target hour.
  (ii)  solver_v2 vs the original solver path (solver.classify_routine, then solve_hour per hour on
        the same frames as solver.py's main loop): ROUTINE set, moves and per-case coverage, for the
        30 replay hours and every +1h target hour.
  (iii) watch flags vs the definition m x threshold < forecast <= threshold on typical cells:
        independent recount and a random sample of watch and non-watch rows.
  (iv)  assembled forecasts reproduce the evaluation (Phase 3 F1 at m = 1).
Also SHA-256 of every v2 output.

RUN: python src/evaluation/promotion_v2_parity.py
"""

from __future__ import annotations

import hashlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import ROOT, Step, load_panel  # noqa: E402
from src.evaluation.margin_tuning import DOWNSTREAM_HOLIDAY, DOWNSTREAM_NONHOLIDAY, flag_counts  # noqa: E402

V2 = ROOT / "data" / "processed" / "v2"
OUT = ROOT / "data" / "experiments" / "promotion_v2"
REPLAY_HOURS = pd.DatetimeIndex(DOWNSTREAM_NONHOLIDAY + DOWNSTREAM_HOLIDAY)
WATCH_MARGINS = {1: 0.955, 2: 0.95, 3: 0.95, 4: 0.94}


def v1_frame(fc_h: pd.DataFrame, thr: pd.Series, h: int) -> pd.DataFrame:
    """The frame diagnosis_agent.attach_forecast_and_risk() would build, from the same forecasts."""
    df = pd.DataFrame({"CellID": fc_h["CellID"].to_numpy(), "datetime": fc_h["origin_datetime"].to_numpy(),
                       "congestion_threshold": thr.reindex(fc_h["CellID"]).to_numpy(),
                       f"forecast_{h}h": fc_h["forecast"].to_numpy(),
                       f"target_datetime_{h}h": fc_h["target_datetime"].to_numpy()})
    df[f"predicted_risk_{h}h"] = df[f"forecast_{h}h"] > df["congestion_threshold"]
    return df


def test_diagnosis(fc, thr, diag):
    from src.diagnosis.diagnosis_agent import diagnose
    rows = []
    for scope, h_list in (("30 replay hours", (1, 2, 3, 4)), ("every +1h target hour", (1,))):
        for h in h_list:
            f = fc[fc.horizon == h]
            if scope == "30 replay hours":
                f = f[f.target_datetime.isin(REPLAY_HOURS)]
            hours = f.target_datetime.unique()
            with redirect_stdout(io.StringIO()):
                v1 = diagnose(v1_frame(f, thr, h), h)
            v1 = v1.rename(columns={f"target_datetime_{h}h": "target_datetime"})[["CellID", "target_datetime", "classification", "reason"]]
            v2 = diag[(diag.horizon == h) & diag.target_datetime.isin(hours)][["CellID", "target_datetime", "classification", "reason"]]
            m = v1.merge(v2, on=["CellID", "target_datetime"], how="outer", suffixes=("_v1", "_v2"), indicator=True)
            both = m["_merge"] == "both"
            rows.append({"test": "(i) diagnosis", "scope": scope, "horizon": h, "target_hours": len(hours),
                         "flags_v1": len(v1), "flags_v2": len(v2),
                         "only_v1": int((m["_merge"] == "left_only").sum()), "only_v2": int((m["_merge"] == "right_only").sum()),
                         "class_mismatch": int((m.loc[both, "classification_v1"] != m.loc[both, "classification_v2"]).sum()),
                         "reason_mismatch": int((m.loc[both, "reason_v1"] != m.loc[both, "reason_v2"]).sum()),
                         "routine_v1": int((v1.classification == "ROUTINE").sum()),
                         "anomalous_v1": int((v1.classification == "ANOMALOUS").sum())})
            print(rows[-1])
    return rows


def test_solver(fc, thr, diag, mv2, cov2):
    from src.optimization.solver import classify_routine, solve_hour
    f = fc[fc.horizon == 1]
    df = pd.DataFrame({"CellID": f["CellID"].to_numpy(), "datetime": f["origin_datetime"].to_numpy(),
                       "congestion_threshold": thr.reindex(f["CellID"]).to_numpy(),
                       "forecast": f["forecast"].to_numpy(), "target_datetime": f["target_datetime"].to_numpy()})
    # as solver.attach_forecast_and_risk (after its merge and rename)
    df["deficit"] = (df["forecast"] - df["congestion_threshold"]).clip(lower=0)
    df["spare"] = (df["congestion_threshold"] - df["forecast"]).clip(lower=0)
    df = df.dropna(subset=["forecast"])
    rf = classify_routine(df)
    routine_v1 = rf[rf["is_routine"]]
    r2 = diag[(diag.horizon == 1) & (diag.classification == "ROUTINE")]
    k1 = set(zip(routine_v1.CellID, routine_v1.target_datetime))
    k2 = set(zip(r2.CellID, r2.target_datetime))
    by_hour_df = dict(tuple(df.groupby("target_datetime")))
    by_hour_rt = dict(tuple(routine_v1.groupby("target_datetime")))
    mv2_by_hour = dict(tuple(mv2.groupby("target_datetime")))
    cov2_idx = cov2.set_index(["CellID", "target_datetime"])["covered"]
    rows = []
    all_hours = sorted(df.target_datetime.unique())
    for scope, hours in (("30 replay hours", list(REPLAY_HOURS)), ("every +1h target hour", all_hours)):
        n_moves1 = n_moves2 = key_mismatch = 0
        max_amt = max_cov = 0.0
        hours_with_routine = 0
        for hour in hours:
            hour = pd.Timestamp(hour)
            if hour not in by_hour_rt:
                assert hour not in mv2_by_hour
                continue
            hours_with_routine += 1
            cong = by_hour_rt[hour]
            sol = solve_hour(cong, by_hour_df[hour])          # exactly solver.py's main-loop call
            s2 = mv2_by_hour.get(hour, pd.DataFrame(columns=["congested_cell", "source_cell", "amount_moved"]))
            d2 = {(int(c), int(n)): v for c, n, v in zip(s2.congested_cell, s2.source_cell, s2.amount_moved)}
            d1 = {(int(c), int(n)): v for (c, n), v in sol.items()}
            n_moves1 += len(d1); n_moves2 += len(d2)
            key_mismatch += len(set(d1) ^ set(d2))
            for k in set(d1) & set(d2):
                max_amt = max(max_amt, abs(d1[k] - d2[k]))
            got = {}                                             # solver.py's per-case coverage, one pass
            for (cc, n), v in sol.items():
                got[cc] = got.get(cc, 0.0) + v
            c2 = cov2_idx.xs(hour, level="target_datetime").reindex(cong.CellID).to_numpy()
            c1 = np.array([got.get(c, 0.0) for c in cong.CellID])
            max_cov = max(max_cov, float(np.max(np.abs(c1 - c2))))
        in_scope = [pd.Timestamp(x) for x in hours]
        rows.append({"test": "(ii) solver", "scope": scope, "target_hours": len(hours), "hours_with_routine": hours_with_routine,
                     "routine_v1": sum(1 for k in k1 if k[1] in set(in_scope)),
                     "routine_set_mismatch": len({k for k in k1 ^ k2 if k[1] in set(in_scope)}),
                     "moves_v1": n_moves1, "moves_v2": n_moves2, "move_key_mismatch": key_mismatch,
                     "max_abs_amount_diff": max_amt, "max_abs_coverage_diff": max_cov})
        print(rows[-1])
    return rows


def test_watch(fc, thr_df, watch, rng):
    thr = thr_df.set_index("CellID")["congestion_threshold"]
    rows = []
    for h in (1, 2, 3, 4):
        f = fc[fc.horizon == h]
        th = thr.reindex(f.CellID).to_numpy()
        typ = (f.segment == "typical").to_numpy()
        m = WATCH_MARGINS[h]
        band = (f.forecast.to_numpy() > m * th) & (f.forecast.to_numpy() <= th)
        w = watch[watch.horizon == h]
        for kind in ("validation", "test"):
            k = (f.forecast_kind == kind).to_numpy()
            rows.append({"test": "(iii) watch count", "horizon": h, "kind": kind, "margin": m,
                         "recount": int((band & typ & k).sum()),
                         "watch_file": int((w.forecast_kind == kind).sum()),
                         "hotspot_rows_in_band_excluded": int((band & ~typ & k).sum())})
        # random samples: watch rows satisfy the definition; typical non-watch rows do not
        ws = w.sample(min(1000, len(w)), random_state=int(rng.integers(1 << 31)))
        tws = thr.reindex(ws.CellID).to_numpy()
        ok_w = ((ws.forecast.to_numpy() > m * tws) & (ws.forecast.to_numpy() <= tws)).all() and (ws.segment == "typical").all() \
            and (~ws.CellID.isin(thr_df.loc[thr_df.is_hotspot, "CellID"])).all()
        keyset = set(zip(w.CellID, w.origin_datetime))
        nw = f[typ & ~band].sample(1000, random_state=int(rng.integers(1 << 31)))
        ok_nw = not any((c, o) in keyset for c, o in zip(nw.CellID, nw.origin_datetime))
        rows.append({"test": "(iii) watch sample", "horizon": h, "watch_sample_ok": bool(ok_w), "non_watch_sample_ok": bool(ok_nw),
                     "sample_size": len(ws)})
        print(rows[-2:])
    return rows


def test_reproduce(fc):
    """+1h and +4h test, Phase 3 thresholds, m = 1: composite (H0) and pooled (P0) F1 vs the report."""
    panel = load_panel()
    thr = pd.read_parquet(V2 / "thresholds_v2.parquet").set_index("CellID")["congestion_threshold"]
    rows = []
    for h in (1, 4):
        f = fc[(fc.horizon == h) & (fc.forecast_kind == "test")]
        ti = panel.hours.get_indexer(pd.DatetimeIndex(f.target_datetime))
        y = panel.activity[f.CellID.to_numpy() - 1, ti]
        th = thr.reindex(f.CellID).to_numpy()
        act = y > th
        hot = (f.segment == "hotspot").to_numpy()
        fl = f.forecast.to_numpy() > th
        rows.append({"test": "(iv) reproduce", "horizon": h, "f1_all": flag_counts(fl, act)["f1"],
                     "f1_hotspot": flag_counts(fl[hot], act[hot])["f1"], "rows": len(f)})
        print(rows[-1])
    return rows


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with Step("load v2 files"):
        fc = pd.read_parquet(V2 / "cell_forecasts_v2.parquet")
        thr_df = pd.read_parquet(V2 / "thresholds_v2.parquet")
        thr = thr_df.set_index("CellID")["congestion_threshold"]
        diag = pd.read_parquet(V2 / "diagnosis_results_v2.parquet")
        watch = pd.read_parquet(V2 / "watch_flags_v2.parquet")
        mv2 = pd.read_parquet(V2 / "reallocation_results_v2.parquet")
        cov2 = pd.read_parquet(V2 / "reallocation_coverage_v2.parquet")
    with Step("(i) diagnosis parity"):
        r1 = test_diagnosis(fc, thr, diag)
    with Step("(ii) solver parity"):
        r2 = test_solver(fc, thr, diag, mv2, cov2)
    with Step("(iii) watch flags"):
        r3 = test_watch(fc, thr_df, watch, rng)
    with Step("(iv) reproduce evaluation"):
        r4 = test_reproduce(fc)
    for name, rows in (("parity_diagnosis.csv", r1), ("parity_solver.csv", r2), ("parity_watch.csv", r3), ("parity_reproduce.csv", r4)):
        pd.DataFrame(rows).to_csv(OUT / name, index=False)
    hashes = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(V2.glob("*.parquet"))]
    pd.DataFrame(hashes).to_csv(OUT / "v2_sha256.csv", index=False)
    print(pd.DataFrame(hashes).to_string(index=False))
    pd.DataFrame(Step.log).to_csv(OUT / "timing_parity.csv", index=False)
