# BalanceGrid improvements: re-run on the second laptop

## Summary of what can be claimed

How to read this:
- **Intervals** are 90% paired day-bootstrap intervals; "n.s." means the
  interval includes 0.
- **Thresholds** are the Phase 3 definition unless stated.
- **Multiplicity:** no correction is made for the many intervals computed,
  so isolated significant results are tentative.
- **Details:** see "What can be claimed" for every caveat.

**Milan:**
- **w1:** test Dec 17 - Jan 1, the holiday lull.
- **w2:** test Nov 25 - Dec 7.
- **Not independent:** both windows come from the same 62 days.

| Claim | Number | Interval / significance | Main caveat |
|---|---|---|---|
| **Tuned model (production features) beats seasonal-naive on MAE** | w1 +1h −19.6, +2h −11.5 | [−27.6, −12.8], [−20.4, −4.1]; **n.s. at +3h / +4h** | Size chosen on Dec 10-16; base-rate shift into the lull |
| **…and on flag F1** (m = 1) | +0.102 (+1h), +0.057 (+2h) | sig; **+3h borderline**; **+4h n.s.** | Hotspots n.s. (worse than naive at +3h / +4h under Phase 2 thresholds) |
| Final 23-feature model vs naive, F1 | +0.110 … +0.038 (+1h … +4h) | +1h to +3h sig; **+4h borderline and threshold-dependent** ([+0.004, +0.072], n.s. under Phase 2 thresholds) | **Optimistic:** feature groups were chosen on the test windows |
| **A typical-cell flagging margin raises F1** | w1 +0.023 to +0.027 (every horizon); w2 +0.022 / +0.036 (+1h / +4h) | sig under both threshold definitions | Costs precision (w1 +1h 0.64 → 0.52; flagged 3.9% → 6.3% vs 5.3% actual); the operating point is a policy choice |
| Separate hotspot model (+1h) | hotspot F1 +0.092 | [+0.061, +0.122]; n.s. at +2h to +4h | Single comparison; MAE never improves |
| Hotspots kept at m = 1 | −0.022 hotspot F1 from a margin (w2 +1h) | sig only there; w1 borderline | **A conservative default, not a finding.** On Trentino a margin *helps* hotspots |
| **Diagnosis rule separates reliable flags** | precision ROUTINE 0.718 vs ANOMALOUS 0.529 | +0.189 [+0.156, +0.219]; holds under both threshold definitions | One window. Precision of flags with flagged neighbours is partly inflated by neighbouring squares sharing base-station load (limitation a) |
| **Solver relief, under the one-for-one assumption** | net reduction of actual excess: V0 14.6%, B (m 0.96) 20.2%; ceiling 47.2% (1-hop), 72.1% (2-hop) | B − V0 +70,211 [+40,521, +103,949]; C − B net n.s. but C adds about 86% more donor harm than B | 24 hours. **Moves between grid squares are not physical capacity moves** (limitation a) |
| Holiday exemption passes lower-precision flags | precision 0.10-0.19 lower on Dec 25-26 | no test (3 days) | Indicative only; net solver effect slightly positive |

**Transfer, Milan → Trentino** (local history used for scale and
thresholds; not zero-shot):
- **At m = 1: no measurable cost.** MAE at most about 2% worse (+1h +0.09
  [−0.04, +0.19]; +4h +0.14 [−0.04, +0.32]); F1 n.s.
- **At each model's own tuned margin: a small but significant F1 cost.**
  −0.014 [−0.020, −0.007] at +1h and −0.027 [−0.037, −0.016] at +4h. The
  cost is probably understated, because the in-domain references trained on
  only 16 days.
- **Typical cells at +4h:** MAE significantly worse (+0.19 [+0.02, +0.35]).
- **Hotspots:** the transferred model is significantly *better* at every
  operating point.
- **Both beat seasonal-naive**, each at its own margin (naive's margin tuned
  the same way): +1h +0.094 / +0.108, +4h +0.022 / +0.050.
- **Milan's margins raise F1 on Trentino**, also when chosen before the
  test window (+0.031 to +0.093).
- **Scope:** the same operator and period (two regions of one dataset),
  one direction, one 13-day normal window. **The diagnosis rule and solver
  were not evaluated on Trentino.** w1 (the holiday peak, where 42% of
  cell-hours exceed) is a level shift, not a congestion result; there,
  transfer costs more.

**Dataset limitations** (Barlacchi et al. 2015):
- **(a) Base-station spreading:** activity is spread from base-station
  coverage areas over grid squares by area, so neighbouring squares share
  load and square-to-square transfers are not capacity moves.
- **(b) Record counts:** "internet" counts records (connection start / end,
  then every 15 min or 5 MB), not data volume.
- **(c) Unknown constant k:** all totals mix record types and are scaled by
  an unspecified constant k, so units are arbitrary and Milan and Trentino
  levels are not comparable.
- **(d) Country code 0** is undocumented (24% of rows, 2% of activity, in
  one Trentino file).
- **(e) Limited validation:** the paper's validation is a few example
  squares, and our checks are those examples.

**Not claimable from this data:**
- real radio-capacity relief;
- robust holiday behaviour (3 days);
- an unbiased estimate of the final feature set (no untouched hold-out
  remains);
- general transfer across cities, operators or years.

**Promotion:** v2 (out-of-sample forecasts for Dec 10 - Jan 1, advisory
watch flags, V0 solver) is built in a separate database.
- **Parity:** the diagnosis and the solver match v1 with 0 mismatches.
- **Default mode:** the v1 API is unchanged.
- **Watch flags:** their measured precision is 0.27-0.32 (test) and
  0.39-0.40 (validation).
- **How to run it:** `docs\RUNNING.md`.

---

This report re-runs the BalanceGrid evaluation study on a second machine and
compares every result with the reference numbers from the first laptop. Each
phase lists what was done, the design decisions and constants chosen, the
results next to the references, and what could not be verified.

Machine: Intel Core Ultra 5 125H (14 cores / 18 threads), 15.6 GB RAM,
Windows 11, Python 3.12.8, pandas 3.0.5, numpy 2.5.2, LightGBM 4.7.0,
scikit-learn 1.9.0, OR-Tools 9.15. Free memory, not CPU, is the constraint
on this machine, so peak memory is reported for every heavy step.

---

## Part A: pipeline parity

**Done.** The pipeline code on this laptop already matched the A1-A4
specification. A1 (loader), `validate_known_cells.py` and
`congestion_threshold.py` were re-run into `data/experiments/parity/` using
copies whose only change is the data path. A2-A4 were not re-run (by
decision): their existing outputs match the reference exactly. The SQLite
database was backed up (`data/experiments/db_backup/`, size and SHA-256
checked) and repopulated with the corrected `populate_db.py`.

| Check | Reference | This laptop |
|---|---|---|
| Aggregated rows | 14,880,000 | 14,880,000 |
| Missing (cell, hour) rows filled | 2,515 (0.017%) | 2,515 (0.017%) |
| Re-run vs existing `cdr_activity_aggregated.parquet` | – | bitwise identical (same file hash) |
| Cell 1, 2013-11-01 00:00 | 2.084285 / 1.104749 / 0.591930 / 0.429290 / 57.799009 | identical |
| Bocconi weekday vs weekend | 2214.09 vs 1169.73 | 2214.09 vs 1169.73 |
| Navigli daytime vs evening | 3819.55 vs 4888.58 | 3819.55 vs 4888.58 |
| Duomo average, percentile | 4251.72, 99.3rd | 4251.72, 99.3rd |
| Thresholds Bocconi / Navigli / Duomo | 4523.6 / 6394.1 / 8837.6 | identical; flags file bitwise identical |
| Rows flagged | 10.01% | 10.01% |
| +1h flagged / ROUTINE share | 845,101 / 90.5% | 845,101 / 90.55% |
| Solver cases / hours / coverage / fully resolved | 765,232 / 1,188 / 39.6% / 46.0% | 765,232 / 1,188 / 39.63% / 46.0% |
| DB cases with forecast = 0 (after repopulate) | near zero | 0 (was 845,101) |

pandas 3.0.5 reproduces the loader output bit for bit, so library version is
not a source of difference for ingestion.

---

## Phase 1: diagnosis of the forecaster as a congestion flagger

Scripts: `src/evaluation/common.py` (shared, memory-lean data preparation),
`src/evaluation/flag_quality.py` (steps `replicate`, `flags`, `bootstrap`,
`insample`, `channels`, `holiday`), `src/evaluation/capacity_check.py`.
Outputs: `data/experiments/phase1/` (CSV/JSON results, logs, timing files).
`lightgbm_model.py`, the parquet outputs and the database were not modified.

### Design decisions and constants

| Decision | Choice | Reason |
|---|---|---|
| Split | Train origins 2013-11-02 00:00 to 2013-12-16 23:00 (10.8M rows); test origins 2013-12-17 00:00 to 2014-01-01 19:00 (3.8M rows, 380 h) | Identical to `lightgbm_model.py` with `TRAIN_DAYS = 45`. Every horizon uses the same rows, because the original drops any origin without a +4h target. |
| Model (production) | 60 trees, lr 0.05, 15 leaves, `min_child_samples` 50, `random_state` 42, log1p target, `CellID` categorical, seasonal-naive value as a feature | Same as the original. |
| Row order and arithmetic | Rows cell-major and hour-ascending; neighbour average and per-cell mean computed with the original's pandas calls | LightGBM samples rows to build its bins, so row order and last-bit float differences can change results. With these choices, replication is exact. |
| Feature dtype | float64 for the replication and the flag analysis; float32 for new fits (capacity check) | Measured: float32 changes the +1h 60-tree MAE by 0.00001, and 500-tree MAE by 0.006, but moves individual predictions by up to 43.5 units, which can flip single flags. |
| Threads | Default for the replication; `n_jobs = 12` for new fits | Measured: `n_jobs = 12` gives predictions identical to the default within 7e-13. |
| Evaluation thresholds | Per-cell 90th percentile of `total_activity` over every hour before the cutoff (2013-11-01 00:00 to 2013-12-16 23:00); exceedance is strictly `>` | Uses only data known at training time. Including Nov 1 reproduces the reference +1h F1 0.438 and flagged share 3.5%; excluding it gave 0.437 and 3.45%. |
| Hotspot cells | Top 200 cells by mean activity over the training rows | Required by the rules. **Finding:** the original's "98th percentile over test rows" definition uses training-period means too (`cell_historical_mean` is mapped from train), and selects exactly the same 200 cells. Both definitions give identical numbers. |
| Slices | By **target** date (the hour being forecast). Holidays in the test window: Dec 25, Dec 26, Jan 1 | Slices with fewer than 5 target days (weekend: 4, holiday: 3) are marked *indicative*. |
| MAPE | Actual and forecast clipped below at 1 | Same convention as the original. |
| Significance | Paired day-level bootstrap of LightGBM minus baseline; 5,000 resamples, seed 0, 90% percentile intervals | Days are the unit because errors are correlated within a day. Pairing removes shared day-to-day variation. |
| Weekly-naive | Activity at target minus 168 h | Available for all 3.8M test rows at every horizon (checked: 0 missing). |
| Capacity check | +1h, 60 vs 500 trees, all other settings unchanged | 500 is the reference comparison point, not a tuned value (tuning is Phase 2). |

### Step 1: replication (exact)

The original `lightgbm_model.py` writes no files. Run unchanged, it printed the
same numbers my script reproduces:

| Horizon | LightGBM MAE | Naive MAE | Reference | Hotspot LightGBM / naive | Reference |
|---|---|---|---|---|---|
| +1h | **55.24** | **64.03** | 55.24 / 64.03 | **658.73 / 640.71** | 658.7 / 640.7 |
| +2h | 59.72 | 64.08 | – | 707.35 / 641.60 | – |
| +3h | 62.23 | 64.20 | – | 746.49 / 643.19 | – |
| +4h | **63.52** | **64.42** | 63.52 / 64.42 | 755.18 / 645.55 | – |

Typical-cell MAE, +1h to +4h: LightGBM 42.92 / 46.51 / 48.27 / 49.40 vs naive
52.26 / 52.30 / 52.38 / 52.56. Every figure matches to the printed precision.

### Step 2: flag quality on the test window

**All cells (training-period thresholds; 5.08% of test cell-hours actually exceed):**

| Horizon | Forecaster | Precision | Recall | F1 | Flagged share | MAE | MAPE |
|---|---|---|---|---|---|---|---|
| +1h | LightGBM-60 | 0.537 | 0.369 | **0.438** | 3.50% | 55.24 | 17.8% |
| +1h | Seasonal-naive | 0.402 | 0.426 | 0.414 | 5.39% | 64.03 | 20.8% |
| +1h | Weekly-naive | 0.241 | 0.411 | 0.304 | 8.70% | 100.50 | 34.1% |
| +2h | LightGBM-60 | 0.460 | 0.331 | 0.385 | 3.65% | 59.72 | 20.1% |
| +3h | LightGBM-60 | 0.428 | 0.312 | 0.361 | 3.71% | 62.23 | 21.5% |
| +4h | LightGBM-60 | 0.407 | 0.300 | 0.345 | 3.74% | 63.52 | 22.6% |
| +2h to +4h | Seasonal-naive | 0.40 | 0.426 | 0.412-0.413 | 5.4% | 64.1-64.4 | 21% |

Reference (+1h): flags 3.5% vs 5.0% actual; precision 0.54, recall 0.37; F1
0.438 vs 0.412 naive. Reproduced: precision 0.537, recall 0.369, F1 0.438. The
naive F1 is 0.414 here vs 0.412; the difference comes from the threshold
window (0.412 with Nov 1 excluded).

**+1h by slice (LightGBM-60 vs seasonal-naive; \* = indicative, fewer than 5 days):**

| Slice | Days | Cell-hours | LightGBM F1 | Naive F1 | Weekly F1 | LightGBM flagged / actual | LightGBM MAE | Naive MAE |
|---|---|---|---|---|---|---|---|---|
| All | 16 | 3,800,000 | 0.438 | 0.414 | 0.304 | 3.5% / 5.1% | 55.2 | 64.0 |
| Typical cells | 16 | 3,724,000 | 0.440 | 0.413 | 0.303 | 3.6% / 5.1% | 42.9 | 52.3 |
| **Hotspot cells** | 16 | 76,000 | **0.017** | 0.498 | 0.373 | **0.02% / 2.0%** | 658.7 | 640.7 |
| Weekday | 12 | 2,840,000 | 0.438 | 0.429 | 0.299 | 4.2% / 6.1% | 60.0 | 63.0 |
| Weekend\* | 4 | 960,000 | 0.437 | 0.326 | 0.345 | 1.5% / 2.1% | 41.2 | 67.1 |
| Non-holiday | 13 | 3,110,000 | 0.421 | 0.425 | 0.357 | 3.2% / 4.9% | 56.6 | 60.6 |
| Holiday\* | 3 | 690,000 | 0.492 | 0.376 | 0.113 | 5.0% / 6.0% | 49.0 | 79.6 |

On hotspots at +1h, LightGBM flags 16 of 76,000 cell-hours (13 true positives,
3 false), against 1,521 actual exceedances. At +2h it flags none, so F1 is
undefined. At +3h and +4h, F1 is 0.003. Reference hotspot F1: 0.014; here
0.017 (see "not verified").

**Paired day-level bootstrap, LightGBM-60 minus seasonal-naive (90% intervals):**

| Horizon | Slice | F1 difference | Verdict | MAE difference | Verdict |
|---|---|---|---|---|---|
| +1h | All | +0.024 [-0.022, +0.071] | not significant | -8.8 [-18.7, +1.3] | not significant |
| +1h | Typical | +0.027 [-0.018, +0.073] | not significant | -9.3 [-15.5, -3.2] | **LightGBM better** |
| +1h | Hotspot | -0.481 [-0.541, -0.371] | **LightGBM worse** | +18 [-189, +235] | not significant |
| +1h | Non-holiday | -0.004 [-0.042, +0.046] | not significant | -4.0 [-15.2, +7.0] | not significant |
| +2h | All | -0.028 [-0.064, +0.009] | not significant | -4.4 [-13.5, +5.3] | not significant |
| +3h | All | -0.052 [-0.080, -0.021] | **LightGBM worse** | -2.0 [-10.4, +7.0] | not significant |
| +4h | All | -0.067 [-0.094, -0.037] | **LightGBM worse** | -0.9 [-9.0, +7.6] | not significant |
| +2h to +4h | Non-holiday | -0.051 to -0.087 | **LightGBM worse** | ±2 | not significant |
| any | Hotspot | about -0.49 | **LightGBM worse** | +18 to +110 | not significant |

Against weekly-naive, LightGBM is significantly better on F1 at +1h and +2h,
and on MAE at every horizon.

**Main reading.** The reference said "F1 beats naive only at +1h". With a day
bootstrap, even the +1h F1 lead of the 60-tree model is not significant (the
interval includes 0), and from +3h on the 60-tree model is significantly
*worse* than naive at flagging. On non-holiday days it ties naive at +1h, as
the reference found.

**These are results for an under-sized model, not a finding about the
method.** The step 3 capacity check, and Phase 2, show the same LightGBM
approach at an adequate size beats seasonal-naive. Every "LightGBM" figure in
Phase 1 refers to the 60-tree production configuration.

**Existing in-sample `cell_forecasts.parquet`** (+1h, full-period thresholds; labelled in-sample):

| Measure | Reference | This laptop |
|---|---|---|
| Actual exceedance share | 10.0% | 10.01% |
| Flagged share (all target hours) | – | 5.67% |
| Gap explained by target hours with no forecast (25 per cell) | about 0.09 pp | 0.083 pp of a 4.34 pp gap |
| Recall (all target hours / rows with a forecast) | about 0.32 | 0.316 / 0.318 |
| Precision / F1 | – | 0.557 / 0.403 |

Even trained and evaluated on the same rows, the production model flags only
half the actual exceedances. Missing forecasts explain almost none of that.

**Channel shares of `total_activity`** (full 62 days):

| Scope | SMS in | SMS out | Call in | Call out | Internet | Reference internet |
|---|---|---|---|---|---|---|
| All cells | 6.00% | 3.26% | 3.96% | 4.56% | **82.22%** | about 82% |
| Bocconi 4259 | 4.58% | 2.43% | 2.66% | 3.06% | **87.26%** | 87% |
| Navigli 4456 | 4.52% | 2.13% | 2.95% | 2.85% | **87.55%** | 87% |
| Duomo 5060 | 8.33% | 3.64% | 4.86% | 5.96% | **77.21%** | 77% |

### Step 3: capacity check (+1h, float32, `n_jobs = 12`)

| Measure | 60 trees | 500 trees | Reference (60 → 500) |
|---|---|---|---|
| MAE, all cells | 55.24 | **44.05** | 55.3 → 43.8 |
| MAE, hotspot cells | 658.7 | **431.7** | 657 → 423 |
| MAE, typical cells | 42.9 | 36.1 | – |
| Precision / recall | 0.537 / 0.369 | 0.613 / 0.401 | – / 0.62 / 0.40 |
| F1 | 0.438 | **0.485** | 0.438 → 0.486 |
| Flagged share vs actual | 3.50% vs 5.08% | **3.33% vs 5.08%** | about 3.3% vs 5.0% |
| Hotspot F1 (flagged vs actual) | 0.017 (0.02% vs 2.0%) | 0.337 (1.46% vs 2.0%) | – |
| Mean bias (forecast minus actual), all | -26.1 | +4.5 | – |
| Mean bias, hotspot | **-514.6** | +90.6 | – |
| Mean bias, typical | -16.2 | +2.7 | – |
| Fit time | 17.2 s | **82.9 s** | – |
| Peak memory (fit step) | 3.12 GB RAM / 4.34 GB commit | 2.79 GB RAM / 4.00 GB commit | – |

**Reading.** The 60-tree model is strongly biased low, by about -515 on
hotspots. With 60 trees at learning rate 0.05, the log-target model hasn't
moved far from its starting point, so it under-predicts large cells most.
500 trees remove almost all of that bias and cut hotspot MAE by a third. Yet
the flagged share **falls** (3.50% to 3.33%) while the actual share is 5.08%,
and recall only goes from 0.37 to 0.40. An almost unbiased point forecast
still under-flags. That is a property of the decision rule: a point forecast
of a noisy quantity crosses a high per-cell threshold less often than the
actual value does. It is not only a capacity problem. This matches the
reference conclusion.

### Optional: holiday exemption in the diagnosis rule

Using the existing `diagnosis_results_1h.parquet` (read-only): without the
holiday exemption, **1,086 of 19,203** flagged cells on Dec 25 (5.7%) and
**933 of 12,650** on Dec 26 (7.4%), **2,019 in total**, would have been
ANOMALOUS under the 30% neighbour rule. All of them are currently ROUTINE,
so they are sent to the solver without human review.

### Time and memory

| Step | Wall time | Peak RAM | Peak commit |
|---|---|---|---|
| Original `lightgbm_model.py` (unchanged, 4 horizons) | 186 s | 8.43 GB | **17.10 GB** (page file grew) |
| `flag_quality.py replicate` (4 horizons + 2 extra +1h fits) | 171 s | 4.38 GB | 5.61 GB |
| – one 60-tree fit (float64) | 18-24 s | about 4.1 GB | about 5.3 GB |
| `flag_quality.py flags` | 125 s | 2.58 GB | 3.26 GB |
| `flag_quality.py bootstrap` | 71 s | 2.59 GB | 3.28 GB |
| `insample` / `channels` / `holiday` | 3 s / 1 s / 1 s | under 1.7 GB | under 2.4 GB |
| `capacity_check.py` 500-tree fit (float32) | 83 s fit, 102 s step | 2.79 GB | 4.00 GB |
| Probe: 500 trees in float64 | 98 s fit | 4.07 GB | 5.30 GB |

The lean implementation needs about a third of the original's peak memory, so
full-grid fits are safe on this laptop with heavy apps closed. Rough
extrapolation for Phase 2 (not measured): at 63 leaves and about 2,000 trees,
expect around 10-20 minutes per horizon here, against the reference's 13
minutes.

### Not verified

- **The other laptop's evaluation scripts were not found** on this machine.
  Every figure here comes from scripts recreated from the description, so
  differences in their details cannot be ruled out.
- **500-tree gap to the reference.** MAE 44.05 vs 43.8 (+0.25, 0.6%) and
  hotspot MAE 431.7 vs 423 (+8.7, 2%). Checked and ruled out: float32 vs
  float64 (44.052 vs 44.058) and thread count (identical). Remaining
  candidates I can't test without the original script include a slightly
  different setup (its 60-tree figure was 55.3, while the original script
  gives exactly 55.24) or a different LightGBM version. Directionally
  identical.
- **Hotspot F1 0.017 vs 0.014.** It rests on 13 true and 3 false positives, so
  1-2 rows explain it. Not pinned down.
- **Training targets cross the cutoff by up to 4 hours.** This is inherited
  from the original: the last training origins (Dec 16 20:00-23:00) have
  targets on Dec 17 00:00-03:00. Kept for exact replication. The effect is
  negligible (at most 4 of 380 test hours overlap a training target).
- **Weekend (4 days) and holiday (3 days) slices are indicative only.**

### Recommendation

The numbers justify **(iv) a tuned flagging margin** as the next step,
together with the Phase 2 model-capacity fix:

- **Not (i), no change.** Hotspot flagging is near zero (F1 0.017, LightGBM
  significantly worse than naive at every horizon). At +3h and +4h the model is
  significantly worse than seasonal-naive overall, and the +1h F1 lead is not
  significant.
- **(iv) first.** Under-flagging persists even when the forecast is almost
  unbiased (500 trees: bias +4.5, flagged 3.3% vs 5.1% actual), so the
  decision rule needs fixing as well as the model. A per-horizon margin
  chosen on Dec 10-16 is cheap, keeps the forecast values the solver needs
  for its deficit, and is already designed as Phase 3.
- **(ii) a quantile objective as the fallback.** It addresses the same cause
  more directly by forecasting an upper quantile, but it changes the numbers
  the solver uses as deficits. Worth testing if a margin cannot close the gap,
  especially on hotspots.
- **Not (iii), a direct classifier, yet.** It would optimise the flag but give
  no magnitude for the solver, so it needs a second model. Nothing here shows
  that a margin or quantile cannot do the job.
- **Separately (diagnosis rule):** the holiday exemption sends 2,019
  otherwise-isolated holiday flags straight to the solver. Worth reviewing
  when the diagnosis rules are revisited (Phase 4).

### Limitations recorded at review

- Phase 1's conclusions describe the **60-tree production model** only. Its
  +1h F1 lead over seasonal-naive is within noise, and it loses from +3h. Do
  not read these as final results for the forecaster (see Phase 2).
- **Holiday exemption:** without it, 2,019 cells flagged on Dec 25-26
  (1,086 + 933) would have been ANOMALOUS. All of them currently go straight
  to the solver as ROUTINE. The rule is unchanged; an exemption on/off
  comparison is planned for the sensitivity phase (Phase 4).

---

## Phase 2: model size and features

Scripts: `src/evaluation/phase2_common.py` (windows, feature groups, sample,
fitting, size rule, evaluation), `src/evaluation/tune_trees.py` (size),
`src/evaluation/feature_ablation.py` (feature groups, rules, report).
Orchestration: `data/experiments/phase2/run_phase2.sh`. Outputs:
`data/experiments/phase2/`: CSV results, saved test **and validation**
predictions (`preds/`, for Phase 3) and truncated models (`models/`).
Nothing outside `data/experiments/`, `src/evaluation/` and this report was
changed.

### How the data was split, and how Dec 10-16 was used

| Role | Window 1 (main) | Window 2 (no holidays) |
|---|---|---|
| Train | targets 2013-11-02 → 12-09 (about 9.1M rows per horizon) | targets 11-02 → 11-17 (about 3.8M rows) |
| Validation (size only) | targets 12-10 → 12-16 | targets 11-18 → 11-24 |
| Test | origins 12-17 00:00 → 01-01 19:00 (16 target days, 3.8M cell-hours; 3 holidays, 4 weekend days) | origins 11-25 00:00 → 12-07 19:00 (13 target days, 3.08M cell-hours; 0 holidays, 3 weekend days) |
| Thresholds for evaluation | activity 11-01 → 12-16 (as in Phase 1) | activity 11-01 → 11-24 |
| Hotspots (top 200 by training mean) | 11-02 → 12-09; 4 cells differ from the Phase 1 set | 11-02 → 11-17 |

**Dec 10-16 was used once, for model size only.**
- The validation loss curve (L2 on `log1p`, the training objective) chose
  num_leaves (on the sample) and the tree count per horizon (on the full
  grid).
- The final models **are** those selection fits, truncated at the chosen tree
  count. Nothing was refit on Dec 10-16, so it stays unseen for Phase 3.
- Dec 10-16 was not used for thresholds, features or flag decisions.
- Consequence: every Phase 2 model, including the 60-tree "prod60" comparison
  row, trains on about 7 fewer days than Phase 1's 45-day model.

**Window 2 avoids tuning size on its own test days.** Its tree count was
re-derived on Nov 18-24 with the same rule. **One look-ahead remains:**
num_leaves (31) was chosen on window 1's validation week (Dec 10-16), which
lies after window 2's test period. That is one hyperparameter, and the leaf
choice was a close call (see below).

### Design decisions and constants

| Decision | Choice | Reason |
|---|---|---|
| Learning rate | 0.1 | As specified. |
| Other LightGBM settings | `min_child_samples` 50 (kept from production), defaults otherwise, `random_state` 42, `n_jobs` 12, float32 features | Only size is being tuned. |
| Tree cap | 3,000 | As specified. |
| Size rule | Smallest tree count with validation loss ≤ 1.005 × the minimum over 3,000 trees | As specified. Plain early stopping would not trigger: the best was at, or within 25 trees of, the cap at every horizon. |
| num_leaves | **One value for all horizons:** lowest mean (over horizons) of best validation loss relative to the best leaf setting at that horizon | A per-horizon choice would add selection noise for little gain. |
| Leaf-search sample | 1,000 cells: 20 hotspot + 980 typical (2.0%), seed 0 | Stratified as required; about 3 minutes per 3,000-tree fit vs about 13 on the full grid. |
| Hotspot-only model | Trained on the 200 hotspot cells; its own num_leaves **per horizon** (same grid) and tree count by the same rule | Fits on 200 cells are cheap, so a per-horizon choice costs little. |
| Information set | Every feature at origin t uses activity up to t-1 (the production convention; hour t itself is never a feature) | Kept for comparability. Note: the "+1h" model is therefore two steps beyond the latest observed hour. |
| G1 | `lag_168h` = A[t-168]; weekly-naive = A[t+h-168] | Missing (NaN) for window 2 training rows before Nov 9; LightGBM handles missing values natively. |
| G2 | Mean and population std (ddof 0) over A[t-w … t-1], w = 3, 6, 24 | Computed from cumulative sums; checked by construction. |
| Momentum | A[t-1]-A[t-2] and A[t-1]-A[t-4] | The reference left momentum undefined; these are the simplest short-term slopes. |
| G3 | Max over the 8 neighbours of A[t-1]; mean over the 24-cell 5×5 ring (Chebyshev distance ≤ 2) of A[t-1], averaging existing cells only at edges | Spot-checked against a direct per-cell computation (corner, centre and landmark cells). |
| G4 | Target date is a holiday; seasonal-naive source date is a holiday | Training targets (Nov 2 → Dec 9) contain **one** holiday, Dec 8, and it was a **Sunday**, so the flag is confounded with "Sunday". The model cannot learn a holiday effect from that. Window 2 training has no holiday, so G4 is constant there. |
| Group-keep rule (pre-declared) | Keep a group if it is significantly **better** than "tuned" on ≥ 1 of {MAE, F1, hotspot MAE, hotspot F1} × {+1h, +4h} and significantly **worse** on none (window 1, full grid, 90% paired day bootstrap) | Declared before any feature result was seen; applied in code from saved results. |
| Momentum rule (pre-declared, as specified) | Drop momentum only if removing it (G2 vs G2+MOM) is not significantly worse on any of the 8 checks | Applied in code from saved results. |
| Final model size | Frozen at the tuned size (31 leaves; 2,690 / 2,749 / 2,745 / 2,695 trees) | As specified: no re-tuning for the larger feature set. |

### Model size

**Leaf search** (stratified sample, best validation loss relative to the best
leaf setting at each horizon):

| num_leaves | +1h | +2h | +3h | +4h | Mean |
|---|---|---|---|---|---|
| 15 | 1.0082 | 1.0107 (cap reached) | 1.0089 | 1.0185 | 1.0116 |
| **31** | **1.0000** | 1.0015 | **1.0000** | **1.0000** | **1.0004** |
| 63 | 1.0003 | **1.0000** | 1.0067 | 1.0104 | 1.0044 |
| 127 | 1.0035 | 1.0065 | 1.0067 | 1.0082 | 1.0062 |

**Chosen: 31 leaves** [sample]. The reference chose 63. Here 31 and 63 differ
by 0.4% in mean validation loss, so it is a close call.

**Tree counts** [full grid], 31 leaves, learning rate 0.1:

| Window | +1h | +2h | +3h | +4h | Best at cap? | Fit time for 3,000 trees | Peak commit |
|---|---|---|---|---|---|---|---|
| w1 | 2,690 | 2,749 | 2,745 | 2,695 | yes, all | 7.5-13.6 min | 5.0 GB |
| w2 | 2,589 | 2,590 | 2,414 | 2,427 | within 25 trees | 7.3-9.6 min | 4.3 GB |

Reference: about 1,980-2,322 trees at 63 leaves, about 13 minutes per horizon.
Here, 31 leaves needs more trees, and a full 3,000-tree selection fit takes
7.5-13.6 minutes per horizon. The spread is CPU throttling on this laptop,
not memory.

**Hotspot-only model** [200 cells]: leaves 63 / 127 / 127 / 127 and 125 / 168 /
137 / 125 trees. With 200 cells it overfits after a few hundred trees.

### Results, window 1 (test Dec 17 - Jan 1)

Actual exceedance share: 5.08% overall, 1.96% on hotspots.

**MAE, all cells:**

| Model | +1h | +2h | +3h | +4h | Reference +1h / +4h |
|---|---|---|---|---|---|
| Seasonal-naive | 64.03 | 64.08 | 64.20 | 64.42 | – |
| prod60 (60 trees, trained before Dec 10; under-sized) | 55.37 | 59.59 | 61.90 | 63.34 | 55.6 / 63.4 |
| tuned (31 leaves, about 2,700 trees) | 44.43 | 52.60 | 57.37 | 61.78 | 43.5 / 59.9 |
| tuned + G2 + momentum | 43.05 | – | – | 59.63 | 42.5 / 57.2 (tuned + rolling stats) |
| **final** (+ G1, G2, G3, momentum) | **41.85** | **48.42** | **52.54** | **55.70** | – |
| tuned pooled + hotspot-only | 43.38 | 52.09 | 56.91 | 60.52 | – |

**F1, all cells** (flagged share in brackets; actual share 5.08%):

| Model | +1h | +2h | +3h | +4h | Reference +1h / +4h |
|---|---|---|---|---|---|
| Seasonal-naive | 0.414 (5.4%) | 0.413 | 0.413 | 0.412 | 0.428 at +1h |
| prod60 | 0.440 (3.5%) | 0.386 | 0.362 | 0.340 (3.6%) | 0.451 / 0.357 |
| tuned | 0.514 (3.5%) | 0.468 | 0.439 | 0.419 (3.2%) | 0.534 / 0.444 |
| tuned + G2 + momentum | 0.519 | – | – | 0.428 | 0.540 / 0.452 |
| **final** | **0.520** (3.6%) | **0.476** | **0.454** | **0.438** (3.4%) | – |
| tuned pooled + hotspot-only | 0.515 | 0.469 | 0.440 | 0.420 | – |

**Hotspot cells (200):**

| Model | MAE +1h | MAE +4h | F1 +1h | F1 +4h | Flagged share +1h / +4h (actual 1.96%) |
|---|---|---|---|---|---|
| Seasonal-naive | 639.5 | 644.3 | 0.502 | 0.502 | 2.2% / 2.2% |
| prod60 | 658.9 | 742.5 | 0.008 | 0.003 | 0.0% / 0.0% |
| tuned | 445.0 | 701.9 | 0.470 | 0.421 | 1.8% / 2.1% |
| final | 381.2 | 601.4 | 0.536 | 0.464 | 1.8% / 1.8% |
| **tuned pooled + hotspot-only** | 392.3 | 638.7 | **0.613** | **0.517** | 1.5% / 1.9% |
| Reference | – | – | tuned 0.542, + rolling 0.600, hotspot-only 0.643 | tuned 0.465, + rolling 0.508 | – |

The tuned pooled model **now flags hotspots**: 1.8% of hotspot cell-hours
against 1.96% actual. The 60-tree model flagged 16 of 76,000.

**Paired day-level bootstrap** (90% intervals; "better" or "worse" means the
interval excludes 0):

| Comparison | Slice | +1h | +2h | +3h | +4h |
|---|---|---|---|---|---|
| tuned vs prod60, F1 | all | **+0.074** | **+0.082** | **+0.078** | **+0.079** |
| tuned vs prod60, MAE | all | **-10.9** | -7.0 n.s. | -4.5 n.s. | -1.6 n.s. |
| tuned vs prod60, F1 | hotspot | **+0.462** | **+0.460** | **+0.414** | **+0.418** |
| final vs tuned, MAE | all | **-2.6** | **-4.2** | **-4.8** | **-6.1** |
| final vs tuned, F1 | all | +0.005 n.s. | +0.008 n.s. | +0.014 n.s. | +0.019 n.s. |
| final vs tuned, F1 | non-holiday | +0.016 n.s. | **+0.020** | **+0.027** | **+0.033** |
| final vs tuned, MAE / F1 | hotspot | **-64 / +0.066** | **-78 / +0.044** | **-80 / +0.073** | **-101 / +0.043** |
| final vs seasonal-naive, F1 | all | **+0.106** | **+0.063** | **+0.041** | +0.026 n.s. |
| final vs seasonal-naive, F1 | non-holiday | **+0.105** | **+0.066** | **+0.044** | **+0.033** |
| final vs seasonal-naive, F1 | hotspot | +0.034 n.s. | +0.003 n.s. | -0.015 n.s. | -0.038 n.s. |
| final vs seasonal-naive, MAE | hotspot | **-258** | **-154** | -95 n.s. | -43 n.s. |
| pooled+hotspot vs tuned, F1 | hotspot | **+0.144** | **+0.124** | **+0.137** | **+0.096** |
| pooled+hotspot vs tuned, MAE | hotspot | **-52.6** | -25.7 n.s. | -22.9 n.s. | **-63.2** |

**Slices**, final vs tuned vs seasonal-naive (\* = fewer than 5 days, indicative):

| Slice (days) | F1 +1h final / tuned / naive | F1 +4h final / tuned / naive | MAE +1h final / tuned / prod60 |
|---|---|---|---|
| Weekday (12) | 0.519 / 0.513 / 0.429 | 0.436 / 0.418 / 0.427 | 45.2 / 48.5 / 60.2 |
| Weekend\* (4) | 0.530 / 0.525 / 0.326 | 0.456 / 0.427 / 0.326 | 31.9 / 32.3 / 41.1 |
| Non-holiday (13) | 0.530 / 0.515 / 0.425 | 0.458 / 0.425 / 0.425 | 38.7 / 40.9 / 56.8 |
| Holiday\* (3) | **0.478** / 0.513 / 0.376 | **0.362** / 0.400 / 0.370 | 55.9 / 60.4 / **48.8** |

**Holiday trade-off reproduced (indicative, 3 days).** On holidays, both
larger models have higher MAE than prod60 (+1h: tuned 60.4, final 55.9,
prod60 48.8; reference: tuned 57.3 vs production 48.7). The final model's
holiday F1 also falls below tuned, at both horizons. Holiday-specific
behaviour cannot be learned from one Sunday holiday in training.

### Feature groups (one at a time on tuned, frozen size, full grid)

**Rule outcomes (window 1, applied in code):**

| Group | +1h MAE / F1 / hot MAE / hot F1 | +4h MAE / F1 / hot MAE / hot F1 | Kept? |
|---|---|---|---|
| G1 weekly lag | n.s. / n.s. / better / n.s. | better / n.s. / better / n.s. | **yes** |
| G2 rolling mean/std | better / n.s. / better / better | better / better / better / better | **yes** |
| G3 wider spatial | better / n.s. / better / n.s. | better / n.s. / n.s. / n.s. | **yes** |
| G4 holiday flags | better / n.s. / better / n.s. | better / n.s. / n.s. / **worse** | **no** |
| Momentum (removing it) | worse / worse / worse / worse | n.s. / n.s. / n.s. / worse | **kept** (removing it is worse on 5 of 8) |

**Final set:** production features + G1 + G2 + G3 + momentum (23 features).

**Ranking across windows** (MAE gain over tuned, all cells; rank 1 = best):

| Group | w1 +1h | w1 +4h | w2 +1h | w2 +4h |
|---|---|---|---|---|
| G2 + momentum | 1.38 (1) | 2.15 (2) | 0.88 (1) | 0.31 (3) |
| G1 | 1.00 (2) | **4.70 (1)** | 0.81 (2) | **1.57 (1)** |
| G2 | 0.88 (3) | 1.90 (3) | 0.68 (3) | 0.31 (2) |
| G4 | 0.64 (4) | 0.55 (4) | 0.10 (4) | -0.22 (5) |
| G3 | 0.28 (5) | 0.12 (5) | 0.07 (5) | 0.10 (4) |

**The ranking holds across both windows.** G1 and G2 (with or without
momentum) are the top three in every column, and G3 and G4 are the bottom
two in every column. On F1, G1 gives the largest gain in window 2 (+0.023
at +1h, +0.046 at +4h, both significant). In window 2 the weekly-naive
baseline beats seasonal-naive (F1 0.426 vs 0.400), and without G1 the tuned
model only ties seasonal-naive at +4h (0.400). So **the weekly lag matters
most outside the holidays**.

**Differences from the reference:**
- **G1:** the reference found the weekly lag hurt MAE. Here it helps in both
  windows, significantly at +4h.
- **G3:** the reference saw no effect on a cell sample. On the full grid it
  is significantly better on window 1 MAE, but the gains are small (0.1-0.3
  MAE). In window 2 it is significantly **worse** on +1h F1 (-0.002).
- **G3 is the weakest group kept.** It passes the pre-declared rule on window
  1 only.

### Pooled vs hotspot-only

*Superseded by "Phase 3 (d, e)" and Phase 4 (d): against the final pooled model the hotspot model is significant at +1h only (D3).*

The hotspot-only model, combined with the tuned pooled model for typical
cells, is significantly better on hotspot F1 at every horizon (+0.10 to
+0.14). It gives the best hotspot F1 of any model here (0.613 at +1h, 0.517
at +4h), and its hotspot flagged share is close to actual (1.5-1.9% vs
1.96%). Overall F1 barely changes (+0.001), because hotspots are 2% of
cell-hours. Reference: hotspot-only 0.643 vs pooled 0.600 at +1h (on the
rolling-stats model), and n.s. at +4h. Here the gain holds at +4h too.

**Not tested:** the hotspot-only model used the **production** features.
Combining it with the final feature set is untested.

### Time and memory

| Step | Data | Time | Peak commit |
|---|---|---|---|
| Leaf search (16 fits × 3,000 trees) | sample | 12.5 min total | 4.0 GB |
| Size selection w1 (4 × 3,000 trees) | full grid | 64 min | 5.0 GB |
| Hotspot-only (16 fits) | 200 cells | 4.5 min | 3.6 GB |
| Feature fits w1 (14 fits, about 2,700 trees) | full grid | 6.4-16.4 min each, mean 10.5 | 5.7 GB |
| Size selection + feature fits w2 (14 fits) | full grid | 3.2-9.6 min each | 4.3 GB |
| Report (all bootstraps) | – | 4 min | – |

Commit headroom stayed at or above 6.1 GB before every full-grid fit, just
above the 6 GB stop rule, so no stop was needed. The chain stopped once, when the previous Claude Code
session ended. It was resumed as a detached process, and completed steps were
skipped.

### Not verified, and caveats

- **Selection bias in the headline numbers.** Every keep/drop decision (G1-G4,
  momentum) was made from bootstrap results on the **test** windows (window 1
  for the rules, window 2 for the ranking check), and those same windows also
  report the final model's numbers. The final model's test figures are
  therefore **optimistically biased** by that selection. A clean estimate
  would need a further held-out period, which this 62-day dataset does not
  leave.
- **Size chosen on the old feature set.** The 31 leaves and the per-horizon
  tree counts were chosen with the **production** features, not the
  23-feature final set. The final model inherits a size it was never tuned
  for (checked before Phase 3, see "Pre-Phase-3 checks").
- **Sample vs full grid:** only num_leaves comes from the sample. Every tree
  count, feature result and comparison is on the full grid.
- **num_leaves 31 vs the reference's 63** was a close call (0.4%). The
  final-feature model was not re-tuned for 23 features.
- **The reference's higher tuned F1** (0.534 / 0.444 vs 0.514 / 0.419 here)
  is not explained. Candidates: 63 vs 31 leaves, and threshold details.
  Not chased, per instruction.
- **Window 2 caveats:** its leaves come from window 1 (look-ahead in one
  hyperparameter). Its thresholds come from only 24 days, so hotspots
  exceed their threshold far more often (6.9% vs 1.96% in window 1).
- **Indicative slices:** weekend (window 1: 4 days; window 2: 3 days) and
  holiday (3 days).
- **Phase 3 predictions:** validation predictions for Dec 10-16 are saved for
  final, tuned, hotspot-only, prod60 and each feature group.

### Recommendation

*Superseded by "Pre-Phase-3 checks": the full-grid leaf rule chose 63 leaves ("final_63l", 2,153-2,193 trees), and item 2 by Phase 3: the hotspot model is used at +1h only.*

1. **Adopt the tuned size plus the final feature set** (production + G1 + G2
   + G3 + momentum, 31 leaves, about 2,700 trees per horizon).
   - It is significantly better than the tuned model on MAE at every
     horizon, and on hotspot MAE and F1 at every horizon.
   - It is significantly better than seasonal-naive on F1 at +1h to +3h
     overall, and at every horizon on non-holiday days.
   - **G3 is optional:** it passes the rule on window 1 only, gains little,
     and slightly hurts +1h F1 in window 2. Dropping it would simplify the
     model at almost no cost. Your call; I have not refit without it.
2. **Use the hotspot-only model for the 200 hotspot cells.** It gives the
   largest hotspot flagging gain measured (+0.10 to +0.14 F1, significant at
   every horizon). Before adopting it, it should be refit with the final
   feature set, which I have not done.
3. **Proceed to Phase 3 (flagging margin).** Even the best model flags 3.4-3.6%
   of hours against 5.08% actual (window 1), and 5.3% vs 8.1% (window 2).
   Under-flagging persists after the capacity and feature fixes, so the
   decision rule still needs work.
4. **Holidays remain a known weak spot.** On the 3 test holidays, the larger
   models are worse than prod60 on MAE, and the final model's holiday F1 is
   below tuned. Holiday features cannot be learned from this training data.
   Treat holiday forecasts as lower-confidence (indicative, 3 days), and
   consider this in the Phase 4 holiday-exemption comparison.

### Pre-Phase-3 checks (full grid, judged on Dec 10-16 validation loss only)

Script: `src/evaluation/pre_phase3_checks.py`; outputs in
`data/experiments/phase2/pre_phase3/`. Criteria were pre-declared:
- **63 leaves is "better"** if its best validation loss is more than 0.5%
  below 31 leaves'.
- **An inherited tree count is "out"** if it differs from the new 0.5%-rule
  choice by more than 15%.
Test numbers are for information only.

**Check 1: +1h, final features, 31 vs 63 leaves.**

| | 31 leaves | 63 leaves |
|---|---|---|
| Best validation loss | 0.017276 (at the 3,000-tree cap) | **0.016943** (at 2,988 trees) |
| Relative to 31 leaves | – | **-1.93%** (exceeds the 0.5% tolerance) |
| Chosen trees (0.5% rule) | 2,597 | 2,153 |
| Inherited 2,690 vs new choice | 3.6% (within 15%) | 24.9% (would need re-deriving) |
| Fit time for 3,000 trees / peak commit | 15.3 min / 5.85 GB | 20.5 min / 5.88 GB |
| Test MAE (information only) | 41.86 | 42.21 |

**Result: a better configuration was found.** On the full grid with the
final features, 63 leaves beats the sample-based choice of 31 by 1.9% in
validation loss. The sample leaf choice does not transfer, which may explain
part of the F1 gap to the reference (which used 63 leaves). The test MAE
(information only) does not show the gain at +1h. Per instruction, nothing
was switched until the user decided.

**Decision (user): re-apply the Phase 2 leaf rule on the full grid with the
final features.** The rule: one leaf value for all horizons, chosen by the
lowest mean of relative best validation loss. Candidates were 31 and 63 only;
15 and 127 were not re-tested on the full grid.

| Horizon | Best val. loss, 31 leaves | Best val. loss, 63 leaves | 63 vs 31 | Trees at 63 (0.5% rule) | Fit time (3,000 trees) |
|---|---|---|---|---|---|
| +1h | 0.017276 | 0.016943 | -1.9% | 2,153 | 20.5 min |
| +2h | 0.020387 | 0.019955 | -2.1% | 2,154 | 13.4 min |
| +3h | 0.022504 | 0.022019 | -2.2% | 2,193 | 13.7 min |
| +4h | 0.023919 | 0.023461 | -1.9% | 2,181 | 22.0 min |

**Chosen: 63 leaves**, better at every horizon. The final pooled model used
from Phase 3 on is "final_63l": the 23 final features, 63 leaves,
2,153 / 2,154 / 2,193 / 2,181 trees, peak commit 6.3 GB. That is close to the
reference size (63 leaves, about 1,980-2,322 trees). **The Phase 2 headline
tables above describe the 31-leaf model.** Test MAE (information only) for
the 63-leaf model is 42.21 / 48.58 / 52.78 / 55.43, against 41.85 / 48.42 /
52.54 / 55.70 at 31 leaves, so the validation gain does not show on test
MAE.

**Hotspot model (user decision): chosen per horizon by validation loss.**
The final-feature hotspot model is used at +1h, and the old-feature hotspot
model at +2h, +3h and +4h.

**Check 2: +4h, final features, 31 leaves.** The 0.5% rule chooses 2,590
trees against the inherited 2,695, a **4.1%** difference. The inherited tree
count holds (best loss 0.02392, at the cap; 8.6 min; 5.81 GB).

**Check 3: hotspot-only model refit with the final features.** Size was
chosen per horizon by the same procedure: 31 / 31 / 127 / 127 leaves and
568 / 425 / 111 / 92 trees. On **validation loss**, the new model is
better than the old-feature hotspot model only at +1h (0.0184 vs 0.0196).
It is **worse** at +2h (0.0233 vs 0.0219), +3h (0.0253 vs 0.0244) and +4h
(0.0268 vs 0.0260). The test comparison on hotspot rows below points the
other way at most horizons (paired day bootstrap, Phase 2 thresholds); the
test numbers are for information, not for choosing. This conflict is noted
in the decision below.

| Horizon | Hotspot MAE, new vs old | Verdict | Hotspot F1, new vs old | Verdict |
|---|---|---|---|---|
| +1h | 390.3 vs 392.3 | n.s. | **0.648** vs 0.613 | better |
| +2h | 477.2 vs 538.6 | better | 0.600 vs 0.585 | n.s. |
| +3h | 550.6 vs 604.2 | better | **0.576** vs 0.551 | better |
| +4h | 597.4 vs 638.7 | n.s. | **0.562** vs 0.517 | better |

On test, the final-feature hotspot model is never significantly worse, and
it is better on hotspot F1 at three horizons and on hotspot MAE at two. But
judged by validation loss, the rule for choosing, it wins only at +1h.
Which hotspot model Phase 3 uses is therefore part of the pending decision.
The
final pooled model alone is significantly worse than the hotspot-only model
on hotspot F1 at +1h to +3h (for example 0.536 vs 0.613 at +1h), although
its hotspot MAE is lower at +1h (381 vs 392).

### Phase 3 planning: what a margin-based flag triggers downstream

Today a cell is flagged when `forecast > threshold`. The solver uses
`deficit = max(forecast - threshold, 0)` and
`spare = max(threshold - forecast, 0)`, and the diagnosis counts flagged
neighbours for the 30% rule. With a margin m < 1 (flag when
`forecast > m × threshold`), cells with `m × threshold < forecast ≤ threshold`
become flagged and the following happens downstream:

1. **They get deficit 0.** The solver moves nothing to them, yet
   `covered >= deficit` marks them **fully resolved**. That inflates the
   resolved share and hides the risk.
2. **They still have spare capacity** (`threshold - forecast > 0`), so the
   solver can take capacity **from** a cell that is flagged as at risk.
3. **They count as at-risk neighbours** in the 30% rule. More cells become
   ROUTINE, which changes the ROUTINE/ANOMALOUS mix even for cells whose own
   flag did not change.

**Options:**

| Option | What changes | Pros | Cons |
|---|---|---|---|
| A. Advisory flags | Margin flags are shown as a "watch" status; solver and diagnosis keep `m = 1` | No change to solver numbers; no fake "resolved"; easy to roll back | Watch flags trigger no action; operators must act on them |
| B. Margin as an effective capacity target | `deficit = max(forecast - m × threshold, 0)`, `spare = max(m × threshold - forecast, 0)` | Consistent: flagged ⇔ deficit > 0; builds a safety buffer of (1 - m) × threshold | Total deficit rises and spare falls, so coverage drops; m is chosen for flag F1, not for capacity, so the buffer size has no operational meaning |
| C. Margin deficit with flagged cells excluded as donors | Deficit as in B, spare unchanged at threshold, but flagged cells cannot donate | Keeps real spare for unflagged cells; no flagged cell gives capacity away | Asymmetric and harder to explain; still a margin-sized deficit |
| D. Quantile forecast for the deficit (Phase 1 option ii) | Flag and deficit both from an upper-quantile forecast | One principled number for both | Needs new models; changes every deficit |

**My suggestion for Phase 3:** report the margin results under option A
first, since it doesn't touch the solver. Then quantify B and C on a fixed
subset of hours: flagged count, ROUTINE/ANOMALOUS mix, total deficit,
coverage and how many cells become "resolved with zero deficit". Choose
after seeing those numbers.

**Precision-recall trade-off:** the relative cost of a missed congestion
versus a false alarm is unknown, so Phase 3 should report, per horizon and
for the pooled and hotspot segments:
- the full precision-recall curve over the margin grid
- the F1-maximising margin
- the margins reaching fixed recall levels (for example 0.5, 0.6, 0.7), with
  the precision and flagged share each one costs
- all of it chosen on Dec 10-16 and reported on the test window

**One more thing for Phase 3:** thresholds used when tuning margins on Dec
10-16 should come from data **before** Dec 10. The test thresholds use data
up to Dec 16, which would leak validation actuals into margin tuning.

---

## Phase 3: flagging margin

Script: `src/evaluation/margin_tuning.py` (steps `flags`, `downstream`);
runner `data/experiments/phase3/run_phase3.sh`. Outputs in
`data/experiments/phase3/`: full precision-recall curves (`pr_curves.csv`,
validation and test), chosen margins, test results, bootstrap, base rates,
downstream results by hour and in summary. No existing file, parquet output
or database was changed. `diagnosis_agent.py` and `solver.py` were imported,
not edited.

**Use of Dec 10-16.**
- **First use:** model size, in Phase 2 and the pre-Phase-3 checks.
- **Second and final use:** choosing every margin in this phase.

Nothing else is chosen on it, and no model is refit on it.

### Design decisions and constants (declared before results)

| Item | Choice |
|---|---|
| Models | **Pooled:** "final_63l" (23 features, 63 leaves). **Hotspot model**, per horizon by validation loss: final-feature model at +1h, old-feature model at +2h, +3h and +4h. |
| (a) Thresholds, one definition for tuning and testing | Per-cell 90th percentile of `total_activity` over **2013-11-01 00:00 → 2013-12-09 23:00**, all data before the validation week, so validation actuals never enter a threshold. |
| Flag rule | forecast > m × threshold |
| Margin grid | 0.600 → 1.400, step 0.005 (161 values). The sort-based counting was verified against a brute-force count at every 7th grid margin (synthetic data) and at 6 margins on real validation data. |
| F1-best margin | Maximum validation F1; ties broken by the margin closest to 1. |
| Recall-target margin r | The **largest** m whose validation recall is ≥ r, which gives the highest precision at that recall. |
| Strategies | **N0/N1:** seasonal-naive at m = 1 and with its own global m. **W0/W1:** weekly-naive, the same. **P0/P1/P2:** final pooled at m = 1, global m, and separate hotspot/typical m. **H0/H1/H2:** pooled + hotspot model, the same three. S2 margins each maximise their own segment's validation F1. |
| Significance | Paired day-level bootstrap of F1 differences, 5,000 resamples, seed 0, 90% intervals. |
| (b) Material base-rate shift | Relative difference > 20% between validation and test. |
| (f) Downstream hours | **Non-holiday (24):** Dec 18 (Wed), Dec 20 (Fri), Dec 21 (Sat), Dec 29 (Sun) × 03, 08, 12, 15, 18, 21:00. That is two weekdays and two weekend days spread over both test weeks, avoiding the holiday-adjacent Dec 23-24, 27 and 30-31. **Holiday (6):** Dec 25 and Jan 1 × 12, 15, 18:00; the last +1h test target is Jan 1 20:00. |
| Downstream variants | **V0:** today's rule (m = 1). **A, advisory:** flags at m; the diagnosis uses them, but deficit and spare stay at m = 1. **B:** deficit = max(f − m·thr, 0), spare = max(m·thr − f, 0). **C:** deficit as in B, spare = max(thr − f, 0), but flagged cells cannot donate. |
| Downstream rules | Diagnosis rule unchanged (holiday exemption, 30% neighbour rule) and solver unchanged, at +1h with the final pooled model. Run at the F1-best margin (0.96) and the recall-0.6 margin (0.98). |

### (a) Thresholds

The Phase 3 thresholds barely differ from the Phase 2 ones (Nov 1 → Dec 16).
Per-cell ratio, Phase 3 / Phase 2:
- **Median** 1.0007; **5th percentile** 0.978; **95th percentile** 1.033.
- **Range** 0.889 to 1.552. No cell has a zero threshold.

For reference, +1h F1 at m = 1 under each definition:

| | Phase 3 thresholds (used) | Phase 2 thresholds (reference) |
|---|---|---|
| Final pooled, all / hotspot | 0.540 / 0.587 | 0.522 / 0.546 |
| Pooled + hotspot, all / hotspot | 0.540 / 0.679 | 0.523 / 0.648 |
| Seasonal-naive, all / hotspot | 0.430 / 0.524 | 0.414 / 0.502 |

### (b) Base rates: **material shift, so margin transfer is uncertain**

| Segment | Validation (Dec 10-16) | Test (Dec 17 - Jan 1) | Relative difference |
|---|---|---|---|
| All cells | **10.16%** | 5.27% | -48% |
| Hotspot cells | **9.35%** | 2.11% | -77% |
| Typical cells | 10.18% | 5.33% | -48% |

**The three test-window exceedance shares in this report** (all cells; added
at review):

| Share | Thresholds from | Hours counted | Used in |
|---|---|---|---|
| **5.08%** | Nov 1 00:00 → Dec 16 23:00 (Phase 1/2 definition) | +h target hours of test origins Dec 17 00:00 → Jan 1 19:00 (at +1h: Dec 17 01:00 → Jan 1 20:00) | Phases 1 and 2 |
| **5.27%** | Nov 1 00:00 → Dec 9 23:00 (Phase 3 definition) | The same +h target hours | Phase 3 tables |
| **5.22%** | Nov 1 00:00 → Dec 9 23:00 (Phase 3 definition) | Every calendar hour Dec 17 00:00 → Jan 1 23:00 (16 full days) | Check 1 (rates by day and week) |

**Which period is unusual.** The thresholds are per-cell 90th
percentiles, so about 10% of hours exceed them in a normal period.
- **Dec 10-16 (10.16%) is the normal week.** It matches the threshold
  period itself (10.20%).
- **The test window (5.27%) is the holiday lull**, with half the usual
  exceedances overall and about a fifth on hotspots.

This was an inference when first written. **Check 1 (post-Phase-3 checks)
confirms it:** by week, exceedance falls from 7.9% (Dec 16-22) to 4.5%
(Dec 23-29) and 1.7% (Dec 30 - Jan 1). Hotspots drop below 0.6% per day
from Dec 24.

**What this means for margins.** A margin tuned on the normal week flags
more cell-hours on test than actually exceed (6.3-7.6% vs 5.27%), because
the test period has unusually few exceedances. It does not show that the
margin is too low for normal operations. The same shift explains why recall
targets land low on test (validation recall 0.6 / 0.7 gives test recall
0.546 / 0.636 at +1h). Margin transfer between these two periods is
uncertain; check 5 tests it on a second pair of periods.

### (c) Margins chosen on validation

| Model, segment | Rule | +1h | +2h | +3h | +4h |
|---|---|---|---|---|---|
| Final pooled, all (P1) | F1-best | 0.960 | 0.950 | 0.950 | 0.945 |
|  | recall ≥ 0.5 / 0.6 / 0.7 | 1.000 / 0.980 / 0.955 | 0.990 / 0.970 / 0.945 | 0.985 / 0.965 / 0.940 | 0.980 / 0.960 / 0.940 |
| Final pooled, hotspot (P2) | F1-best | 0.970 | 0.950 | 0.965 | 0.950 |
| Final pooled, typical (P2) | F1-best | 0.960 | 0.950 | 0.950 | 0.945 |
| Pooled + hotspot, hotspot (H2) | F1-best | 0.970 | 0.970 | 0.970 | 0.965 |
|  | recall ≥ 0.5 / 0.6 / 0.7 | 1.020 / 1.005 / 0.985 | 1.015 / 1.000 / 0.985 | 1.015 / 0.995 / 0.980 | 1.010 / 0.995 / 0.980 |
| Seasonal-naive (N1) | F1-best | 0.955 | 0.955 | 0.955 | 0.955 |
| Weekly-naive (W1) | F1-best | 0.920 | 0.920 | 0.920 | 0.920 |

Every F1-best margin is below 1, consistent with the under-flagging found
in Phases 1 and 2.

**Precision-recall curve** (final pooled; validation → test). The full curve
for every model, segment and horizon is in `pr_curves.csv`.

| Margin | +1h all: precision / recall / flagged | +1h hotspot: precision / recall / flagged | +4h all: precision / recall / flagged |
|---|---|---|---|
| 0.900 | 0.439→0.359 / 0.875→0.781 / 20.3%→11.5% | 0.482→0.331 / 0.941→0.914 / 18.2%→5.8% | 0.425→0.318 / 0.827→0.724 / 19.8%→12.0% |
| 0.950 | 0.561→0.488 / 0.737→0.653 / 13.3%→7.0% | 0.598→0.448 / 0.823→0.796 / 12.9%→3.7% | 0.537→0.430 / 0.662→0.583 / 12.5%→7.1% |
| 0.960 | 0.589→0.518 / 0.697→0.620 / 12.0%→6.3% | 0.622→0.478 / 0.784→0.760 / 11.8%→3.3% | 0.560→0.455 / 0.617→0.547 / 11.2%→6.3% |
| 0.975 | 0.631→0.563 / 0.633→0.565 / 10.2%→5.3% | 0.669→0.536 / 0.723→0.696 / 10.1%→2.7% | 0.596→0.495 / 0.543→0.491 / 9.3%→5.2% |
| **1.000** | 0.700→0.637 / 0.510→0.468 / 7.4%→3.9% | 0.742→0.615 / 0.581→0.562 / 7.3%→1.9% | 0.655→0.563 / 0.424→0.397 / 6.6%→3.7% |
| 1.025 | 0.764→0.709 / 0.382→0.376 / 5.1%→2.8% | 0.802→0.673 / 0.440→0.425 / 5.1%→1.3% | 0.710→0.626 / 0.308→0.306 / 4.4%→2.6% |
| 1.050 | 0.815→0.770 / 0.272→0.290 / 3.4%→2.0% | 0.848→0.734 / 0.308→0.301 / 3.4%→0.9% | 0.759→0.683 / 0.213→0.233 / 2.9%→1.8% |

Recall transfers from validation to test far better than precision does,
because precision depends on the base rate. That favours choosing margins by
**recall target** when the operator's cost ratio is unknown.

### (d, e) Test results by strategy (actual exceedance 5.27% all, 2.11% hotspot)

**F1, all cells** (flagged share in brackets):

| Strategy | +1h | +2h | +3h | +4h |
|---|---|---|---|---|
| N0 seasonal-naive m=1 | 0.430 (5.6%) | 0.429 | 0.429 | 0.428 |
| N1 seasonal-naive global m | 0.447 (8.3%) | 0.446 | 0.446 | 0.445 |
| W0 weekly-naive m=1 | 0.323 (9.0%) | 0.322 | 0.322 | 0.322 |
| W1 weekly-naive global m | 0.309 (16.9%) | 0.309 | 0.309 | 0.308 |
| P0 final pooled m=1 | 0.540 (3.9%) | 0.500 | 0.481 | 0.465 (3.7%) |
| **P1 final pooled global m** | **0.564** (6.3%) | **0.523** (7.0%) | **0.506** (7.1%) | **0.492** (7.6%) |
| P2 final pooled hot/typ m | 0.564 | 0.523 | 0.506 | 0.492 |
| H0 pooled+hotspot m=1 | 0.540 | 0.500 | 0.481 | 0.466 |
| H1 pooled+hotspot global m | 0.565 | 0.523 | 0.506 | 0.492 |
| **H2 pooled+hotspot hot/typ m** | **0.565** | **0.523** | **0.506** | **0.493** |

**F1, hotspot cells** (precision / recall at +1h):

| Strategy | +1h | +2h | +3h | +4h |
|---|---|---|---|---|
| N0 seasonal-naive m=1 | 0.524 | 0.524 | 0.524 | 0.523 |
| N1 seasonal-naive global m | 0.516 | 0.516 | 0.516 | 0.516 |
| P0 final pooled m=1 | 0.587 (0.615 / 0.562) | 0.566 | 0.539 | 0.526 |
| P1 final pooled global m | 0.587 (0.478 / 0.760) | 0.560 | 0.540 | 0.513 |
| P2 final pooled hot/typ m | 0.600 (0.515 / 0.719) | 0.560 | 0.551 | 0.518 |
| **H0 pooled+hotspot m=1** | **0.679** (0.702 / 0.657) | **0.609** | **0.585** | **0.548** |
| H1 pooled+hotspot global m | 0.654 (0.539 / 0.831) | 0.576 | 0.542 | 0.510 |
| H2 pooled+hotspot hot/typ m | 0.668 (0.577 / 0.795) | 0.607 | 0.571 | 0.544 |

**Paired day bootstrap** (F1 difference [90% interval]):

| Comparison | Slice | +1h | +2h | +3h | +4h |
|---|---|---|---|---|---|
| P1 vs P0 | all | **+0.025** | **+0.023** | **+0.025** | **+0.027** |
| P1 vs P0 | non-holiday | **+0.025** | **+0.024** | **+0.027** | +0.028 n.s. |
| P1 vs P0 | hotspot | 0.000 n.s. | -0.006 n.s. | 0.000 n.s. | -0.013 n.s. |
| H0 vs P0 | hotspot | **+0.092** | +0.043 n.s. | +0.045 n.s. | +0.022 n.s. |
| H1 vs H0 | hotspot | -0.025 n.s. | **-0.033** | **-0.043** | **-0.038** |
| H2 vs H0 | hotspot | -0.010 n.s. | -0.002 n.s. | -0.014 n.s. | -0.004 n.s. |
| H2 vs H0 | all | **+0.025** | **+0.023** | **+0.024** | **+0.027** |
| P1 vs N1 (both tuned) | all | **+0.117** | **+0.076** | **+0.060** | **+0.047** |
| H2 vs N1 | hotspot | **+0.152** | **+0.091** | +0.054 n.s. | +0.028 n.s. |
| P0 vs N1 | all | **+0.093** | **+0.053** | +0.035 n.s. | +0.020 n.s. |

**Reading:**
- **A global margin helps overall F1 significantly at every horizon**
  (+0.023 to +0.027), at the cost of flagging more than actually exceed on
  test (6.3-7.6% vs 5.27%; see base rates).
- **The tuned model beats tuned seasonal-naive at every horizon** (P1 vs
  N1), which closes the Phase 1 question fairly: both baselines got their
  own margin.
- *Superseded by "Threshold sensitivity of the significance calls": borderline at +2h and threshold-dependent at +3h / +4h; firm only in Milan w2 at +1h.* **On hotspots, a global margin hurts the hotspot model** (H1 vs H0,
  significant at +2h to +4h), because the hotspot base rate falls most
  between validation and test (it falls to the holiday lull, see (b)).
- **H2 vs H0 on hotspots is not significant at any horizon** (-0.010,
  -0.002, -0.014, -0.004). Separate margins do not *improve* hotspot
  flagging; they only avoid the damage a global margin does, while keeping
  the overall gain.
- **The hotspot model beats the final pooled model on hotspots
  significantly only at +1h** (H0 vs P0: +0.092). At +2h to +4h the gain is
  **not significant** (+0.043, +0.045, +0.022). The case for a separate
  hotspot model rests on +1h.
- **Holiday (indicative, 3 days):** +1h F1 is 0.486 (P0) / 0.502 (P1) on
  holidays vs 0.553 / 0.579 on non-holidays. At +4h on holidays the model is
  no better than seasonal-naive (0.376 / 0.386 vs 0.378). **Weekend
  (indicative, 4 days):** the margin barely changes F1 (0.554 → 0.558 at
  +1h).

### (f) Downstream: advisory flags first, then margin-adjusted solver

Totals over the 24 non-holiday hours (+1h, final pooled):

| Variant | Margin | Flagged (actually exceeding) | ROUTINE / ANOMALOUS | ROUTINE with zero deficit | Total deficit | Covered | Fully resolved (all / positive deficit only) | Solver time |
|---|---|---|---|---|---|---|---|---|
| V0 current | 1.00 | 16,306 (11,331) | 14,311 / 1,995 | 0 | 704,353 | 354,585 | 46.7% / 46.7% | 0.9 s |
| A advisory | 0.96 | 25,517 (14,502) | 23,319 / 2,198 | **7,589** | 754,074 | 403,804 | **67.2%** / 51.3% | 1.7 s |
| B margin deficit + spare | 0.96 | 25,517 | 23,319 / 2,198 | 0 | 1,220,640 | 570,111 | 41.6% / 41.6% | 1.2 s |
| C margin deficit, flagged cannot donate | 0.96 | 25,517 | 23,319 / 2,198 | 0 | 1,220,640 | **674,415** | 52.0% / 52.0% | 1.2 s |
| A advisory | 0.98 | 20,489 (12,923) | 18,350 / 2,139 | 3,081 | 737,661 | 387,533 | 58.3% / 49.9% | 1.1 s |
| B | 0.98 | 20,489 | 18,350 / 2,139 | 0 | 931,620 | 459,635 | 44.5% / 44.5% | 1.0 s |
| C | 0.98 | 20,489 | 18,350 / 2,139 | 0 | 931,620 | 501,924 | 50.5% / 50.5% | 0.9 s |

All 19,203 cell-hours that actually exceeded their threshold in these hours
are the reference: m = 0.96 catches 14,502 of them, against 11,331 at m = 1.

**Reading:**
- **Advisory flags (A) produce the predicted artefact.** At m = 0.96, a third
  of ROUTINE cases (7,589 of 23,319) have zero deficit and are counted as
  "fully resolved" with nothing moved. That lifts the resolved share from
  46.7% to 67.2%, which is entirely artificial; on cells with a positive
  forecast deficit it is 51.3%. A watch status must not count toward
  "resolved".
- **The margin also changes the diagnosis mix:** the ROUTINE share rises
  from 87.8% to 91.4% (more flagged neighbours), and ANOMALOUS rises only
  from 1,995 to 2,198.
- **B** raises total deficit by 73% and lowers the resolved share, because
  each cell's own spare also shrinks to m × threshold.
- **C** keeps the threshold-based spare for unflagged cells, so it moves the
  most capacity (674k, 1.9× V0) and has the highest resolved share among
  cells with a positive deficit (52%). **Under B and C the deficit is
  margin-defined** (forecast minus m × threshold), not the real excess over
  the threshold, so "covered" and "resolved" here are measured against that
  margin-defined deficit. What the moves achieve against actual activity is
  measured separately (check 4, outcome replay).
- **Solver time is negligible:** about 1-2 s for 24 hours, in every variant.

**Holiday subset** (6 hours, indicative): the holiday exemption makes
**every** flagged cell ROUTINE (5,744 at m = 1; 8,327 at m = 0.96; 0
ANOMALOUS). Coverage is low (34-44%), and A again inflates the resolved share
(43% → 61%).

### (g) Conflict recorded for Phase 4

Holiday forecasts are the **least reliable**: +1h F1 0.486 on holidays vs
0.553 on non-holidays, and at +4h no better than seasonal-naive. Higher MAE
was also seen in Phase 2. Yet flagged holiday cells get **no human check at
all**: the exemption makes every one ROUTINE (8,327 in 6 holiday hours at
m = 0.96, 0 ANOMALOUS). The least trustworthy forecasts are the ones
automated without review. The rule is unchanged here; this is input for the
Phase 4 exemption on/off comparison.

### (h) Time

| Step | Time | Peak commit |
|---|---|---|
| Fit, final pooled (63 leaves, about 2,150-2,190 trees; from the 3,000-tree selection fits) | 13.4-22.0 min per horizon for 3,000 trees, about 9.6-16.0 min at the chosen size | 6.3 GB |
| Fit, hotspot model (200 cells) | 11-21 s per 3,000-tree fit | 3.6 GB |
| Prediction, +1h final pooled, full test (3.8M rows, 2,153 trees) | 271 s, about **0.7 s per forecast hour for all 10,000 cells** | – |
| Margin tuning (every curve and margin, all forecasters, one horizon) | 0.4-0.7 s | – |
| `flags` step overall (features, curves, test results, 5,000-resample bootstraps) | 9.3 min | 4.35 GB |
| `downstream` step (30 hours × 7 variants, including solver) | 24 s | 3.3 GB |

### Not verified, and caveats

- **Base-rate shift (b)** is the main caveat. The validation week is a
  normal period (about 10% exceedance, as the 90th-percentile thresholds
  imply). The test window is the holiday lull (5.27%), confirmed by check 1.
  So test flagged shares and recalls differ from what was targeted on
  validation, and test results understate how a margin behaves in normal
  weeks. Margin transfer is checked on window 2 (check 5).
- **Selection bias (Phase 2)** still applies: the feature set was chosen on
  the test windows.
- **Holiday (3 days) and weekend (4 days) slices are indicative.**
- **The downstream comparison** uses 24 + 6 hours and the +1h model only.
  "Coverage" ratios aren't comparable across variants, since deficits differ
  by definition; compare the absolute "covered" and "fully resolved
  (positive deficit)" columns.
- **The +1h test MAE of the 63-leaf model** (42.21) is slightly worse than
  31 leaves (41.85) despite better validation loss. It was not used for any
  choice.

### Recommendation

1. **Model:** use the final pooled model (63 leaves) for typical cells and
   the per-horizon hotspot model for the 200 hotspot cells (strategy H).
2. *Superseded by the post-Phase-3 decision (H3) and "What can be claimed": typical-cell margin only, hotspots at m = 1 as a conservative default.* **Margins:** separate margins (H2) — typical about 0.945-0.96, hotspot
   about 0.965-0.97 by horizon. That gains significant overall F1 at every
   horizon (+0.023 to +0.027 over m = 1) without hurting hotspots, which a
   global margin does. On the test window they flag more than actually
   exceeds (about 6.3-7.6% vs 5.3%), because that window is the holiday
   lull; the margins were tuned on a normal week. Treat them as provisional
   until check 5 shows how they transfer between normal periods.
3. **Operating point:** since the miss/false-alarm cost is unknown, choose by
   recall target rather than F1. Recall transfers from validation to test
   better than precision. The recall table above shows what each target
   costs in precision and flag volume.
4. **Downstream:** do **not** run margin flags through today's deficit
   definition (A as a solver input). It fabricates "fully resolved" cases.
   Either keep margin flags as a separate **watch** status that is excluded
   from the resolved count, or, if flags should trigger reallocation, use
   **C** (margin-defined deficit, threshold-based spare, flagged cells cannot
   donate). C moved the most capacity, with the highest resolved share
   against its own margin-defined deficit, in the test hours. Its effect on
   actual activity is assessed in check 4.
5. **Holidays:** carry the recorded conflict into Phase 4. The holiday
   exemption sends every flagged holiday cell to the solver unchecked, on
   the least reliable forecasts.

### Post-Phase-3 checks (bounded, before Phase 4)

Script: `src/evaluation/phase3_checks.py`; outputs in
`data/experiments/phase3/checks/`. Checks 1-4 use saved predictions and
actual activity only (no fitting). No pipeline file, parquet output or
database was changed.

**Check 1: exceedance rate by period** (Phase 3 thresholds, actual
activity). Days up to Dec 9 are in-sample for the thresholds, so they average
about 10% by construction.

| Period | All cells | Hotspot | Typical | Non-holiday weekdays only (all cells) |
|---|---|---|---|---|
| Nov 2 - Dec 9 (threshold period) | 10.20% | 10.28% | 10.20% | 12.28% |
| **Dec 10-16 (validation)** | **10.16%** | 9.35% | 10.18% | 12.65% |
| **Dec 17 - Jan 1 (test)** | **5.22%** | **2.08%** | 5.28% | 6.37% |

| Week (Monday start) | All cells | Hotspot |
|---|---|---|
| Nov 4 | 13.6% | 12.2% |
| Nov 11 | 11.4% | 13.5% |
| Nov 18 | 11.7% | 14.1% |
| Nov 25 | 8.1% | 9.3% |
| Dec 2 (includes the Dec 8 holiday) | 7.7% | 5.6% |
| Dec 9 | 10.4% | 9.5% |
| Dec 16 | 7.9% | 5.0% |
| **Dec 23** (Dec 25-26 holidays) | **4.5%** | **0.4%** |
| **Dec 30** (3 days, Jan 1 holiday) | **1.7%** | **0.1%** |

Weekends sit at 0.7-8.8%; weekdays in November reach 12-18%. **The
validation week is a normal week; the test window is the holiday lull**,
which deepens from Dec 21. Hotspots almost stop exceeding from Dec 24 (0.04-0.3%
per day). The one exception is Dec 25 for typical cells (12.4%; possibly
residential activity). This confirms the reading in (b). Full daily table:
`rates_daily.csv`.

**Check 2: the downstream reference recount.** The 24 non-holiday hours
contain **19,203** actual exceedances. That is confirmed twice: from the
saved per-hour results, and by an independent recount from raw activity that
matches hour by hour. At m = 1: TP 11,331, FP 4,975, FN 7,872, from 16,306
flags. The Dec 25 flagged-case count in `diagnosis_results_1h.parquet` is
also 19,203, but that comes from a different date, the v1 in-sample
forecasts, the full-period thresholds and a separate code path. **It is a
coincidence, not a mix-up.** The report figure stands.

**Check 3: H3 / P3 (typical-cell margin from Dec 10-16, hotspots at m = 1).**
No new tuning: the typical-cell margins are those already chosen (0.960 /
0.950 / 0.950 / 0.945).

| Strategy | F1 all +1h / +2h / +3h / +4h | F1 hotspot +1h / +2h / +3h / +4h |
|---|---|---|
| P0 pooled, m = 1 | 0.540 / 0.500 / 0.481 / 0.465 | 0.587 / 0.566 / 0.539 / 0.526 |
| P1 pooled, global m | 0.564 / 0.523 / 0.506 / 0.492 | 0.587 / 0.560 / 0.540 / 0.513 |
| P3 pooled, typical m, hotspot m = 1 | 0.564 / 0.523 / 0.506 / 0.492 | 0.587 / 0.566 / 0.539 / 0.526 |
| H0 pooled + hotspot, m = 1 | 0.540 / 0.500 / 0.481 / 0.466 | 0.679 / 0.609 / 0.585 / 0.548 |
| H2 hot/typical margins | 0.565 / 0.523 / 0.506 / 0.493 | 0.668 / 0.607 / 0.571 / 0.544 |
| **H3 typical m, hotspot m = 1** | **0.565 / 0.523 / 0.506 / 0.493** | **0.679 / 0.609 / 0.585 / 0.548** |

Bootstrap (90% intervals):
- **H3 vs H0:** better overall at every horizon (+0.023 to +0.027); equal on
  hotspots by construction.
- **H3 vs H2:** not significant overall or on hotspots (+0.002 to +0.014).
- **H3 vs P1:** better on hotspots at +1h (+0.092) and +2h (+0.049); n.s. at
  +3h and +4h.
- **P3 vs P1:** not significant overall or on hotspots.

**Reading:** dropping the hotspot margin costs nothing measurable, and H3 is
simpler than H2.

**Caution (added at review).** H3's hotspot results equal H0's **by
construction** (both use m = 1 on hotspots). So "no hotspot difference vs H0"
is not evidence for anything. The evidence for keeping hotspots at m = 1
comes from the strategies that **do** put a margin on hotspots:
- *Superseded by "Threshold sensitivity of the significance calls": in window 1 this is borderline (+2h) and threshold-dependent (+3h / +4h).* **H1 (global margin) vs H0 on hotspots:** significantly worse at +2h to +4h
  in window 1 (−0.033, −0.043, −0.038).
- **The same in window 2:** the global margin is significantly worse at +1h
  (−0.022).
- **H2 (separate hotspot margin) vs H0:** not significant at any horizon. A
  hotspot margin never helped on Milan. *(On Trentino a margin does help hotspots; see Phase 5 "Margin gain in w2".)*

Recall-target typical margins (hotspots at m = 1), test precision / recall
on typical cells:

| Target | +1h | +2h | +3h | +4h |
|---|---|---|---|---|
| Recall ≥ 0.5 | m 1.000: 0.637 / 0.467 | m 0.990: 0.574 / 0.464 | m 0.985: 0.539 / 0.466 | m 0.980: 0.509 / 0.471 |
| Recall ≥ 0.6 | m 0.980: 0.579 / 0.545 | m 0.970: 0.514 / 0.539 | m 0.965: 0.482 / 0.539 | m 0.960: 0.455 / 0.545 |
| Recall ≥ 0.7 | m 0.955: 0.503 / 0.635 | m 0.945: 0.445 / 0.624 | m 0.940: 0.416 / 0.624 | m 0.935: 0.393 / 0.630 |

Every target lands about 0.04-0.08 below its validation recall on test (the
holiday-lull effect). Overall F1 at the recall-0.6 margins (0.562 / 0.527 /
0.509 / 0.497) is within 0.005 of the F1-best margins, with fewer flags
(5.0-6.3% vs 6.3-7.5%).

**Check 4: outcome replay on actual activity** (same 24 + 6 hours, +1h).
Moves come from the unchanged solver and flag rules, applied to actual
activity at the target hour. **Assumption:** activity units transfer
one-for-one between neighbouring cells with no radio effects; the dataset has
no real capacity figures. "A" here means **the moves of V0** with margin
flags as watch-only. (Phase 3's earlier "A" let watch flags into the ROUTINE
set.) Perfect foresight: the same pipeline, including the diagnosis rule,
with actual activity used as the forecast.

Totals over the 24 non-holiday hours (actual excess over threshold before any
move: 1,261,738):

| Variant | Moved | Actual excess after | Reduction | Relief on actual exceeders | Moved to non-exceeders (share) | Donor harm (donors pushed over) | Donors already over threshold (amount donated) |
|---|---|---|---|---|---|---|---|
| V0 / A watch-only (9,211 watch flags at m = 0.96; 4,183 at 0.98) | 354,585 | 1,077,017 | 184,721 (14.6%) | 247,474 | 65,684 (18.5%) | 62,753 (1,345) | 1,909 (34,576) |
| B, m = 0.96 | 570,111 | 1,006,806 | 254,932 (20.2%) | 311,903 | 173,084 (30.4%) | 56,971 (1,232) | 1,586 (31,838) |
| C, m = 0.96 | 674,415 | 1,003,320 | 258,418 (20.5%) | 364,419 | 201,603 (29.9%) | **106,001 (2,089)** | 1,499 (51,986) |
| B, m = 0.98 | 459,635 | 1,038,916 | 222,822 (17.7%) | 285,928 | 111,686 (24.3%) | 63,106 (1,331) | 1,836 (34,313) |
| C, m = 0.98 | 501,924 | 1,037,380 | 224,358 (17.8%) | 311,716 | 120,508 (24.0%) | 87,359 (1,742) | 1,736 (44,888) |
| **Perfect foresight** | 596,143 | 665,595 | **596,143 (47.2%)** | 596,143 | 0 | 0 | 0 |

**Net vs gross, with V0 as baseline and the perfect-foresight ceiling**
(added at review).
- **"Reduction" in this report is always NET:** actual excess after the moves
  vs before. That is gross relief minus the harm done to donors.
- **Gross relief** is check (ii): the sum over cells that actually exceeded of
  min(received, actual excess).
- Percentages are of total actual excess before any move.
- "Of ceiling" divides by the perfect-foresight value. **The ceiling is
  solver-limited, not the true optimum.** It uses the same pipeline (including
  the diagnosis rule, under which ANOMALOUS cells receive nothing), solves one
  hour at a time with no carry-over, and assumes activity units transfer
  one-for-one with no radio effects.

Non-holiday hours (24; actual excess before any move 1,261,738):

| Variant | Net reduction (% of excess; % of ceiling) | Gross relief (% of excess; % of ceiling) | Donor harm (% of excess) | Moved | To non-exceeders (share of moved) | Donors pushed over / all donors | Donors already over threshold (amount donated) |
|---|---|---|---|---|---|---|---|
| **V0 current (baseline)** | 184,721 (14.6%; 31.0%) | 247,474 (19.6%; 41.5%) | 62,753 (5.0%) | 354,585 | 65,684 (18.5%) | 1,345 / 10,338 | 1,909 (34,576) |
| A watch-only, m = 0.96 or 0.98 | identical to V0 | identical | identical | identical | identical | identical | identical |
| B, m = 0.96 | 254,932 (20.2%; 42.8%) | 311,903 (24.7%; 52.3%) | 56,971 (4.5%) | 570,111 | 173,084 (30.4%) | 1,232 / 15,073 | 1,586 (31,838) |
| C, m = 0.96 | 258,418 (20.5%; 43.3%) | 364,419 (28.9%; 61.1%) | 106,001 (8.4%) | 674,415 | 201,603 (29.9%) | 2,089 / 14,282 | 1,499 (51,986) |
| B, m = 0.98 | 222,822 (17.7%; 37.4%) | 285,928 (22.7%; 48.0%) | 63,106 (5.0%) | 459,635 | 111,686 (24.3%) | 1,331 / 12,652 | 1,836 (34,313) |
| C, m = 0.98 | 224,358 (17.8%; 37.6%) | 311,716 (24.7%; 52.3%) | 87,359 (6.9%) | 501,924 | 120,508 (24.0%) | 1,742 / 12,201 | 1,736 (44,888) |
| Perfect foresight (ceiling) | 596,143 (47.2%; 100%) | 596,143 (47.2%; 100%) | 0 | 596,143 | 0 | 0 / 12,797 | 0 |

Holiday hours (6; excess before 526,351; indicative):

| Variant | Net (% excess; % ceiling) | Gross (% excess; % ceiling) | Donor harm (% excess) | To non-exceeders | Pushed over / already over |
|---|---|---|---|---|---|
| V0 / A | 28,782 (5.5%; 17.4%) | 44,549 (8.5%; 26.9%) | 15,767 (3.0%) | 22.8% | 274 / 955 |
| B, m = 0.96 | 37,887 (7.2%; 22.9%) | 54,971 (10.4%; 33.2%) | 17,084 (3.2%) | 32.3% | 251 / 1,098 |
| C, m = 0.96 | 38,862 (7.4%; 23.4%) | 64,615 (12.3%; 39.0%) | 25,752 (4.9%) | 31.2% | 372 / 1,028 |
| B, m = 0.98 | 33,107 (6.3%; 20.0%) | 50,082 (9.5%; 30.2%) | 16,975 (3.2%) | 27.3% | 257 / 1,059 |
| C, m = 0.98 | 34,278 (6.5%; 20.7%) | 55,033 (10.5%; 33.2%) | 20,755 (3.9%) | 27.0% | 351 / 1,013 |
| Perfect foresight | 165,725 (31.5%; 100%) | 165,725 (31.5%; 100%) | 0 | 0% | 0 / 0 |

So **V0 achieves 31% of the solver-limited ceiling** (net), and **B at
m = 0.96 achieves 43%**. Earlier wording such as "B and C remove 20% of actual
excess" refers to the **net** reduction. V0 sends **18.5%** of its moved
capacity to cells that did not actually exceed (table above).

**Uncertainty** (added at review; `data/experiments/phase4/followup/replay_bootstrap.csv`).
- **Method:** paired bootstrap, **unit = 1 hour** (24 non-holiday hours,
  resampled with replacement; 5,000 resamples; seed 0). Totals over 24 hours,
  90% intervals.
- **Sensitivity check:** resampling **days** instead (4 days of 6 hours each;
  very few units, so the intervals are coarse).

| m | Comparison | Net-reduction difference [hour CI] / [day CI] | Donor-harm difference [hour CI] / [day CI] |
|---|---|---|---|
| 0.96 | B − V0 | **+70,211** [+40,521, +103,949] / [+30,241, +114,390] | −5,782 [−14,363, +2,599] n.s. / [−11,481, −506] |
| 0.96 | C − V0 | **+73,697** [+44,233, +106,304] / [+32,838, +116,165] | **+43,248** [+23,591, +65,110] / [+14,383, +70,374] |
| 0.96 | C − B | +3,486 [−1,266, +8,302] n.s. / n.s. | **+49,030** [+27,579, +73,229] / [+22,039, +78,774] |
| 0.98 | B − V0 | **+38,101** [+20,758, +58,081] / [+15,297, +61,990] | +353 [−4,445, +5,542] n.s. / n.s. |
| 0.98 | C − V0 | **+39,637** [+23,110, +58,058] / [+18,626, +59,953] | **+24,606** [+12,258, +37,521] / [+5,868, +42,177] |
| 0.98 | C − B | +1,536 [−3,680, +6,789] n.s. / n.s. | **+24,253** [+13,559, +35,965] / [+9,292, +40,878] |

**Reading:**
- **B and C both remove significantly more actual excess than V0**, under
  either resampling unit.
- **C is not significantly better than B on net reduction, but causes
  significantly more donor harm.**
- **B's donor harm is not significantly different from V0's** at the hour
  level. At the day level, B − V0 at m = 0.96 is just significant in B's
  favour (fewer harmed); with only 4 day-units, read that as indicative.

**Assumption, repeated:** activity units transfer one-for-one with no radio
effects. 24 hours.

**Holiday hours** (6, indicative; excess before 526,351): reductions are
V0 / A 5.5%, B 7.2%, C 7.4% at m = 0.96, and perfect foresight 31.5%.
Moved to non-exceeders: 23% (V0), 31-32% (B, C).

**Reading:**
- In every row, actual reduction = relief − donor harm, so the accounting
  balances.
- **Margins help on actual activity.** B and C at m = 0.96 remove 38-40%
  more actual excess than V0 (255-258k vs 185k).
- **B and C achieve about the same reduction, but C does it with 86% more
  donor harm.** C moves more capacity (674k vs 570k), and much of the extra
  pushes donors over their own threshold. **B is the more efficient of the
  two.**
- About **30% of margin-driven moves go to cells that did not actually
  exceed**, against 18.5% under V0. That is the price of the extra recall.
- **Forecast spare is unreliable at the donor end.** Even V0 has 1,909
  donors that were actually over threshold when they donated, a donor-side
  forecast error no flag rule addresses.
- **Perfect foresight reaches only 47%.** The rest is limited by neighbours'
  real spare and by the diagnosis rule (ANOMALOUS cells receive nothing), so
  the forecast accounts for at most the gap between 15-20% and 47%.

**Check 5: margin transfer on window 2** (+1h and +4h).

**Configuration.** The deployed configuration (23 final features, 63
leaves) was refitted on window 2: training targets Nov 2-17, with the tree
count by the 0.5% rule on Nov 18-24. That is **the second use of Nov
18-24**; the first was Phase 2's tree counts. The result is **999 trees at
+1h** (best at 1,583) and **657 at +4h** (best at 1,055). With only 16 days
of training data, the validation loss turns up well before the cap.
- **Fits:** 6.9 and 7.1 min for 3,000 trees; peak commit 5.05 GB; headroom
  7.2 / 6.6 GB at fit time. The first launch stopped at 5.5 GB under the 6 GB
  rule, and was re-run after memory was freed.
- **Thresholds:** from Nov 1-17 only (**17 days**). Hotspots from window 2
  training means.
- **Strategies:** pooled model only (no window 2 hotspot model), so
  P0 / P1 / P2 / P3, plus window 1's own global margin applied to window 2.
- **Secondary:** the existing window 2 "tuned" model (production features,
  31 leaves) is also reported.
- Margins were tuned on Nov 18-24 and tested on Nov 25 - Dec 7 (13 days, no
  holidays).

| | +1h | +4h |
|---|---|---|
| Base rate, validation → test, all cells | 11.84% → 8.94% (−24%) | 11.84% → 8.99% (−24%) |
| Base rate, validation → test, hotspots | 13.70% → 7.90% (−42%) | 13.70% → 7.94% (−42%) |
| F1-best global margin, window 2 (window 1) | **0.950 (0.960)** | **0.935 (0.945)** |
| Hotspot / typical margin, window 2 (window 1) | 0.960 / 0.950 (0.970 / 0.960) | 0.960 / 0.935 (0.950 / 0.945) |
| F1, m = 1 → window 2's own global m | 0.552 → 0.574 | 0.484 → 0.521 |
| F1 gain, window 2's own m (bootstrap) | **+0.022 [+0.009, +0.036]** | **+0.036 [+0.014, +0.061]** |
| F1 gain, window 1's margin applied to window 2 | **+0.027 [+0.016, +0.038]** | **+0.042 [+0.022, +0.064]** |
| Hotspot F1: P1 global / P2 hot-typ / P3 typical-only vs m = 1 | **−0.022** / **−0.010** / 0 | −0.016 n.s. / +0.004 n.s. / 0 |
| Flagged share vs actual (window 2 global m) | 12.9% vs 8.9% | 14.8% vs 9.0% |
| Precision / recall, m = 1 → global m | 0.629 / 0.492 → 0.486 / 0.702 | 0.582 / 0.415 → 0.419 / 0.688 |

Secondary model (production features, 31 leaves): the same margins (0.950 /
0.935), larger gains (+0.044 / +0.069), and the same hotspot harm from a
global margin at +1h.

**Reading:**
- **The margins transfer.** The margin chosen on a different, holiday-free
  period lands within 0.010 of window 1's at both horizons. Window 1's
  margin, applied unchanged to window 2's test, gives a significant gain of
  the same size (+0.027 / +0.042).
- **The gain over m = 1 reproduces** on a test period without holidays
  (+0.022 / +0.036, significant).
- *Superseded by "What can be claimed": hotspots at m = 1 is a conservative default, firm only here (w2 +1h).* **Hotspots should stay at m = 1.** A global margin hurts hotspot F1
  significantly at +1h in window 2 as well, and P3 (typical-only) is neutral
  by construction.
- **Flagging more than actually exceeds is expected, not a sign of a broken
  margin.** It also happens in window 2, where the test period is not a
  lull: 12.9-14.8% flagged against 8.9-9.0% actual. The F1-best margin
  trades precision for recall (0.49 → 0.70).
- **Window 2 caveats:** its thresholds come from only 17 days, which include
  the busiest November weeks, so both its slices sit above 10% (11.8% and
  8.9%). Its validation-to-test shift is milder than window 1's (−24% vs
  −48%).
- **The hotspot −0.022 at +1h is significant** (90% interval −0.031 to
  −0.013): a global margin hurts window 2's hotspots at +1h. At +4h it is
  not significant (−0.016, interval −0.032 to +0.002).

**Margin plateaus, and why one validation week is a noisy guide** (added at
review). Window 1's margins gave a slightly *larger* gain on window 2 than
window 2's own (+0.027 vs +0.022 at +1h; +0.042 vs +0.036 at +4h). The
validation F1 curve is flat near its peak, so the "best" margin is
poorly determined. The table shows the margins within 0.002 and 0.005 of
the best validation F1 (all cells, final pooled model), with the test F1 across
that range (`data/experiments/phase4/margin_plateaus.csv`):

| Window, horizon | Val. best (F1) | Within 0.002 | Test F1 across it | Within 0.005 | Test F1 across it | Test-best margin (F1) | Test F1 at m = 1 |
|---|---|---|---|---|---|---|---|
| w1 +1h | 0.960 (0.639) | 0.950-0.965 | 0.559-0.565 | 0.945-0.970 | 0.555-0.565 | 0.970 (0.565) | 0.540 |
| w1 +2h | 0.950 (0.614) | 0.945-0.960 | 0.520-0.526 | 0.935-0.965 | 0.511-0.527 | 0.965 (0.527) | 0.500 |
| w1 +3h | 0.950 (0.599) | 0.940-0.955 | 0.500-0.508 | 0.935-0.960 | 0.494-0.509 | 0.965 (0.509) | 0.481 |
| w1 +4h | 0.945 (0.594) | 0.940-0.950 | 0.489-0.495 | 0.930-0.955 | 0.479-0.496 | 0.960 (0.497) | 0.465 |
| w2 +1h | 0.950 (0.606) | 0.945-0.960 | 0.571-0.579 | 0.935-0.965 | 0.562-0.580 | 0.965 (0.580) | 0.552 |
| w2 +4h | 0.935 (0.566) | 0.925-0.945 | 0.513-0.526 | 0.920-0.955 | 0.509-0.528 | 0.955 (0.528) | 0.484 |

**Reading:**
- **Tuning on a single validation week is noisy.** The validation optimum
  is a plateau 0.010-0.020 wide (within 0.002) or 0.025-0.035 wide (within
  0.005). Within the wider plateau, test F1 varies by up to 0.019, roughly
  half the size of the gain over m = 1.
- **In all six cases, the margin that would have been best on test sits
  0.01-0.02 above the validation optimum.** That is why window 1's (slightly
  higher) margins did a little better on window 2. Every margin on the
  plateau still beats m = 1 clearly. **This is weak evidence:** the six cases
  come from only two windows, and within a window the horizons share the same
  days and are strongly correlated. So it is closer to two observations than
  six. The ranges are kept; leaning to their upper end is a tentative
  preference, not a finding.
- **Recommended margins, expressed as ranges** (typical cells, where the
  plateaus of both windows overlap): **+1h 0.95-0.96, +2h 0.945-0.96, +3h
  0.94-0.955, +4h 0.94-0.945.** Leaning to the upper end is supported by
  the test-best margins in both windows. +2h and +3h rest on window 1 only.

### Decision table (post-Phase-3 checks)

| Decision | Options | Evidence for | Evidence against | Unverified |
|---|---|---|---|---|
| **Margin policy** | global / hot+typical (H2) / **typical only, hotspots m = 1 (H3)** | H3 = H2 overall at every horizon (window 1). Hotspot difference n.s. everywhere (check 3). A global margin hurts hotspot F1 in both windows (significant at +2h to +4h in window 1, +1h in window 2). *Later qualified: the window 1 hotspot-margin cost is borderline (+2h) and threshold-dependent (+3h / +4h); it is firm only in Milan w2 at +1h ("Threshold sensitivity of the significance calls").* Typical-only margins transfer: window 2 picks 0.950 / 0.935 vs window 1's 0.960 / 0.945, and window 1's margin gives a significant gain on window 2 (check 5). H3 needs one fewer parameter than H2. | Flags exceed actual exceedances (6.3-7.6% vs 5.3% window 1; 12.9-14.8% vs 9% window 2), so precision falls (0.64 → 0.52 at +1h, window 1). Recall targets land 0.04-0.08 low on the window 1 test. | The operator's miss vs false-alarm cost. +2h / +3h transfer (window 2 checked +1h / +4h only). Behaviour on a normal (non-lull) period long enough to tune and test separately. |
| **Hotspot model, per horizon** | pooled only / **hotspot model at +1h** / hotspot model at all horizons | Significant at +1h on hotspots: +0.092 vs pooled (window 1). With typical-only margins, H3 vs P1 is significant at +1h (+0.092) and +2h (+0.049). Validation loss picked the final-feature hotspot model only at +1h. | At +2h to +4h the hotspot model vs pooled is **not significant** (+0.043, +0.045, +0.022). It adds a second model per horizon to maintain. | No window 2 hotspot model, so transfer of the +1h hotspot gain is untested. Hotspot behaviour outside the holiday lull: window 1 test hotspots barely exceed from Dec 24 (check 1). |
| **Solver variant** | V0 / **watch status (A)** / **B** / C | On actual activity, B and C remove 20% of excess vs V0's 14.6% (check 4). B achieves the same as C (255k vs 258k) with **46% less donor harm** (57k vs 106k; 1,232 vs 2,089 donors pushed over). A as watch-only leaves the solver identical to V0 and adds 9,211 watch flags (m = 0.96) without fake "resolved" counts. | About 30% of B / C moved capacity goes to cells that did not actually exceed (vs 18.5% for V0). Every variant uses donors that were already over threshold (about 1,500-1,900). C adds about 86% more donor harm than B (106k vs 57k; 69% more than V0). Phase 3's earlier "A" (watch flags entering ROUTINE) fabricates "resolved" cases. | The one-for-one activity-unit assumption (no radio effects). Only 24 + 6 hours at +1h. The holiday subset is indicative. Perfect foresight only reaches 47%, so the solver's neighbour and diagnosis limits cap any variant. |

**Suggested combination for review:**
- **H3 margins** (typical-cell margin from validation, hotspots at m = 1).
- **Hotspot model at +1h only**; pooled model at +2h to +4h, unless the +2h
  gain (+0.049 with H3) is judged worth a second model.
- **Margin flags as a watch status first.** **B** if flags are to drive
  moves; not C, because of its donor harm.
- **Separately:** the donor-side error (donors already over threshold) and
  the holiday-exemption conflict go to Phase 4.

---

## Phase 4: sensitivity of design choices (bounded)

Script: `src/evaluation/phase4_sensitivity.py`; outputs in
`data/experiments/phase4/`. No pipeline file, parquet output or database was
changed. `diagnosis_agent` and `solver` were imported, not edited.

**Choices.**
- **Common setup:** the final pooled model (63 leaves) at +1h, flag rule
  m = 1 (today's rule), Phase 3 thresholds (Nov 1 - Dec 9) unless the check
  varies them, and the same 24 non-holiday + 6 holiday downstream hours.
- **Neighbour fraction:** computed for all hours at once with grid shifts,
  verified against `diagnosis_agent.get_neighbors` on 300 random (cell, hour)
  pairs.
- **Hotspot refits** in (d) use a **frozen size** (31 leaves, 568 trees, from
  the 2% final-feature model), so Dec 10-16 was not used again.

| Step | Time | Peak commit |
|---|---|---|
| Margin plateaus | 18 s | 3.2 GB |
| (a) rule | 29 s | 3.3 GB |
| (b) holiday | 32 s | 3.3 GB |
| (c) percentile | 26 s | 3.3 GB |
| (d) hotspot cutoff (3 fits, 1.5-6 s each) | 29 s | 3.4 GB |

No full-grid fit was needed.

### (a) Does the diagnosis rule mean anything? (first outcome-based check)

**Hour set, thresholds and margin.**
- **Hour set:** all 311 non-holiday test hours (+1h target hours from Dec 17
  01:00 to Jan 1 20:00, excluding Dec 25, Dec 26 and Jan 1).
- **Thresholds:** Phase 3 (Nov 1 - Dec 9).
- **Margin:** m = 1.
- **This is not the same hour set as the downstream V0 row** (24 hours), so
  the two tables below differ. The reconciliation follows.

Test-window non-holiday hours (13 days, 311 hours), +1h flags at m = 1:

| Neighbour fraction | Class | Flags (share) | Precision (actually exceeded) | Mean actual excess per flag | Mean excess when it did exceed |
|---|---|---|---|---|---|
| 0.2 | ROUTINE | 110,595 (92.8%) | 0.659 | 58.2 | 88.4 |
| 0.2 | ANOMALOUS | 8,518 (7.2%) | 0.454 | 65.6 | 144.6 |
| **0.3 (current)** | **ROUTINE** | 102,071 (85.7%) | **0.671** | 58.2 | 86.7 |
| **0.3 (current)** | **ANOMALOUS** | 17,042 (14.3%) | **0.485** | 62.2 | 128.4 |
| 0.4 | ROUTINE | 89,282 (75.0%) | 0.687 | 57.8 | 84.2 |
| 0.4 | ANOMALOUS | 29,831 (25.0%) | 0.517 | 61.5 | 118.9 |

**Precision, ROUTINE minus ANOMALOUS** (paired day bootstrap, 13 days):
+0.205 [+0.176, +0.241] at 0.2; **+0.186 [+0.164, +0.210] at 0.3**; +0.169
[+0.149, +0.189] at 0.4. All three are significant.

**Reconciliation with the V0 downstream row** (added at review). The same
rule was run on exactly the 24 downstream hours. Flags and true positives
per class sum to the V0 row:

| Set | Fraction | ROUTINE: flags / TP / precision | ANOMALOUS: flags / TP / precision | All flags: flags / TP / precision |
|---|---|---|---|---|
| **24 downstream hours (V0 set)** | 0.2 | 15,345 / 10,853 / 0.707 | 961 / 478 / 0.497 | 16,306 / 11,331 / 0.695 |
| **24 downstream hours (V0 set)** | **0.3** | **14,311 / 10,275 / 0.718** | **1,995 / 1,056 / 0.529** | **16,306 / 11,331 / 0.695** |
| **24 downstream hours (V0 set)** | 0.4 | 12,735 / 9,340 / 0.733 | 3,571 / 1,991 / 0.558 | 16,306 / 11,331 / 0.695 |
| All 311 non-holiday hours | 0.2 | 110,595 / 72,865 / 0.659 | 8,518 / 3,864 / 0.454 | 119,113 / 76,729 / 0.644 |
| All 311 non-holiday hours | 0.3 | 102,071 / 68,468 / 0.671 | 17,042 / 8,261 / 0.485 | 119,113 / 76,729 / 0.644 |
| All 311 non-holiday hours | 0.4 | 89,282 / 61,297 / 0.687 | 29,831 / 15,432 / 0.517 | 119,113 / 76,729 / 0.644 |

On the V0 set, ROUTINE precision is **0.718**, matching the reviewer's
estimate of about 0.72. The 0.671 above belongs to the larger 311-hour set,
whose overall flag precision is lower (0.644 vs 0.695).

**Bootstrap with HOURS as the unit** (flags within an hour are correlated;
5,000 resamples). Precision, ROUTINE minus ANOMALOUS:

| Fraction | 24-hour V0 set | 311-hour set |
|---|---|---|
| 0.2 | **+0.210** [+0.172, +0.257] | **+0.205** [+0.182, +0.227] |
| 0.3 | **+0.189** [+0.156, +0.219] | **+0.186** [+0.167, +0.205] |
| 0.4 | **+0.176** [+0.143, +0.205] | **+0.169** [+0.152, +0.186] |

All are significant with either unit and on either set. Time: 29 s; peak
commit 3.3 GB.

Where moves go (24 hours). In the counterfactual, every flag is sent to the
solver, and capacity is split by the class each flag would have had:

| Fraction | Moves under the rule | Share to non-exceeders (rule) | Counterfactual share to non-exceeders: ROUTINE-class / ANOMALOUS-class | Net reduction under the rule |
|---|---|---|---|---|
| 0.2 | 400,031 | 19.7% | 19.7% / **42.0%** | 205,122 |
| 0.3 | 354,585 | 18.5% | 18.5% / **36.4%** | 184,721 |
| 0.4 | 282,797 | 16.7% | 16.8% / **31.8%** | 150,914 |

**Reading: the rule is informative, but not for the reason its name
suggests.**
- **Isolated (ANOMALOUS) flags are worse as flags.** They are wrong about
  half the time (precision 0.49 vs 0.67), and if routed, twice the share of
  their capacity would go to cells that did not exceed (36% vs 18.5%).
- **When an isolated flag is real, it is a bigger overload** (mean excess
  128 vs 87). So by mean actual excess per flag, ANOMALOUS flags are *not*
  smaller (62 vs 58).
- **Holding them back for a human is therefore sensible.** They are
  unreliable, and the real ones are large.
- **A stricter rule (0.4)** raises precision on both sides, but routes 20%
  less capacity and lowers net reduction (151k vs 185k). The 0.3 setting
  sits between; this is a policy trade-off, not a data-decided optimum.

### (b) Holiday exemption on vs off

The flags whose class would change (ROUTINE → ANOMALOUS) if the exemption
were removed, against the flags that would stay ROUTINE. Precision means
the flag actually exceeded:

| Source | Day | Changing flags: count / precision / mean excess | Staying flags: count / precision / mean excess |
|---|---|---|---|
| Final model, +1h, Phase 3 thresholds | Dec 25 | 1,376 / **0.556** / 38.9 | 16,235 / 0.745 / 38.0 |
| Final model | Dec 26 | 795 / **0.357** / 28.0 | 7,113 / 0.468 / 19.3 |
| Final model | Jan 1 (21 h) | 391 / 0.197 / 21.3 | 2,077 / 0.195 / 8.6 |
| Existing `diagnosis_results_1h.parquet` (v1 in-sample, full-period thresholds) | Dec 25 | 1,086 / **0.599** / 42.3 | 18,117 / 0.726 / 30.4 |
| Existing diagnosis output | Dec 26 | 933 / **0.193** / 16.7 | 11,717 / 0.349 / 6.5 |

**6 holiday downstream hours.** With the exemption, all 5,744 flags are
ROUTINE, and 487 would change without it.
- **The 487 changing flags** receive 7,600 units, 28.0% of it to
  non-exceeders.
- **The staying flags** receive 58,120 units, 22.1% to non-exceeders.
- **Exemption on vs off:** net reduction 28,782 vs 25,584; donor harm 15,767
  vs 14,118.

**Reading (corrected at review): the exemption lets lower-precision flags
through and wastes more of their moves, but its net solver effect in the 6
hours was slightly positive.** The supported claim is lower flag precision
and more wasted moves, **not** a worse net outcome. Indicative only (3 days,
6 hours).
- **Lower precision:** on Dec 25 and 26, the flags it protects from review
  have 0.10-0.19 lower precision, in both forecast sources. On Jan 1 there is
  no difference.
- **More wasted moves:** a larger share of their moves goes to cells that
  did not exceed.
- **Still slightly positive on net**, in the 6 hours: +3.2k net reduction,
  +1.6k donor harm. Their real exceedances are larger (mean excess 39 vs 38
  on Dec 25, 28 vs 19 on Dec 26).
- **Caveat:** 3 holiday days, 6 solver hours, no bootstrap (too few days).
  Indicative only.
- **For the decision:** the exemption trades human review of
  lower-precision flags for a small net gain.

### (c) Congestion percentile 85 / 90 / 95 (policy, not data)

Each definition is judged against **its own** actuals.

| Percentile | Flagged / actual share (full test window) | Precision / recall / F1 | Flags in 24 h: ROUTINE / ANOMALOUS (ROUTINE share) | Solver coverage of forecast deficit | Fully resolved | Net reduction / gross relief / donor harm (% of own actual excess) | Moved to non-exceeders |
|---|---|---|---|---|---|---|---|
| 85 | 6.7% / 7.8% | 0.666 / 0.573 / 0.616 | 24,948 / 2,095 (92.3%) | 42.0% | 39.4% | 15.4% / 20.5% / 5.1% | 16.8% |
| **90** | 3.9% / 5.3% | 0.637 / 0.468 / 0.540 | 14,311 / 1,995 (87.8%) | 50.3% | 46.7% | 14.6% / 19.6% / 5.0% | 18.5% |
| 95 | 1.5% / 2.9% | 0.612 / 0.320 / 0.420 | 5,447 / 1,174 (82.3%) | 58.4% | 54.3% | 12.4% / 15.9% / 3.5% | 16.1% |

**Reading:**
- **These rows measure different targets.** A higher F1 at 85 does not make
  85 better; it is an easier, more frequent target.
- **Higher percentiles** mean fewer, rarer flags, a higher solver coverage
  share (less deficit chasing the same spare), lower recall and a smaller
  share of own excess removed.
- **The percentile defines what "congestion" means operationally.** It is a
  **policy choice the data cannot pick**. Nothing here argues against 90.

### (d) Hotspot cutoff 1% / 2% / 5%

+1h hotspot-only model, refitted per cutoff with a frozen size, against the
pooled model **on the same cells** (paired day bootstrap, 16 test days).
The 2% refit reproduces the saved model exactly (F1 0.679, MAE 390.35).

| Cutoff | Cells | Training rows | Actual share | F1 pooled → hotspot-only | F1 difference [90%] | MAE pooled → hotspot-only | MAE difference [90%] | Fit |
|---|---|---|---|---|---|---|---|---|
| 1% | 100 | 91,100 | 2.6% | 0.583 → 0.711 | **+0.128 [+0.105, +0.154]** | 510.4 → 538.9 | +28.5 [−19.0, +80.8] n.s. | 1.5 s |
| 2% | 200 | 182,200 | 2.1% | 0.587 → 0.679 | **+0.092 [+0.061, +0.122]** | 388.2 → 390.3 | +2.2 [−19.8, +26.8] n.s. | 2.8 s |
| 5% | 500 | 455,500 | 2.2% | 0.566 → 0.602 | **+0.036 [+0.014, +0.053]** | 263.9 → 267.9 | +4.1 [−6.7, +17.7] n.s. | 6.0 s |

**Reading:**
- **The hotspot model's advantage is a flagging advantage, not an accuracy
  one.** F1 improves significantly at every cutoff, but MAE never does.
- **The flag gain shrinks as the set widens** (+0.128 → +0.092 → +0.036).
  The benefit is concentrated in the very busiest cells.
- **2% is a reasonable middle; 1% would give a larger gain on fewer cells.**
  Choosing between them is a scope decision. These are +1h results only, on
  the holiday-lull test window, where hotspots barely exceed after Dec 24
  (check 1).

### Solver reach: 2-hop transfers (bounded check, not Phase 6)

Script: `src/evaluation/phase4_followup.py reach2`.
- **Setup:** perfect foresight (actual activity used as the forecast), the
  same diagnosis rule (1-hop neighbour fraction), the same 24 non-holiday
  hours. The solver's LP was copied with the neighbour set as a parameter;
  `solver.py` is unchanged.
- **Validation of the copy:** with 1-hop neighbours it reproduces the
  original solver exactly (596,143 net in both).

| Neighbour set for transfers | Net reduction of actual excess (of 1,261,738) | Solver time (24 h) |
|---|---|---|
| 1 hop (8 cells), current solver | 596,143 (**47.2%**) | 3.5 s (original) / 1.4 s (copy) |
| **2 hops (up to 24 cells, Chebyshev distance ≤ 2)** | 909,852 (**72.1%**) | 5.3 s |
| 2 hops, every flag routed (no diagnosis hold-back) | 1,038,522 (82.3%) | 6.3 s |

**Reading:**
- **Allowing 2-hop transfers raises the solver-limited ceiling by about 25
  points** (47% → 72%). Spare capacity one ring further out is the main
  limit on the current solver, not the forecast.
- **This is an upper bound**: perfect foresight, one-for-one units, and no
  cost or loss for longer transfers.
- **It is also a bound on the grid representation, not on the network.**
  Square values are area shares of base-station coverage areas (dataset
  limitation a), so a 1-hop or 2-hop "transfer" may stay within one base
  station's coverage area. Neither ceiling is a physical capacity figure.
- **The extra 10 points from routing every flag** is what the diagnosis
  hold-back costs under perfect foresight. That is not a recommendation:
  with real forecasts, held-back flags have much lower precision (check a).
- Implementing multi-hop transfers belongs to Phase 6 (not started).
- **Time:** 46 s for the whole step; peak commit 2.1 GB.

---

## Dataset limitations (Barlacchi et al. 2015)

The source paper is "A multi-source dataset of urban life in the city of
Milan and the Province of Trentino", *Scientific Data* 2, 150055 (2015).
The points below were checked against the paper's text (PMC4622222). The
country-code share was measured on one Trentino file.

**(a) Activity is spread over grid squares by area, not measured per square.**
- **How it is built:** each record belongs to a radio base station's coverage
  area. A square's value is the sum over coverage areas of the area's records
  times the share of the area that overlaps the square
  (S_i(t) = Σ_v R_v(t) · |A_v ∩ i| / |A_v|).
- **Consequence:** neighbouring squares under the same coverage area share
  its load by construction, so their series are partly the same signal
  split by area.
- **For the solver:** moving "activity" from one square to a neighbour is
  **not a physical capacity move**. Squares under one coverage area are
  served by the same base station, so a transfer between them changes
  nothing on the network.
- **What this applies to:**
  - every solver figure (Phase 3 downstream, check 4 outcome replay, the
    variant comparisons B / C / V0, the Phase 4 checks);
  - the **1-hop and 2-hop perfect-foresight ceilings** (47.2%, 72.1%).
  
  They hold only under the stated one-for-one assumption, on this
  area-weighted representation.
- **It also affects the neighbour rule and the neighbour features:** part of
  the co-flagging that makes a flag ROUTINE comes from shared coverage
  areas, not from independent congestion in neighbouring squares.
- **Measured (see "Shared-coverage check"):** exact twins make up 4.4% of
  neighbour pairs and sit in quiet outer areas. None are within 10 squares
  of Duomo; 0.2% of the highest-activity decile and 28% of the
  lowest-activity decile belong to a twin group. Partial sharing cannot be
  separated from similar neighbourhoods without the coverage-area geometry,
  which is not in the dataset.

**(b) "Internet" counts records, not data volume. Totals mix record types.**
- **What a record is:** an internet record is written when a connection
  starts or ends, and within a connection after every 15 minutes or 5 MB.
  So the internet channel counts records, not bytes.
- **What `total_activity` sums:** SMS, call and internet record counts. These
  are different record types, all scaled by **a constant k set by Telecom
  Italia and not disclosed** ("which hides the true number of calls, SMS and
  connections").
- **Consequence:** thresholds, deficits, spare and "capacity" are in
  arbitrary record-count units. Internet records make up 82% of Milan's
  total (Part A), so `total_activity` mostly tracks internet session
  records. Because k is unknown, absolute levels cannot be compared between
  Milan and Trentino (Phase 5 uses a scale-free formulation for this reason).

**(c) Country code 0 is undocumented.**
- **What the paper says:** only that the field is "the phone country code of
  the nation". Code 0 is not explained.
- **Measured on one Trentino file (Nov 4):** code 0 is **24.1% of rows** but
  **2.1% of total activity**, almost all of it incoming SMS (23.5% of
  `smsin`), with no internet. Code 39 (Italy) is 97.3% of activity.
- **How the pipeline handles it:** it sums all country codes, so code 0 is
  included in every total.
- **Not checked:** Milan, because `data\raw` was not processed. Whether code 0
  means unknown, roaming or something else cannot be determined from the
  data.

**(d) The paper's own validation is limited, and our checks are its examples.**
- **What the paper validates:** it illustrates the telecom data with six
  squares: Bocconi (4259), Navigli (4456), Duomo Milan (5060), Duomo Trento
  (5200), Mesiano (5085) and Bosco della città (4703).
- **What our checks are:** Part A's three Milan checks and Phase 5's three
  Trentino check cells are **exactly these examples**. They confirm that our
  loading reproduces the paper's patterns. They are not independent evidence
  that the data measure network load.

---

## Shared-coverage check (do neighbouring squares share base-station load?)

Script: `src/evaluation/shared_coverage_check.py` (steps `stage1`,
`stage1map`, `stage2`). Outputs are in `data\experiments\shared_coverage\`.
Milan only. `data\raw\cdr_activity_aggregated.parquet` was read only.

**Why it matters.** Under the paper's construction (dataset limitation a),
equal-size squares that lie fully inside the same coverage area must have
identical series in every channel. Such "twin" squares share one
measurement, so a transfer between them is not a capacity move. A twin
neighbour also confirms a flag only trivially.

### Stage 1: how common is sharing?

**Decisions and constants** (declared before results):
- **Pairs:** 39,402 unordered 8-neighbour pairs (78,804 directed neighbour
  relations).
- **Twin:** all five channels within relative tolerance at all 1,488 hours;
  two zeros match. Primary tolerance 1e-9, sensitivity 1e-6 and 1e-4.
- **Near-twin:** matches at ≥ 99% of hours. The loader fills missing hours
  with 0, which can break an exact twin at a few hours.
- **Proportional:** not a twin; the ratio of hourly totals has a
  coefficient of variation below 1e-6, and there is no hour where exactly
  one of the two is 0.
- **Exclusion:** cells with zero variance, or mean total activity below
  1.0 per hour. No cell was excluded.
- **Twin groups:** connected components of the twin pairs.

| Measure | Result |
|---|---|
| Twin pairs, tolerance 1e-9 / 1e-6 / 1e-4 | **1,722** / 1,722 / 1,723 |
| Near-twin pairs (not exact) | 9, all in one block of seven cells (5530, 5629-5631, 5729-5731), differing at 3 of 1,488 hours |
| Proportional pairs (not twin) | 3 (273-274, 273-374, 5413-5514) |
| **Share of pairs that are twin or proportional** | **4.4%** |
| Twin groups / size distribution | 246 groups: 88 of size 2, 47 of 3, 34 of 4, …, largest 28 |
| Cells in a twin group | 1,141 (11.4% of cells) |
| **Share of total activity carried by those cells** | **3.8%** |
| Duomo, Bocconi, Navigli | 0 twin and 0 proportional neighbours; neighbour correlation 0.82-0.99 |

**Where the twins are.** They are in quiet outer areas, which is consistent
with large coverage areas there:

| Distance from Duomo (squares) | 0-10 | 10-20 | 20-30 | 30-45 | 45+ |
|---|---|---|---|---|---|
| Share of cells in a twin group | 0% | 1.4% | 10.5% | 10.1% | 16.4% |

| Activity decile (1 = quietest) | 1 | 2 | 3 | 5 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| Share in a twin group | 28% | 18% | 18% | 13% | 4.7% | 1.9% | 0.2% |

**Ordinary pairs** (not twin, not proportional): correlation of the hourly
total-activity series.

| Percentile | 1% | 10% | 25% | 50% | 75% | 90% | 95% |
|---|---|---|---|---|---|---|---|
| Correlation | 0.49 | 0.79 | 0.89 | 0.961 | 0.993 | 0.9997 | 0.99997 |

- **7.3% of ordinary pairs correlate at ≥ 0.9999**, and 14% at ≥ 0.999.
  Such pairs are more common among quiet cells (10.5% of pairs in the
  lowest activity decile vs 0.2% in the highest).
- **1,225 non-twin pairs match exactly at some hours** but not at all of
  them (including the 9 near-twins).
- Both observations are consistent with **partial** shared coverage, but
  also with genuinely similar neighbourhoods.

**Pre-declared rule and decision.**
- **The rule:** sharing is "material" if more than 10% of neighbour pairs,
  or more than 10% of activity, involve twin or proportional pairs. **The
  10% threshold is arbitrary.**
- **Decision: not material under the strict definition** (4.4% of pairs,
  3.8% of activity).
- **This is a lower bound.** The strict test detects only full containment
  in one coverage area. The share of *cells* in twin groups (11.4%) is
  above 10%, but the rule is on pairs and activity.

Time: 32 s; peak commit 3.0 GB.

### Stage 2: solver and diagnosis rule without shared-coverage pairs

**Decisions** (declared before results):
- **Setup:** as Phases 3-4. The final pooled model at +1h with saved
  predictions, Phase 3 thresholds, the same 24 non-holiday and 6 holiday
  hours, and the unchanged diagnosis rule for the solver runs.
- **Solver:** the unchanged `solve_hour` without restrictions. The
  validated LP copy (`solve_hour_k`) is used with an allowed-neighbour set.
- **Outcome replay:** under the one-for-one activity-unit assumption.
- **Restrictions** (pairs forbidden in both directions):

| Restriction | Definition | Forbidden 1-hop pairs (share of 39,402) | Forbidden pairs within 2 hops (share) |
|---|---|---|---|
| S-noNT | exact twins + proportional | 1,725 (4.4%) | 1,725 (1.5%) |
| **S (strict)** | S-noNT + the 9 near-twins (treated as twins) | **1,734 (4.4%)** | 2,915 (2.5%), including 2,912 same-twin-group pairs |
| L9999 (loose proxy) | S + ordinary pairs with correlation ≥ 0.9999 | 4,458 (11.3%) | 8,319 (7.1%) |
| L999 (looser proxy) | S + ordinary pairs with correlation ≥ 0.999 | 7,022 (17.8%) | 13,328 (11.4%) |

- **Same-group pairs at 2 hops:** for the 2-hop ceiling, every pair inside
  the same twin group (within distance 2) is also forbidden under S and the
  L brackets. The L brackets also forbid distance-2 pairs whose correlation
  passes the proxy.
- **The loose proxy also removes genuinely similar neighbours.** It is an
  over-restrictive upper bracket, not an estimate of true sharing.

**Solver** (24 non-holiday hours; actual excess before any move 1,261,738;
net reduction under the one-for-one assumption):

| Variant | Restriction | Net reduction (share) | Gross relief | Donor harm | Original moves between forbidden pairs: amount / count |
|---|---|---|---|---|---|
| V0 | none | 184,721 (14.6%) | 247,474 | 62,753 | – |
| V0 | S | 184,165 (14.6%) | 247,448 | 63,283 | 26 of 354,585 (0.007%) / 10 of 11,254 moves |
| V0 | L9999 | 184,811 (14.6%) | 247,378 | 62,567 | 168 (0.05%) / 82 |
| V0 | L999 | 184,381 (14.6%) | 247,344 | 62,963 | 410 (0.12%) / 191 (1.7%) |
| B m = 0.96 | none | 254,932 (20.2%) | 311,903 | 56,971 | – |
| B m = 0.96 | S | 254,726 (20.2%) | 311,900 | 57,174 | 3 of 570,111 (0.0005%) / 21 of 16,724 |
| B m = 0.96 | L9999 | 253,754 (20.1%) | 311,768 | 58,014 | 186 (0.03%) / 91 |
| B m = 0.96 | L999 | 254,639 (20.2%) | 311,704 | 57,065 | 555 (0.10%) / 224 (1.3%) |
| Perfect foresight, 1-hop | none / S / L9999 / L999 | 596,143 / 596,143 / 596,143 / 596,130 (**47.2%** in every case) | same | 0 | 0 / 0 / 3 / 40 moves |
| Perfect foresight, 2-hop | none / S / L9999 / L999 | 909,852 / 909,852 / 909,852 / 909,851 (**72.1%** in every case) | same | 0 | 0 / 0 / 2 / 11 moves |

- **Unrestricted runs reproduce the report exactly:** 184,721 (V0),
  254,932 (B), 596,143 (1-hop ceiling) and 909,852 (2-hop ceiling).
- **Differences in net reduction** (restricted − unrestricted, hour-level
  bootstrap):
  - V0: −556 (S), +90 (L9999), −340 (L999), all n.s.
  - B: −206 [−484, −3] (S, nominally significant but 0.08% of B's net),
    −1,178 (L9999, n.s.), −293 (L999, n.s.).
  - Ceilings: 0 to −13.
- **Holiday hours** (6, indicative): changes of at most 113 units (0.02
  percentage points); V0 5.5%, B 7.2%, 1-hop ceiling 31.5% and 2-hop ceiling
  55.7% are unchanged to one decimal.
- A restriction can *raise* net reduction slightly (V0 L9999), because the
  LP maximises moved deficit, not the net outcome.

**Perfect-foresight ceilings in activity units** (net reduction of actual
excess; `stage2_solver_summary.csv`):

| Ceiling | Hours (excess before) | none | S-noNT | S | L9999 | L999 |
|---|---|---|---|---|---|---|
| 1-hop | 24 non-holiday (1,261,738) | 596,143.27 | 596,143.27 | 596,143.27 | 596,143.12 | 596,130.07 |
| 2-hop | 24 non-holiday (1,261,738) | 909,851.63 | 909,851.63 | 909,851.63 | 909,851.63 | 909,850.90 |
| 1-hop | 6 holiday (526,351) | 165,724.57 | 165,724.57 | 165,724.57 | 165,723.08 | 165,719.12 |
| 2-hop | 6 holiday (526,351) | 293,011.74 | 293,011.74 | 293,011.74 | 293,011.74 | 293,011.56 |

**Under the strict restrictions (S-noNT and S), both ceilings are identical
to the unrestricted run** (no move used a forbidden pair).
- **The loose brackets differ slightly:**
  - L9999: −0.15 units (1-hop) and 0 (2-hop) non-holiday; −1.50 (1-hop)
    holiday.
  - L999: −13.20 (1-hop) and −0.73 (2-hop) non-holiday; −5.46 and −0.18
    holiday.
- **All are below 0.004% of the ceiling.**

**Diagnosis rule: where twins sit in the flags** (+1h, m = 1; twin cells
are 11.4% of cells):

| Set | Flags in twin cells | True positives in twin cells | Exceedances in twin cells | **Actual excess (units) in twin cells** | Flag precision: twin vs other cells |
|---|---|---|---|---|---|
| 24 downstream hours | 13.6% | 14.0% | 13.8% | **5.5%** | 0.712 vs 0.692 |
| 311 non-holiday hours | 13.2% | 12.7% | 13.6% | **5.5%** | 0.621 vs 0.648 |

**Hypothesis check: supported.** Twin cells carry about 13-14% of
count-based quantities (flags, true positives, exceedances) but only 5.5%
of the actual excess in activity units. So they weigh about 2.5× more in
count-based claims (flag precision, F1) than in unit-based claims (net
reduction).

**De-duplicated neighbour fraction.**
- **D1 (primary):** neighbours in the cell's own twin group are excluded,
  because they are the same measurement. Each other twin group among the
  neighbours counts as one unit, flagged if any member is flagged. The
  denominator is the number of units. A cell with no remaining units gets
  fraction 0 (ANOMALOUS); this affects 65 flags (24 hours) and 458 flags
  (311 hours).
- **D2 (sensitivity):** the cell's own-group neighbours also count as one
  unit.
- **Bootstrap:** hours as the unit, 5,000 resamples, seed 0, 90% intervals.

| Set | Fraction | Definition | Precision ROUTINE / ANOMALOUS | Gap [90%] | Flags changing class |
|---|---|---|---|---|---|
| 24 hours | 0.3 | original | 0.718 / 0.529 | **+0.189** [+0.156, +0.220] | – |
| 24 hours | 0.3 | D1 | 0.721 / 0.533 | **+0.188** [+0.168, +0.208] | 317 of 16,306 |
| 24 hours | 0.3 | D2 | 0.720 / 0.526 | **+0.194** [+0.167, +0.220] | 159 |
| 311 hours | 0.3 | original | 0.671 / 0.485 | **+0.186** [+0.167, +0.205] | – |
| 311 hours | 0.3 | D1 | 0.675 / 0.485 | **+0.190** [+0.174, +0.207] | 2,584 of 119,113 |
| 311 hours | 0.3 | D2 | 0.673 / 0.485 | **+0.188** [+0.170, +0.206] | 1,364 |
| 24 / 311 hours | 0.2 | original → D1 | – | +0.210 → +0.198 / +0.205 → +0.207 | 174 / 1,426 |
| 24 / 311 hours | 0.4 | original → D1 | – | +0.176 → +0.174 / +0.169 → +0.173 | 340 / 2,956 |

**The gap does not shrink materially.** At 0.3 it moves by −0.001 to
+0.005 (24 hours) and +0.002 to +0.004 (311 hours), well inside the
intervals. It stays significant at every fraction under both definitions.

### What this means for the report's claims

- **Solver claims** (V0 / B relief, B − V0, the 47.2% and 72.1% ceilings,
  all under the one-for-one assumption):
  - **Unaffected by the shared-coverage pairs this data can detect.**
  - Moves between twin or proportional pairs are 0.0005-0.007% of the moved
    amount; even the over-restrictive L999 bracket touches at most 0.12%.
    Removing them changes net reduction by at most about 0.5% and leaves
    both ceilings at 47.2% and 72.1%.
  - **The reason:** moves happen where congestion is, in the dense centre,
    and twins are in quiet outer cells.
  - **This does not remove dataset limitation (a).** Partial sharing
    between central squares, and the fact that square-to-square moves are
    not physical capacity moves, remain unverified. The one-for-one
    assumption still applies to every solver number.
- **Diagnosis-rule claim** (ROUTINE vs ANOMALOUS precision):
  - **Robust to counting each twin group once.** The gap stays at
    0.188-0.194 vs 0.189 originally.
  - Twins carry 13-14% of flags, so they could in principle have moved
    count-based claims. They did not, because their flags behave like
    other flags (precision 0.71 vs 0.69, and 0.62 vs 0.65).
- **Count-based vs unit-based:** twin cells weigh more in count-based
  measures (13-14%) than in unit-based ones (5.5%), as hypothesised. Even
  so, neither kind of claim changes materially.

### What this check does not show

- **It rules out exact sharing for most pairs, not sharing in general.**
  95.6% of neighbour pairs are not exact twins or proportional. But
  partial sharing (squares split across the same coverage areas) cannot be
  separated from real co-movement of neighbouring areas: ordinary
  neighbours are highly correlated anyway (median correlation 0.961; 7.3%
  at ≥ 0.9999).
- **Even without any sharing, a square-to-square move is not a physical
  capacity move.** The data are scaled record counts on a grid, not
  base-station capacity. Every solver number still holds only under the
  one-for-one activity-unit assumption.

### What could not be verified

- **The coverage-area geometry is not in the dataset.** So:
  - full containment (exact twins) is the only sharing that can be proven;
  - partial sharing cannot be separated from genuinely similar
    neighbourhoods;
  - the loose brackets (correlation ≥ 0.9999 / 0.999) are an
    over-restrictive proxy, not a measurement.
- **The twin test assumes equal-size squares** (the Milan grid's 235 m
  squares). Unequal overlap with the same set of areas would show as
  "proportional"; only 3 such pairs were found.
- **The materiality threshold (10%) is arbitrary.** "Not material" holds
  for the strict definition only, and is a lower bound.
- **Grid orientation** (which grid row is north) was not verified. It
  affects only the text map, not any count.
- **The solver results cover 24 + 6 hours at +1h only**, as in Phases 3-4.

Time: Stage 2 65 s; peak commit 3.3 GB. No model was fitted.

---

## What can be claimed

**Strength codes:**
- **sig** = 90% paired day-bootstrap interval excludes 0.
- **n.s.** = not significant.
- **w1** = test Dec 17 - Jan 1 (16 days, includes 3 holidays and the lull).
- **w2** = test Nov 25 - Dec 7 (13 days, no holidays).

**Optimistic** marks a number that was selected or tuned in a way that
favours it.

**Multiplicity.** No correction is made for the many 90% intervals
computed in this study (hundreds across phases, horizons, slices and
thresholds), so some significant results are expected by chance alone, and
an isolated significant result is tentative. **[single]** marks a claim
whose significance rests on a single comparison (one horizon, one slice,
one window).

**Thresholds.** Calls below use the Phase 3 thresholds (Nov 1 - Dec 9)
unless stated. **"Threshold-dependent"** marks a call that changes under the
Phase 2 thresholds (Nov 1 - Dec 16); see "Threshold sensitivity of the
significance calls" (Phase 5 review, second round). **"Borderline"** marks
an interval end within 0.001 of zero, where the call can flip with the
bootstrap draws alone.

| Claim | Number | Evidence strength | Caveat | Optimistic? |
|---|---|---|---|---|
| **(i) Tuned-size model with PRODUCTION features beats seasonal-naive on MAE** (no feature selection) | 44.4 / 52.6 / 57.4 / 61.8 vs 64.0 / 64.1 / 64.2 / 64.4 | **sig at +1h** (−19.6 [−27.6, −12.8]) and **+2h** (−11.5 [−20.4, −4.1]); **n.s. at +3h** (−6.8 [−16.1, +1.5]) and **+4h** (−2.6 [−12.7, +6.3]). Hotspots: sig at +1h only. w1 full grid, Phase 3 thresholds, paired day bootstrap. | Size chosen on Dec 10-16 (validation, not test). Base-rate shift. | No |
| **(ii) Final 23-feature model beats seasonal-naive on MAE** | 42.2 / 48.6 / 52.8 / 55.4 vs 64.0 / 64.1 / 64.2 / 64.4 | sig at every horizon (−21.8 [−30.2, −15.4] to −9.0 [−17.9, −1.6]). Hotspots: sig at +1h and +2h. w1 full grid. | **Feature groups were selected on the same test windows.** The 63-leaf choice came from validation, and its test MAE is slightly worse than 31 leaves'. | **Yes**, optimistic |
| **(i) Production-feature model beats seasonal-naive on F1** | m = 1: 0.532 / 0.486 / 0.457 / 0.438 vs 0.430 / 0.429 / 0.429 / 0.428. Both with their own margin tuned on Dec 10-16: 0.563 / 0.518 / 0.496 / 0.482 vs 0.447 / 0.446 / 0.446 / 0.445 | m = 1: sig at +1h (+0.102), +2h (+0.057); **+3h borderline** (+0.028, interval [−0.000, +0.058] on rerun); **n.s. at +4h** (+0.010). Both tuned: sig at +1h (+0.116), +2h (+0.071), +3h (+0.050); **n.s. at +4h** (+0.037 [−0.001, +0.078]). Hotspots: n.s. at every horizon, either way; **threshold-dependent:** under Phase 2 thresholds the production model is significantly *worse* than naive on hotspots at +3h / +4h at m = 1 (−0.088 / −0.081). | The production model's margin was tuned on Dec 10-16 here: the same kind of use as Phase 3, with no model refit. Base-rate shift. | No |
| **(ii) Final 23-feature model beats seasonal-naive on F1** | m = 1: 0.540 / 0.500 / 0.481 / 0.465; both tuned: 0.564 / 0.523 / 0.506 / 0.492 vs 0.447 / 0.446 / 0.446 / 0.445 | sig at +1h to +3h, both ways (m = 1: +0.110 / +0.071 / +0.052; both tuned: +0.117 / +0.076 / +0.060). **+4h borderline and threshold-dependent:** m = 1 +0.038 [+0.004, +0.072] sig under Phase 3 thresholds, but +0.031 [−0.001, +0.066] n.s. under Phase 2 (the 31-leaf Phase 2 model: +0.026 n.s. → +0.032 sig); both tuned +0.047 sig under both. Hotspots: sig at +1h only, and **threshold-dependent** (n.s. under Phase 2). | Feature selection on the test windows. | **Yes**, optimistic |
| **Feature groups help** (weekly lag, rolling stats, momentum; spatial marginal) | Final vs tuned MAE −2.6 to −6.1 (w1, every horizon); F1 n.s. overall, sig on hotspots | sig on MAE, w1 full grid. The ranking holds on w2. Overall F1 at +4h is **threshold-dependent** (n.s. under Phase 2, +0.022 [+0.001, +0.042] sig under Phase 3). | Keep/drop decisions were made on the **same test windows** that report these numbers. G3 is marginal (passes on w1 only; slightly worse +1h F1 on w2). | **Yes**: selection bias. |
| **Separate hotspot model helps** **[single]** (+1h, w1) | Hotspot F1 +0.092 at +1h (2% cutoff); +0.128 at 1%, +0.036 at 5% | sig at +1h only (w1). n.s. at +2h to +4h (+0.043, +0.045, +0.022). **Threshold-dependent:** under Phase 2 thresholds also sig at +2h / +3h (+0.067 / +0.058). +1h holds under both. MAE n.s. at every cutoff. | No w2 hotspot model, so transfer untested. Test hotspots barely exceed after Dec 24. Validation chose the final-feature hotspot model only at +1h. | Mild (one horizon of four is significant; the cutoff sweep is post hoc). |
| **A flagging margin helps** | F1 +0.023 to +0.027 (w1, every horizon); +0.022 / +0.036 (w2, +1h / +4h). Recommended ranges: +1h 0.95-0.96, +2h 0.945-0.96, +3h 0.94-0.955, +4h 0.94-0.945 | sig on w1 (all horizons) and w2 (+1h, +4h). **The two windows are not independent replications:** both come from the same 62 days, and Nov 18-24 was used twice (tree counts, then margins). | **Cost (P0 → P1, all cells):** w1 +1h precision 0.637 → 0.518, flagged 3.9% → 6.3% (actual 5.3%); w1 +4h 0.563 → 0.418, 3.7% → 7.6%; w2 +1h 0.629 → 0.486, 7.0% → 12.9% (actual 8.9%); w2 +4h 0.582 → 0.419, 6.4% → 14.8% (actual 9.0%). One noisy validation week per window. | No, but the operating point is a policy trade-off |
| **Hotspots at m = 1 is a conservative default, not a finding** **[single]** (firm only in w2 +1h) | A global margin changes Milan hotspot F1 by −0.022 (w2, +1h); −0.033 to −0.043 (w1, +2h to +4h) | **Firm only in Milan w2 at +1h** (−0.022, sig under both threshold definitions). **w1: borderline** (+2h interval end at 0.000; +3h / +4h sig under Phase 3 thresholds only, n.s. under Phase 2). **On Trentino a margin *helps* hotspots** (+0.034 to +0.095, sig). | Kept as the default because a margin never significantly helped Milan hotspots and the one firm result is a cost. H3 = H0 on hotspots by construction, so that equality is not evidence. | No, but weak |
| **The solver relieves actual congestion, under the one-for-one assumption** | Under the one-for-one assumption: net reduction of actual excess V0 14.6%, B 20.2% (m = 0.96); B − V0 +70,211 [+40,521, +103,949]; 1-hop ceiling 47.2%; **2-hop ceiling 72.1%** | Outcome replay, 24 non-holiday hours (+6 holiday); hour-level bootstrap (day level as sensitivity). B − V0 holds under both threshold definitions (+64,354 under Phase 2). **C − B net reduction is threshold-dependent** (n.s. under Phase 3; +7,062 sig under Phase 2); C's extra donor harm holds under both. | **One-for-one activity-unit assumption with no radio effects.** **Activity is spread over squares from base-station coverage areas by area (dataset limitation a), so moves between squares are not physical capacity moves; this applies equally to the 1-hop and 2-hop ceilings.** Units are record counts scaled by an unknown k (limitation b). 30-hour sample, one hour at a time. About 18-30% of moves go to cells that did not exceed. Donors already over threshold are used (about 1,500-1,900). | Likely optimistic in absolute terms; the comparison between variants is fairer |
| **The diagnosis rule separates reliable from unreliable flags** | Precision ROUTINE vs ANOMALOUS: 0.718 vs 0.529 on the 24-hour V0 set; 0.671 vs 0.485 on all 311 non-holiday hours | sig with hours as the bootstrap unit (+0.189 [+0.156, +0.219]; +0.186 [+0.167, +0.205]) and with days, at fractions 0.2 / 0.3 / 0.4. Holds under both threshold definitions (+0.223 / +0.195 under Phase 2). One window. | First outcome-based check. Isolated flags that are real are larger (128 vs 87). The threshold is a policy choice. Part of the neighbour co-flagging comes from shared base-station coverage areas (dataset limitation a). | No |
| **The holiday exemption lets lower-precision flags through** | Changing flags' precision 0.10-0.19 lower (Dec 25-26); a larger share of their moves goes to non-exceeders (28% vs 22%); **net solver effect slightly positive** (+3.2k, under the one-for-one assumption) | Consistent across two forecast sources; no significance test | **3 holiday days, 6 solver hours.** Indicative only. The claim is lower precision and more wasted moves, **not** a worse net outcome. | n/a |
| **Congestion percentile** | 85 / 90 / 95 change the target, not the quality | Descriptive | F1 across definitions measures different targets. A policy choice. | n/a |
| **P5: The scale-free reformulation costs nothing on Milan** | SF vs final_63l MAE: −1.79 / −5.10 (w1 +1h / +4h), +0.04 n.s. / −2.44 (w2); overall F1 −0.003 to −0.008 n.s.; hotspots better on MAE and F1 | sig on MAE in 3 of 4 cases, F1 n.s. in all 4 (w1 and w2, +1h and +4h) | SF uses the frozen 2,153 / 2,181 trees; w2 final_63l used 999 / 657. The final feature set carries Phase 2's selection bias into both. | Mild (same feature-selection bias as (ii)) |
| **P5: "No measurable transfer cost" holds at m = 1 only.** At m = 1 a Milan-trained model is at most about 2% worse on MAE than an in-domain model in a normal period; **with margins it is significantly worse on F1** (Milan's w2 margins: −0.012 at +1h, −0.023 at +4h; each model at its own Trentino-tuned margin: −0.014 / −0.027; at equal flagged share: −0.016 / −0.032), and **at +4h on typical cells MAE is significantly worse (+0.19)** | Trentino w2, transfer − in-domain [90%]: +1h (train Nov 2-24) MAE +0.09 [−0.04, +0.19] (upper end 1.7% of 11.20), F1 −0.005 [−0.015, +0.007]; +4h (train Nov 2-17) MAE +0.14 [−0.04, +0.32] (2.4% of 12.99), F1 +0.011 [−0.016, +0.046] | n.s. overall on MAE and F1 at +1h (both training spans) and +4h; these are the bounds. **+4h typical cells: transfer significantly worse on MAE** (+0.19 [+0.02, +0.35]). Hotspots: transfer significantly *better* (+1h F1 +0.033 / +0.026; +4h MAE −2.1, F1 +0.050). **At the margin operating point** (margins chosen before Nov 25), transfer is significantly worse on F1: −0.012 [−0.021, −0.002] at +1h, −0.023 [−0.036, −0.006] at +4h. | Bounds, not equality. **Both references are weak** (16 or 23 days, weekly lag missing before Nov 8, size untuned), so the bound is against an in-domain model of the same limited kind. Same operator and period, two regions of one dataset; Milan → Trentino only; one 13-day window. Local history supplies s_c and thresholds (not zero-shot). | No, but scope is narrow |
| **P5: The transferred model beats seasonal-naive on Trentino** (F1 part **[single]**: +1h only) | w2 +1h MAE −6.53, F1 +0.094 (Nov 2-17); −6.72, +0.096 (Nov 2-24). w2 +4h MAE −4.98, F1 −0.009 | **sig at +1h** (MAE and F1); +4h: MAE sig, **F1 n.s.** | One window, one direction. | No |
| **P5: Milan's margins raise F1 on Trentino** | **With margins chosen before Nov 25** (Milan w2: 0.950 / 0.935): transfer +0.031 [+0.023, +0.042] / +0.059 [+0.049, +0.073]; in-domain +0.051 / +0.093. With Milan's w1 margins (0.955 / 0.945, which use Dec 10-16, after this test window): +0.031 / +0.057 and +0.049 / +0.086 | sig with either margin set, both models, both horizons; also on hotspots. Holds under both Trentino threshold definitions. | **Cost:** precision −0.10 to −0.14; flagged share +4 to +6 points (transfer ends 15.4% / 15.5% vs actual 14.2% / 14.5%). Hotspots gain on Trentino, unlike Milan, so the hotspot exemption does not transfer as a rule. | No |
| **P5: Transfer under a seasonal level shift costs something** (one window, w1) | Trentino w1 transfer − in-domain MAE +2.66 / +4.71, F1 −0.008 / −0.017 (+1h / +4h); at +4h transfer is worse than naive | sig | **w1 is not a congestion result:** with thresholds from Nov 1 - Dec 9, 42% of cell-hours exceed (up to 67% in the last days) and 82% persist day to day. w1 F1 values (0.84-0.89) are not comparable with w2. | n/a |

**Summary of strength.**
- **Firm**, holding without the selected feature set:
  - the production-feature model beats seasonal-naive on MAE at +1h and +2h,
    and on F1 at +1h to +3h, including when both get a tuned margin;
  - a typical-cell margin raises F1, in both windows (not independent);
  - the diagnosis rule's precision split;
  - under the one-for-one assumption (and on the area-weighted grid,
    dataset limitation a), B removes more actual excess than V0, with no
    significant increase in donor harm;
  - (P5, narrow scope) the scale-free model costs nothing on Milan. On
    Trentino's normal window it beats naive at +1h, is at most about 2%
    worse on MAE than a weak in-domain model at m = 1, and gains F1 from
    Milan's margins, including margins chosen before the test window.
    **"No measurable transfer cost" holds at m = 1 only:** with margins
    (Milan's, its own, or at equal flagged share) transfer is significantly
    worse on F1 by about 0.012-0.032, and at +4h on typical cells its MAE is
    significantly worse (+0.19).
- **Not firm:**
  - anything at +4h for the production-feature model vs naive;
  - every hotspot comparison against naive;
  - the final model's margin over the production-feature model (optimistic,
    selection bias);
  - the absolute size of solver relief (one-for-one assumption, area-weighted
    grid);
  - the final model's F1 lead over naive at +4h (borderline,
    threshold-dependent);
  - most window 1 hotspot calls (threshold-dependent): the hotspot model at
    +2h / +3h, and the cost of a hotspot margin at +2h to +4h. Keeping
    hotspots at m = 1 is a **conservative default**: a margin hurts them
    firmly only in Milan w2 at +1h (−0.022 under both definitions), the w1
    evidence is borderline, and on Trentino a margin helps them;
  - every claim marked [single], given that no multiplicity correction is
    made;
  - C vs B net reduction (threshold-dependent);
  - (P5) F1 vs naive at +4h on Trentino; transfer at the margin operating
    point (significantly, slightly worse: −0.012 / −0.023 F1); anything from
    Trentino w1 as a congestion result; transfer in the other direction or
    across operators.

**What cannot be claimed from this data:**
- Real radio-capacity relief. The units are scaled record counts, not
  capacity, and square values are area shares of base-station coverage
  areas (see "Dataset limitations").
- Robust holiday behaviour: 3 days.
- A clean, unbiased estimate of the final feature set's test performance:
  no untouched hold-out period remains in 62 days.
- General transfer to another city or network. Phase 5 shows transfer
  between two regions of the same operator's dataset, over the same period,
  in one direction (Milan → Trentino), on one 13-day normal window, with
  local history for scale and thresholds. **The diagnosis rule and the
  solver were not evaluated on Trentino.**

---

## Promotion proposal (stages 1 and 2 implemented)

Stage 1 (compute only) and stage 2 (v2 database, API, Streamlit) are
described in "Promotion, stage 1" and "Promotion, stage 2" at the end of
this report. Outputs and the v2 database go to `data\processed\v2\`, not to
`data\raw` as D8 below first proposed.

How the chosen configuration would enter the pipeline. Each decision is
flagged **D1-D9** with a recommendation. Nothing below has been done.
Run-time and memory estimates come from the measurements in this report.

### Configuration

| # | Decision | Options | Recommendation and reason |
|---|---|---|---|
| D1 | Feature set | (a) production + G1 + G2 + momentum + **G3** (23 features); (b) the same without G3 | **(a) keep G3.** It is the only configuration whose size, margins and outcome replay were all evaluated. Dropping G3 is defensible (it passes the rule in window 1 only, and its gains are tiny), but it would need a refit and re-checks of size and margins. |
| D2 | Pooled model | 63 leaves, learning rate 0.1, `min_child_samples` 50, trees 2,153 / 2,154 / 2,193 / 2,181 (+1h to +4h), training targets before Dec 10 | As chosen on validation (Phase 2, pre-Phase-3 checks). |
| D3 | Hotspot model | +1h only: top 200 cells, final features, 31 leaves, 568 trees. Pooled model at +2h to +4h | Significant only at +1h (+0.092). At +2h to +4h it is n.s. and would add three more models. |
| D4 | Margin | Typical cells only; hotspots at m = 1. Ranges: +1h 0.95-0.96, +2h 0.945-0.96, +3h 0.94-0.955, +4h 0.94-0.945 | **As a watch status, with no effect on the solver.** Suggested starting values: 0.955 / 0.95 / 0.95 / 0.94. Thresholds as in Phase 3 (per-cell 90th percentile of data before the training cutoff). |
| D5 | Solver | V0 (unchanged) / watch only / B / C | **V0 for moves, plus watch flags** (operators see them; they never count as "resolved"). **B** is the candidate second step once watch flags have been reviewed in use: significantly more net relief than V0, and donor harm not significantly different. **Not C** (significantly more donor harm than B for the same net relief). |

### D6: refit policy

| Option | What it does | Effect on later evaluation |
|---|---|---|
| **R1 Keep the models as trained** (targets before Dec 10) | Use the evaluated models exactly | Every number in this report applies. Forecasts from Dec 10 on are out-of-sample. Ignores the 23 newest days. |
| R2 Refit once on all 62 days with frozen sizes | More data | No out-of-sample period remains: every forecast becomes in-sample, and nothing reported here describes the refit models. Sizes tuned for about 38 days may not suit 62. |
| R3 Rolling refits (for example weekly, on all data to date, frozen sizes) | The right policy for a live system | Each week's forecasts stay out-of-sample. For this historical data it is the same as option F2 below. |

**Recommendation: R1** for this dataset (it keeps the evaluation honest).
Adopt R3 as the policy for any live deployment.

### D7: the in-sample problem (what the dashboard shows as "forecasts")

Any forecast file covering all 62 days is **in-sample** for a model trained
on those days, so the dashboard would show fitted values as predictions.
Today's `cell_forecasts.parquet` (v1) already does this.

| Option | Coverage | Cost | Trade-offs |
|---|---|---|---|
| **F1 Out-of-sample forecasts only for Dec 10 - Jan 1**, from saved predictions (pooled `final_63l` validation and test; `hotspot_only_final` at +1h) | 23 days (Dec 10-16 labelled "validation": used to choose size and margins; Dec 17 - Jan 1 "test") | About 2 min to assemble, about 3 GB; **no fitting** | Honest and cheap. Nov 2 - Dec 9 has no forecasts. The test window is the holiday lull. |
| F2 Expanding-window forecasts for every day (weekly refits on data before each week, frozen sizes) | About Nov 16 - Jan 1 (earlier weeks have too little history) | About 7 cutoffs × 4 horizons × 10-16 min ≈ **5-7.5 h**, peak about 6.3 GB per fit | Nearly full coverage, all out-of-sample. Early weeks train on 2-3 weeks of data with sizes tuned for 5 weeks, so they are weaker. A new procedure, not yet evaluated. |
| F3 In-sample forecasts for all days, labelled "in-sample" | 62 days | About 40-65 min for 4 fits + about 20 min of prediction; 6.3 GB | Full coverage and simple, but **not predictions**. Must be visibly labelled; easy to misread as performance. |

**Recommendation: F1 now**, with each case tagged `validation` or `test`.
Add F2 later if a full-period demo is needed. Avoid F3 for anything shown
as a forecast.

A related point: the **v1 pipeline's thresholds** (`cdr_with_congestion_flags`)
are computed on all 62 days, so they are also look-ahead. The v2 path should
use the Phase 3 thresholds (data before the training cutoff).

### D8: code and schema changes for a watch status

**Design choice: a separate `WatchFlag` table, not a new classification
value.** Adding `classification = "WATCH"` would silently mislabel cases in
existing code:
- `/hours` counts every non-ANOMALOUS case as routine.
- The React UI (`statusOf`) shows anything not ANOMALOUS as routine.
- The Streamlit badge is red for anything not ROUTINE.
- `main.py` / `report_agent` write the anomalous report for anything not
  ROUTINE.
- Solver inputs filter on ROUTINE.

A separate table leaves all existing endpoints and clients unchanged.

| Component | Change (proposal) | Breaks existing? |
|---|---|---|
| Forecasts | New `src/forecasting/generate_forecasts_v2.py` → `data/raw/cell_forecasts_v2.parquet` (adds `model_version`, `forecast_kind`). `generate_forecasts.py` stays, so v1 remains reproducible. | No |
| Thresholds | New `thresholds_v2.parquet` (per-cell 90th percentile, Nov 1 - Dec 9) | No |
| Diagnosis | New `diagnosis_agent_v2.py`: same rules (30% neighbour fraction, holiday exemption unchanged), vectorised. Writes `diagnosis_results_v2_1h.parquet` (alerts: ROUTINE / ANOMALOUS) and `watch_flags_v2_1h.parquet` (typical cells with m × threshold < forecast ≤ threshold) | No |
| Solver | New `solver_v2.py` that reads the v2 inputs and **imports** `solve_hour` unchanged (V0). The slow per-case coverage loop is replaced. | No |
| Database | **New file `balancegrid_v2.db`**, selected by an environment variable in `database.py`. `models.py` gains a `WatchFlag` table (id, cell_id, target_datetime, horizon, forecast, threshold, margin, model_version) and nullable `model_version` / `forecast_kind` columns on `DiagnosedCase`. The v1 DB is untouched. SQLite `create_all` does **not** add columns to existing tables, which is another reason for a new file. | No (v1 DB unchanged) |
| Loader | New `populate_db_v2.py` (forecast column `forecast_1h`; loads watch flags) | No |
| API | New `GET /watch?target_datetime=…` or `?date=…`. `/hours` gains an additive `watch` count field. All existing endpoints unchanged. | No (additive fields are ignored by current clients) |
| Streamlit dashboard | Optional watch table and filter | No |
| React UI | Optional watch layer (outline only); never counted as resolved | No |
| Report agent | Unchanged (watch flags get no report) | No |

### D9: backups, run time and memory

**Backups:**
- Copy every edited file (`models.py`, `database.py`, `main.py`, front-end
  files) to `backup_initial/promotion/` before editing.
- The v1 DB and v1 parquet files are never overwritten; v2 uses new file
  names.
- Record a SHA-256 of each output.

**Estimated run time and memory (option F1 path):**

| Step | Time | Peak commit |
|---|---|---|
| Thresholds v2 | about 15 s | about 3 GB |
| Forecasts v2 (assembled from saved predictions) | about 2 min | about 3 GB |
| Diagnosis v2 (23 days, vectorised) | about 1 min | about 3 GB |
| Solver v2, V0 (552 hours; measured about 0.15 s per hour) | about 2 min | about 3 GB |
| `populate_db_v2.py` (v1 took 232 s and 4.5 GB for 845k cases; v2 has fewer cases plus watch rows) | about 2 min | about 3-4 GB |
| API / UI changes | seconds | – |

Compute is about 10 minutes in total on this laptop, plus the code changes.
Under F2, add 5-7.5 h of refits; under F3, add about 1-1.5 h.

**Recommended package:** D1(a) keep G3; D2 as trained; R1; D3 hotspot model
at +1h; D4 typical-cell margins as a watch status; D5 V0 + watch (B later);
F1 forecasts; D8 separate `WatchFlag` table, new v2 scripts and a new v2
database. Awaiting approval before any implementation. *Superseded by "Promotion, stage 1" and "Promotion, stage 2" (both implemented).*

---

## Phase 5: Trentino generalisation (data preparation and checks only)

No model has been fitted. Nothing was read from or written to `data\raw`.
- **Scripts:** `data\experiments\phase5\scripts\` (`inspect_trentino.py`,
  `grid_checks.py`, `threshold_checks.py`), plus `cdr_loader_tn.py`, a copy
  of `cdr_loader.py` whose diff shows only the input folder (`data\trentino`),
  the file pattern (`-tn-`) and the output path (`data\experiments\phase5`).
- **Outputs:** `data\experiments\phase5\`.

### Raw format (vs the Milan loader)

All 62 files match the Milan loader's assumptions:
- tab-separated, no header, 8 fields (`CellID, timestamp_ms, countrycode,
  smsin, smsout, callin, callout, internet`); empty fields mean no activity;
- epoch milliseconds in UTC, converted to Europe/Rome (local Nov 1 00:00 →
  Jan 1 23:50);
- 144 ten-minute timestamps in every file.

There are 171,361,027 raw rows, with 159-230 country codes per file.
**Observation:** Dec 31 and Jan 1 have about twice the rows of other days
(4.7-4.8M vs 2.1-2.5M) and the most country codes (226-230). The data does
not show *who* generated them, so this is recorded as an observation only.
The scan took 12.9 min, peak commit 3.4 GB.

### Grid geometry (from `trentino-grid.geojson`, not from an ID formula)

| Item | Result |
|---|---|
| Squares in the file | **6,575** (matches the paper); `cellId` 38-11,454; EPSG:4326, about 1 km squares |
| Adjacency | Squares sharing at least one vertex (edge or corner; coordinates rounded to 1e-9°). 5,923 squares have 8 neighbours; boundary squares have 2-7. Neighbour centroids are 0.99-1.42 km apart. |
| **117 × 98 numbering (tested as a hypothesis)** | **Supported by the geometry.** Every geometric neighbour pair differs in ID by 1, 116, 117 or 118. **No** pair the formula predicts as adjacent is missing from the geometry. Row correlates with latitude (+0.9994) and column with longitude (+0.9998); rows 0-97, columns 0-116. The file fills 6,575 of the 11,466 grid positions (the province's shape). Adjacency is nevertheless taken from the geometry (`trentino_adjacency.csv`). |
| Reporting cells | 6,259, **all** in the file; none reporting outside it |
| Squares that never report | **316** (`nonreporting_squares.csv`) |

**Squares that never report: what the evidence supports.**
- They have **no record at all** in 62 days × 144 ten-minute slots, in any
  country code.
- The raw format writes **no row** when a cell has no activity in a slot
  (the Milan loader's gap-filling relies on this).
- They sit inside the province: centroids from 10.47-11.86°E and 45.72-46.49°N,
  within the reporting cells' extent.
- Their reporting neighbours are very quiet (median 6.5 per hour vs 63.6 for
  all reporting cells), and 31 of them have no reporting neighbour.

That is consistent with zero activity, and equally with no data or coverage;
**the data cannot tell the two apart.** **Decision:** exclude them from
evaluation (an all-zero series has no meaningful threshold), and do not count
them as neighbours in neighbour features, which use reporting neighbours
only. Including them as zero-activity neighbours would assume the
zero-activity reading.

### Check cells (paper)

| Cell | Location | Mean per hour (rank of 6,259) | Share of cell 5200 | Weekday / weekend daily | Ratio | Neighbours' mean per hour |
|---|---|---|---|---|---|---|
| 5200, Trento centre | 11.116°E, 46.070°N | 11,851 (**1**) | 100% | 321,616 / 193,539 | **1.66** | 4,597 |
| 5085, engineering department | 11.141°E, 46.061°N; 2.2 km from 5200 | 1,799 (14) | 15.2% | 47,046 / 33,748 | **1.39** | 3,619 |
| 4703, forest | 10.740°E, 46.040°N; **29.2 km west of 5200** | 172 (1,051) | **1.45%** | 4,258 / 3,826 | **1.11** | 151 |

- **5200 passes:** it is the busiest cell.
- **5085 passes:** a clear weekday pattern.
- **4703 is inconclusive.** It is far quieter than the centre (1.45% of
  5200), which is all the paper's description implies, and it has the
  flattest weekday/weekend ratio of the three. But it is not unusually quiet
  for Trentino (rank 1,051 of 6,259, in an area averaging 151 per hour), and
  its location cannot be confirmed as forest from this data.

**Scale:** Trentino's busiest cell averages 11,851 per hour against Milan
Duomo's 4,252. This fits the paper's undisclosed per-dataset scaling
constant: absolute levels are not comparable between the cities, so the
formulation must be scale-free.

### Thresholds before fitting (item 4)

Per-cell 90th percentile, zero-filled to 6,259 cells × 1,488 hours. Only
**0.14%** of reporting cell-hours have no record. Built from the scan's
hourly totals (the same aggregation as the loader; checked against the
loader output when it runs).
- **Window 1:** thresholds from Nov 1 - Dec 9.
- **Window 2:** thresholds from Nov 1 - Nov 24.

| Window | min | 1% | 5% | 10% | 25% | median | 75% | 90% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|---|
| w1 | 0.00 | 0.26 | 4.07 | 9.93 | 28.8 | 75.8 | 159.8 | 333.7 | 1,462 | 26,136 |
| w2 | 0.00 | 0.24 | 3.84 | 8.95 | 26.5 | 72.1 | 154.9 | 320.3 | 1,441 | 26,875 |

| Low-threshold rule | Window 1: cells (share) / share of actual exceedances | Window 2: cells (share) / share of actual exceedances |
|---|---|---|
| threshold = 0 | 0 / 0 | 0 / 0 |
| ≤ 0.5 | 94 (1.5%) / 1.6% | 99 (1.6%) / 2.0% |
| **≤ 1** | **148 (2.4%) / 2.5%** | **155 (2.5%) / 2.9%** |
| ≤ 5 | 364 (5.8%) / 6.5% | 392 (6.3%) / 7.0% |
| ≤ 10 | 631 (10.1%) / 11.8% | 693 (11.1%) / 13.5% |

Seasonal-naive flags at +1h (no model), with and without low-threshold cells:

| Window | Cells kept | Precision | Recall | F1 | Flagged share | Actual share |
|---|---|---|---|---|---|---|
| w1 | all 6,259 | 0.895 | 0.818 | 0.855 | 38.6% | **42.2%** |
| w1 | threshold > 1 (6,111) | 0.895 | 0.818 | 0.855 | 38.5% | 42.2% |
| w1 | threshold > 5 (5,895) | 0.894 | 0.818 | 0.854 | 38.4% | 42.0% |
| w2 | all 6,259 | 0.532 | 0.460 | 0.493 | 12.3% | **14.2%** |
| w2 | threshold > 1 (6,104) | 0.529 | 0.457 | 0.491 | 12.3% | 14.2% |
| w2 | threshold > 5 (5,867) | 0.526 | 0.455 | 0.488 | 12.2% | 14.1% |

**Reading:**
- **No cell has a zero threshold.** Near-zero cells exceed roughly in
  proportion to their number, so they do not distort the metrics: excluding
  them changes F1 by at most 0.005.
- **Proposed minimum-activity rule:** exclude cells with a training-period
  threshold ≤ 1 activity unit (148 cells in window 1, 155 in window 2). Their
  thresholds (mean training activity about 0.25 per hour) are at noise level,
  and a single small burst counts as an "exceedance". I would report results
  with and without them, since the effect is negligible either way.
- **Main finding: window 1 is a regime shift in Trentino.** With thresholds
  from Nov 1 - Dec 9, **42.2% of Dec 17 - Jan 1 cell-hours exceed their
  90th percentile** (nominal 10%). Activity in the winter-holiday weeks is far
  above November's. Flagging there becomes nearly trivial: seasonal-naive
  reaches F1 0.855 because exceedances last for whole days. **Window 1
  mixes region shift and a strong season shift**, so it says little about
  transfer. Window 2 (14.2%) is much closer to the nominal 10% and should be
  the primary transfer test.
- **Time:** 2.4 s (grid checks) and 17.6 s (thresholds); peak commit ≤ 2.0 GB.

### Loader run and parity

The loader copy ran after memory was freed (headroom 9.1 GB). It wrote
`data\experiments\phase5\cdr_activity_aggregated_tn.parquet`:
- 9,313,392 rows = 6,259 cells × 1,488 hours, Nov 1 00:00 → Jan 1 23:00;
- **12,701 missing cell-hours filled with 0 (0.136%)**; no NaN; hourly
  spacing.
- **Time:** 182 s; peak commit 2.45 GB (lower than Milan's, since there are
  fewer cells).
- `data\raw` is unchanged (still 69 files).

**Parity with the scan-based panel** used for the thresholds above:
- same cells and shape;
- maximum absolute difference per cell-hour 3.6e-12 (floating-point
  rounding); 0 cell-hours differ by more than 1e-6;
- window 1 and window 2 thresholds identical to within 1.8e-12.

The threshold tables above therefore hold for the loader output.

**Reporting-neighbour counts per reporting cell** (geometric adjacency,
reporting neighbours only):

| Neighbours | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Cells | 3 | 9 | 59 | 164 | 249 | 254 | 407 | 5,114 |

No reporting cell has zero reporting neighbours; 568 border at least one
non-reporting square. The rule had any cell had none: its neighbour features
would be missing (NaN, handled natively by LightGBM), and in the diagnosis
rule its neighbour fraction would count as 0 (ANOMALOUS). It applies to no
cell here.

### Transfer experiment: Milan → Trentino

Script: `src/evaluation/phase5_transfer.py` (steps `fit`, `eval`); outputs
in `data\experiments\phase5\transfer\` (`metrics.csv`, `bootstrap.csv`,
`trentino_tuned_margins.csv`, `fit_log.json`, predictions, models).

**What this is: model transfer with local history, NOT zero-shot.**
Trentino's own history (training period only) supplies each cell's scale
s_c and its congestion thresholds. Only the fitted model is transferred.

**Formulation (scale-free):**
- **s_c** = the cell's mean activity over its city's training target hours,
  floored at 1.
- **Target:** log1p(A[t+h]) − log1p(s_c).
- **Level features:** log1p(x) − log1p(s_c), for lags, seasonal and weekly
  naive values, rolling means, and the neighbour mean, maximum and 2-step
  ring mean.
- **Rolling standard deviations:** log1p(std) − log1p(s_c).
- **Momentum:** the difference divided by s_c.
- **Calendar features:** unchanged.
- **Excluded:** CellID and `cell_historical_mean`, leaving 21 features.
- **Trentino neighbours:** geometric adjacency, reporting cells only. The
  2-step ring comes from adjacency of adjacency; it was checked against a
  direct computation on 200 random (cell, hour) pairs.

**Sizes and their caveats:**
- **Frozen size:** 63 leaves; 2,153 trees at +1h, 2,181 at +4h. **This size
  was chosen on Milan for the unscaled target and was never tuned for the
  scale-free target.** Milan's Dec 10-16 was not used again.
- **Window 2's Milan model** trains only on Milan targets **Nov 2-17**
  (window 2's training period). **Its frozen size comes from the Dec 10-16
  choice, which lies after this test window.**
- **The Trentino in-domain model** uses the same training span in each
  window, for parity.

**Windows:**
- **w1:** train before Dec 10; test Dec 17 - Jan 1. Trentino's
  winter-holiday peak, so it mixes **region shift and season shift** (42% of
  cell-hours exceed their threshold).
- **w2 (primary):** train before Nov 18; test Nov 25 - Dec 7; no holidays;
  thresholds from data before Nov 25.
- **Hotspots:** top 2% by s_c (125 Trentino cells).
- **"excl. thr≤1":** drops cells whose threshold is ≤ 1 (148 / 155 Trentino
  cells; Milan has none).
- **Bootstrap:** paired day bootstrap, 90% intervals. The w1 holiday slice
  is indicative (3 days).

**Fits** (headroom checked before each, 6.0-7.5 GB):

| Model | Rows | Time | Peak commit |
|---|---|---|---|
| Milan SF w1 +1h / +4h | 9.11M / 9.08M | 395 s / 381 s | 4.4 / 4.5 GB |
| Milan SF w2 +1h / +4h | 3.83M / 3.80M | 195 s / 185 s | 3.5 / 3.6 GB |
| Milan models predicted on Trentino (val and test), 4 models | – | 89-111 s each | 4.5 GB |
| Trentino SF w1 +1h (3,000 trees with validation curve) / +4h | 5.70M / 5.68M | 298 s / 266 s | 4.3 / 4.2 GB |
| Trentino SF w2 +1h / +4h | 2.40M / 2.38M | 112 s / 116 s | 3.7 GB |
| Evaluation (no fitting) | – | 955 s | 4.1 GB |

Three launch attempts stopped at the 6 GB headroom check before fitting (logs
kept). Nothing was fitted on those attempts. The run resumed after the
working set was reduced (float32 feature arrays, slimmer training frames) and
memory was freed.

#### B. Reformulation control (Milan, scale-free vs the existing final model)

| Window, horizon | MAE: SF vs final_63l (diff [90%]) | F1: SF vs final_63l (diff) | Hotspot MAE / F1 diff |
|---|---|---|---|
| w1 +1h | 40.42 vs 42.21 (**−1.79** [−2.67, −0.90]) | 0.533 vs 0.540 (−0.006 n.s.) | **−56.3** / **+0.081** |
| w1 +4h | 50.33 vs 55.43 (**−5.10**) | 0.458 vs 0.465 (−0.007 n.s.) | **−129.6** / **+0.056** |
| w2 +1h | 44.31 vs 44.28 (+0.04 n.s.) | 0.509 vs 0.516 (−0.008 n.s.) | **−44.6** / **+0.032** |
| w2 +4h | 51.30 vs 53.74 (**−2.44**) | 0.440 vs 0.444 (−0.003 n.s.) | **−71.0** / **+0.036** |

**The scale-free reformulation costs nothing on Milan.** MAE is equal or
significantly better, overall F1 differs by n.s. −0.003 to −0.008, and
hotspots are significantly better on both MAE and F1. (The w2 final_63l
model used 999 / 657 trees chosen on Nov 18-24; the SF model uses the frozen
2,153 / 2,181.) **Reformulation cost and transfer cost are therefore
separate, and the first is about zero.**

#### Transfer results on Trentino (m = 1)

| Window, horizon | Seasonal-naive MAE / F1 | Milan-trained SF (transfer) MAE / F1 | Trentino-trained SF (in-domain) MAE / F1 | Actual share |
|---|---|---|---|---|
| **w2 +1h (primary)** | 18.01 / 0.493 | **11.48 / 0.588** | 11.39 / 0.580 | 14.2% |
| w2 +4h | 18.11 / 0.497 | 13.13 / 0.488 | 12.99 / 0.477 | 14.5% |
| w1 +1h (holiday shift) | 25.48 / 0.855 | 28.46 / 0.879 | 25.79 / 0.887 | 42.2% |
| w1 +4h (holiday shift) | 25.87 / 0.856 | 36.27 / 0.838 | 31.56 / 0.855 | 42.8% |

**Paired day bootstrap** (difference = first named − second named):

| Window, horizon | Slice | Transfer − in-domain: MAE / F1 | Transfer − naive: MAE / F1 | In-domain − naive: MAE / F1 |
|---|---|---|---|---|
| w2 +1h | all | +0.09 n.s. / +0.008 n.s. | **−6.53** / **+0.094** | **−6.62** / **+0.087** |
| w2 +1h | excl. thr≤1 | +0.09 n.s. / **+0.020** | **−6.70** / **+0.108** | **−6.79** / **+0.088** |
| w2 +1h | hotspot | −0.46 n.s. / **+0.033** | **−112.5** / **+0.110** | **−112.1** / **+0.076** |
| w2 +4h | all | +0.14 n.s. / +0.011 n.s. | **−4.98** / −0.009 n.s. | **−5.13** / −0.020 n.s. |
| w2 +4h | hotspot | **−2.12** / **+0.050** | **−102.2** / **+0.052** | **−100.0** / +0.003 n.s. |
| w1 +1h | all | **+2.66 / −0.008** (transfer worse) | +2.98 n.s. / **+0.025** | +0.32 n.s. / **+0.032** |
| w1 +1h | hotspot | **−9.86** / +0.004 n.s. | **−79.9** / **+0.096** | **−70.0** / **+0.092** |
| w1 +4h | all | **+4.71 / −0.017** (transfer worse) | **+10.40 / −0.018** (worse than naive) | **+5.69** / −0.001 n.s. |
| w1 +4h | hotspot | **−9.35** / −0.002 n.s. | **−49.7** / +0.031 n.s. | **−40.4** / **+0.032** |

**Reading:**
- **In a normal period (w2), the Milan-trained model transfers with no
  measurable cost, at m = 1 only** (superseded by "Phase 5 review, third round": at fair margins transfer is significantly worse on F1). It matches the Trentino-trained model on MAE (n.s.) and
  F1 (n.s. overall; significantly better excluding near-zero cells and on
  hotspots). It beats seasonal-naive clearly at +1h (MAE −6.5, F1 +0.094).
  At +4h it still wins on MAE, but F1 vs naive is n.s.
- **In Trentino's holiday peak (w1), transfer costs something.** The
  Milan-trained model is significantly worse than the Trentino-trained one
  overall (MAE +2.7 / +4.7; F1 −0.008 / −0.017), and at +4h it is
  significantly worse than seasonal-naive on both MAE and F1. On hotspots it
  remains *better* than the in-domain model on MAE in both windows.
- **Even the in-domain model does not beat seasonal-naive on MAE in w1.**
  The season shift (42% of cell-hours above November thresholds) hurts every
  learned model there; it is not specific to transfer.
- **Near-zero-threshold cells** (thr ≤ 1) change little. Excluding them
  slightly favours the transferred model in w2.

#### D. Sizing sensitivity (w1 +1h, first use of Trentino Dec 10-16)

The Trentino-trained model's validation curve on Trentino's own Dec 10-16
bottoms out at **562 trees**; the 0.5% rule picks **426**, against the frozen
**2,153**.

| | Frozen 2,153 trees | Sized on Trentino (426 trees) |
|---|---|---|
| Test MAE / F1 | 25.79 / 0.887 | 25.83 / 0.889 |
| Transfer − this model, MAE / F1 | +2.66 / −0.008 | +2.63 / −0.009 |

**The Milan-vs-Trentino gap does not change.** The frozen size is about 5×
what Trentino's own validation would choose, but at learning rate 0.1 the
extra trees neither help nor hurt test performance here. (This is labelled
separately from the frozen comparison above.)

#### E. Margins on Trentino (precision / recall cost)

| Window, horizon | Forecaster | F1 at m = 1 → Milan margin | Precision / recall at m = 1 → Milan margin | Flagged vs actual (Milan margin) |
|---|---|---|---|---|
| w2 +1h | Transfer | 0.588 → **0.619** (m 0.955) | 0.695 / 0.509 → 0.595 / 0.644 | 15.4% vs 14.2% |
| w2 +1h | In-domain | 0.580 → **0.630** | 0.794 / 0.457 → 0.674 / 0.591 | 12.5% vs 14.2% |
| w2 +1h | Seasonal-naive | 0.493 → 0.519 | – | – |
| w2 +4h | Transfer | 0.488 → **0.546** (m 0.945) | 0.628 / 0.400 → 0.529 / 0.563 | 15.5% vs 14.5% |
| w2 +4h | In-domain | 0.477 → **0.563** | 0.779 / 0.344 → 0.637 / 0.505 | 11.5% vs 14.5% |
| w1 +1h | Transfer / in-domain | 0.879 / 0.887 → 0.886 / 0.890 | – | – |

- **Milan's margins carry over to Trentino:** every model's F1 rises at the
  Milan margins (0.950-0.960 at +1h; 0.940-0.945 at +4h), and the gain is
  largest in w2.
- *Superseded by "Phase 5 review, second round" and "Phase 5 review, third round": with look-ahead-free or own-tuned margins, transfer is significantly worse.* **At the Milan margin, transfer vs in-domain F1 is n.s. in w2** (−0.011
  +1h, −0.018 +4h). In w1 it is significantly lower (−0.004, −0.010).

**Separate sensitivity: margins tuned on Trentino Dec 10-16** (w1 only; for
w2, tuning on Nov 18-24 would need thresholds that include those days):
- **Margins chosen:** 0.975 at +1h for both models; 0.950 (transfer) and
  0.965 (in-domain) at +4h. These sit higher than Milan's, consistent with
  Trentino's much higher base rate in that period.
- **w1 test F1:** transfer 0.885 / 0.854, in-domain 0.891 / 0.863 (+1h /
  +4h). That is barely different from Milan's margins.

#### F. Slices (w1; holiday indicative, 3 days; weekend 4 days)

+1h F1 (transfer / in-domain / naive):
- **Non-holiday:** 0.873 / 0.883 / 0.846.
- **Holiday:** 0.898 / 0.900 / 0.881.
- **Weekend:** 0.883 / 0.898 / 0.848.

The in-domain model leads slightly in every slice. Results with and without
the thr≤1 cells, and by hotspot vs typical cells, are in the tables above and
in `metrics.csv`.

#### What Phase 5 supports, and what it does not

- **Supported:**
  - With local history for scale and thresholds, a Milan-trained scale-free
    model **transfers to Trentino at no measurable cost in a normal period**
    (w2, +1h and +4h, **at m = 1 only**; see the third review round for the
    fair margin comparison, where transfer is significantly worse).
    *Refined in the second review round:* at
    most about 2% worse on MAE and 0.015 worse on F1 (90% bounds); a small
    significant cost on typical cells at +4h (MAE) and at the margin
    operating point (F1 −0.012 / −0.023).
  - It beats seasonal-naive at +1h.
  - The scale-free reformulation itself costs nothing on Milan.
- **Not supported:**
  - Transfer under Trentino's holiday-season shift, where it is
    significantly worse than an in-domain model and, at +4h, worse than
    seasonal-naive.
  - Any zero-shot claim (Trentino history is used for s_c and thresholds).
- **Caveats:**
  - One normal window (13 days), and windows from the same 62 days.
  - Model size never tuned for the scale-free target (the sensitivity
    suggests it does not matter at +1h).
  - The 4703 "forest" identification remains inconclusive.
  - The Dec 31 / Jan 1 row jump is recorded as an observation only.

### Phase 5 review follow-ups

Script: `src/evaluation/phase5_followup.py` (steps `rates`, `train`,
`fit2b`, `cmp2b`, `margins`). Outputs are in
`data\experiments\phase5\followup\`. Nothing in `data\raw`, the pipeline or
the database was touched.

#### Scope of the transfer result

- **Both datasets come from the same operator (Telecom Italia Big Data
  Challenge) and the same period (Nov 1 2013 - Jan 1 2014).** This is
  transfer **between two regions of one dataset**, not between operators,
  countries or years.
- **One direction only:** Milan → Trentino. Trentino → Milan was not tested.
- **One normal test window:** w2, Nov 25 - Dec 7, 13 days. w1 is not a
  normal window (see the base rates below).
- **Not evaluated on Trentino:** the diagnosis rule and the solver. The
  Trentino result is about **forecasting and flagging only**.

#### Base rates (Trentino)

Actual and flagged exceedance share at m = 1 (cell-hours above the cell's
90th-percentile threshold). w1 thresholds come from Nov 1 - Dec 9; w2
thresholds from Nov 1 - Nov 24.

| Window, horizon | Slice | Actual | Seasonal-naive flagged | Transfer flagged | In-domain flagged |
|---|---|---|---|---|---|
| w2 +1h | all | 14.2% | 12.3% | 10.4% | 8.2% |
| w2 +1h | hotspot | 13.1% | 12.7% | 10.8% | 9.5% |
| w2 +1h | typical | 14.3% | 12.3% | 10.4% | 8.2% |
| w2 +4h | all | 14.6% | 12.5% | 9.3% | 6.4% |
| w2 +4h | hotspot | 13.1% | 12.7% | 9.6% | 7.9% |
| w2 +4h | typical | 14.6% | 12.5% | 9.3% | 6.4% |
| w1 +1h | all | 42.2% | 38.6% | 37.7% | 39.1% |
| w1 +1h | hotspot | 8.9% | 8.9% | 6.9% | 6.8% |
| w1 +1h | typical | 42.9% | 39.2% | 38.3% | 39.8% |
| w1 +4h | all | 42.8% | 39.2% | 35.7% | 38.0% |
| w1 +4h | hotspot | 9.0% | 9.0% | 6.1% | 6.0% |
| w1 +4h | typical | 43.5% | 39.8% | 36.3% | 38.7% |

Weekly actual exceedance share (each window's own thresholds; weeks start
on Monday and are cut to the test window):

| Window | Week | Days | All | Hotspot | Typical |
|---|---|---|---|---|---|
| w2 | Nov 25 - Dec 1 | 7 | 11.7% | 11.5% | 11.7% |
| w2 | Dec 2 - Dec 7 | 6 | 17.5% | 14.5% | 17.6% |
| w1 | Dec 17 - Dec 22 | 6 | 21.2% | 10.6% | 21.4% |
| w1 | Dec 23 - Dec 29 | 7 | 49.9% | 7.5% | 50.8% |
| w1 | Dec 30 - Jan 1 | 3 | 67.2% | 8.7% | 68.4% |

**Day-to-day persistence:** of the cell-hours that exceed, the share whose
cell also exceeded at the same hour the day before is **81.9% in w1** and
**46.2% in w2**.

**Why seasonal-naive F1 is 0.855 in w1 and 0.493 in w2.** Seasonal-naive
flags a cell-hour when the same hour yesterday exceeded. Its F1 is driven by
how common exceedance is and how persistent it is from one day to the next:
- In w1, 42% of cell-hours exceed, rising week by week to 67%, and 82% of
  them repeat yesterday's state. Yesterday's value is right most of the time.
- In w2, 14% exceed and only 46% repeat. Yesterday is a much weaker guide.

**w1 flags mainly track the seasonal level shift, not congestion.** With
thresholds from Nov 1 - Dec 9, by Dec 23 - Jan 1 half to two thirds of
typical cell-hours sit above their "90th percentile". That is a seasonal
rise in level across most of the region (hotspots do not rise: 7.5-10.6%),
not 10%-tail congestion events. **w1 F1 values (0.84-0.89 for every
forecaster) are therefore not comparable with w2 F1 values (0.48-0.59), and
w1 is not a congestion result.** w1 remains useful only as a test of
behaviour under a level shift, where the transfer cost appears.

#### Training span of the w2 models

| Model | Training origins | Days | Rows | Origins without the 168h lag |
|---|---|---|---|---|
| Milan-trained, w2 +1h | Nov 2 00:00 - Nov 17 22:00 | 15.96 | 3,830,000 | Nov 2 00:00 - Nov 7 23:00 (37.6%) |
| Trentino-trained, w2 +1h | same | 15.96 | 2,397,197 | same (37.6%) |
| Milan-trained, w2 +4h | Nov 2 00:00 - Nov 17 19:00 | 15.83 | 3,800,000 | same (37.9%) |
| Trentino-trained, w2 +4h | same | 15.83 | 2,378,420 | same (37.9%) |

The data starts on Nov 1 and the first training origin is Nov 2 00:00 (24h
of lags). **The weekly lag and weekly-naive feature are missing (NaN) for
every origin before Nov 8**, the panel's first 7 days, which is 6 of the
16 training days. So the w2 models learn the weekly pattern from only about
10 days with the feature present, and with 16 days in total, they are weak
in-domain references.

#### Stronger in-domain reference: +1h pair trained on Nov 2-24

> **Labelled re-use:** Nov 18-24 was used before in window 2, for Milan's
> tree counts (pre-Phase-3 sizing) and Milan's margins (Phase 3). Here it is
> used again, **as training data**. The test week (Nov 25 - Dec 7), the
> thresholds (data before Nov 25) and the model size (frozen: 63 leaves,
> 2,153 trees) are unchanged. Each cell's scale s_c is recomputed from the
> longer training period, as the scale-free definition requires.

| Model | Training origins | Days | Rows | Origins without the 168h lag | Fit time | Peak commit |
|---|---|---|---|---|---|---|
| Milan-trained, +1h, Nov 2-24 | Nov 2 00:00 - Nov 24 22:00 | 22.96 | 5,510,000 | Nov 2-7 (26.1%) | 595 s | 3.6 GB |
| Trentino-trained, +1h, Nov 2-24 | same | 22.96 | 3,448,709 | Nov 2-7 (26.1%) | 266 s | 3.7 GB |

Headroom before the fits was 8.8 and 9.1 GB. The whole step (both panels,
two fits, predictions) took 1,228 s, with a peak commit of 4.5 GB. Milan's
prediction on Trentino took 94 s.

Trentino w2 test, +1h, m = 1 (actual share 14.2%):

| Forecaster | MAE | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|---|
| Seasonal-naive | 18.01 | 0.532 | 0.460 | 0.493 | 12.3% |
| Milan-trained, Nov 2-17 | 11.48 | 0.695 | 0.509 | 0.588 | 10.4% |
| Trentino-trained, Nov 2-17 | 11.39 | 0.794 | 0.457 | 0.580 | 8.2% |
| **Milan-trained, Nov 2-24** | **11.29** | 0.694 | 0.512 | **0.589** | 10.5% |
| **Trentino-trained, Nov 2-24** | **11.20** | 0.797 | 0.474 | **0.595** | 8.5% |

Paired day bootstrap (first named − second named; 90% intervals):

| Comparison | Slice | MAE diff | F1 diff |
|---|---|---|---|
| Milan-trained: Nov 2-24 − Nov 2-17 | all | **−0.19 [−0.24, −0.14]** | +0.001 [−0.003, +0.006] n.s. |
| Trentino-trained: Nov 2-24 − Nov 2-17 | all | **−0.19 [−0.33, −0.08]** | **+0.015 [+0.008, +0.020]** |
| Trentino-trained: Nov 2-24 − Nov 2-17 | hotspot | −0.63 [−1.46, +0.14] n.s. | **+0.018 [+0.009, +0.028]** |
| **Transfer − in-domain, Nov 2-24** | all | +0.09 [−0.04, +0.19] n.s. | −0.005 [−0.015, +0.007] n.s. |
| Transfer − in-domain, Nov 2-24 | excl. thr≤1 | +0.09 [−0.04, +0.20] n.s. | +0.009 [−0.002, +0.023] n.s. |
| Transfer − in-domain, Nov 2-24 | hotspot | **−2.61 [−3.50, −1.76]** | **+0.026 [+0.010, +0.045]** |
| Transfer − in-domain, Nov 2-17 (previous) | all | +0.09 [−0.10, +0.25] n.s. | +0.008 [−0.007, +0.026] n.s. |
| In-domain Nov 2-24 − naive | all | **−6.81** | **+0.101** |
| Transfer Nov 2-24 − naive | all | **−6.72** | **+0.096** |

- **The longer training period helps both models a little:** MAE about −0.19
  for each, significant. F1 improves significantly only for the
  Trentino-trained model (+0.015).
- *At m = 1 only; superseded at the margin operating point by "Phase 5 review, third round".* **The transfer conclusion does not change.** With the stronger in-domain
  reference, transfer vs in-domain is still n.s. overall on MAE and F1.
  Its F1 point difference turned from +0.008 to −0.005, and its earlier
  significant F1 advantage on the thr≤1-excluded slice is now n.s.
  On hotspots the Milan-trained model is still significantly better.
- **Still weak references:** 23 days of training, a quarter of them without
  the weekly lag, and a model size not tuned for either city. "No measurable
  transfer cost" means: no cost against an in-domain model of the same
  limited kind.

#### Margin gain in w2 (m = 1 vs Milan's margins), paired day bootstrap

Saved w2 predictions; F1 gain = F1 at the margin − F1 at m = 1, with the 90%
day-bootstrap interval. The middle of Milan's range is the headline; the
range ends were also run (`w2_margin_gain.csv`).

| Horizon | Model | Margin | F1 m = 1 → margin | Gain [90%] | Precision | Recall | Flagged (actual) |
|---|---|---|---|---|---|---|---|
| +1h | Transfer | 0.955 | 0.588 → 0.619 | **+0.031 [+0.024, +0.041]** | 0.695 → 0.595 | 0.509 → 0.644 | 10.4% → 15.4% (14.2%) |
| +1h | In-domain | 0.955 | 0.580 → 0.630 | **+0.049 [+0.037, +0.065]** | 0.794 → 0.674 | 0.457 → 0.591 | 8.2% → 12.5% (14.2%) |
| +4h | Transfer | 0.945 | 0.488 → 0.546 | **+0.057 [+0.048, +0.069]** | 0.628 → 0.529 | 0.400 → 0.563 | 9.3% → 15.5% (14.5%) |
| +4h | In-domain | 0.945 | 0.477 → 0.563 | **+0.086 [+0.067, +0.111]** | 0.779 → 0.637 | 0.344 → 0.505 | 6.4% → 11.5% (14.5%) |

- **Every gain is significant**, at every margin in Milan's ranges (+1h
  0.950 / 0.960: +0.031 / +0.031 transfer, +0.051 / +0.047 in-domain; +4h
  0.940: +0.059 transfer, +0.090 in-domain). On hotspots: +0.034 to +0.040
  (+1h) and +0.061 to +0.066 (+4h) for transfer, +0.061 to +0.063 and
  +0.092 to +0.096 for in-domain, all significant.
- **Cost:** precision falls by 0.10-0.14. Flagged share rises by 4-6 points.
  For the transferred model it ends slightly **above** the actual share
  (15.4% vs 14.2%; 15.5% vs 14.5%). For the in-domain model it stays below.
- **Hotspots:** Milan's policy keeps hotspots at m = 1. On Trentino w2 the
  margin *helps* hotspots. This is the opposite of Milan (where a margin
  cost hotspot F1). It is one window, but it means the hotspot exemption
  does not transfer as a rule.
- The margin was not tuned on Trentino. The gain is a transfer of Milan's
  operating point, measured on one 13-day window.

### Phase 5 review, second round (no fitting)

Script: `src/evaluation/phase5_review.py` (steps `bounds`, `marginspre`,
`thrsens`, `dec31`). Outputs are in `data\experiments\phase5\review\`.
Everything comes from saved predictions. Time: under 2 minutes per step;
peak commit 3.9 GB (`thrsens`).

#### Transfer vs in-domain as bounds (Trentino w2)

Transfer − in-domain, 90% paired day-bootstrap interval. MAE is also shown
as a share of the in-domain model's MAE (the upper end of the interval).

| Horizon, pair, margin | Slice | MAE diff [90%] | Upper end as % of in-domain MAE | F1 diff [90%] |
|---|---|---|---|---|
| +1h, Nov 2-24, m = 1 | all | +0.09 [−0.04, +0.19] | 1.7% of 11.20 | −0.005 [−0.015, +0.007] |
| +1h, Nov 2-17, m = 1 | all | +0.09 [−0.09, +0.25] | 2.2% of 11.39 | +0.008 [−0.008, +0.026] |
| +4h, Nov 2-17, m = 1 | all | +0.14 [−0.04, +0.32] | 2.4% of 12.99 | +0.011 [−0.016, +0.046] |
| +4h, Nov 2-17, m = 1 | **typical** | **+0.19 [+0.02, +0.35]** (transfer worse) | – | +0.011 [−0.017, +0.045] |
| +4h, Nov 2-17, m = 1 | hotspot | **−2.12 [−4.09, −0.17]** (transfer better) | – | **+0.050 [+0.012, +0.098]** |

No +4h pair was trained on Nov 2-24, so +4h rests on the Nov 2-17 pair only.

**Wording for the claim:** on Trentino's normal window, with local history
for scale and thresholds, transfer is **at most about 2% worse on MAE**
(+1h: up to +0.19, 1.7%; +4h: up to +0.32, 2.4%) and **at most about 0.015
(+1h) / 0.016 (+4h) worse on F1 at m = 1** (90% intervals). **At +4h the
transfer cost is significant on typical cells** (MAE +0.19 [+0.02, +0.35]),
offset by a significant gain on hotspots. At the margin operating point the
F1 bound is wider (see the next table). **Both references are weak** (16 or
23 days of training, weekly lag missing before Nov 8, size not tuned for
either city), so these bounds compare transfer with an in-domain model of
the same limited kind.

#### Margins chosen before Nov 25 (removes the look-ahead)

Milan's recommended ranges were set using Dec 10-16, which lies after
Trentino's w2 test window. Recomputed with Milan's **window-2** margins,
chosen on Nov 18-24 (0.950 at +1h, 0.935 at +4h), from saved predictions,
with the same bootstrap (`w2_margin_gain_pre_nov25.csv`):

| Horizon | Model | Gain, margin before Nov 25 [90%] | Gain, Milan w1 margin (earlier) | Precision → | Flagged (actual) |
|---|---|---|---|---|---|
| +1h | Transfer, m 0.950 | **+0.031 [+0.023, +0.042]** | +0.031 (m 0.955) | 0.695 → 0.584 | 10.4% → 16.1% (14.2%) |
| +1h | In-domain, m 0.950 | **+0.051 [+0.037, +0.068]** | +0.049 | 0.794 → 0.659 | 8.2% → 13.1% |
| +4h | Transfer, m 0.935 | **+0.059 [+0.049, +0.073]** | +0.057 (m 0.945) | 0.628 → 0.509 | 9.3% → 16.9% (14.5%) |
| +4h | In-domain, m 0.935 | **+0.093 [+0.072, +0.121]** | +0.086 | 0.779 → 0.609 | 6.4% → 12.8% |

Hotspots: the gains stay significant (+0.034 / +0.061 at +1h, +0.061 /
+0.095 at +4h, transfer / in-domain).

**Transfer vs in-domain, both at the margin:**

| Horizon | Margin before Nov 25 | Milan w1 margin (earlier) |
|---|---|---|
| +1h, all | **−0.012 [−0.021, −0.002]** (transfer worse) | −0.011 [−0.020, +0.001] n.s. |
| +4h, all | **−0.023 [−0.036, −0.006]** (transfer worse) | −0.018 [−0.033, +0.001] n.s. |
| +4h, hotspot | **+0.016 [+0.003, +0.032]** (transfer better) | +0.024 [+0.008, +0.043] |

**Does the conclusion change?**
- **The margin gain does not.** It is the same size or slightly larger with
  margins chosen before the test window, for both models.
- **The transfer comparison at the margin does.** With look-ahead-free
  margins, the transferred model is **significantly, if slightly, worse**
  than the in-domain model on F1 (−0.012 at +1h, −0.023 at +4h). With Milan's
  w1 margins the same comparison was borderline n.s. So "no measurable
  transfer cost" holds at m = 1 only. At the margin operating point, there
  is a small cost of about 0.01-0.02 F1, because the transferred model
  already flags more and the margin pushes it further past the actual share
  (16.1-16.9% flagged vs 14.2-14.5%).

#### Threshold sensitivity of the significance calls

Every F1, precision and replay call in "What can be claimed" was rerun under
both threshold definitions, with the same margins and the same bootstrap
(`threshold_sensitivity.csv`, `threshold_sensitivity_changed.csv`):
- **Window 1:** Phase 2 (Nov 1 - Dec 16) vs Phase 3 (Nov 1 - Dec 9).
- **Milan window 2:** Nov 1-17 (used) vs Nov 1-24.
- **Trentino w2:** before Nov 25 (used) vs before Nov 18.

Margins are always the ones chosen with the leak-free definition, because
the Phase 2 thresholds include the validation week. The rerun reproduces
the reported Phase 3 point estimates exactly (for example +0.038 at +4h; B −
V0 +70,211). MAE claims do not depend on thresholds.

**Calls that change between the two definitions:**

| Claim | Comparison | Phase 2 thresholds | Phase 3 thresholds |
|---|---|---|---|
| **(ii) final vs naive F1, +4h, all** (the case raised) | final_63l, m = 1 | +0.031 [−0.001, +0.066] n.s. | **+0.038 [+0.004, +0.072]** sig |
|  | Phase 2 final (31 leaves), m = 1 | +0.026 [−0.007, +0.060] n.s. | **+0.032 [+0.001, +0.066]** sig |
| (ii) final vs naive F1, +1h, hotspots | m = 1 | +0.044 [−0.016, +0.138] n.s. | **+0.063 [+0.009, +0.159]** sig |
|  | both tuned | +0.061 [−0.003, +0.163] n.s. | **+0.071 [+0.009, +0.169]** sig |
| (i) production model vs naive F1, hotspots, m = 1 | +3h / +4h | **−0.088 / −0.081, sig worse** | −0.071 / −0.061 n.s. |
| Feature groups (final 31 leaves vs tuned) F1, +4h, all | m = 1 | +0.019 [−0.003, +0.040] n.s. | **+0.022 [+0.001, +0.042]** sig |
| Hotspot model (H0 vs P0), hotspots | +2h / +3h | **+0.067 / +0.058 sig** | +0.043 / +0.045 n.s. |
| Keep hotspots at m = 1 (H1 vs H0), w1 | +3h / +4h | −0.032 / −0.029 n.s. | **−0.043 / −0.038 sig** |
| Outcome replay, C − B net reduction (m 0.96) | 24 h | **+7,062 [+2,284, +12,217] sig** | +3,486 [−1,249, +8,197] n.s. |
| Milan w2, w1 margin (0.96) on hotspots, +1h | (not in the table) | −0.010 sig (Nov 1-17) | −0.011 [−0.020, +0.000] n.s. (Nov 1-24) |

**Calls that hold under both definitions:**
- (i) MAE, (ii) MAE, and (i) / (ii) F1 on all cells at +1h to +3h.
- The margin gain P1 vs P0 (w1 every horizon; w2 +1h / +4h).
- The w2 hotspot cost of a margin (−0.022 under both).
- The diagnosis rule's precision split (+0.223 / +0.195 under Phase 2;
  +0.189 / +0.186 under Phase 3).
- B − V0 net reduction (+64,354 / +70,211), C − V0 donor harm, and C − B
  donor harm.
- Every Trentino call that was run: transfer vs in-domain, transfer vs naive
  and the margin gains, with w2 thresholds from Nov 1-24 (before Nov 25, the
  Phase 5 definition) vs Nov 1-17 (before Nov 18). In-domain vs naive was
  added in the third review round ("Trentino threshold sensitivity" there)
  and also holds.

**Calls on the edge of the interval, regardless of thresholds.** Two calls
in the table changed with the bootstrap draws alone. The rerun reproduces
their point estimates, but the interval end lands at about 0:
- **(i) production model F1 at +3h:** +0.028 [−0.000, +0.058], reported
  before as "lower bound +0.000, borderline".
- **Keep hotspots at m = 1, +2h:** −0.033 [−0.068, +0.000], reported before
  as significant.

Both are now marked borderline.

**Reading.** The headline +4h F1 claim, and most hotspot claims in window
1, depend on the threshold definition. Hotspot exceedances in the window 1
test are rare after Dec 24 (check 1), so small threshold changes move them.
The all-cell claims at +1h to +3h, the margin gain, the diagnosis rule and
B − V0 are robust to it.

#### Dec 31 (observation)

+1h, final pooled model, Phase 3 thresholds, per test day
(`dec31_by_day.csv`). **Dec 31 is not in the holiday list** (Nov 1, Dec 8,
Dec 25, Dec 26, Jan 1), so the diagnosis rule treats it as a normal day.

| Day | MAE model / naive | F1 model / naive | Recall | Flagged / actual share |
|---|---|---|---|---|
| **Dec 31 (Tue)** | 41.7 / 47.6 | **0.297** / 0.079 | 0.225 | 1.0% / 2.0% |
| Other non-holiday days (12) | 29.4-57.4 (median 35.6) | 0.342-0.617 (median 0.538) | 0.33-0.60 | – |
| Dec 25 / Dec 26 / Jan 1 (holidays) | 63.2 / 42.5 / 66.1 | 0.550 / 0.484 / 0.118 | – | – |

- **Dec 31 has the lowest F1 of any non-holiday test day** (0.297; next
  lowest Dec 27 at 0.342 and Dec 30 at 0.360). Recall is 0.23: the model
  misses three quarters of the day's few exceedances.
- **Its MAE is unremarkable** (41.7, below Dec 23-24 and Dec 30). The
  problem is flagging, not average error.
- **Seasonal-naive almost fails completely** on Dec 31 (F1 0.079), because
  Dec 30 is a poor guide to New Year's Eve.
- **The model has never seen a New Year's Eve:** training targets end on
  Dec 9, so no earlier Dec 31 is in its training data. Dec 31 is also not in
  the holiday list, so neither the model nor the diagnosis rule treats it
  specially. The diagnosis rule is unchanged.
- Only one such day exists in the data, so this is an observation, not a
  finding. If the holiday list is revisited, Dec 31 is a candidate. Adding
  it would make its flags ROUTINE by exemption, which, given this low
  precision and recall, is the opposite of what a low-reliability day needs.

### Phase 5 review, third round (no fitting)

Script: `src/evaluation/promotion_v2_followup.py` (steps `fairmargin`,
`tnthr`). Outputs are in `data\experiments\promotion_v2\`. Saved
predictions only: the Nov 2-17 models' Trentino validation predictions
(Nov 18-24) already existed, so no prediction pass was needed. Time: 19 s
and 12 s; peak commit 3.1 GB.

#### Fair margin comparison on Trentino w2

> **Re-used week:** Trentino's Nov 18-24 was used before, as the margin
> tuning week here, and Milan's Nov 18-24 was used for Milan's window-2
> tree counts and margins. Each model's **own** global margin is chosen on
> Trentino Nov 18-24 (F1-best, same rule as Phase 3), with thresholds from
> **Nov 1-17**. It is then tested on Nov 25 - Dec 7 with the same
> thresholds. As a check, the Phase 5 test thresholds (Nov 1-24) were also
> applied.

**Margins chosen on Trentino Nov 18-24:**

| Model | +1h | +4h |
|---|---|---|
| Transfer | 0.945 | 0.925 |
| In-domain | 0.930 | 0.910 |
| Seasonal-naive | 0.925 | 0.925 |

The actual share was 8.5% in the tuning week and 13.9-14.5% on test, a
base-rate rise.

**Test, all cells, thresholds Nov 1-17** (actual share 13.9% at +1h, 14.2%
at +4h):

| Horizon | Rule | Transfer: F1 (precision / recall, flagged) | In-domain: F1 (precision / recall, flagged) |
|---|---|---|---|
| +1h | m = 1 | 0.602 (0.706 / 0.524, 10.3%) | 0.595 (0.799 / 0.474, 8.3%) |
| +1h | own margin | 0.629 (0.585 / 0.681, 16.2%) | **0.643** (0.614 / 0.675, 15.3%) |
| +1h | equal share = actual (13.9%) | 0.630 | **0.645** |
| +4h | m = 1 | 0.502 (0.643 / 0.412, 9.1%) | 0.493 (0.788 / 0.359, 6.5%) |
| +4h | own margin | 0.561 (0.507 / 0.626, 17.5%) | **0.588** (0.558 / 0.620, 15.8%) |
| +4h | equal share = actual (14.2%) | 0.556 | **0.588** |

**Transfer − in-domain, paired day bootstrap** (test thresholds Nov 1-17;
the Nov 1-24 values agree to within 0.002):

| Comparison | +1h, all | +1h, hotspot | +4h, all | +4h, hotspot |
|---|---|---|---|---|
| m = 1 | +0.006 [−0.007, +0.023] n.s. | **+0.030** | +0.009 [−0.017, +0.041] n.s. | **+0.048** |
| **Each at its own tuned margin** | **−0.014 [−0.020, −0.007]** | **+0.015** | **−0.027 [−0.037, −0.016]** | **+0.017** |
| Equal flagged share = actual share (margins set on test) | **−0.016 [−0.023, −0.009]** | – | **−0.032 [−0.043, −0.020]** | – |
| Equal flagged share = transfer's own-margin share | **−0.011 [−0.017, −0.005]** | – | **−0.024 [−0.034, −0.013]** | – |
| Transfer vs naive, own margins | **+0.094** | **+0.108** | **+0.022** | **+0.080** |
| In-domain vs naive, own margins | **+0.108** | **+0.093** | **+0.050** | **+0.063** |

The equal-share rows set the margin on test data. They are a diagnostic
that compares the two models' ranking of cell-hours at the same flag
volume, not a deployable operating point.

**Reading:**
- **The m = 1 parity is an operating-point effect.** At m = 1 the
  transferred model flags more (10.3% vs 8.3%) and the in-domain model
  under-flags further, so their F1 values meet.
- **Once each model gets a fair margin, or both flag the same volume, the
  in-domain model is significantly better on all cells**: by 0.014-0.016
  at +1h and 0.027-0.032 at +4h. This agrees with the Milan-margin result
  (−0.012 / −0.023).
- **On hotspots the transferred model stays significantly better** at every
  operating point.
- **Both models still beat seasonal-naive with their own margins**,
  including at +4h (+0.022 transfer, +0.050 in-domain), where at m = 1 the
  comparison with naive was n.s. Seasonal-naive got its **own** margin too,
  tuned the same way (Trentino Nov 18-24, thresholds Nov 1-17, F1-best):
  0.925 at both horizons. So this is a like-for-like comparison.
- **The in-domain references trained on only 16 days** (Nov 2-17, with the
  weekly lag missing before Nov 8) and a model size never tuned for
  Trentino. A stronger in-domain model would probably widen the gap, so
  **the transfer cost measured here may be understated.**

**Conclusion for the claim:** "no measurable transfer cost" holds **at
m = 1 only**. At any fair margin, transfer costs about 0.01-0.03 F1 on all
cells in this window, probably understated. The MAE bound (at most about 2%
worse at m = 1) is unchanged.

#### Trentino threshold sensitivity (transfer vs in-domain vs naive, m = 1)

The second-round check had already compared Trentino w2 thresholds from
Nov 1-24 (the Phase 5 definition) with Nov 1-17. Here **in-domain vs naive**
is added (`tn_threshold_sensitivity.csv`):

| Horizon | Comparison | All: Nov 1-24 | All: Nov 1-17 | Hotspot: Nov 1-24 | Hotspot: Nov 1-17 |
|---|---|---|---|---|---|
| +1h | transfer − in-domain | +0.008 n.s. | +0.006 n.s. | **+0.033** | **+0.030** |
| +1h | transfer − naive | **+0.094** | **+0.094** | **+0.110** | **+0.126** |
| +1h | in-domain − naive | **+0.087** | **+0.087** | **+0.076** | **+0.096** |
| +4h | transfer − in-domain | +0.011 n.s. | +0.009 n.s. | **+0.050** | **+0.048** |
| +4h | transfer − naive | −0.009 n.s. | −0.009 n.s. | **+0.052** | **+0.076** |
| +4h | in-domain − naive | −0.020 n.s. | −0.018 n.s. | +0.003 n.s. | +0.028 n.s. |

**No Trentino call changes between the two definitions.** This is **weak
evidence of robustness**: the two definitions share 17 of their 24 days
(Nov 1-17), so the change is a mild perturbation of the thresholds. A
stronger test would need a threshold period with no overlap, which this
dataset does not provide before the test window.

---

## Promotion, stage 1: compute only (implemented)

Approved package: D1a, D2, D3 (hotspot model at +1h only), R1, F1, D4 (watch
flags on typical cells, hotspots at m = 1, margins 0.955 / 0.95 / 0.95 /
0.94), D5 (V0). **Stage 2 (v2 database with WatchFlag, API, dashboard) is
not started.** *Superseded by "Promotion, stage 2" (implemented).*

**What was changed: only new files.**
- **New scripts:**
  - `src/forecasting/thresholds_v2.py`
  - `src/forecasting/generate_forecasts_v2.py`
  - `src/diagnosis/diagnosis_agent_v2.py` (diagnosis and watch flags)
  - `src/optimization/solver_v2.py`
  - `src/evaluation/promotion_v2_parity.py`
- **Not touched:** existing pipeline files, the API, database code, the
  front end, the v1 database, v1 parquet files and `data\raw` (still 69
  files, newest dated Sep 7).

**Decision: outputs go to `data\processed\v2\`, not `data\raw`.** D8
proposed `data\raw\cell_forecasts_v2.parquet`, which conflicts with the
standing rule not to write to `data\raw`. Parity results and hashes are in
`data\experiments\promotion_v2\`.

**Other decisions:**
- **Forecasts are stored long** (one row per cell, origin and horizon), with
  `segment`, `model_version` and `forecast_kind`.
- **Diagnosis and watch flags are written for all four horizons.** v1 saved
  +1h only.
- **The solver runs at +1h**, as v1 did.
- **Watch flags are a separate file.** They never enter the diagnosis or the
  solver.
- **v2 reuses v1's rule constants by import** (holiday list, 30% fraction,
  grid) and the unchanged `solve_hour`, so the two cannot drift apart.

### Coverage and counts

**Days covered:** 23 target days, Dec 10 - Jan 1.
- **"validation (used for model size and margin selection)":** targets
  Dec 10 00:00 - Dec 16 23:00 (7 days). These days are **out-of-training**
  (the models were trained on targets before Dec 10), but they are **not an
  untouched test**: they chose the tree counts, the leaf count and every
  margin. The label is carried in `v2_metadata.json` (`forecast_kind_label`)
  and in every v2 API response.
- **The "test" label is not fully untouched either.** Dec 17 - Jan 1 was not
  used for size or margins, but Phase 2 selected the feature groups on it.
- **"test":** from Dec 17 (the first test target is Dec 17 01:00 at +1h and
  04:00 at +4h; the last is Jan 1 20:00 at +1h and 23:00 at +4h).
- Targets between the two splits (for example Dec 17 00:00 at +1h) have no
  forecast, as in the evaluation.
- **Nov 2 - Dec 9 has no forecasts** (option F1).

| Output | Rows |
|---|---|
| `thresholds_v2` | 10,000 cells; 200 hotspots; no zero threshold |
| `cell_forecasts_v2` | 21,920,000 = 4 horizons × 10,000 cells × 548 origins (168 validation + 380 test). At +1h, 109,600 hotspot rows come from `hotspot_only_final`. |
| `diagnosis_results_v2`, +1h | validation 107,605 ROUTINE / 16,764 ANOMALOUS; test 130,116 / 17,019 |
| `diagnosis_results_v2`, +2h / +3h / +4h (test) | 123,599 / 18,152; 122,764 / 18,571; 122,072 / 18,930 ROUTINE / ANOMALOUS |
| `watch_flags_v2` (validation / test) | +1h 87,144 / 104,885; +2h 98,379 / 123,112; +3h 98,431 / 126,405; +4h 120,657 / 161,344 |
| `reallocation_results_v2` (+1h) | 195,576 moves over 390 target hours that have ROUTINE cases (the other 158 of 548 have none) |
| `reallocation_coverage_v2` (+1h) | 237,721 ROUTINE cases. Validation: deficit 5.59M, covered 3.16M, 50.9% fully resolved. Test: 5.92M, 3.03M, 49.6%. Forecast-defined deficits, one-for-one units. |

### Hours with no ROUTINE case (+1h)

**158 of 548** +1h target hours have no ROUTINE case. **41 of them fall in
the validation week** (Dec 10-16); the other 117 are in test. Across all
158 hours there are only 60 flags, all ANOMALOUS; the remaining hours have
no flag at all. Full list: `data\experiments\promotion_v2\v2_hours_without_routine.csv`.

| Date | Hours (target, local) |
|---|---|
| Dec 10 | 00-06 |
| Dec 11, 12, 13 | 01-06 |
| Dec 14, 15 (weekend) | 04-08 |
| Dec 16, 17 | 01-06 |
| Dec 18, 19, 20 | 00-06 |
| Dec 21 (Sat) | 04-08 |
| Dec 22 (Sun) | 04-09 |
| Dec 23 | 00-06 |
| Dec 24 | 00-07 |
| Dec 25 (holiday) | 02-07 |
| Dec 26 (holiday) | 00, 01, 06-08 |
| Dec 27 | 03-06, 22, 23 |
| Dec 28 | 00, 01, 04-09 |
| Dec 29 | 04-11, 13, 22, 23 |
| Dec 30 | 00-07, 10, 22, 23 |
| Dec 31 | 00-08, 23 |
| Jan 1 (holiday) | 01, 06-11 |

**Checked: these are low-traffic hours, and nothing is dropped**
(`v2_hours_check.csv`):
- **Hour of day of the 158:** 00:00 10, 01:00 16, 02:00 14, 03:00 15,
  04:00 21, 05:00 21, 06:00 23, 07:00 12, 08:00 9, 09:00 4, 10:00 3,
  11:00 2, 13:00 1, 22:00 3, 23:00 4. **89% fall between 00:00 and 08:00.**
- **Traffic:** mean actual activity 214 per cell vs 469 at the other 390
  hours. Their median activity ranks at the 17th percentile of all 548
  hours.
- **Actual exceedance:** 0.2% of cell-hours vs 9.4% at the other hours.
- **No dropped cases:**
  - every one of the 548 hours has forecasts for all 10,000 cells;
  - an independent recount of forecast > threshold equals the diagnosis
    flag count at every hour;
  - all 60 flags in these hours are ANOMALOUS.

  No ROUTINE case means no flag had 30% flagged neighbours. Nothing was
  filtered out.
- **Two hours had real exceedances the model did not forecast:**
  - **Dec 31 23:00:** 6.8% of cells actually exceeded, 0 flags (the New
    Year's Eve miss; see the Dec 31 observation);
  - **Dec 30 10:00:** 1.7% exceeded, 0 flags.

  This is a forecast miss, not a pipeline drop.

These are night and early-morning hours (03:00-06:00 accounts for 80 of the
158), plus late-December daytime hours during the holiday lull. On holidays
every flag is ROUTINE by exemption, so a holiday hour without ROUTINE cases
has no flags at all.

### Watch-set precision

The share of watch flags whose target hour actually exceeded the threshold
(`data\processed\v2\watch_precision_v2.csv` and `v2_metadata.json`). Flags at
m = 1 are shown for reference.

| Horizon | Watch: validation | Watch: test | Flags (m = 1): validation | Flags (m = 1): test |
|---|---|---|---|---|
| +1h | 0.400 (34,815 / 87,144) | 0.318 (33,349 / 104,885) | 0.701 | 0.638 |
| +2h | 0.393 | 0.293 | 0.680 | 0.603 |
| +3h | 0.399 | 0.286 | 0.666 | 0.581 |
| +4h | 0.385 | 0.269 | 0.655 | 0.563 |

**About 3 in 10 (test) to 4 in 10 (validation) watch flags precede an
actual exceedance**, roughly half the precision of an m = 1 flag. That is
consistent with watch being advisory. The test values are lower partly
because the test window is the holiday lull (fewer exceedances).

### Parity tests

| Test | Scope | Result |
|---|---|---|
| **(i)** v2 diagnosis vs the existing `diagnosis_agent.diagnose()`, same forecast input | **30 replay hours**, each horizon +1h to +4h (22,060 / 21,777 / 20,797 / 20,199 flags) | **0 mismatches** in the flagged set, classification and reason text |
|  | Every +1h target hour (548 hours, 271,504 flags) | **0 mismatches** |
| **(ii)** `solver_v2` vs the original solver path (`solver.classify_routine`, then `solve_hour` on the same frames as `solver.py`'s main loop) | 30 replay hours (26 have ROUTINE cases; 15,125 moves) | ROUTINE set identical; **same moves** (0 key mismatches, maximum amount difference 0.0); coverage identical to 1e-14 |
|  | Every +1h target hour (390 with ROUTINE cases; 195,576 moves) | identical; maximum amount difference 0.0; coverage to 1e-13 |
| **(iii)** watch flags vs the definition | Independent recount, every horizon and split | counts equal (for example +1h test 104,885). No hotspot cell. 1,098-2,336 hotspot rows per horizon and split fall in the band and are correctly excluded. |
|  | Random samples: 1,000 watch rows and 1,000 typical non-watch rows per horizon | all satisfy (or correctly fail) m × threshold < forecast ≤ threshold |
| (iv) The assembled forecasts reproduce the evaluation | test, m = 1, Phase 3 thresholds | +1h F1 0.540 all / 0.679 hotspot (H0 in the report: 0.540 / 0.679); +4h 0.465 / 0.526 (P0: 0.465 / 0.526) |

### Time and memory

| Step | Time | Peak commit |
|---|---|---|
| `thresholds_v2` | 12 s | 2.1 GB |
| `generate_forecasts_v2` | 31 s | 4.5 GB |
| `diagnosis_agent_v2` (4 horizons + watch) | 9 s | 3.6 GB |
| `solver_v2` (548 hours, LP 21.7 s) | 24 s | 1.8 GB |
| Parity tests (v1 row-wise code on 271k flags dominates) | 265 s | 3.3 GB |

Commit headroom before the run was 11.7 GB. The four compute steps take
**76 s** in total, against about 10 minutes estimated in D9.

**SHA-256** of every output: `data\experiments\promotion_v2\v2_sha256.csv`.

*Superseded by "Promotion, stage 2" (implemented).* **Not done (stage 2, awaiting approval):** `balancegrid_v2.db` and the
WatchFlag table, `populate_db_v2.py`, the API `/watch` endpoint and the
`/hours` watch count, dashboard and UI changes.

---

## Promotion, stage 2: v2 database, API and dashboard (implemented)

**What changed.**
- **New files:**
  - `src/db/models_v2.py` (its own declarative Base, with a `WatchFlag` table)
  - `src/db/database_v2.py`
  - `src/db/populate_db_v2.py`
  - test harnesses `src/evaluation/promotion_v2_apitest.py` and
    `promotion_v2_dashtest.py`
- **Edited, with backups first** in `backup_initial/promotion/`
  (SHA-256 checked):
  - `src/api/main.py`
  - `src/dashboard/app.py` (+23 lines, additive)
- **Not touched:** `models.py`, `database.py`, `populate_db.py`, the React
  front end, the v1 database, the v1 parquet files and `data\raw` (still 69
  files).

**How v2 is selected and isolated.**
- **Selection:** the API uses v2 only when the environment variable
  `BALANCEGRID_DB=v2` is set. With no variable, it imports `database.py` /
  `models.py` exactly as before.
- **Location:** the v2 database is `data\processed\v2\balancegrid_v2.db`.
  `database_v2.py` refuses any path equal to the v1 file or under
  `data\raw`, and creates tables only from `models_v2.Base`, a separate
  metadata. So `create_all` in v2 mode never sees the v1 file.
- **Decision:** the v2 database sits in `data\processed\v2\`, not next to
  the v1 database in `data\raw`, because of the rule not to write there.

**What the v2 API adds** (only when `BALANCEGRID_DB=v2`):
- **`forecast_kind` on every case, hour and report.** `/cases` rows and
  `/cases/{id}` also carry horizon, model_version and segment, and
  `/cases/{id}` adds the label "validation (used for model size and margin
  selection): out-of-training, but not an untouched test".
- **`/hours` adds `watch`,** the +1h watch count. Hours that have watch flags
  but no case are listed with zero case counts (435 hours in total).
- **`GET /watch`** (filters: target_datetime, date, horizon, forecast_kind,
  limit, offset). The response carries:
  - `status: "advisory"` and the definition;
  - the margin;
  - the measured watch precision for validation and test (for example +1h:
    0.400 / 0.318);
  - the forecast_kind labels;
  - on every item, `advisory: true`, its `forecast_kind` and its precision.
- **The route does not exist in default mode.**

**Watch flags never count as resolved, never appear in /cases, and are
never sent to the solver.** This is checked on the loaded database:
- 0 watch flags coincide with a +1h case;
- 0 receive moves;
- 0 sit on hotspot cells;
- `watch_flags` has no covered / resolved column;
- the solver inputs came only from ROUTINE cases (stage 1).

### Loading (`populate_db_v2.py`)

Headroom before the run was 9.9 GB; the script refuses to run below 6 GB.

| Step | Rows | Time | Peak commit |
|---|---|---|---|
| Read v2 parquet files | – | 0.3 s | 1.1 GB |
| Build records | – | 13.6 s | 2.0 GB |
| Write `balancegrid_v2.db` | cases 271,504 (237,721 ROUTINE, 33,783 ANOMALOUS; +1h, as v1); reallocation sources 195,576; **watch flags 920,357** (all four horizons) | 19.3 s | 1.9 GB |

- **Cases:** 124,369 validation and 147,135 test.
- **Fully resolved ROUTINE cases** (forecast-defined deficit, one-for-one
  units): 54,755 of 107,605 validation, 64,552 of 130,116 test.
- **File:** the database is 180 MB.

### Safety tests

**v1 database, before and after** (`data\experiments\promotion_v2\api\`):
- SHA-256 `2ee5f78f…5bed1`, mtime 2026-10-03 21:23:56.560.
- **Identical** across all five runs:
  - the baseline (old `main.py`);
  - the edited `main.py` in default mode;
  - the edited `main.py` in v2 mode;
  - the dashboard against the v1 API;
  - the dashboard against the v2 API.

**Default-mode responses.** Twelve fixed GET requests were sent to the old
`main.py` (baseline) and to the edited one, with no environment variable:
- `/hours`;
- six `/cases` variants (filters, offset, date, hour, include_sources);
- `/cases/1`, `/cases/12345` and a missing id (404);
- `/reports`;
- `/watch` (404);
- `/openapi.json`.

**All 12 responses are identical, including status codes and the OpenAPI
schema.** POST `/cases/{id}/report` was not sent, because it writes to the
database by design.

**v2 mode.** All requests return 200. Spot checks:
- a v2 case: `forecast_kind` "validation", horizon 1, model `final_63l`;
- `/watch` at 2013-12-18 12:00: 862 flags, `status` "advisory", precision
  validation 0.400 / test 0.318;
- +4h on Dec 12: 23,516 flags, precision 0.385 / 0.269.

**Dashboard (Streamlit, run headless with `AppTest` against a live API on
port 8000):**
- **v1 API:** no exceptions; the same subheaders and table columns as
  before; no watch section and no forecast_kind column.
- **v2 API:** no exceptions; a "Forecast kind" column in the case table; the
  case detail shows the forecast_kind label; a "Watch flags (advisory, +1h)"
  table with the advisory note and the measured precision (validation 40%,
  test 32%).
- The React front end was not changed.

**How to run v2:**

```
python src/db/populate_db_v2.py                  # only if headroom > 6 GB
set BALANCEGRID_DB=v2                            # PowerShell: $env:BALANCEGRID_DB = "v2"
uvicorn src.api.main:app
streamlit run src/dashboard/app.py
```

Without the variable, the API and dashboard run on v1 as before.

**Known gaps:**
- `tests/test_smoke.py` imports functions that no longer exist
  (`solve_dispatch`, a dict-based `diagnose`). It was already broken before
  this work and was not changed.
- *(Superseded by the next subsection: POST /report was then tested on a
  copy.)* The v2 API was tested with GET requests only. Report generation in v2 mode
  writes to the v2 database and was not exercised.

### POST /cases/{id}/report in v2 mode (on a copy)

Script: `src/evaluation/promotion_v2_reporttest.py`; result in
`data\experiments\promotion_v2\api\report_test_v2_copy.json`.

**Setup:**
- The test runs against a **copy** of `balancegrid_v2.db`
  (`data\experiments\promotion_v2\api\balancegrid_v2_COPY_for_report_test.db`).
- It uses FastAPI's in-process TestClient with `get_session` overridden to
  the copy, and the startup hook not run.
- LLM credentials are removed from the environment, so `report_agent` uses
  its offline template and nothing is sent to an external service.
- **The v1 database was not used.**

| Check | Result |
|---|---|
| POST for a ROUTINE case (id 256533, test) | 200; the template report names the source cell and "covers 100% of the predicted deficit" |
| POST for an ANOMALOUS case (id 71506, validation) | 200; the report says "ESCALATE" |
| POST again on the same cases | 200; the report is updated, not duplicated (2 report rows in the copy) |
| POST for a missing id | 404 |
| GET `/cases/{id}` after POST | report present; `forecast_kind` carried |
| GET `/reports` | 2 reports, each with `forecast_kind` |
| Original `balancegrid_v2.db` | **unchanged** (SHA-256 `422fc3bb…1dab86`, same mtime) |
| v1 `balancegrid.db` | **unchanged** (`2ee5f78f…5bed1`) |

Not tested: report generation with a real LLM (it would send case data to
an external service).

### Dashboard watch table: what it shows

Confirmed from the headless run against the v2 API (`dash_v2.json`):
- **Advisory label:** the section is titled "Watch flags (advisory, +1h)";
  every row has `Status = advisory`; the caption repeats "advisory: never
  counted as resolved, never in /cases, never sent to the solver".
- **Measured watch precision, separately for validation and test:** the
  caption reads "Measured share that actually exceeded: validation 40%,
  test 32%".
- **Other columns:** the margin (0.955), the watch definition, and a
  "Forecast kind" column per row.

No change was needed. The React front end was not touched.

---

## Scoring from raw activity (v2, one origin at a time)

**This replays hours inside the 62-day dataset (Nov 1 2013 - Jan 1 2014).
It is not live forecasting.**

**Scripts:**
- `src/forecasting/score_v2.py` (commands `install-models`, `score`);
- the parity harness `src/evaluation/scoring_parity.py`.

**Outputs:**
- `data\processed\v2\scored\<origin>\`;
- harness results and notes in `data\experiments\scoring\`.

No model was fitted. Nothing in `data\raw`, the v1 database, the v1 parquet
files, the API or any saved v2 output was changed.

### Design

- **One code path.** `score_origin(panel, models, origin)` is used by both
  the command line and the parity harness. The harness loads the panel and
  the models once.
- **Information set.** Features at origin t use activity up to t-1 only;
  hour t is never used. The lead time from the last observed hour is
  therefore **h+1 hours (2, 3, 4, 5 hours for +1h to +4h)**.
  `score_origin` slices the panel to hours before t before computing
  anything. The rolling statistics use cumulative sums from the first hour
  (Nov 1), exactly as the evaluation's feature store does, which is
  required for bit-identical float32 features.
- **Activity source.** Only `CellID`, `datetime` and `total_activity` are
  read from `data\raw\cdr_with_congestion_flags.parquet`, through the
  evaluation's own `load_panel`. No threshold or flag column of that file is
  read; thresholds come only from `thresholds_v2`. Checked: the five-channel
  sum in `cdr_activity_aggregated.parquet` is **bit-identical** to
  `total_activity` on all 14,880,000 rows.
- **Models.** The five files were **copied** (not moved) from
  `data\experiments\phase2\models\` to `data\processed\v2\models\`, and
  their SHA-256 hashes match the originals.

  | Model | Trees | Leaves |
  |---|---|---|
  | `final_63l` +1h / +2h / +3h / +4h | 2,153 / 2,154 / 2,193 / 2,181 | 63 |
  | `hotspot_only_final` +1h | 568 | 31 |

  `models_manifest.json` records:
  - LightGBM 4.7.0;
  - the 23 features in model order, with `CellID` (index 8) categorical;
  - float32 features;
  - the target transform: log1p in training; prediction = clip(expm1, 0),
    cast to float32;
  - `cell_historical_mean`: `thresholds_v2.training_mean`, with the hash of
    that file;
  - the hotspot set (200 cell ids; top 2% by training mean, Nov 2 - Dec 9);
  - the training period (targets Nov 2 - Dec 9; sizes chosen on Dec 10-16);
  - the forecast_kind rules and the information-set convention.

  Hashes are checked again every time the models are loaded.
- **Rules.** Watch flags and diagnosis call `diagnosis_agent_v2.diagnose`.
  The V0 solver at +1h calls `solver_v2.run`, which imports `solve_hour`
  unchanged. Solver results hold only under the one-for-one activity-unit
  assumption: moves between grid squares are not physical capacity moves.
- **Outputs per origin:**
  - `forecasts.parquet`, `watch_flags.parquet` and `diagnosis.parquet`;
  - `solver_moves.parquet` and `coverage.parquet`;
  - `manifest.json`: targets with their labels, input and model hashes,
    LightGBM version, counts and timings.

  The script refuses to overwrite an existing folder unless `--force` is
  given.

### Labelling rules (by target T = origin + h)

These apply to the forecasts, watch flags, diagnosis and solver outputs
alike.

| Label | Rule | Meaning |
|---|---|---|
| `in-sample` | T before Dec 10 00:00 | A training target of the models. The thresholds (Nov 1 - Dec 9) and `cell_historical_mean` also use data after the origin (look-ahead). Not a measure of prediction quality. |
| `validation` | Dec 10 00:00 ≤ T < Dec 17 00:00 | Used for model size and margin selection; out-of-training, but not an untouched test |
| `unevaluated` | T ≥ Dec 17 00:00 from an origin before Dec 17 00:00 (targets Dec 17 00:00-03:00) | Between validation and test; no saved reference |
| `test` | T from Dec 17 to Jan 1 23:00, origin ≥ Dec 17 00:00 | Not used for size or margins; used for feature-group selection in Phase 2 |
| `forecast` | T after Jan 1 23:00 | Forecast only; no actuals available |

- **Valid origins:** Nov 2 00:00 to Jan 2 00:00. Earlier origins are
  rejected with a message (checked: Nov 1 23:00 is refused).
- **Checked from the command line:** an origin on Dec 16 22:00 gives
  validation, unevaluated, unevaluated, unevaluated. An origin on Jan 1
  22:00 gives test, then forecast for the three later horizons.

### Parity results

**30 parity origins with a saved reference.**
- **Validation (10):** Dec 10 02:00, Dec 10 08:00, Dec 11 12:00, Dec 12
  17:00, Dec 13 20:00, Dec 14 11:00, Dec 15 15:00, Dec 15 19:00, Dec 16
  07:00, Dec 16 19:00.
- **Test (20):** 14 replay targets minus one hour (including Dec 25 and
  Jan 1), plus Dec 17 00:00, Dec 24 10:00, Dec 27 15:00, Dec 31 22:00,
  Jan 1 01:00 and Jan 1 19:00.
- Every origin was checked at every horizon: 120 origin × horizon cases.

| Check | Required | Result |
|---|---|---|
| Features vs `build_rows` on the full panel | bit-identical float32 | **identical in all 120 cases** |
| Forecasts vs saved `cell_forecasts_v2` | bit-identical, at most 1 float32 ULP | **0 ULP (bit-identical)** in all 1,200,000 cell forecasts. 2 cells lie within a relative 1e-6 of their threshold; with no difference, no flag can flip. |
| Flags (68,393), classification, reason text | 0 mismatches | **0 / 0 / 0** |
| Watch flags | 0 mismatches | **0** |
| V0 solver moves at +1h (14,731 moves) | 0 mismatches | **0 key mismatches; maximum amount difference 0.0** |
| forecast_kind vs the saved v2 label | agree | **agree** |

**6 weekly-boundary origins with no saved reference:** Nov 2 00:00 and
Nov 7 20:00 to Nov 8 00:00, all in-sample.
- **Features:** bit-identical to `build_rows`.
- **Forecasts:** bit-identical (0 ULP) to the original evaluation path
  (`phase2_common.predict` on `build_rows` features, same model files).
- **The NaN handling matches training:**
  - `lag_168h` is NaN for every cell until Nov 8 00:00;
  - `weekly_naive` (A[t+h-168]) becomes available as t+h reaches Nov 8
    00:00 (for example origin Nov 7 21:00: NaN at +1h and +2h, present at
    +3h and +4h).

**Further checks:**
- **Leakage test** (extra): at two origins (Dec 13 20:00, Dec 29 14:00),
  activity and neighbour means at every hour from the origin onward were
  replaced with random values. All outputs (forecasts, diagnosis, watch
  flags, solver moves) were **identical**. Scoring does not read hours at
  or after the origin.
- **Command line vs harness:** `score --origin "2013-12-18 11:00"` wrote
  forecasts and solver moves (673) identical to the harness result. A
  second run without `--force` was refused.
- **Protected files:** v1 database SHA-256 (`2ee5f78f…5bed1`) and mtime
  unchanged; `data\raw` file count (69), sizes and mtimes unchanged,
  before and after.

### Time and memory

| Step | Time | Peak commit |
|---|---|---|
| Model copy and hash check (`install-models`, rerun with the copies in place) | 5.2 s | 1.3 GB |
| Channel-sum bit-identity check (both parquet files, read-only) | 6.5 s | – |
| Harness: load panel / feature store / models / saved v2 outputs | 10.5 / 4.4 / 5.0 / 1.7 s | 1.9 / 3.3 / 3.7 / 4.9 GB |
| Harness: 30 parity origins (score + reference + comparisons) | 381 s (about 13 s per origin) | 5.3 GB |
| Harness: 6 boundary origins / leakage test | 89 s / 25 s | 5.0 / 5.5 GB |
| **Command line, one origin** | **about 30 s** (load panel 10-20 s, load models 5 s, score 5.5-7 s) | **about 2.7 GB** |

Commit headroom before the harness was 5.9 GB. That is just under the 6 GB
rule, which applies to model fits; no fit was run, and the harness peaked
at 5.45 GB of commit.

### What could not be verified

- **Origins with no saved reference** are checked only for feature and
  forecast identity with the evaluation path (the 6 boundary origins), not
  for flags or solver moves against an independent source. This covers:
  in-sample origins, the `unevaluated` targets, test targets from origins
  Jan 1 20:00-22:00 (outside the saved v2 test set), and `forecast`
  targets.
- **Bit-identity depends on the environment:** LightGBM 4.7.0, numpy and
  pandas as installed. Another library version, or computing the rolling
  sums from a later start hour, could change the last float bits and, at
  boundary splits, a forecast.
- **The models and the panel come from the study's own files.** The model
  files live under ignored folders (not in git). The panel is loaded in
  full and truncated in memory; the leakage test shows that hours from the
  origin onward are not used.
- **Not live:** there is no live data feed, no handling of late or missing
  hours beyond the loader's zero-fill, and no check on data after Jan 1.
  `forecast`-labelled outputs cannot be evaluated.
- **Look-ahead for in-sample origins:** these outputs use thresholds and
  cell means computed with later data, so they must not be read as
  prediction quality.
