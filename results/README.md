# Archived results

`outputs/` is git-ignored, so this folder keeps the run outputs that are expensive or impossible to regenerate. Everything here was written by `scripts/archive_results.py` unless noted.

## Contents

| Folder | Source in `outputs/` | What it is | Cost to regenerate |
|---|---|---|---|
| `benchmark/` | `outputs/` | Experiment 1: 9 models, 11 tasks, 5 rolling origins (thesis chapters) | about 1 day (AutoARIMA) |
| `timesfm_tuning/` | `outputs/timesfm_tuning/` | Experiment 3: TimesFM context sweep and flag ablation | a few hours |
| `combination/` | `outputs/combination/` | R1: TimesFM, Chronos-2, TabPFN-TS, SeasonalNaive with q10–q90, 20 origins (10 for M4 and ILI) | 22.3 h (TabPFN-TS 22.0 h) |
| `combination_stats/` | `outputs/combination_stats/` | SeasonalNaive, AutoETS, AutoTheta on the R1 windows | about 3.3 h |
| `extras/` | various | Phase C analysis tables, Experiment 2 features, combiner weights and summary metrics, cached series features | minutes |
| `logs/` | `*.log` | Run logs with wall-clock timings (stored as `.txt` because `*.log` is git-ignored) | – |
| `pilots_2026-10/` | none (written here) | Exploratory pilots behind the October 2026 direction decision; see below | minutes, except `corruption.py` (~40 min) |

Each run folder holds `forecasts/` (one zstd parquet per dataset and model), `metrics/` (including `mase_scales.csv`) and `metadata/environment.json`. Archiving checks that the split files reproduce the original forecast table exactly.

## Restore

```bash
uv run python scripts/archive_results.py restore
```

This rebuilds `outputs/<run>/forecasts/all_datasets_all_models.parquet` plus the metrics, metadata and extras. The combined forecasts of the combiners are not archived; regenerate them with `scripts/run_combination.py` (about 35 s). Then `scripts/run_combination_analysis.py` reproduces `outputs/combination/analysis/`.

## Pilots (`pilots_2026-10/`)

Run from the repository root after restoring `outputs/`, e.g. `uv run python results/pilots_2026-10/shared_error.py`. Text outputs are in `out/`. The original scratchpad copies were lost in a server reset on 2026-10-26; these scripts were recreated from the session record and rerun on that date.

| Script | Question | Result |
|---|---|---|
| `shared_error.py` | How much FM error is shared, and is it predictable from past windows? | 84–97% of member-error variance is in one common component on every task; a series' mean error barely correlates between consecutive windows (−0.16 to 0.37); per-series bias correction from past windows makes MASE worse on most tasks |
| `probabilistic_pooling.py` | Quantile averaging vs mixture pooling | Equal mixture 1.043× the best single FM's quantile loss per task (geometric mean), quantile averaging 1.082×, TimesFM 1.050×; 80% coverage 0.73–0.79 (Chronos-2), 0.76–0.89 (TabPFN-TS) |
| `timesfm_context_selection.py` | Choose TimesFM's context per task from past windows (Experiment 3 sweep, windows 0–3) | Weather 0.231 → 0.167 (−28%), ETTm1 0.527 → 0.460 (−13%), ETTh1 and M4 unchanged |
| `chronos2_multivariate.py` | Chronos-2 multivariate / cross-learning vs univariate, 20 origins | First run (2026-10-15): Weather −8.3%, ETTm1 −5.9%, ETTh1 −3.1%, ETTm2 0%, ETTh2 +2.2%, ECL (cross-learning, batches of 100) +14% |
| `corruption.py`, `corruption_analysis.py` | Do damaged contexts (20% MCAR, a 96-point block, 1% spikes) make the FMs disagree more? | First run (2026-10-15): no; errors stay as correlated or become more so, equal mean stays 1.00–1.20× the best member; TabPFN-TS degrades most under missing data. The rerun uses a deterministic seed, so its damage pattern differs from the first run |
