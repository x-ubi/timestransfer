# TimesTransfer

Benchmark pipeline for a master's thesis experiment comparing time-series foundation models, tabular transfer models, and classical baselines.

## Scope

- Datasets: M4 Hourly plus the full LongHorizon2 suite (ETT, ECL, Traffic, Weather, Exchange, ILI), all at h48, plus ETTh1 h96.
- Split: rolling-origin evaluation, 5 non-overlapping `horizon`-length windows per series ending at the series end (window 0 is the final holdout).
- Models: Linear Regression, TabPFN-TS (v2 and v3.5 checkpoints), TimesFM 2.5, Chronos-2, Prophet, AutoARIMA, AutoETS, AutoTheta, and SeasonalNaive.
- Metrics: MAE, RMSE, sMAPE, and MASE at overall, per-window, per-horizon-step, and per-series scope, plus the geometric mean of per-task scores relative to SeasonalNaive.
- Required outputs: `outputs/metrics/metrics.csv` and `outputs/metadata/environment.json`.

## Setup

```bash
uv python install 3.12
uv run pytest
```

TabPFN-TS runs locally on the GPU and downloads its checkpoints on first use. The v2 checkpoint (`tabpfn_ts`) downloads without a license gate; the v3.5 checkpoint (`tabpfn_ts_v3p5`) requires a one-time license acceptance: log in at https://ux.priorlabs.ai, accept the TabPFN-3.5 license on the Licenses tab, and on a headless VM set `TABPFN_TOKEN` to the account's API key. Telemetry is disabled for benchmark runs (`TABPFN_DISABLE_TELEMETRY=1`). The first benchmark run downloads the TimesFM and Chronos-2 checkpoints from the Hugging Face Hub.

## Run

```bash
uv run python scripts/run_benchmark.py --config configs/experiment.yaml
```

Useful filters:

```bash
uv run python scripts/run_benchmark.py --config configs/experiment.yaml --datasets m4_hourly ett_h1_h48
uv run python scripts/run_benchmark.py --config configs/experiment.yaml --models seasonal_naive prophet chronos2
uv run python scripts/run_benchmark.py --config configs/experiment.yaml --series-limit 4   # fast smoke
```

After a benchmark run, two standalone analysis scripts consume the persisted outputs (so plots and analyses can be iterated without re-running models):

```bash
# Per-series figures: train-history tail | actual continuation vs model forecasts.
uv run python scripts/plot_forecasts.py --config configs/experiment.yaml

# Per-series characteristics (ACF, STL trend/seasonal strength, CV, spectral entropy)
# joined with per-series metrics -> Spearman correlations, heatmap, scatter plots.
uv run python scripts/analyze_series.py --config configs/experiment.yaml --metric smape
```

## Model configuration and variants

Each entry under `models:` in the config is a named model variant. The YAML key is the model name that flows into metrics, forecasts, and plots; the optional `runner:` field selects the implementation (it defaults to the entry name). This makes hyperparameter sweeps plain config:

```yaml
models:
  timesfm_2p5_ctx4096:
    enabled: true
    runner: timesfm_2p5
    max_context: 4096
```

## TimesFM tuning experiment

`configs/timesfm_tuning.yaml` runs a TimesFM 2.5 inference-tuning study on ETTh1/ETTm1/Weather/M4 Hourly with results isolated under `outputs/timesfm_tuning/`:

- a context-length sweep (512 ... 16256): contexts must be multiples of the 32-point input patch, and `compile()` rounds the horizon up to the 128-point output patch while enforcing context + horizon <= 16384, so 16256 is the largest legal context. M4 Hourly trains are <= 960 points, so contexts beyond 1024 only matter on the long LongHorizon2 series (ETTm1 and Weather exercise the full grid).
- a ForecastConfig flag ablation vs the ctx1024 baseline (`normalize_inputs`, `use_continuous_quantile_head`, `force_flip_invariance`, `infer_is_positive`, `fix_quantile_crossing`). The two quantile-related flags mainly affect quantile outputs, so with point-forecast scoring small or zero deltas are expected there.

```bash
uv run python scripts/run_benchmark.py --config configs/timesfm_tuning.yaml
```

The installed timesfm package exposes no weight fine-tuning API; tuning here means inference-time configuration.

## Outputs

- `outputs/metrics/metrics.csv`: overall, per-window, horizon-wise, and per-series MAE, RMSE, sMAPE, and MASE. Overall, horizon, and series rows pool all windows; `window` rows score each window. Failed or skipped models get a `model_status` row with the reason; a model is scored on a task only if every window succeeded.
- `outputs/metrics/relative_to_seasonal_naive.csv`: per-task MASE and MAE divided by SeasonalNaive's.
- `outputs/metrics/aggregate_relative_scores.csv`: geometric mean of those ratios per model, with the number of tasks it covers.
- `outputs/metrics/window_spread.csv`: mean, std, min, and max of window-level MASE and sMAPE per task and model.
- `outputs/metrics/overall_by_dataset.csv`: overall metrics for quick thesis tables.
- `outputs/metrics/model_comparison_wide.csv`: one row per dataset/horizon with model metrics as columns.
- `outputs/metrics/horizon_metrics.csv`: horizon-wise metric rows.
- `outputs/metrics/series_metrics.csv`: per-series metric rows (input for the characteristics analysis).
- `outputs/metrics/mase_scales.csv`: per-series, per-window MASE denominators (in-sample seasonal-naive MAE on that window's train split).
- `outputs/forecasts/*.parquet`: point forecasts for successful models, with a `window` column.
- `outputs/metadata/environment.json`: package versions, GPU/CUDA checks, config values, model statuses, dataset summary, and contamination notes.
- `outputs/figures/*.png`: sMAPE, RMSE, and MASE comparison plots.
- `outputs/figures/forecasts/*.png`: per-series history + actuals vs forecasts (from `scripts/plot_forecasts.py`).
- `outputs/analysis/`: series features, feature-vs-metric Spearman correlations, heatmap, and scatter figures (from `scripts/analyze_series.py`).
- `outputs/timesfm_tuning/`: the same output tree for the TimesFM tuning experiment.

## Metrics

MASE follows the M4 convention: each per-series absolute error is divided by that series' in-sample mean absolute error of the seasonal-naive forecast on the train split (lag-1 naive when a series is shorter than one season; Exchange uses `mase_seasonality: 1` because FX is treated as non-seasonal). Each window uses the scale of its own training history. Aggregate rows average the scaled errors across series (and windows), so MASE stays comparable across series of very different magnitudes. sMAPE uses the 0-200 formulation. The cross-task summary divides each model's task-level MASE (and MAE) by SeasonalNaive's and takes the geometric mean over tasks, as in the Chronos and TabPFN-TS papers.

## Datasets

A full run with 5 windows and seasonal AutoARIMA takes on the order of a day on 16 CPU cores and one GPU (AutoARIMA on ECL/Traffic dominates). Filtered runs overwrite the combined outputs in `outputs_dir`, so point partial runs at a separate `outputs_dir`.

| name | source group | freq | season | horizon | series |
|---|---|---|---|---|---|
| m4_hourly | M4 Hourly | hourly (integer ds) | 24 | 48 | 414 |
| ett_h1_h48 | ETTh1 | hourly | 24 | 48 | 7 |
| ett_h1_h96 | ETTh1 | hourly | 24 | 96 | 7 |
| ett_h2_h48 | ETTh2 | hourly | 24 | 48 | 7 |
| ett_m1_h48 | ETTm1 | 15 min | 96 | 48 | 7 |
| ett_m2_h48 | ETTm2 | 15 min | 96 | 48 | 7 |
| ecl_h48 | ECL | hourly | 24 | 48 | 321 |
| traffic_h48 | TrafficL | hourly | 24 | 48 | 862 |
| weather_h48 | Weather | 10 min | 144 | 48 | 21 |
| exchange_h48 | Exchange | daily | 7 | 48 | 8 |
| ili_h48 | ILI | weekly | 52 | 48 | 7 |

All datasets evaluate the full panel over `n_windows` rolling origins (default 5; stride = horizon, so test windows do not overlap). For window k the test span is the `horizon` observations ending `k * horizon` steps before the series end; the model sees only earlier data. On M4 Hourly, window 0 is the official M4 test set and earlier windows come from the M4 training data. The uniform h48 horizon keeps models comparable across domains; ETTh1 h96 is the long-horizon variant. Note this deviates from the standard LongHorizon protocol (96/192/336/720 with normalized data), so numbers are not directly comparable to long-horizon papers.

LongHorizon2 loading notes: the datasetsforecast library registers ETTh1/ETTh2/ETTm1/ETTm2/ECL/TrafficL/Weather (and mimics the Google/TiDE truncation of ETT to its first `n_time` rows); Exchange and ILI ship in the same archive but are not registered, so they are melted directly from the extracted `Y_df.csv`.

If the per-series models (Prophet, AutoARIMA, AutoETS, AutoTheta) make the full run too slow on a given machine, the documented fallback is `series_limit: 100` for `ecl_h48` and `traffic_h48` (a deterministic first-100-ids subset).

## Model notes

**TabPFN-TS.** The TabPFN-TS pipeline (`tabpfn-time-series` 1.3.0, local mode) fits TabPFN separately for each series. Each timestamp is one row with running-index, calendar (sin/cos), and FFT-detected seasonal features; no lag features. The point forecast is the predictive median. `tabpfn_ts` uses the published configuration (Hoo et al., arXiv:2501.02945 v4): the TabPFN-v2 regressor and the most recent 4096 observations. The pipeline code is the 1.3.0 release, not the paper's code version (e.g. its seasonal feature keeps up to 12 periods; the paper uses k=5), and `tabpfn` 9 downloads `tabpfn-v2-regressor.ckpt`, which may not be byte-identical to the paper's `tabpfn-v2-regression-2noar4o2.ckpt`. `tabpfn_ts_v3p5` uses the release's pinned TabPFN v3.5 checkpoint (synthetic-only pretraining) with the shipped 32768 context; it needs the TabPFN-3.5 license accepted. M4 Hourly has integer time indices, so it is mapped onto synthetic hourly timestamps (as for Prophet); its calendar features therefore encode only the period-24 cycle, not real clock or weekday time. Run metadata records the package versions, checkpoint file name, and checkpoint SHA-256.

**Linear Regression.** Pooled direct multi-horizon regression: each series is converted into supervised rows (origin time, forecast horizon) with normalized lag values, the horizon, seasonal phase, and log scale as features, capped at `train_row_cap` rows per task and window (200k). The target is the future value divided by the train-series scale, and predictions are multiplied back before scoring. Missing lags are median-imputed.

**Statistical baselines.** AutoARIMA (StatsForecast default stepwise seasonal search), AutoETS, AutoTheta, and SeasonalNaive run per series through StatsForecast; these are the comparators used across the TimesFM/Chronos/TabPFN-TS papers, and SeasonalNaive also anchors MASE. AutoARIMA/AutoETS/AutoTheta cap training at the last `max_train_length` points (default 5000) to stay tractable on the 52k-57k-point ETTm/Weather series. Seasonal AutoARIMA is skipped for season lengths above 52 (ETTm: 96, Weather: 144; one ETTm1 fit did not finish in ~35 min) and recorded as `skipped`.

**Prophet** (https://facebook.github.io/prophet/) fits each series independently with `uncertainty_samples=0` (point forecasts only) and Prophet's automatic seasonality selection; training is capped at the last `max_train_length` points (default 4320) and parallelized across series. M4 Hourly has integer timestamps, so the pipeline maps them onto a synthetic hourly datetime axis; the daily pattern is preserved exactly, but weekday alignment is arbitrary per series, so Prophet's weekly component on M4 is phase-shifted noise - a caveat, not a bug.

**Chronos-2** (amazon/chronos-2, arXiv:2510.15821; `chronos-forecasting` 2.x) runs zero-shot, univariate, with cross-learning disabled so each series is forecast independently of the batch. It uses its full 8192-point context and the median quantile as the point forecast; its 1024-step native prediction length covers every horizon here directly.

**TimesFM 2.5** uses a 1024-point context. Hugging Face checkpoints for TimesFM and Chronos-2 are pinned to commit revisions in the config and recorded in run metadata.

## Foundation Models And Benchmark Contamination

M4 Hourly should be treated as a pipeline and comparability benchmark, not as a clean zero-shot generalization benchmark for TimesFM. Public TimesFM 2.5 documentation lists GiftEvalPretrain, Wikimedia Pageviews, Google Trends, and synthetic/augmented data. The original TimesFM paper also states that all M4 granularities were included in a TimesFM pretraining corpus. The exact overlap for TimesFM 2.5 cannot be fully audited from public metadata, so thesis text should label this as possible or likely benchmark contamination.

The same caveat applies to Chronos-2: its model card lists a subset of the Chronos datasets (which include M4, ETT, ECL, Traffic, and Weather; see the Chronos paper appendix) and a subset of GIFT-Eval Pretrain. Zero-shot foundation-model numbers on these panels are comparability results, not clean generalization evidence. Per-dataset notes are recorded in `outputs/metadata/environment.json`.

Relevant sources:

- https://huggingface.co/google/timesfm-2.5-200m-pytorch
- https://github.com/google-research/timesfm
- https://arxiv.org/abs/2310.10688
- https://arxiv.org/abs/2403.07815
- https://huggingface.co/amazon/chronos-2
- https://arxiv.org/abs/2510.15821
- https://raw.githubusercontent.com/Nixtla/datasetsforecast/main/datasetsforecast/m4.py
- https://github.com/PriorLabs/TabPFN
- https://github.com/PriorLabs/tabpfn-time-series
- https://arxiv.org/abs/2501.02945
