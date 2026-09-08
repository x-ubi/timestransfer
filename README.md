# TimesTransfer

Benchmark pipeline for a master's thesis experiment comparing time-series foundation models, tabular transfer models, and classical baselines.

## Scope

- Datasets: M4 Hourly plus the full LongHorizon2 suite (ETT, ECL, Traffic, Weather, Exchange, ILI), all at h48, plus ETTh1 h96.
- Split: fixed final-horizon holdout per series.
- Models: Linear Regression, TabPFNRegressor, TimesFM 2.5, Chronos-Bolt, Prophet, AutoARIMA, AutoETS, AutoTheta, and SeasonalNaive.
- Metrics: MAE, RMSE, sMAPE, and MASE at overall, per-horizon-step, and per-series scope.
- Required outputs: `outputs/metrics/metrics.csv` and `outputs/metadata/environment.json`.

## Setup

```bash
uv python install 3.12
uv run pytest
```

TabPFN may require accepting non-commercial license terms. On a headless VM, set `TABPFN_TOKEN` after accepting the license in the Prior Labs UI. The first benchmark run downloads the TimesFM and Chronos-Bolt checkpoints from the Hugging Face Hub.

## Run

```bash
uv run python scripts/run_benchmark.py --config configs/experiment.yaml
```

Useful filters:

```bash
uv run python scripts/run_benchmark.py --config configs/experiment.yaml --datasets m4_hourly ett_h1_h48
uv run python scripts/run_benchmark.py --config configs/experiment.yaml --models seasonal_naive prophet chronos_bolt
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

- `outputs/metrics/metrics.csv`: overall, horizon-wise, and per-series MAE, RMSE, sMAPE, and MASE. Failed models get a `model_status` row with the error.
- `outputs/metrics/overall_by_dataset.csv`: overall metrics for quick thesis tables.
- `outputs/metrics/model_comparison_wide.csv`: one row per dataset/horizon with model metrics as columns.
- `outputs/metrics/horizon_metrics.csv`: horizon-wise metric rows.
- `outputs/metrics/series_metrics.csv`: per-series metric rows (input for the characteristics analysis).
- `outputs/metrics/mase_scales.csv`: per-series MASE denominators (in-sample seasonal-naive MAE on train).
- `outputs/forecasts/*.parquet`: point forecasts for successful models.
- `outputs/metadata/environment.json`: package versions, GPU/CUDA checks, config values, model statuses, dataset summary, and contamination notes.
- `outputs/figures/*.png`: sMAPE, RMSE, and MASE comparison plots.
- `outputs/figures/forecasts/*.png`: per-series history + actuals vs forecasts (from `scripts/plot_forecasts.py`).
- `outputs/analysis/`: series features, feature-vs-metric Spearman correlations, heatmap, and scatter figures (from `scripts/analyze_series.py`).
- `outputs/timesfm_tuning/`: the same output tree for the TimesFM tuning experiment.

## Metrics

MASE follows the M4 convention: each per-series absolute error is divided by that series' in-sample mean absolute error of the seasonal-naive forecast on the train split (lag-1 naive when a series is shorter than one season; Exchange uses `mase_seasonality: 1` because FX is treated as non-seasonal). Aggregate rows average the scaled errors across series, so MASE stays comparable across series of very different magnitudes. sMAPE uses the 0-200 formulation.

## Datasets

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

All datasets evaluate the full panel and hold out the last `horizon` observations per series. The uniform h48 horizon keeps models comparable across domains; ETTh1 h96 is the long-horizon variant. Note this deviates from the standard LongHorizon protocol (96/192/336/720 with normalized data), so numbers are not directly comparable to long-horizon papers.

LongHorizon2 loading notes: the datasetsforecast library registers ETTh1/ETTh2/ETTm1/ETTm2/ECL/TrafficL/Weather (and mimics the Google/TiDE truncation of ETT to its first `n_time` rows); Exchange and ILI ship in the same archive but are not registered, so they are melted directly from the extracted `Y_df.csv`.

If the per-series models (Prophet, AutoARIMA, AutoETS, AutoTheta) make the full run too slow on a given machine, the documented fallback is `series_limit: 100` for `ecl_h48` and `traffic_h48` (a deterministic first-100-ids subset).

## Model notes

**TabPFN adaptation.** TabPFN is tabular, so the pipeline converts each series into supervised rows. Each row represents an origin time and forecast horizon. Features include normalized lag values, the horizon, seasonal phase, and log scale. The target is the future value divided by the train-series scale. Predictions are multiplied back by the same scale before scoring. Linear Regression uses the same feature table with median imputation, making it a direct simple baseline for the TabPFN framing.

**Statistical baselines.** AutoARIMA (lightweight non-seasonal bounded search), AutoETS, AutoTheta, and SeasonalNaive run per series through StatsForecast; these are the comparators used across the TimesFM/Chronos/TabPFN-TS papers, and SeasonalNaive also anchors MASE. AutoARIMA/AutoETS/AutoTheta cap training at the last `max_train_length` points (default 5000) to stay tractable on the 52k-57k-point ETTm/Weather series.

**Prophet** (https://facebook.github.io/prophet/) fits each series independently with `uncertainty_samples=0` (point forecasts only) and Prophet's automatic seasonality selection; training is capped at the last `max_train_length` points (default 4320) and parallelized across series. M4 Hourly has integer timestamps, so the pipeline maps them onto a synthetic hourly datetime axis; the daily pattern is preserved exactly, but weekday alignment is arbitrary per series, so Prophet's weekly component on M4 is phase-shifted noise - a caveat, not a bug.

**Chronos-Bolt** (amazon/chronos-bolt-base) runs zero-shot like TimesFM with a 2048-point context and the median quantile as the point forecast. Its native prediction length is 64, so h96 forecasts continue autoregressively beyond that - expect a quality drop on ett_h1_h96 and treat it as a property of the model.

## Foundation Models And Benchmark Contamination

M4 Hourly should be treated as a pipeline and comparability benchmark, not as a clean zero-shot generalization benchmark for TimesFM. Public TimesFM 2.5 documentation lists GiftEvalPretrain, Wikimedia Pageviews, Google Trends, and synthetic/augmented data. The original TimesFM paper also states that all M4 granularities were included in a TimesFM pretraining corpus. The exact overlap for TimesFM 2.5 cannot be fully audited from public metadata, so thesis text should label this as possible or likely benchmark contamination.

The same caveat applies to Chronos-Bolt: the Chronos training corpus includes M4, ETT, ECL, Traffic, and Weather (see the Chronos paper appendix). Zero-shot foundation-model numbers on these panels are comparability results, not clean generalization evidence. Per-dataset notes are recorded in `outputs/metadata/environment.json`.

Relevant sources:

- https://huggingface.co/google/timesfm-2.5-200m-pytorch
- https://github.com/google-research/timesfm
- https://arxiv.org/abs/2310.10688
- https://arxiv.org/abs/2403.07815
- https://huggingface.co/amazon/chronos-bolt-base
- https://raw.githubusercontent.com/Nixtla/datasetsforecast/main/datasetsforecast/m4.py
- https://github.com/PriorLabs/TabPFN
