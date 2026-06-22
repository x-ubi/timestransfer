# TimesTransfer

Benchmark pipeline for a master's thesis experiment comparing time-series foundation models, tabular transfer models, and classical baselines.

## Scope

- Datasets: M4 Hourly h48 and ETTh1 h48.
- Split: fixed final-horizon holdout per series.
- Models: Linear Regression, TabPFNRegressor, TimesFM 2.5, and AutoARIMA.
- Required outputs: `outputs/metrics/metrics.csv` and `outputs/metadata/environment.json`.

ETTh1 h96 is intentionally not implemented yet; it is reserved for the next approved step.

## Setup

```bash
uv python install 3.12
uv run pytest
```

TabPFN may require accepting non-commercial license terms. On a headless VM, set `TABPFN_TOKEN` after accepting the license in the Prior Labs UI.

## Run

```bash
uv run python scripts/run_benchmark.py --config configs/experiment.yaml
```

The default config runs full M4 Hourly h48 and ETTh1 h48 with `train_row_cap: 10000`.

Useful filters:

```bash
uv run python scripts/run_benchmark.py --config configs/experiment.yaml --datasets m4_hourly ett_h1_h48
uv run python scripts/run_benchmark.py --config configs/experiment.yaml --models linear_regression auto_arima
```

## Outputs

- `outputs/metrics/metrics.csv`: overall and horizon-wise MAE, RMSE, and sMAPE. Failed models get a `model_status` row with the error.
- `outputs/metrics/overall_by_dataset.csv`: overall metrics for quick thesis tables.
- `outputs/metrics/model_comparison_wide.csv`: one row per dataset/horizon with model metrics as columns.
- `outputs/metrics/horizon_metrics.csv`: horizon-wise metric rows.
- `outputs/forecasts/*.parquet`: point forecasts for successful models.
- `outputs/metadata/environment.json`: package versions, GPU/CUDA checks, config values, model statuses, dataset summary, and the TimesFM/M4 contamination note.
- `outputs/figures/*.png`: simple sMAPE comparison plots.

## Datasets

M4 Hourly h48 is preserved from Phase 1: all 414 hourly series use their final 48 observations as the test set.

ETTh1 h48 uses the hourly Electricity Transformer Temperature dataset as seven independent univariate series. The final 48 observations are held out for each series. This gives a second non-M4 benchmark with the same horizon length as M4, so model behavior can be compared across domains without changing forecast length. ETTh1 h96 is a standard ETT-style horizon, but it is deliberately held for the next implementation step.

## TabPFN Time-Series Adaptation

TabPFN is tabular, so the pipeline converts each series into supervised rows. Each row represents an origin time and forecast horizon. Features include normalized lag values, the horizon, seasonal phase, and log scale. The target is the future value divided by the train-series scale. Predictions are multiplied back by the same scale before scoring.

Linear Regression uses the same feature table with median imputation, making it a direct simple baseline for the TabPFN framing. AutoARIMA is run separately as a classical per-series statistical baseline through StatsForecast. The default AutoARIMA config uses a lightweight non-seasonal bounded search so the full M4 benchmark remains practical.

## TimesFM And M4 Contamination

M4 Hourly should be treated as a pipeline and comparability benchmark, not as a clean zero-shot generalization benchmark for TimesFM. Public TimesFM 2.5 documentation lists GiftEvalPretrain, Wikimedia Pageviews, Google Trends, and synthetic/augmented data. The original TimesFM paper also states that all M4 granularities were included in a TimesFM pretraining corpus. The exact overlap for TimesFM 2.5 cannot be fully audited from public metadata, so thesis text should label this as possible or likely benchmark contamination.

Relevant sources:

- https://huggingface.co/google/timesfm-2.5-200m-pytorch
- https://github.com/google-research/timesfm
- https://arxiv.org/abs/2310.10688
- https://raw.githubusercontent.com/Nixtla/datasetsforecast/main/datasetsforecast/m4.py
- https://github.com/PriorLabs/TabPFN
