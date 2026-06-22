# TimesTransfer

Minimal Phase 1 benchmark pipeline for a master's thesis experiment comparing time-series foundation models and tabular transfer models on M4 Hourly.

## Phase 1 Scope

- Dataset: M4 Hourly only, loaded through `datasetsforecast`.
- Default run: deterministic subset of 32 series, final 48 observations held out per series.
- Models: Linear Regression, TabPFNRegressor if available, and TimesFM 2.5 if available.
- Required outputs: `outputs/metrics/metrics.csv` and `outputs/metadata/environment.json`.

Tourism Monthly, ARIMA, plots, and larger sweeps are intentionally left for Phase 2.

## Setup

```bash
uv python install 3.12
uv run pytest
```

The default install keeps TabPFN and TimesFM optional so dependency, license, CUDA, or checkpoint problems do not prevent the basic benchmark from running. To install the optional model packages:

```bash
uv sync --extra tabpfn --extra timesfm
```

TabPFN may require accepting non-commercial license terms. On a headless VM, set `TABPFN_TOKEN` after accepting the license in the Prior Labs UI.

## Run

```bash
uv run python scripts/run_benchmark.py --config configs/experiment.yaml
```

The default config uses `series_limit: 32` and `train_row_cap: 10000`. Increase these after the first successful pipeline run.

## Outputs

- `outputs/metrics/metrics.csv`: overall and horizon-wise MAE, RMSE, and sMAPE. Failed optional models get a `model_status` row with the error.
- `outputs/forecasts/*.parquet`: point forecasts for successful models.
- `outputs/metadata/environment.json`: package versions, GPU/CUDA checks, config values, model statuses, dataset summary, and the TimesFM/M4 contamination note.

## TabPFN Time-Series Adaptation

TabPFN is tabular, so the pipeline converts each series into supervised rows. Each row represents an origin time and forecast horizon. Features include normalized lag values, the horizon, seasonal phase, and log scale. The target is the future value divided by the train-series scale. Predictions are multiplied back by the same scale before scoring.

Linear Regression uses the same feature table with median imputation, making it a direct simple baseline for the TabPFN framing.

## TimesFM And M4 Contamination

M4 Hourly should be treated as a pipeline and comparability benchmark, not as a clean zero-shot generalization benchmark for TimesFM. Public TimesFM 2.5 documentation lists GiftEvalPretrain, Wikimedia Pageviews, Google Trends, and synthetic/augmented data. The original TimesFM paper also states that all M4 granularities were included in a TimesFM pretraining corpus. The exact overlap for TimesFM 2.5 cannot be fully audited from public metadata, so thesis text should label this as possible or likely benchmark contamination.

Relevant sources:

- https://huggingface.co/google/timesfm-2.5-200m-pytorch
- https://github.com/google-research/timesfm
- https://arxiv.org/abs/2310.10688
- https://raw.githubusercontent.com/Nixtla/datasetsforecast/main/datasetsforecast/m4.py
- https://github.com/PriorLabs/TabPFN
