# Combining zero-shot TSFMs: findings, thesis direction, and experiment/implementation plan

Status: research brief and work plan (2026-10-03). Phase A1–A3 are implemented and the A4 smoke run passed on branch `ensemble_research` (2026-10-03; see the A4 note in §5). R1 is done as a single run of all four models (see the R1 note in §5). Phase B (B1–B3) is implemented and a first R2 run is done (see the R2 note in §5). Phase C analyses are implemented and run on the FM and FM+SeasonalNaive pools (see the Phase C note in §5); the AutoETS/AutoTheta run for the statistical pools is in progress. Phases D–G have not been started.

When TabPFN-TS runs separately, write it elsewhere with `scripts/run_benchmark.py --outputs-dir`.
Audience: an agent continuing this master's-thesis work. Read this whole file before acting.

## 0. Context and ground rules

- Code repository: `/home/ubuntu/timestransfer` (branch `benchmark_comparison`). Read its `AGENTS.md` first.
- Thesis repository (Overleaf checkout): `/home/ubuntu/thesis`. Read its `AGENTS.md` before editing thesis files. Keep the two Git histories separate.
- The existing thesis contains Experiments 1–3 (`tex/experiments.tex`, `tex/results.tex`): a 9-model benchmark, a feature–error analysis, and a TimesFM inference-sensitivity study. The professor considers this an engineering baseline and asked for a scientific contribution in one of: improving a TSFM without pretraining, ensembling TSFMs, fine-tuning, or data issues (noise, missing values). This document selects **ensembling/combination** as the main line.
- Budget: about 8 weeks from 2026-10-03, 10–20 h/week of the student's time, with heavy AI-agent use. One RTX A5000 (24 GB), 16 CPU cores.
- **Do not commit.** Implement in the working tree; the user reviews and commits. Never add Co-Authored-By or other AI attribution lines to commit messages.
- **Never overwrite `outputs/`** (the thesis numbers depend on it). Write new runs to separate `outputs_dir`s. Filtered runs overwrite the combined outputs of their own `outputs_dir`.
- Distinguish established results, assumptions, and hypotheses. Verify any literature claim against the primary source before it goes into the thesis.

## 1. Existing benchmark (what the stored outputs contain)

- Tasks (11, univariate, H=48 unless noted): `m4_hourly` (414 series), `ett_h1_h48`, `ett_h1_h96` (H=96), `ett_h2_h48`, `ett_m1_h48`, `ett_m2_h48` (7 series each), `ecl_h48` (321), `traffic_h48` (862), `weather_h48` (21), `exchange_h48` (8), `ili_h48` (7). Seasonal lags: 24 (hourly), 96 (15-min), 144 (10-min), 7 (Exchange; MASE uses lag 1), 52 (ILI).
- Protocol: 5 non-overlapping rolling origins, stride = H. Window 0 is the final holdout; window k ends kH steps before the series end. Models see only data before their window.
- Models: TimesFM 2.5 (ctx 1024), Chronos-2 (ctx 8192, univariate, no cross-learning), TabPFN-TS (v2 checkpoint, ctx 4096), AutoARIMA (skipped for s>52), AutoETS, AutoTheta, Prophet, pooled linear regression, SeasonalNaive.
- Stored files:
  - `outputs/forecasts/all_datasets_all_models.parquet`, columns `dataset, model, unique_id, horizon, ds, y_true, y_pred, window, forecast_horizon`. Point forecasts (medians) only; quantiles were discarded.
  - `outputs/metrics/mase_scales.csv` (`dataset, window, unique_id, mase_scale`) holds the per-series, per-window MASE denominators.
  - `outputs/timesfm_tuning/` holds the TimesFM context sweep (512…16256) and flag ablation on ETTh1/ETTm1/Weather/M4.
- Headline (geometric mean over tasks of task MASE / SeasonalNaive task MASE): TimesFM 0.748, Chronos-2 0.785, TabPFN-TS 0.840, AutoARIMA 0.926 (8 tasks), AutoTheta 1.163, AutoETS 1.212, Prophet 1.451, linear regression 1.757. TimesFM's lead rests largely on ILI (rel. MASE 0.275 vs 0.84/0.80). Without ILI the order is Chronos-2 0.780, TimesFM 0.827, TabPFN-TS 0.844.

## 2. Findings from the post-hoc analysis of stored forecasts

All numbers come from the stored forecasts; nothing was re-run. FM = {TimesFM 2.5, Chronos-2, TabPFN-TS}. "Relative MASE" = task MASE / SeasonalNaive task MASE, aggregated by geometric mean over the 11 tasks. Definitions are in §2.8 so the analysis can be reproduced. The analysis scripts lived in a temporary scratchpad and are not in the repo, so re-implement them from §2.8.

### 2.1 FM errors are highly correlated, so naive averaging gains little
- Pearson correlation of signed scaled errors between FM pairs is about 0.8–0.97 on most tasks.
- Exceptions:
  - ETTm1: TabPFN-TS vs others ≈ 0.41–0.44.
  - ILI: TimesFM vs others ≈ 0.08–0.16.
- Equal-weight mean of the 3 FMs: 0.761, worse than TimesFM's 0.748. Median of the 3: 0.772. TimesFM+Chronos-2 mean: 0.746.

### 2.2 Large ex-post headroom, but not predictable per series
- Ex-post oracle (pick the best FM per series and window): 0.605, against 0.748 for the best single FM. Adding SeasonalNaive, AutoETS and AutoTheta to the oracle pool lowers it further: 0.632 → 0.594 → 0.567 on windows 0–3.
- **Winner persistence is close to chance.** P(same FM wins a series in windows k and k+1) vs the chance level from marginal win shares:

  | Task | Persistence | Chance |
  |---|---|---|
  | ECL | 0.37 | 0.35 |
  | M4 | 0.42 | 0.34 |
  | Traffic | 0.41 | 0.34 |
  | Weather | 0.45 | 0.38 |
  | ETTh1/48 | 0.29 | 0.37 |
  | ETTh2 | 0.21 | 0.34 |
  | ILI | 0.89 | 0.84 |

- Interpretation: most of the series-level oracle gap is window-specific noise.
- Realistic per-series selection on windows 0–3, choosing each series' FM by mean MASE on all earlier windows: 0.765, against TimesFM 0.799 and Chronos-2 0.829 on the same windows. Excluding ILI it reaches 0.844, worse than Chronos-2's 0.828.

### 2.3 The usable signal is at task × horizon level
Combiners are evaluated on windows 0–3, with weights for window k fitted only on windows k+1..4:

| Method | All 11 tasks | Excl. ILI |
|---|---|---|
| TimesFM | 0.7985 | 0.8933 |
| Chronos-2 | 0.8290 | 0.8280 |
| Equal mean of 3 FMs | 0.8054 | 0.8374 |
| Task-level selection (best past FM per task) | 0.7531 | 0.8375 |
| Inverse past-MASE^4 weights (per series) | 0.7521 | – |
| NNLS weights per task, 1 horizon bin | 0.7384 | 0.8193 |
| **NNLS weights per task, 3 horizon bins** | **0.7321** | **0.8040** |
| NNLS weights per task, 6 horizon bins | 0.7373 | 0.8025 |

### 2.4 The benefit is robustness (regret), not typical-case gain
- Per task-window (44 pairs), the 3-bin NNLS stack against:
  - TimesFM: wins 64%, Wilcoxon p = 0.065;
  - Chronos-2: wins 43%, p = 0.70;
  - TabPFN-TS: wins 77%, p = 2e-6;
  - equal mean: wins 57%, p = 0.23.
- Regret = method's task rel. MASE / that of the ex-post best single FM on that task (windows 0–3):

  | Method | Max regret | Mean regret | Tasks with regret > 1.10 |
  |---|---|---|---|
  | TimesFM | 1.53 | 1.12 | 4 |
  | Chronos-2 | 3.22 | 1.25 | 3 |
  | TabPFN-TS | 3.07 | 1.30 | 6 |
  | Equal mean | 2.10 | 1.15 | 4 |
  | NNLS, 3 horizon bins | **1.10** | **1.02** | 1 |

### 2.5 Adding statistical models to the combination pool makes it worse (at 3–4 past windows)
- NNLS with 3 horizon bins:

  | Pool | Geomean | Excl. ILI | Max regret | Mean regret |
  |---|---|---|---|---|
  | FM only | 0.732 | 0.804 | 1.10 | 1.02 |
  | FM + SeasonalNaive | 0.745 | 0.813 | 1.20 | 1.04 |
  | FM + SeasonalNaive + AutoETS + AutoTheta | 0.779 | 0.853 | 1.31 | 1.09 |

- This holds even though the statistical models' errors often correlate less with the FMs. Mean correlation with the FMs: Theta 0.24 on M4, SeasonalNaive 0.29 on Weather and 0.31 on ETTm1.
- Hypothesis: there are too few past windows to estimate the extra weights, and the extra members are individually weak (the forecast-combination puzzle). This is to be retested with 20 origins.

### 2.6 Failure mode: late-horizon degradation and amplitude damping on high-frequency tasks
- Scaled error relative to SeasonalNaive, by horizon third (early / mid / late):
  - ETTm1, TimesFM: 0.38 / 0.93 / 2.68
  - ETTm1, Chronos-2: 0.36 / 0.87 / 2.17
  - Weather, TimesFM: 0.30 / 0.70 / 1.10
- Median ratio of within-window forecast std to actual std:
  - Weather: TimesFM 0.25, Chronos-2 0.42, TabPFN-TS 0.51
  - ETTm2: 0.33 / 0.39 / 0.42
  - ETTh2: 0.52 / 0.54 / 0.69
  - SeasonalNaive is about 0.8–1.15.
- Caveat: with stride = H = 48, horizon position is partly confounded with time of day (windows 0, 2, 4 and windows 1, 3 cover different halves of the day on 15-min data).
- The TimesFM context sweep is stable at task level (Weather prefers long contexts, ETTm1 short ones) but not per window.

### 2.7 No evidence that data-quality issues drive FM errors in this benchmark
- ECL: 30/321 series contain zero-filled runs (≥3 zeros) in their training history. Their median rel. MASE is not worse (Chronos-2 0.59 with vs 0.61 without; Mann–Whitney p = 0.37).
- The only clear data-related effect is metric-driven: zero-inflated Weather variables under sMAPE (already in the thesis).

### 2.8 Definitions used
- Series-window MASE = mean over horizon steps of |ŷ−y| / mase_scale(i,k).
- Task MASE = mean over series-windows.
- Relative = divided by the SeasonalNaive task MASE on the same windows.
- Aggregate = geometric mean over tasks.
- NNLS stack:
  - per task and per horizon bin (bins = equal splits of 1..H);
  - fitted with `scipy.optimize.nnls` on all rows (series × steps) of past windows, with both the member forecasts and y divided by mase_scale;
  - weights renormalized to sum to 1; equal weights if all weights are zero.
- Inverse-MASE weights: per series, w_m ∝ (mean past series-window MASE of m)^(−p).
- Winner persistence: for k = 0..3, the share of series whose argmin-MASE FM is the same in windows k and k+1, averaged over k. Chance = Σ(marginal win share)².
- Error correlation: Pearson on signed scaled errors pooled per task.

## 3. Thesis direction

**Working title:** *When and why does combining zero-shot time-series foundation models help? Diversity, weighting granularity, and robustness.* Optional second arm: *adapt or combine?*, comparing Chronos-2 LoRA fine-tuning with zero-shot combination.

**Rationale:**
- The benchmark shows complementary failures: TimesFM is strong on ILI, Chronos-2 on ECL/Weather/Traffic, and TabPFN-TS on ETTm1, while each fails badly somewhere.
- Those failures are unpredictable at series level but exploitable at task and horizon level.
- The work needs almost no GPU, its outputs can be iterated on cheaply, and it is contamination-tolerant: it compares models relative to one another on identical data. Contamination can still bias learned weights, hence the external check in Phase D.
- Under the "transfer learning" thesis title it frames combination as pooling knowledge transferred from heterogeneous pretraining (real corpora vs synthetic-only for TabPFN-TS). The fine-tuning arm keeps a literal transfer-learning component.

**Research questions:**
1. **Diversity.** How correlated are TSFM errors, and does diversity (ambiguity decomposition) explain per-task combination gains? Does adding low-correlation statistical members help once weights are well estimated?
2. **Granularity.** At which level should combination weights be learned: global, task, task × horizon, or series? Can series-level selection be rescued by a feature-based meta-learner (FFORMA-style, leave-one-dataset-out), or is the series-level signal noise?
3. **Data requirements.** How many past origins are needed before learned weights beat equal weights (the forecast-combination puzzle)?
4. **Robustness.** Is the main benefit a reduction in regret and worst-case error? Test with Diebold–Mariano tests and bootstrap confidence intervals.
5. **Probabilistic pooling.** Quantile averaging (Vincentization) vs linear (mixture) pooling of TSFM predictive distributions, evaluated by WQL and coverage.
6. **Failure modes.** Does combination fix the late-horizon degradation and amplitude damping of §2.6?
7. **Optional: adapt vs combine.** Does LoRA fine-tuning of Chronos-2, using only pre-test data, beat zero-shot combination?

**Directions considered and not chosen as the main line:**
- Test-time adaptation / improving a frozen TSFM: crowded in 2025–26 (ORCA, GITCO, PaCTS, TimesFM-ICF, RACE), with typical gains of 1–6% and disputed protocols.
- Fine-tuning: feasible on Chronos-2, but studies report about 2–3% LoRA gains, full fine-tuning hurting, and small datasets favouring zero-shot; contamination muddies evaluation. Kept as the optional arm.
- Data issues: genuinely open (native NaN handling vs impute-then-forecast), but not motivated by this benchmark (§2.7). This is the fallback direction if the professor prefers it.

## 4. Verified technical facts (installed environment)

- **GPU.** `nvidia-smi` fails ("Driver/library version mismatch"), but torch CUDA works: torch 2.12.1+cu130, RTX A5000; TimesFM and Chronos-2 run on cuda.
- **TimesFM** (`timesfm` 2.0.1, checkpoint `google/timesfm-2.5-200m-pytorch`, revision `1d952420…`):
  - `forecast()` returns `(point, quantiles)`, with quantiles of shape (N, H, 10): channel 0 = mean, channels 1–9 = q0.1…q0.9, point = channel 5 (median).
  - The repo discards the quantiles (`src/timestransfer/models.py` ~L244–247).
  - No training code in the installed package. Upstream added a LoRA example (HF Transformers `TimesFm2_5ModelForPrediction` + PEFT, April 2026). TimesFM 3.0 has been released (August 2026) and is not used here.
  - NaNs: leading NaNs are stripped and interior/trailing NaNs linearly interpolated (no learned mask).
  - **Batching:** the runner uses the default `per_core_batch_size` (1). Passing `forecast_config_overrides={'per_core_batch_size': 64}` cut ECL (321 series, ctx 1024) from 31.4 s to 3.6 s.
- **Chronos-2** (`chronos-forecasting` 2.3.2, revision `29ec3766…`):
  - Trained on 21 quantile levels (0.01, 0.05, 0.1…0.9, 0.95, 0.99); `predict_quantiles(quantile_levels=...)`. The repo requests only [0.5] (`models.py` ~L686–690).
  - Native NaN masking.
  - Fine-tuning: `Chronos2Pipeline.fit(inputs, prediction_length, validation_inputs=None, finetune_mode="full"|"lora", lora_config=None, context_length=None, learning_rate=1e-6, num_steps=1000, batch_size=256, ...)`. LoRA needs `peft` and otherwise falls back to full fine-tuning; the docstring suggests lr ≈ 1e-5 for LoRA.
- **TabPFN-TS** (`tabpfn-time-series` 1.3.0):
  - `predict_df(..., quantiles=[...])`; the repo requests only [0.5] (`models.py` ~L116).
  - Drops NaN-target rows; the running-index and periodic features count rows, not time.
  - `tabpfn_ts_v3p5` in `configs/experiment.yaml` needs a license token; keep it disabled.
- **Harness:**
  - `configs/*.yaml` `models:` entries are named variants; `runner:` selects a function in `MODEL_RUNNERS` (`runner.py` ~L197–224).
  - Per-dataset `n_windows` and `series_limit` overrides are supported (`runner.py` ~L258–264), and `project.outputs_dir` redirects all outputs.
  - Runners return long frames (`dataset, model, unique_id, horizon, ds, y_true, y_pred`). Extra columns pass through to the parquet files.
  - An in-loop ensemble runner would need a refactor; do combination **post hoc** on saved forecasts.
- **Measured timings** (single window, includes some warm-up, ±2×):

  | Model | Measurement | Per window, all 11 tasks (1,668 series) |
  |---|---|---|
  | TimesFM, batch 64 | 3.6 s for 321 series | ~20 s |
  | Chronos-2, ctx 8192 | 6.5 s for 321 series | ~35 s |
  | TabPFN-TS, ctx 4096 | ~3 s per series | ~1.4 h |
  | TabPFN-TS, ctx 1024 | ~0.9 s per series | ~25 min |

  - SeasonalNaive is negligible. AutoETS/AutoTheta were not measured.
  - The original full run took about a day, dominated by AutoARIMA on ECL/Traffic, which is not needed here.

## 5. Implementation and run plan

### Phase A — prepare runs (week 1)
- **A1. Quantile outputs.** Extend `run_timesfm_2p5`, `run_chronos2` and `run_tabpfn_ts` to save q0.1…q0.9 as extra columns (e.g. `q10…q90`) alongside `y_pred` (the median):
  - Apply the same nonnegative clipping as for the median.
  - Enforce monotone quantiles.
  - Keep existing outputs bit-compatible when quantiles are disabled.
  - Add unit tests.
- **A2. Probabilistic metrics** in `metrics.py`: scaled weighted quantile loss (WQL, as in Chronos/GIFT-Eval) and empirical coverage of the 80% interval (q10–q90). Add tests.
- **A3. New config `configs/combination.yaml`:**
  - `outputs_dir: outputs/combination`;
  - `n_windows: 20`, with dataset overrides of 10 for `m4_hourly` (shortest training length 700 at window 0) and `ili_h48` (918);
  - models: `timesfm_2p5` (with `per_core_batch_size: 64`), `chronos2`, `tabpfn_ts`, `seasonal_naive`; all others disabled.
- **A4. Smoke run:** `uv run python scripts/run_benchmark.py --config configs/combination.yaml --series-limit 4`.
  - Done 2026-10-03 with `--outputs-dir outputs/combination_smoke` in 63 min wall time, almost all of it TabPFN-TS. All 4 models succeeded on all windows (20; 10 for M4 and ILI).
  - Windows 0–4 against `outputs/forecasts/`:
    - SeasonalNaive and TabPFN-TS are bit-identical.
    - TimesFM (now with batch 64) and Chronos-2 differ by at most 3.7e-5 in MASE units (float32/batching noise).
  - Quantiles: none missing; monotone after sorting.
  - `y_pred` keeps the raw median. For Chronos-2 this means `q50` (sorted) differs from `y_pred` on 0.9% of rows (max 0.07 MASE units), where its raw quantiles cross.
- **Run R1:**
  - **R1a:** TimesFM + Chronos-2 + SeasonalNaive, about 30 min.
  - **R1b:** TabPFN-TS overnight in a *separate* `outputs_dir`, because filtered runs overwrite combined outputs: about 28 h full, or about 11 h with `series_limit: 100` for ECL and Traffic. The full-vs-capped choice is open for the user to decide.
  - **Acceptance check:** windows 0–4 of R1 reproduce the stored forecasts in `outputs/forecasts/` up to float tolerance (batching may introduce tiny differences; report them).
  - **Done 2026-10-05** as one run (no R1a/R1b split) of `configs/combination.yaml` at commit `8b2f975` into `outputs/combination`.
    - Wall time 22.3 h, of which TabPFN-TS took 22.0 h.
    - All models ok on all tasks and windows; 5.62M forecast rows; FM quantiles complete and monotone.
  - **Acceptance check passed** (windows 0–4, all series):
    - SeasonalNaive, Chronos-2 and TabPFN-TS are bit-identical to `outputs/forecasts/`.
    - TimesFM (batch 64 instead of 1) differs by at most 2.4e-4 in MASE units (mean 2e-6).

### Phase B — combination module (week 2)
- **B1. `src/timestransfer/combination.py`:**
  - Load one or more forecast parquet files into a wide table: (dataset, unique_id, window, horizon) × member, plus y and mase_scale.
  - Rolling protocol: weights or selections for window k are fitted only on windows > k (strictly earlier data), with a configurable minimum number of past windows (default 3) and an optional cap on the number of past windows.
- **B2. Combiners:**
  - equal mean; median;
  - task-level best-member selection; series-level selection (expanding and last-window);
  - inverse past-MASE weights (power p ∈ {1, 2, 4}, optional top-k);
  - NNLS weights at four levels: global, task, task × horizon bins ({1, 3, 6} bins), series;
  - optional shrinkage of learned weights toward equal weights;
  - feature-based meta-learner (FFORMA-style: features from `src/timestransfer/series_features.py`, LightGBM or multinomial logistic regression predicting softmax weights), evaluated leave-one-dataset-out.
- **B3. Script and tests.** `scripts/run_combination.py` writes combined forecasts in the standard long format and computes metrics with the existing metric functions into `outputs/combination/metrics/`. Tests:
  - toy-data correctness;
  - **no leakage:** perturbing y in windows ≤ k must not change the weights used for window k;
  - equal-mean reproduction.
- **Run R2:** CPU only, minutes.
  - **Implemented 2026-10-07.** Module `src/timestransfer/combination.py`, script `scripts/run_combination.py`, tests in `tests/test_combination.py`.
    - The meta-learner is a linear softmax model trained on FFORMA's objective (no LightGBM). Its training error per series-window is each member's MASE divided by the mean over members.
    - Its features are the `series_features.py` set without the scale-dependent mean and std. They are computed on the last 2,048 points of each window's history (the full history takes several times longer) and cached in `outputs/combination/combiners/series_features.parquet`, which took 9 min with 16 processes.
    - Outputs go to `outputs/combination/combiners/<name>/`, not `outputs/combination/metrics/`, so the R1 metrics are not overwritten.
  - **First R2 run** (FM pool, min 3 past windows, no cap): evaluated windows 0–16 (0–6 for M4 and ILI); combiners took about 35 s.
    - Geometric-mean relative MASE: nnls_task 0.633, inv_mase_p4 0.633, nnls_task_h3 0.634, TimesFM 0.663, equal mean 0.682, Chronos-2 0.702.
    - Descriptive only: no significance tests or regret analysis yet (C4).

### Phase C — analysis (weeks 3–4)
- **C1. Diversity.** Error-correlation matrices per task. Ambiguity decomposition (exact for squared error: ensemble MSE = weighted mean member MSE − weighted mean squared deviation from the ensemble). Relate diversity to combination gain per task.
- **C2. Granularity.** Compare global, task, horizon and series weighting; winner persistence with 20 origins; meta-learner vs task-level weights.
- **C3. Combination puzzle.** Learning curve of learned vs equal weights as a function of the number of past windows (1, 3, 5, 10). Retest pools that include SeasonalNaive and, optionally, AutoETS/AutoTheta (run them only for this check, on a subset if slow).
- **C4. Robustness and inference.**
  - Regret vs the ex-post best member per task (max and mean).
  - Diebold–Mariano tests per task (pairwise vs the best single model and vs the equal mean), on series-averaged loss per origin, with an autocorrelation-robust variance.
  - Bootstrap over series (stratified by task) for confidence intervals on the aggregate relative MASE.
  - Report multiple-comparison handling.
- **C5. Failure modes.** Horizon-resolved error vs SeasonalNaive and the amplitude ratio per method. Control for time-of-day by grouping by target timestamp phase as well as horizon step.

- **Status 2026-10-07.**
  - Code: `src/timestransfer/combination_analysis.py`, `scripts/run_combination_analysis.py`, tests in `tests/test_combination_analysis.py`. Outputs in `outputs/combination/analysis/`.
  - Inference choices:
    - DM tests use window-level task MASE, Newey–West variance and the HLN small-sample factor, with Holm adjustment across tasks.
    - Bootstrap is stratified by task and reported three ways: resampling series, windows, or both.
  - **Findings that supersede §2 (5-window analysis):**
    - Horizon bins do not beat plain task-level NNLS (0.634/0.635 vs 0.633).
    - The ETTm1 TabPFN-TS low-correlation "exception" disappears with 20 windows (0.77–0.82).
    - Do not cite §2 numbers in the thesis.
  - AutoETS/AutoTheta forecasts: `outputs/combination_stats/` (run with `--models seasonal_naive auto_ets auto_theta`). Rerun the analysis script when it finishes; it adds the statistical pools automatically.

### Phase D — external validity (week 4, runs in the background)
- **D1.** Loader for a small subset of GIFT-Eval *test* datasets at hourly and sub-daily frequency; the Chronos-2 paper states that its corpus excludes the GIFT-Eval test portions. Choose the subset with the user.
- **Run R3:** the same four members, a few hours (TabPFN-TS dominates).
- **D2.** Repeat C1–C4 on the subset.

### Phase E — probabilistic combination (week 5)
- **E1.** Vincentization (average quantiles per level) vs linear pooling (mixture of member CDFs; e.g. interpolate each member's quantile function, sample, and re-estimate quantiles), with equal and learned weights. Metrics: WQL, coverage, interval width. CPU only.

### Phase F — optional: adapt vs combine (weeks 5–6; drop if Phase C slips)
- **F1.** Chronos-2 LoRA via `Chronos2Pipeline.fit`:
  - Train per task (or pooled) only on data before the earliest evaluation window's cutoff; validate on a held-out pre-test slice.
  - Evaluate on the same windows as the combiners.
  - Report GPU time next to accuracy.
  - Estimated cost: tens of minutes per fine-tune, a few hours in total; measure first.
- **Run R4.**

### Phase G — thesis writing (weeks 6–8)
- Experiments 1–3 become the motivating chapter.
- New chapters: combination method and protocol; results for RQ1–RQ6 (and RQ7 if done); discussion and limitations (contamination, dependent windows and series, the horizon/time-of-day confound).
- Follow `/home/ubuntu/thesis/AGENTS.md` and the existing table/figure pipeline (`scripts/build_thesis_artifacts.py`, `scripts/check_thesis_numbers.py`).

### Compute summary
- GPU: about 1–2 days unattended, almost all TabPFN-TS (R1b, R3), plus a few hours for the optional fine-tuning.
- Everything else: minutes per iteration.

## 6. Risks and open decisions
- **TabPFN-TS runtime.** Choose full panels vs a 100-series cap on ECL/Traffic, or ctx 1024. It is the most diverse member, so drop it only as a last resort.
- **Contamination.** M4 and the ECL source collection are in the pretraining lists of TimesFM v1 and Chronos-2; GIFT-Eval Pretrain overlap with the panels is unmapped. Learned weights could favour a member that memorized the data; Phase D is the guard.
- **Dependence.** Origins and series are not independent; use the dependence-aware tests of C4 and phrase claims accordingly.
- **Time-of-day confound** in horizon analyses (§2.6).
- **Open decisions for the user:** TabPFN-TS budget; GIFT-Eval subset; whether to include Phase F.

## 7. Literature (verification status noted)

- **Checked first-hand on arXiv abstract pages (2026-10-03):**
  - Ning et al., TimeRouter, arXiv 2606.11625;
  - Bosch et al., Multi-layer Stack Ensembles for Time Series Forecasting, arXiv 2511.15350;
  - Olivares et al., A More Realistic Evaluation of Cross-Frequency Transfer Learning and Foundation Forecasting Models, arXiv 2509.19465 (statistical ensembles beat FFMs by >20% MASE under their protocol).
- **Located on arXiv by a research subagent.** Re-verify the abstract and any number before citing.
  - **Combination / selection:**
    - Kayaalp et al., Test-Time Efficient Pretrained Model Portfolios, arXiv 2510.06419;
    - Shi et al., ZooCast (One-Embedding-Fits-All), arXiv 2509.04208;
    - Khwaja et al., Toto 2.0 (FFORMA-style "FnF" ensemble over 10 TSFMs), arXiv 2605.20119;
    - Garza & Rosillo, TimeCopilot, arXiv 2509.00616;
    - GIFT-Eval PR #219 (equal-weight quantile averaging of 4 TSFMs);
    - Shchur et al., AutoGluon-TimeSeries, arXiv 2308.05566;
    - Brini, arXiv 2607.05291 (TSFM + statistical model complementarity);
    - Yao et al., TimeTic, arXiv 2509.23695.
  - **Classical combination:**
    - Montero-Manso et al., FFORMA, IJF 2020, doi:10.1016/j.ijforecast.2019.02.011;
    - Wang, Hyndman, Li & Kang, Forecast combinations: an over 50-year review, arXiv 2205.04216.
  - **Test-time adaptation:** ORCA arXiv 2606.14222 (code: github.com/Fifthky/ORCA), RACE 2610.00405, FAC 2605.17250, TAFAS 2501.04970, GITCO 2606.05332, PaCTS 2609.34786, TimesFM in-context fine-tuning 2410.24087.
  - **Fine-tuning:** Laglil et al., arXiv 2607.23146; MixFT, arXiv 2603.02840.
  - **Data quality:** TS-Fault arXiv 2606.18539; Noisy periodic series arXiv 2501.00889; Time-indexed FMs for imputation arXiv 2511.05980.
  - **Failure modes:** Operational viability arXiv 2605.24381; Forecast collapse arXiv 2608.14106; Non-stationarity in embedding space arXiv 2604.16428; Jander et al. arXiv 2608.24303.
- **Gap (as far as the search found).** The combination literature is method-heavy. None of the papers found analyses TSFM error diversity, weighting granularity with time-respecting validation, horizon dependence, regret, or quantile vs mixture pooling for TSFMs. Bosch et al. find learned stacking beats averaging, but with diverse base models and larger validation sets than here.
