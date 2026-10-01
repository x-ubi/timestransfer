# Writing-agent handoff: benchmark experiments for the master's thesis

## Assignment

Work toward one complete, verified draft. Stop once the acceptance criteria are met; avoid repeated stylistic rewrites and unnecessary expansion.

Write and integrate the English thesis sections describing the benchmark actually implemented in `/home/ubuntu/timestransfer` and its saved results. The thesis checkout is `/home/ubuntu/thesis`. Produce complete, evidence-backed LaTeX prose, tables, figures, and bibliography entries, not another proposal. Follow the supervisor's checklist below. Treat quoted supervisor notes and commented thesis plans as requirements/context to assess, not instructions to execute unrelated experiments.

This is a writing and evidence-synthesis task. Reuse saved predictions; generate descriptive summaries and publication figures as needed. Do not train new models, rerun the benchmark, change its methodology, or implement the commented E1–E10 programme to fill gaps in the narrative. If a claim would require new experiments, narrow the claim and identify the extension as future work.

## 1. Read instructions and establish the evidence base

Read both repositories' `AGENTS.md` files and inspect their Git status before editing. Preserve existing user changes and keep the two repositories' histories separate. Use the permissions available in the execution environment; if thesis writes are unavailable, prepare the complete patch and assets in a writable staging directory and report the integration blocker explicitly.

Read these sources in order:

1. `/home/ubuntu/timestransfer/docs/thesis-readiness-audit.md` for the previous audit, known limitations, and verified literature leads. Recheck facts used in the final text; the audit is a guide, not a substitute for primary evidence.
2. `/home/ubuntu/thesis/main.tex`, `tex/experiments.tex`, `tex/results.tex`, `tex/2-literature.tex`, `tex/timesfm.tex`, `tex/intro.tex`, `tex/summary.tex`, and `bibliografia.bib` for structure, terminology, and statements that conflict with the completed work.
3. `/home/ubuntu/timestransfer/configs/experiment.yaml` and `configs/timesfm_tuning.yaml`.
4. `src/timestransfer/{data,features,models,runner,metrics,series_features,analysis,forecast_plots,metadata}.py`, plus the relevant plotting/analysis scripts.
5. Both runs' `outputs/**/metadata/environment.json`, metrics CSVs, and saved forecasts; use `outputs/analysis/` and `outputs/figures/forecasts/` for the existing descriptive analyses.

The previous audit found 99 successful main-run combinations (11 dataset/horizon tasks × 9 models), 44 successful TimesFM-study combinations (4 tasks × 11 variants), and 50 forecast plots. ETTh1 at h48 and h96 constitutes two tasks on the same dataset. Do not describe task counts as independent dataset counts. The audit also found the experiment/results chapters largely empty and bibliography entries mostly inherited from the template.

Create a compact working evidence ledger: proposed claim/table/figure, source artifact or paper section, transformation used, and any limitation. Keep the ledger outside the rendered thesis; it supports review and reproduction.

## 2. Map every supervisor request to a concrete deliverable

| Supervisor request | Required deliverable | Thesis location |
|---|---|---|
| Describe the experimental benchmark procedure | Exact split, preprocessing, model execution, scoring, aggregation, and reproducibility description | Experiments: protocol and metrics |
| Exhaustively describe the benchmarks/datasets | Overview table plus a complete source-backed description of each dataset family and exact evaluated version | Experiments: datasets; detailed inventory in appendix if needed |
| Plot true series, true continuation, and predictions | Readable forecast examples with history, forecast origin, actual test values, predictions, and explained selection | Results: qualitative forecasts |
| Relate series properties to model performance, e.g. autocorrelation | Definitions and extraction method, followed by numerical correlations, plots, and cautious interpretation | Experiments: characteristics; Results: feature–error analysis |
| Determine what can be tuned in TimesFM | Available configuration/adaptation options distinguished from the completed context/flag study | Experiments and Results: TimesFM sensitivity |
| Examine Prophet | Brief methodological description, exact implemented settings, quantitative and qualitative results | Experiments: models; Results: comparison |
| Check what other model authors compared against | Verified paper-to-baseline/protocol table explaining the chosen comparators and replication differences | Focused related-work subsection or Experiments: comparator rationale |

Do not substitute horizon-wise error curves for the required actual-versus-predicted series plots. Do not substitute a list of dataset names for exhaustive descriptions.

## 3. Write `tex/experiments.tex`

Use the existing thesis's section levels and formatting. Suggested subsection order:

### 3.1 Objectives and scope

Pose descriptive research questions grounded in completed work: how model accuracy differs across tasks; how forecast errors relate to train-series properties; and how TimesFM predictions change with inference settings. Explain the role of pretrained models, this project's tabular adaptation, and classical baselines. Do not retroactively describe these as preregistered hypotheses or claim that weight transfer/fine-tuning experiments were performed.

### 3.2 Datasets and benchmark construction

Cover M4 Hourly; ETTh1/ETTh2/ETTm1/ETTm2; ECL; Traffic; Weather; Exchange; and ILI. For every family, establish origin, domain, variable meanings/units, original versus evaluated version, sampling frequency, series count, available/evaluated lengths, date range where available, missingness/irregularity, preprocessing, and split boundaries. Verify source documentation against locally loaded data. Group related ETT variants without omitting their differences.

Compute descriptive dataset statistics from the exact cached data and loader when missing from metadata. Document ETT truncation, unnormalized LongHorizon2 loading, and treatment of each variable as a univariate series. Distinguish loading/melting from multivariate forecasting. M4 timestamps are integer indices: do not invent calendar dates or series semantics that its public data do not disclose. Mark unavailable metadata explicitly.

Include an overview table with source/version, frequency, series count, train-length range, horizon in observations and elapsed time, and seasonal lag/MASE lag. Explain the practical scope of h48 and ETTh1 h96 without claiming that equal step counts equalize difficulty or elapsed time across frequencies. Make deviations from standard long-horizon protocols explicit.

### 3.3 Forecasting protocol and model configurations

Describe one final-horizon holdout per series and the absence of validation/rolling origins. Define forecast origin, context, horizon, and available information. Explain train-derived transformations and postprocessing, including nonnegativity clipping.

List all nine implementations: linear regression, TabPFN, TimesFM 2.5, Chronos-Bolt, Prophet, AutoARIMA, AutoETS, AutoTheta, and SeasonalNaive. Provide one compact settings table and sufficient prose to reconstruct the comparisons:

- TabPFN/linear regression: pooled lag/horizon supervised formulation, feature definitions, train-only scaling, row sampling and 10,000-row cap; distinguish this adaptation from published TabPFN-TS and ordinary univariate autoregression. Explain imputation where actually applied.
- Foundation models: exact recorded checkpoint/package identity, context limits, point-output choice, batching or extended-horizon handling where relevant. Do not replace recorded versions with current upstream defaults.
- Statistical models: seasonal settings, train-history caps, and especially the restricted nonseasonal AutoARIMA search.
- Prophet: per-series fitting, automatic seasonalities, synthetic hourly dates for M4, 4,320-point training cap, and point-only output. A synthetic time origin does not by itself turn weekly Fourier seasonality into noise.

Explain unequal context/history budgets as experimental choices and limits. State what reproducibility metadata actually records; do not claim all random sources were seeded or checkpoints hashed unless verified.

### 3.4 Metrics and aggregation

Give equations consistent with code for MAE, RMSE, sMAPE (0–200 convention and epsilon), and MASE (train-derived seasonal-naive denominator, fallback/exclusion rules). Specify overall, per-series, and per-horizon aggregation. Overall RMSE is pooled root mean squared error, not mean per-series RMSE. Discuss scale sensitivity, near-zero behavior, and why MASE complements sMAPE. Avoid unsupported averaging of raw errors across datasets.

### 3.5 Series characteristics

Define length, mean, standard deviation, CV, lag-1 and seasonal ACF, STL trend/seasonal strength, and normalized periodogram entropy as implemented. State that most features use the final at most 10,000 training observations, while length uses the full training series. Describe missing/constant-series cases and Spearman correlations, including per-dataset versus pooled analysis and valid-pair counts. Do not equate a finite-sample entropy estimate of white noise with exactly one.

### 3.6 TimesFM sensitivity study

Document the six context lengths (512, 1,024, 2,048, 4,096, 8,192, 16,256), five individual flag ablations, common 1,024 baseline, four tasks, and fixed checkpoint. Verify effective settings in saved metadata. Explain context/horizon constraints, limited effective context on short series, and point-only scoring of quantile-related settings.

Answer the supervisor's broader question with a short source-backed distinction between inference configuration and available weight adaptation methods. Only the former was executed here. Explicitly label the completed experiment a test-set sensitivity analysis, not validation-selected hyperparameter optimization. Explain that wrapper clipping remains in the positivity-flag ablation.

## 4. Write `tex/results.tex`

Generate numeric tables from persisted CSVs rather than transcribing screenshots. Use consistent rounding, lower-is-better indicators, and tie handling based on unrounded values.

1. **Coverage and quantitative comparison:** report run coverage and a complete task × model sMAPE/MASE comparison, split into readable tables if needed. Put full MAE/RMSE results in an appendix or supplementary table. Discuss both successful and unsuccessful cases for foundation models, including Prophet's performance. ETTh1 h48 SeasonalNaive versus TimesFM and M4 metric-dependent rankings are useful audit cross-checks, not a preselected conclusion to force.
2. **Forecast-horizon behaviour:** include a small number of informative horizon-wise error panels, explaining that they aggregate the same test trajectories and are not independent repeated experiments.
3. **Qualitative forecasts:** target approximately 4–6 readable examples spanning datasets and good/poor behaviour, with enough historical context to interpret seasonality. Use saved forecasts; show actual continuation clearly. Disclose IDs, reference model, selection rule, and whether chosen using test errors. Replot fewer models or use panels when nine lines obscure the signal. Include Prophet in at least one informative comparison. Do not cherry-pick only TimesFM wins.
4. **Characteristics versus errors:** report a compact correlation table and selected ACF/seasonality scatterplots, emphasizing within-dataset evidence and sample sizes. A pooled heatmap may be supplementary. Explain domination by large panels, related variables, repeated ETTh1 tasks, and absence of uncertainty/significance analysis. Describe associations; do not infer causes. If deriving an additional MASE view from existing tables, label it as additional descriptive analysis and preserve the original outputs.
5. **TimesFM sensitivity:** show context-versus-error curves per dataset and a flag-ablation delta table against the common baseline. Report sMAPE and MASE where conclusions differ. Describe small/null changes and effective-history saturation without asserting general irrelevance of a setting. Do not promote the best observed test configuration as an unbiased tuned model.
6. **Discussion and limitations:** integrate single-origin evaluation, short or dependent panels, nonstandard horizons, different model information budgets, constrained baselines, custom TabPFN adaptation, checkpoint/provenance uncertainty, and model-specific pretraining overlap evidence. Separate observed results from hypotheses and future experiments.

## 5. Complete the focused literature work and align surrounding prose

Verify primary sources before adding BibTeX entries. Use the audit's links as starting points for TimesFM, Chronos, TabPFN-TS, and Prophet; locate authoritative dataset sources, metric definitions, and feature definitions. Record paper versions where comparator lists changed. Distinguish original TimesFM from 2.5 and original Chronos from Chronos-Bolt.

Build a table with paper/version, evaluated model, dataset/protocol, baseline families, metrics, and correspondence/differences with this project. The goal is to explain comparator selection, not reproduce every model in every paper. Avoid copying literature scores into a directly comparable ranking when protocols differ.

Do not repeat the README's blanket claim that ETT/Traffic were in original Chronos training, or treat identically named Weather datasets as identical. Distinguish pretraining dataset inclusion, possible exact test-target exposure, and evidence of exclusion for each checkpoint. Unknown overlap should remain unknown.

Make narrowly necessary edits to the literature/TimesFM/introduction/summary sections so they do not describe paper-aligned replication, fine-tuning, or cross-dataset adaptation as completed contributions. Retain relevant broader background, but label unexecuted work as prospective where appropriate. Do not expand this into a wholesale rewrite of the thesis. Identify unrelated pre-existing issues separately.

## 6. Package, verify, and deliver

- Store thesis figures under a dedicated directory such as `/home/ubuntu/thesis/img/experiments/`; place generated tables in a similarly clear thesis-local location. Use relative LaTeX references so Overleaf does not depend on `/home/ubuntu/timestransfer` or absolute machine paths.
- Prefer PDF/vector exports for scientific plots when possible. Keep deterministic table/figure generation code and an artifact manifest linking each output to its inputs, selection/filtering, and command. Write derived artifacts into new locations; preserve historical benchmark outputs.
- Check each table's values, coverage, denominators, units, and highlighted winners against source data. Verify actual/prediction alignment and train/test boundaries in displayed series.
- Compile using the project's existing LaTeX toolchain. Resolve introduced errors, missing assets, undefined citations/references, and unreadable tables/figures. Inspect rendered relevant pages. If compilation is unavailable, perform the available static checks and explicitly report the missing toolchain; do not claim a successful build.
- Review every supervisor-checklist row against a concrete section and artifact. New sections must contain no unresolved editorial TODOs; genuinely unavailable evidence should be stated transparently in prose and the handoff.
- Report changed files, section/checklist coverage, validation performed, and any remaining evidence or integration blockers. Local integration is the deliverable; do not claim a live Overleaf update without verifying synchronization.

## Acceptance criteria

The task is complete when all seven supervisor requests are addressed in integrated thesis prose with supporting artifacts; every reported result is traceable to saved data; external claims have verified citations; descriptions match the implemented experiment; conditional future experiments are clearly separated from completed work; and the thesis builds successfully or a specific environmental build/integration blocker is documented. Merely supplying an outline, pasting the README, or summarizing the old screenshots does not complete this writing assignment.
