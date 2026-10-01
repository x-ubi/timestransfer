# Thesis experiment readiness audit — 2026-09-08

The current implementation and saved experiments are sufficient to start writing the master's thesis experiment chapter. The supervisor's implementation requests are substantially complete. The exhaustive dataset descriptions, source-backed baseline comparison, and scientific interpretation are not yet complete. Further experiments are conditional on the claims intended, rather than prerequisites for beginning the write-up.

## Evidence checked

- Main run: `outputs/metadata/environment.json`, generated 2026-07-06; 11 dataset/horizon tasks × 9 models = 99 successful combinations, with full configured panels, including all 321 ECL and 862 Traffic series.
- TimesFM study: `outputs/timesfm_tuning/metadata/environment.json`, generated 2026-07-07; 4 tasks × 11 variants = 44 successful combinations.
- Both stored configurations equal the current corresponding YAML configurations.
- Combined forecast files contain 723,600 main-run rows and 237,072 tuning rows, with no duplicate dataset/model/series/horizon keys or nonfinite targets/predictions. Per-combination observation and series counts agree with saved metadata.
- Independently recalculated overall MAE, RMSE, sMAPE, and MASE agree with all 143 saved overall rows within floating-point tolerance. All saved MASE denominators are positive and finite.
- There are 50 per-series forecast PNGs. Inspected `outputs/figures/forecasts/ecl_h48_0.png`: history, forecast origin, held-out observations, and model predictions are present.
- `outputs/analysis/series_features.csv` contains 1,668 dataset/series entries; ETTh1 occurs at two horizons, so these are not 1,668 independent physical series. The joined table has 15,012 rows; the correlation table has 972 rows.
- `.venv/bin/python -m pytest -q`: 45 passed. These tests and artifact checks do not constitute a fresh execution of all pretrained models.
- The local thesis checkout includes `tex/experiments.tex` and `tex/results.tex` in `main.tex`. Experiments contains a heading and commented-out plans; Results contains only a heading. Those plans are not evidence of completed experiments and extend beyond the supervisor's quoted checklist. Live Overleaf synchronization was not verified.

## Supervisor checklist

| Request | Implementation / execution | Remaining thesis work |
|---|---|---|
| Experimental benchmark procedure | Implemented and run in `src/timestransfer/runner.py` | Describe the exact implemented protocol and its limits. |
| Exhaustive benchmark descriptions | Loaders, configuration table, and run summaries exist | Add sources, domain, variables and units, date ranges, length distributions, missingness, preprocessing, exact splits, and reasons for each horizon/seasonality. |
| Actual history and continuation versus predictions | Implemented and generated in `outputs/figures/forecasts/` | Select readable examples and disclose selection rules. The attached aggregate error plots alone do not satisfy this request. |
| Series characteristics versus model performance | Train-only ACF, STL strengths, CV, entropy and Spearman analyses exist | Interpret per-dataset results; label pooled correlations exploratory. |
| What can be tuned in TimesFM? | Context sweep and five inference-flag ablations implemented and run | Describe sensitivity, distinguish configuration from weight adaptation, and qualify null effects. |
| Prophet | Implemented and successful on all 11 tasks | Describe settings and report its results. |
| What other authors compared against | Relevant baseline families implemented | Add an explicit paper-to-baseline/protocol comparison with verified citations. |

## What the experimental procedure actually does

Each series has one final-horizon holdout. There is no rolling-origin evaluation or separate validation split. Most tasks use 48 observations; ETTh1 also uses 96. Equal observation counts do not imply equal elapsed forecast times: 48 steps mean 12 hours for ETTm, 8 hours for Weather, 2 days for hourly panels, and 48 weeks for ILI.

LongHorizon2 is loaded without normalization, with the library's ETT truncation. Each variable is treated as a univariate series. TabPFN and linear regression pool supervised lag/horizon rows within a dataset, cap the table at 10,000 rows, and use train-derived scaling. This is this project's TabPFN adaptation, not a replication of the published TabPFN-TS implementation. Model history limits differ: TimesFM 1,024, Chronos-Bolt 2,048, statistical fitted baselines 5,000, Prophet 4,320; describe the actual information budgets. AutoARIMA is a deliberately restricted nonseasonal search. Negative predictions are clipped when the full training series is nonnegative.

Report sMAPE on its 0–200 scale and include MASE. Raw MAE/RMSE aggregate across variables with different scales; their interpretation requires care. The implementation's overall RMSE is the square root of pooled mean squared error, not mean per-series RMSE. These results are not a replication of the standard long-horizon paper protocol.

## Required qualifications before finalizing claims

1. **Tuning versus sensitivity.** The TimesFM variants use the same final test split as the main benchmark. They can be reported as a descriptive sensitivity study. Selecting the winning variant on those errors and presenting its score as an unbiased tuned test result requires a separate validation/test protocol. All existing plots and descriptive results can already be written up.
2. **Ablation interpretation.** Quantile flags are assessed only through point forecasts. The wrapper also clips negative forecasts independently of TimesFM's `infer_is_positive` flag, so that ablation does not fully remove positivity handling. M4 histories are at most 960 points, so contexts beyond 1,024 do not add observations. Null effects do not prove the settings are universally irrelevant.
3. **Correlation limits.** The pooled analysis is dominated by Traffic/ECL/M4, mixes frequencies/domains, and includes ETTh1 twice. Several panels have only seven variables, often dependent. Prefer within-dataset interpretation and avoid causal or significance claims based solely on these correlations. No uncertainty intervals or multiple-comparison analysis are implemented.
4. **Reproducibility.** Saved metadata records a dirty working tree at commit `39f5848`; current HEAD is `24e9e58`. Configuration agreement does not prove exact historical source identity. Preserve the existing artifacts and document the code provenance before publication. Checkpoint revisions/hashes are not pinned in run metadata, and TabPFN is constructed through its package default. Record the actual experimental package/checkpoint identity rather than today's upstream defaults. Filtered benchmark runs overwrite aggregate outputs in the configured output directory, so future runs need separate directories.
5. **Figure selection.** The plotting default uses the first two sorted IDs plus best/median/worst series by reference-model test sMAPE. Explain this diagnostic selection and avoid presenting it as a random representative sample. Nine forecast lines can be crowded; existing model filters allow clearer thesis figures without model reruns.

## Literature findings and corrections to carry into the thesis

- The original TimesFM paper includes ETS/ARIMA and learned baselines in its Monash comparisons and PatchTST in the ETT comparison. Its protocol is not the present final-h48 protocol. See [TimesFM, Sections 6 and Appendix A](https://arxiv.org/html/2310.10688v4).
- The Chronos paper compares SeasonalNaive, AutoETS, AutoTheta, AutoARIMA and multiple neural models, including DeepAR, TFT, PatchTST and DLinear. The implemented statistical baselines are therefore relevant; using Chronos-Bolt does not reproduce the original Chronos checkpoint comparison. See [Chronos, Table 1](https://arxiv.org/html/2403.07815v2).
- The initial TabPFN-TS paper compares SeasonalNaive, AutoETS, AutoARIMA, AutoTheta, DeepAR, TFT, Chronos-Mini and Chronos-Large. The supervisor's request does not require implementing every comparator. See [TabPFN-TS v1, Section 3](https://arxiv.org/html/2501.02945v1).
- Correct the blanket contamination claim in `README.md` and `src/timestransfer/metadata.py` before reusing it: the cited original Chronos paper places ETT and Traffic in its **zero-shot evaluation** partition, not its training partition; its Weather entry is daily and is not the same as this project's 10-minute Weather panel. M4 Hourly and Electricity Hourly are in-domain there. That paper alone does not establish exact Chronos-Bolt overlap. Dataset inclusion in pretraining also does not by itself establish that held-out targets were seen. See [Chronos, Appendix B, Table 2](https://arxiv.org/html/2403.07815v2).
- The original TimesFM paper documents M4 in pretraining, but this alone does not prove the exact TimesFM 2.5 checkpoint saw these test values. Keep model-version-specific uncertainty explicit. See [TimesFM pretraining](https://arxiv.org/html/2310.10688v4) and the [2.5 model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch).
- Do not turn the README's statement about the installed package into a general claim that TimesFM cannot be fine-tuned. The official repository now documents a Transformers/PEFT LoRA example for TimesFM 2.5. This project has implemented inference configuration experiments only. See the [official TimesFM repository](https://github.com/google-research/timesfm).
- The README's statement that a synthetic weekday origin makes Prophet's weekly component “phase-shifted noise” is not justified. For a separately fitted Fourier seasonal component, a fixed shift of time origin can be absorbed into sine/cosine coefficients. Synthetic dates prevent trustworthy real weekday/holiday interpretation; they do not by themselves destroy a learnable weekly periodicity. The implementation fits each series separately. See [Prophet seasonality documentation](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html).

## Recommended next work

Start writing the actual procedure, dataset descriptions, model configurations, and descriptive results now. Complete the literature mapping and the qualifications above before calling the experimental chapter finished. Add validation only if selecting tuned models, rolling origins/uncertainty if claiming robust model superiority, or independently verified unseen data if claiming clean zero-shot generalization. These are conditional extensions, not implied requirements to implement the commented E1–E10 thesis plan.

Use current saved results rather than the old screenshots. For example, ETTh1 h48 sMAPE is 23.21 for SeasonalNaive and 32.96 for TimesFM; on M4 Hourly it is 8.35 for Chronos-Bolt and 8.61 for TimesFM, while MASE favors TimesFM (0.734 versus 0.835). The conclusion depends on dataset and metric; the expanded comparison does not support a universal TimesFM-wins statement.
