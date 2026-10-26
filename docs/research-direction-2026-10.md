# Thesis research direction (confirmed with the supervisor, October 2026)

Audience: the student and agents continuing the work. This file supersedes §3 of `docs/combination-study-plan.md` for scope; that file remains the reference for the combination-study implementation (Phases A–C). Numbers come from the 20-origin runs in `results/` unless marked as a pilot.

## Scope

Title stays *Transfer learning in time series modelling*. Scope statement:

> This thesis compares three ways of transferring a pretrained time-series foundation model (TimesFM 2.5) to target datasets without pretraining new models: combining it with other zero-shot models (Chronos-2, TabPFN-TS), adapting its inference configuration per task, and parameter-efficient fine-tuning. It shows that zero-shot foundation models share most of their forecast error, explains why this limits combination, and tests whether configuration or weight adaptation does better, on M4 Hourly and the LongHorizon datasets under a rolling-origin protocol.

This covers the supervisor's suggested directions 1 (improving a foundation model), 2 (ensembling) and 3 (fine-tuning). Direction 4 (data issues) appears only as the negative corruption pilot. Chronos-2's related-series (multivariate) mode is an optional comparison, cut first if time runs short.

## Findings so far

Established (Phase C, `results/extras/combination/analysis/`):
- FM error correlations are 0.77–0.94 on every task except ILI.
- Task-level learned weights (`nnls_task`) give geometric-mean relative MASE 0.633, against TimesFM 0.663: ratio 0.955, 95% bootstrap CI 0.934–0.975 (series and windows resampled).
- Against the ex-post best member per task they only match it: mean regret 1.002, max 1.025. Choosing the past-best member per task gives 0.640.
- Horizon-specific weights (0.634–0.635), per-series weights (0.638) and the leave-one-dataset-out FFORMA-style meta-learner (0.661) add nothing over task-level weights. Learning curves are flat: one past window suffices.
- The equal mean is worse than TimesFM: 1.029× (CI 1.007–1.060).
- Series-level winner persistence is close to chance with 20 origins.
- The 5-window pilot numbers in §2 of the plan are superseded; do not cite them.

Pilots (`results/pilots_2026-10/`): see the table in `results/README.md`. In short:
- 84–97% of FM error variance is a shared component, not removable by simple bias correction.
- Mixture pooling beats quantile averaging.
- Choosing TimesFM's context per task from past windows gives −28% on Weather and −13% on ETTm1, closing most of the gap to Chronos-2 on those tasks.
- Damaged inputs do not make the FMs complementary.
- Chronos-2 multivariate mode helps on physically coupled variables (Weather, ETT) and hurts on unrelated series (ECL).

## Contributions and novelty (literature checks of 2026-10-15; repeat before writing)

1. Measurement of shared vs model-specific error across zero-shot FMs and its link to the combination bound: no prior measurement found.
2. Weighting granularity under a time-respecting protocol: classical theory, but contradicts the per-series / per-window weighting used by recent TSFM ensembles (Toto 2.0, TimeRouter, ZooCast).
3. Per-task backtested configuration of TimesFM: the method is incremental (GITCO, context sensitivity studies); new is showing how much it changes model rankings.
4. Combination vs configuration vs fine-tuning under one protocol: no single source found. Fine-tuning vs zero-shot alone is known (Laglil et al., arXiv 2607.23146: LoRA +2–3% on medium/large datasets).

Recent related work to track: ECHO (arXiv 2610.16433, TSFM error correction), Chronos-2 paper Sections 5.2 and 6 (arXiv 2510.15821), Falcon-X Section 5.3 (arXiv 2605.27286).

## Next steps

1. Close the combination study: commit Phase B/C code; rerun `scripts/run_combination_analysis.py` with the statistical pools; add the shared/specific error decomposition and the per-task optimal-weight bound to `combination_analysis.py` with tests. **Done 2026-10-26** (results in `docs/thesis-status-2026-10-26.md`, §1.3).
2. TimesFM configuration study: context {512, …, 16256} × flip invariance on/off, 20 origins, all tasks; small Chronos-2 context grid as a fairness control; selection and weighting through the existing combination module.
3. TimesFM 2.5 LoRA fine-tuning in a separate environment (`transformers` with `TimesFm2_5ModelForPrediction`, `peft`; checkpoint `google/timesfm-2.5-200m-transformers`, checked against the pinned PyTorch checkpoint first). One fine-tune per task on data before the earliest evaluation window. Fallback: `Chronos2Pipeline.fit`.
4. Three-way comparison with DM tests, bootstrap CIs, regret and compute cost.
5. Literature re-check.
6. Writing, including the deferred introduction and summary.
