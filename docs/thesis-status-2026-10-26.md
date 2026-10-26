# Thesis status: results so far, plan, expected results (2026-10-26)

**Thesis:** *Transfer learning in time series modelling* (Jan Budziński, supervisor dr inż. Paweł Zawistowski).
**Direction (confirmed with the supervisor):** compare three ways of transferring a pretrained time-series foundation model, TimesFM 2.5, to new datasets without pretraining a new model:
1. **combining** it with other zero-shot foundation models (Chronos-2, TabPFN-TS);
2. **configuring** it per dataset (context length and inference flags chosen from past data);
3. **fine-tuning** it with LoRA.

This covers three of the supervisor's suggested directions: improving a foundation model, ensembling, and fine-tuning.

Evidence labels used below:
- **Established**: full runs with significance testing.
- **Pilot**: small exploratory run.
- **Hypothesis**: not yet tested.

All metrics are MASE relative to SeasonalNaive, aggregated over the 11 tasks by geometric mean (lower is better), unless stated otherwise. Raw outputs are in `results/` (restore with `uv run python scripts/archive_results.py restore`).

---

## 1. What we already have

### 1.1 Benchmark (thesis Experiments 1–3, already written up)

- 9 models on 11 tasks (M4 Hourly; ETTh1/h2/m1/m2, ECL, Traffic, Weather, Exchange, ILI), horizon 48, 5 rolling origins.
- **Foundation models lead on aggregate:** TimesFM 0.748, Chronos-2 0.785, TabPFN-TS 0.840, AutoARIMA 0.926 (8 tasks); the other baselines are above 1.
- **TimesFM's lead depends on one task.** It rests largely on ILI (0.275 vs 0.84/0.80). Without ILI the order is Chronos-2 0.780, TimesFM 0.827, TabPFN-TS 0.844.
- **Rankings depend on task, metric (sMAPE vs MASE) and forecast origin.** The best model per task is best in only 25 of 55 single-window runs.
- **TimesFM is sensitive to context length:**
  - Weather improves with longer context (MASE 0.266 → 0.184 at 16,256);
  - ETTm1 is best at the shortest (0.554 → 0.403 at 512).

  This is the seed of the configuration arm.

### 1.2 Combination study (20 rolling origins, 10 for M4 and ILI) — **Established**

Setup:
- Members: TimesFM 2.5, Chronos-2, TabPFN-TS, with SeasonalNaive and AutoETS/AutoTheta in extra pools.
- Every combiner is fitted only on earlier windows (no leakage).
- Tests: Diebold–Mariano tests with Holm correction, and bootstrap confidence intervals resampling series, windows or both.

| Result | Number |
|---|---|
| Best combiner (task-level non-negative weights) | 0.633 vs TimesFM 0.663: ratio 0.955, 95% CI 0.934–0.975 |
| Same combiner vs the best single model per task, chosen after the fact | mean regret 1.002, max 1.025: **no gain** |
| Choosing the past-best model per task | 0.640 |
| Plain average of the three models | 1.029× TimesFM (CI 1.007–1.060): **worse than one model** |
| Finer weighting (by horizon / by series / feature-based meta-learner) | 0.634–0.635 / 0.638 / 0.661: **no improvement** over task-level |
| Past windows needed to learn the weights | one is enough (learning curve flat: 0.627 with 1 window vs 0.625 with all) |
| Worst-case regret vs best model per task | combiner 1.03, TimesFM 1.10, Chronos-2 2.90, TabPFN-TS 2.92 |
| Winner persistence per series between windows | close to chance (e.g. ECL 0.38 vs 0.34) |

### 1.3 Why combination gains so little (new today) — **Established, squared-error view**

These numbers use squared scaled error, which the theory works with; the benchmark metric is absolute error (MASE). They come from `outputs/combination/analysis/{shared_error,combination_bound,pairwise_theory}.csv`.

- **The three models make largely the same mistakes.** Splitting each model's error into the part shared by all three (their average error) and a model-specific remainder:

  | Task | Shared share of error |
  |---|---|
  | Weather | 0.94 |
  | ETTh1 | 0.94 |
  | ETTh2 | 0.93 |
  | Exchange | 0.92 |
  | Traffic | 0.91 |
  | ECL | 0.90 |
  | ETTm2 | 0.90 |
  | ETTm1 | 0.89 |
  | M4 | 0.85 |
  | ILI | 0.60 |

  The first principal component of the error covariance gives the same picture (0.85–0.94).
- **Even ideal weights barely help.** The best fixed non-negative weights per task, fitted after the fact (an upper bound), reduce squared error relative to the best single model by a median of **3%** (ratio 0.969). The largest reductions are ILI (0.878), ETTm1 (0.884) and M4 (0.918); several tasks are at 0.99–1.00.
- **Classical combination theory explains it.** Combining two models helps only if their errors are weakly correlated *and* their accuracies are similar. Here error correlations are 0.75–0.97 and accuracy ratios 0.82–1.00, which puts almost every pair in the region of negligible gain. The one low-correlation pair (TimesFM–Chronos-2 on ILI, ρ ≈ 0.01) has very unequal accuracy (ratio 0.38), so it gains little.
- **Statistical models add diversity that can't be exploited out of sample.** With AutoETS, AutoTheta and SeasonalNaive added, the shared share drops (e.g. M4 0.43, ETTm1 0.46). The ideal-weight bound improves to a median of 0.938 (Weather 0.847). Yet weights learned from past windows did worse with these members in earlier runs (0.732 → 0.779 on 5 windows). The 20-origin learning curves for these pools are being recomputed.
- **Shared error isn't removed by simple bias correction** (pilot, `results/pilots_2026-10/shared_error.py`). A series' error barely correlates between consecutive windows (−0.16 to 0.37), and correcting each series for its past bias makes MASE worse on most tasks. More sophisticated correction methods exist (ECHO, arXiv 2610.16433) but were not tested.

### 1.4 Pilots behind the direction decision (`results/pilots_2026-10/`)

| Pilot | Result | Label |
|---|---|---|
| Choosing TimesFM's context per task from past windows (Experiment 3 sweep, 4 tasks, windows 0–3) | Weather 0.231 → **0.167 (−28%)**, ETTm1 0.527 → **0.460 (−13%)**, ETTh1 and M4 unchanged. On those two tasks TimesFM nearly matches Chronos-2 (0.169, 0.441) | Pilot |
| Chronos-2 forecasting all variables jointly (multivariate / cross-learning), 20 origins (rerun today, identical to the first run) | Weather −8.3%, ETTm1 −5.9%, ETTh1 −3.1%, ETTm2 0%, ETTh2 +2.2%, ECL **+14%** (unrelated series grouped together) | Pilot |
| Probabilistic combination | Mixing the three predictive distributions (1.043× best single model's quantile loss) beats averaging their quantiles (1.082×), the method used by leaderboard ensembles | Pilot |
| Damaged inputs (20% missing, missing block, outliers) | The models do **not** become more complementary; TabPFN-TS suffers most from missing data. The rerun with a deterministic seed is in progress | Pilot |

### 1.5 Takeaways

1. Zero-shot foundation models are **largely redundant forecasters**: most of their error is shared. That caps what combining them can achieve, and the cap matches classical combination theory.
2. In practice, combining them amounts to **choosing the right model per task**. That has real value as insurance (worst case 1.03 vs up to 2.9), but no accuracy gain beyond the best model.
3. The levers that remain must reduce the **shared** error, by changing what the model does on the target data: its configuration, its weights, or the information it receives. The pilots suggest configuration can matter a lot on some tasks.
4. Part of the model ranking in the benchmark (TimesFM vs Chronos-2 on high-frequency tasks) appears to reflect default settings rather than the models themselves.

### 1.6 Infrastructure

- **Code:** `src/timestransfer/combination.py` (combiners, rolling protocol) and `combination_analysis.py` (diversity, shared-error split, ideal-weight bound, DM tests, bootstrap, regret, learning curves, failure modes), with tests (all passing).
- **Archive:** all expensive outputs are on GitHub (`results/`, branch `ensemble_research`), verified to restore exactly.

---

## 2. Plans and prospects

### 2.1 Steps

| # | Step | Content | Compute | When |
|---|---|---|---|---|
| 1 | Close the combination study | Shared-error split and bound (done), pools with AutoETS/AutoTheta (running), figure of tasks on the correlation/accuracy plane | CPU, minutes | this week |
| 2 | TimesFM configuration study | Context {512, 1,024, 2,048, 4,096, 8,192, 16,256} × flip invariance on/off on all 11 tasks, 20 origins. Setting chosen per task from past windows via the combination module. Chronos-2 context grid as a fairness control | ~3–10 GPU h (A5000) | from Oct 27 |
| 3 | Fine-tuning | TimesFM 2.5 LoRA in a separate environment (HF `TimesFm2_5ModelForPrediction` + `peft`; first check the HF checkpoint matches ours). One fine-tune per task on data before the evaluation windows. Chronos-2 LoRA via `fit()` as fallback and second model | ~6–18 GPU h | early November |
| 4 | Three-way comparison | Zero-shot vs configured vs fine-tuned vs combined, and their combinations; DM tests, bootstrap CIs, regret, compute cost | CPU | early November |
| 5 | Literature re-check | Error correlation of foundation models, context selection, fine-tuning vs zero-shot, multi-strategy comparisons; follow-ups to ECHO and Chronos-2 | – | before writing |
| 6 | Writing | Benchmark chapters as motivation → combination (redundancy) → configuration → fine-tuning → comparison → discussion; plus the deferred introduction and summary | – | November, until ~Nov 28 |

**Total compute:** about 10–30 GPU hours on this RTX A5000 (two or three overnight runs), or about 5–15 h on an L40S (spec-sheet estimate, 1.5–3× faster). Measure one long-context TimesFM run and one LoRA run first. Keep each comparison on a single GPU type.

### 2.2 Cut first if time runs short

1. Chronos-2 related-series (multivariate) study: keep as a short section from the pilot.
2. External-validity datasets (GIFT-Eval subset).
3. Most of the probabilistic section: keep the mixture vs quantile-averaging result.

### 2.3 Risks

- **Configuration gains don't generalise beyond Weather and ETTm1.** Step 2 settles this within about a week; the thesis then leans more on the diagnosis and the fine-tuning comparison.
- **The TimesFM LoRA setup fails** (HF checkpoint mismatch or dependency conflicts). Fallback: Chronos-2 `fit()`.
- **Possible pretraining contamination** (M4 and ECL are in some models' training data). Report it as a limitation; relative comparisons are less affected than absolute scores.
- **New related papers** in a fast-moving field. The step 5 re-check guards against this.
- **Writing time** is the most likely bottleneck, not compute.

### 2.4 Contribution and novelty (literature checks of 2026-10-15; repeat before writing)

1. **Measurement of shared vs model-specific error** across zero-shot foundation models, linked to the combination bound and classical theory. No prior measurement found.
2. **Weighting granularity under a leakage-free rolling protocol.** This contradicts the per-series / per-window weighting used by recent foundation-model ensembles (Toto 2.0, TimeRouter, ZooCast), at least on this benchmark.
3. **Per-task backtested configuration of TimesFM.** The method itself is incremental (context sensitivity is documented; GITCO, arXiv 2606.05332). What's new is quantifying how it changes model rankings.
4. **Combination vs configuration vs fine-tuning under one protocol.** No single source found. Fine-tuning vs zero-shot alone is known (e.g. Laglil et al., arXiv 2607.23146: LoRA +2–3% on medium/large datasets).

---

## 3. Expected results

| Arm | Expected result | Basis | What it would mean |
|---|---|---|---|
| Combination | Combining ≈ per-task model selection; ~4.5% better than a single fixed model, ≈0% better than the best model per task; ideal fixed weights ≤ ~3% (median) | **Established** | Redundancy, not the weighting method, limits foundation-model ensembles |
| Configuration | Large gains on long high-frequency series (pilot: −13% to −28%), little or none on short series (M4, ILI); TimesFM vs Chronos-2 ranking changes on some tasks | **Pilot** | Default-setting benchmarks mix up model and configuration; per-task configuration is the cheapest effective transfer step |
| Fine-tuning | Small gains (~0–5%) on long datasets, none or negative on short ones | **Hypothesis**, consistent with the literature | Weight adaptation helps only with enough target data |
| Combined strategies | Combination adds little on top of a well-configured or fine-tuned model | **Hypothesis**, follows from the redundancy result | Model-side adaptation, not model pooling, reduces the shared error |
| Cost-effectiveness | configuration > fine-tuning > combination in gain per GPU hour | **Hypothesis** | A practical recommendation order |

### Is this enough for a master's thesis?

**Very likely yes, provided steps 2–4 are completed.**

- **Original contribution:** an empirical-analytical one. It has two parts, the redundancy diagnosis with its theoretical explanation and the controlled three-way comparison, and as far as the literature checks found, both are new. That is the expected level for a master's thesis.
- **Volume:** a 9-model benchmark plus three experimental arms with statistical testing is ample material.
- **Rigour:** leakage-free rolling origins, significance tests and confidence intervals.
- **What would weaken it:**
  - configuration gains confined to two tasks;
  - fine-tuning dropped, which would thin the link to the "transfer learning" title;
  - too little writing time.

  The step 2 results (within about a week) are the right moment to show the supervisor and confirm sufficiency.
