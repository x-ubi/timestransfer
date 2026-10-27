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
- **Statistical models add diversity that can't be exploited out of sample.** With AutoETS, AutoTheta and SeasonalNaive added, the shared share drops (e.g. M4 0.43, ETTm1 0.46). The ideal-weight bound improves to a median of 0.938 (Weather 0.847). Yet weights learned from past windows do worse with these members at every amount of past data on the 20-origin runs:
  - all tasks: task-level weights 0.625 (foundation models only) vs 0.629 (+ AutoETS/AutoTheta) vs 0.632 (+ SeasonalNaive too); equal mean 0.674 vs 0.760;
  - 20-window tasks with 10 past windows: 0.727 vs 0.747.

  The diversity they add is real in sample but is not recovered out of sample. Part of this gap may come from the bound being computed on squared error while the scores use absolute error.
- **Shared error isn't removed by simple bias correction** (pilot, `results/pilots_2026-10/shared_error.py`). A series' error barely correlates between consecutive windows (−0.16 to 0.37), and correcting each series for its past bias makes MASE worse on most tasks. More sophisticated correction methods exist (ECHO, arXiv 2610.16433) but were not tested.

### 1.4 TimesFM configuration study (step 2, 2026-10-26) — **Established**

Setup:
- Grid: TimesFM 2.5 at context {512, 1,024, 2,048, 4,096, 8,192, 16,256} × flip invariance on/off, and Chronos-2 at context {1,024, 2,048, 4,096, 8,192}, on the 20-origin windows of §1.2 (`configs/timesfm_config.yaml`).
- The default variants (TimesFM 1,024, Chronos-2 8,192) reproduce the §1.2 forecasts exactly.
- Selection and weighting use the same rolling protocol as §1.2 (`scripts/run_timesfm_config_selection.sh`).
- Analysis in `scripts/run_config_analysis.py`; tables in `outputs/timesfm_config/analysis/`.
- Ratios below are against TimesFM at context 1,024. CIs are bootstrap intervals resampling series and windows.

| Result | Number |
|---|---|
| One fixed long context for every task (16,256) | 0.639 vs 0.663: ratio **0.965** (CI 0.944–0.987); better on 9 tasks, equal on M4 and ILI (series shorter than 1,024) |
| Largest per-task gains of the long context | Traffic −11.5%, Weather −10.2%, ETTh2 −4.6%, ECL −3.9%, Exchange −3.9%; ETTh1/ETTm1/ETTm2 −0.2% to −1.5% |
| Context chosen per task from past windows | 0.640 (contexts only) and 0.642 (with flip on/off): **no better than always using 16,256** (ratio 1.001, CI 0.980–1.021) |
| Ex-post best configuration per task (unattainable oracle) | 0.629 |
| Flip invariance off | Worse at every context in aggregate; at 16,256 it is worse on 8 of 11 tasks (3 significant after Holm). It halves inference time at long contexts |
| Chronos-2 context | Hardly matters: 0.715 (1,024) → 0.702 (8,192) |
| TimesFM vs Chronos-2 | Chronos-2 is 1.059× TimesFM (default) and 1.098× TimesFM (16,256). At 16,256, TimesFM overtakes Chronos-2 on Weather and Exchange and pulls ahead on Traffic. Chronos-2 stays better on ECL and the four ETT tasks |
| Combination after configuration | TimesFM 16,256 + Chronos-2, task-level weights: 0.624, ratio **0.976** (CI 0.958–0.991) to TimesFM 16,256. Adding TabPFN-TS: 0.623 |
| Original combination vs long context alone | 0.634 vs 0.639: ratio 0.992 (CI 0.967–1.014), **indistinguishable** |
| Best overall | TimesFM 16,256 + Chronos-2 (+ TabPFN-TS), task-level weights: 0.623, ratio **0.940** (CI 0.916–0.962) to TimesFM default |
| Inference cost, all 11 tasks × 20 origins, A5000 | TimesFM 1,024: 5.6 min, 8,192: 21.7 min, 16,256: 42.3 min (no flip: 22.8 min); Chronos-2 8,192: 8.8 min. Logged per-model times include data handling (SeasonalNaive: 4.4 min) |

Per-task DM tests have little power with 17 evaluated windows: no single-task context gain survives the Holm correction; the closest is Traffic (p = 0.007 raw, 0.08 Holm). The aggregate gains are significant under all three bootstrap schemes.

Interpretation:
- **One fixed setting beats per-task tuning.** Context length is a real lever, but the right per-task setting cannot be identified from past windows. Window-to-window noise in the ranking of configurations outweighs their differences.
  - ETTm1 is the one task where selection helps: −10% from choosing 512.
  - On Weather and ETTh1 selection loses part of the long-context gain.
- **The pilot overstated the configuration gains.** The pilot below (Weather −28%, ETTm1 −13%) used 4 windows of the 5-origin benchmark; on 20 origins the gains are −10% (Weather, fixed long context) and −10% (ETTm1, selected).
- **Configuration and combination are partly substitutes.** A longer context alone recovers the whole gain the combination study reported. Combining still adds 2.4% on top of the long-context model, and the combination keeps its insurance role.
- **Longer context does not make TimesFM less redundant with Chronos-2.** The shared error share of the pair is unchanged (0.94–0.99 on the long-series tasks), consistent with §1.3.
- **The benchmark understated TimesFM.** Default-setting benchmarks partly confound model and configuration: three task rankings change, and the aggregate gap to Chronos-2 widens.

### 1.5 Fine-tuning (step 3, 2026-10-27) — **Established for TimesFM, pilot for Chronos-2**

Setup:
- TimesFM 2.5 LoRA (rank 4, all linear layers), trained at context 1,024. One adapter per task.
- Training data: history before the 3 oldest windows. Early stopping uses those 3 windows. Scoring uses the same evaluated windows as §1.2/§1.4.
- Learning rate 1e-4, chosen on a Traffic/ETTm1 pilot (1e-5 and 3e-5 also tried), 1,000 steps, batch 32.
- About 6 min per task on the A5000 (69 min for all 11).
- Outputs, adapters and logs in `results/finetune/`.
- The run is deterministic: a rerun reproduces the forecasts exactly.

| Result | Number |
|---|---|
| LoRA, forecasting at 1,024 | 0.646: ratio **0.974** (CI 0.960–0.986) to TimesFM 1,024; 1.010 (CI 0.988–1.031) to zero-shot 16,256 |
| Same adapter, forecasting at 16,256 | **0.632**: ratio **0.954** (CI 0.930–0.978) to TimesFM 1,024; **0.989** (CI 0.980–0.998) to zero-shot 16,256 |
| Where LoRA helps (vs zero-shot 16,256, adapter at 16,256) | ECL −4.0%, ETTh1 −3.8% (Holm p = 0.003), M4 −3.3% (Holm p = 0.02), ETTh2 −1.2% |
| Where LoRA does nothing | ETTm1, ETTm2, ILI: early stopping kept the zero-shot model (validation loss rose from the first checkpoint). Traffic, Weather, Exchange: within ±0.5% |
| Combination with Chronos-2 (task-level weights) | 0.621, vs 0.624 with zero-shot 16,256 and 0.634 with default TimesFM; +TabPFN-TS 0.620 |
| Chronos-2 LoRA pilot (context 8,192, lr 1e-5, effective batch 256) | ETTm1 −1.2%, Traffic −0.7% vs zero-shot Chronos-2, at ~80 min per task |

Caveats:
- In the combination runs, the weights for windows 14–16 are fitted partly on the early-stopping windows 17–19. There the LoRA forecasts are in-sample for model selection, so the combination numbers are slightly optimistic.
- The Chronos-2 ETTm1 checkpoint was kept although validation loss rose after step 100. `fit()` does not score the untrained model, so its −1.2% is weak evidence.

Interpretation:
- **Fine-tuning and long context are complementary.** LoRA helps where the long context does not (ECL, ETTh1, M4), and vice versa (Traffic, Weather). An adapter trained at 1,024 transfers to inference at 16,256 without loss.
- **Each improvement of the TimesFM member shrinks the gain from combination with Chronos-2:**

  | TimesFM member | Gain from combining |
  |---|---|
  | default | 4.3% |
  | long context | 2.4% |
  | long context + LoRA | 1.8% |

- **Fine-tuning is cheap for TimesFM** (6 min per task), but its gains are small (~1–4%) and limited to some tasks.

### 1.6 Pilots behind the direction decision (`results/pilots_2026-10/`)

| Pilot | Result | Label |
|---|---|---|
| Choosing TimesFM's context per task from past windows (Experiment 3 sweep, 4 tasks, windows 0–3) | Weather 0.231 → **0.167 (−28%)**, ETTm1 0.527 → **0.460 (−13%)**, ETTh1 and M4 unchanged. On those two tasks TimesFM nearly matches Chronos-2 (0.169, 0.441). **Superseded by §1.4** | Pilot |
| Chronos-2 forecasting all variables jointly (multivariate / cross-learning), 20 origins (rerun today, identical to the first run) | Weather −8.3%, ETTm1 −5.9%, ETTh1 −3.1%, ETTm2 0%, ETTh2 +2.2%, ECL **+14%** (unrelated series grouped together) | Pilot |
| Probabilistic combination | Mixing the three predictive distributions (1.043× best single model's quantile loss) beats averaging their quantiles (1.082×), the method used by leaderboard ensembles | Pilot |
| Damaged inputs (20% missing, missing block, outliers) | The models do **not** become more complementary; TabPFN-TS suffers most from missing data. The rerun with a deterministic seed is in progress | Pilot |

### 1.7 Takeaways

1. Zero-shot foundation models are **largely redundant forecasters**: most of their error is shared. That caps what combining them can achieve, and the cap matches classical combination theory.
2. In practice, combining them amounts to **choosing the right model per task**. That has real value as insurance (worst case 1.03 vs up to 2.9), but no accuracy gain beyond the best model.
3. The levers that remain must reduce the **shared** error, by changing what the model does on the target data: its configuration, its weights, or the information it receives.
   - Configuration (§1.4) gives −3.5% from one fixed long context. Per-task tuning adds nothing to that.
4. Part of the model ranking in the benchmark reflects default settings rather than the models themselves. With a long context, TimesFM overtakes Chronos-2 on Weather and Exchange.

### 1.8 Infrastructure

- **Code:** `src/timestransfer/combination.py` (combiners, rolling protocol) and `combination_analysis.py` (diversity, shared-error split, ideal-weight bound, DM tests, bootstrap, regret, learning curves, failure modes), with tests (all passing).
- **Archive:** all expensive outputs are on GitHub (`results/`, branch `ensemble_research`), verified to restore exactly.

---

## 2. Plans and prospects

### 2.1 Steps

| # | Step | Content | Compute | When |
|---|---|---|---|---|
| 1 | Close the combination study | **Done 2026-10-26:** shared-error split, ideal-weight bound, two-model theory, pools with AutoETS/AutoTheta. The figure of tasks on the correlation/accuracy plane moves to step 6 (thesis artifact pipeline) | CPU, minutes | done |
| 2 | TimesFM configuration study | **Done 2026-10-26** (§1.4): grid 3.1 h on the A5000; selection, combination after configuration, DM tests and bootstrap | 3.1 GPU h | done |
| 3 | Fine-tuning | **TimesFM LoRA done 2026-10-27** (§1.5). Open: full Chronos-2 LoRA (~15 GPU h at context 8,192; ~4 h at 2,048), optional long-context TimesFM LoRA on Traffic/Weather | 1.2 GPU h so far (+2.7 h pilots) | in progress |
| 4 | Three-way comparison | Zero-shot vs configured vs fine-tuned vs combined, and their combinations; DM tests, bootstrap CIs, regret, compute cost | CPU | early November |
| 5 | Literature re-check | Error correlation of foundation models, context selection, fine-tuning vs zero-shot, multi-strategy comparisons; follow-ups to ECHO and Chronos-2 | – | before writing |
| 6 | Writing | Benchmark chapters as motivation → combination (redundancy) → configuration → fine-tuning → comparison → discussion; plus the deferred introduction and summary | – | November, until ~Nov 28 |

**Total compute:** about 10–30 GPU hours on this RTX A5000 (two or three overnight runs), or about 5–15 h on an L40S (spec-sheet estimate, 1.5–3× faster). Measure one long-context TimesFM run and one LoRA run first. Keep each comparison on a single GPU type.

### 2.2 Cut first if time runs short

1. Chronos-2 related-series (multivariate) study: keep as a short section from the pilot.
2. External-validity datasets (GIFT-Eval subset).
3. Most of the probabilistic section: keep the mixture vs quantile-averaging result.

### 2.3 Risks

- **Configuration gains don't generalise beyond Weather and ETTm1.** Partly realised (§1.4): the gains are real but moderate (−3.5% aggregate) and come from one fixed long context, not per-task tuning. Fine-tuning must be compared against TimesFM at 16,256, the stronger zero-shot baseline, not only at 1,024.
- **The TimesFM LoRA setup fails** (HF checkpoint mismatch or dependency conflicts). Fallback: Chronos-2 `fit()`.
- **Possible pretraining contamination** (M4 and ECL are in some models' training data). Report it as a limitation; relative comparisons are less affected than absolute scores.
- **New related papers** in a fast-moving field. The step 5 re-check guards against this.
- **Writing time** is the most likely bottleneck, not compute.

### 2.4 Contribution and novelty (literature checks of 2026-10-15; repeat before writing)

1. **Measurement of shared vs model-specific error** across zero-shot foundation models, linked to the combination bound and classical theory. No prior measurement found.
2. **Weighting granularity under a leakage-free rolling protocol.** This contradicts the per-series / per-window weighting used by recent foundation-model ensembles (Toto 2.0, TimeRouter, ZooCast), at least on this benchmark.
3. **Backtested configuration of TimesFM.** The method itself is incremental (context sensitivity is documented; GITCO, arXiv 2606.05332).
   - What's new is the result under a leakage-free protocol: per-task selection does not beat one fixed long context.
   - The long context alone matches the gain of combining foundation models.
   - It changes three task-level rankings against Chronos-2.
4. **Combination vs configuration vs fine-tuning under one protocol.** No single source found. Fine-tuning vs zero-shot alone is known (e.g. Laglil et al., arXiv 2607.23146: LoRA +2–3% on medium/large datasets).

---

## 3. Expected results

| Arm | Expected result | Basis | What it would mean |
|---|---|---|---|
| Combination | Combining ≈ per-task model selection; ~4.5% better than a single fixed model, ≈0% better than the best model per task; ideal fixed weights ≤ ~3% (median) | **Established** | Redundancy, not the weighting method, limits foundation-model ensembles |
| Configuration | Fixed long context −3.5% (Traffic −11.5%, Weather −10%), none on short series (M4, ILI). Per-task selection from past windows no better than the fixed long context. Rankings vs Chronos-2 change on Weather and Exchange | **Established** (§1.4) | Default-setting benchmarks partly confound model and configuration. Configuration is a cheap transfer step, but its best form is a single good default, not per-task tuning |
| Fine-tuning | TimesFM LoRA: −2.6% at 1,024; with the adapter at 16,256, −1.1% beyond zero-shot 16,256 (CI 0.980–0.998), on ECL, ETTh1, M4, ETTh2; nothing on ETTm1/ETTm2/ILI. Not tied to series length: M4 (short) gains, ETTm1 (long) does not | **Established** (TimesFM, §1.5); Chronos-2 **Pilot** | Weight adaptation and context are complementary levers; both are small next to the ranking noise across tasks |
| Combined strategies | Gain from combining with Chronos-2 shrinks as TimesFM improves: 4.3% (default) → 2.4% (long context) → 1.8% (long context + LoRA) | **Established** (§1.4, §1.5) | Model-side adaptation absorbs part of what combination provides; the redundancy result (§1.3) still caps the rest |
| Cost-effectiveness | configuration > fine-tuning > combination in gain per GPU hour | **Hypothesis** | A practical recommendation order |