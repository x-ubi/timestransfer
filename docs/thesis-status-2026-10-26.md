# Thesis status: results so far, plan, expected results (2026-10-26, updated 2026-10-28)

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

All metrics are MASE relative to SeasonalNaive, aggregated over the 11 tasks by geometric mean (lower is better), unless stated otherwise. "Ratio to X" means a score divided by X's score: below 1 is better than X (0.965 = 3.5% lower error). CIs are 95% bootstrap intervals. Raw outputs are in `results/` (restore with `uv run python scripts/archive_results.py restore`).

## Summary (2026-10-28)

All strategies scored on one common evaluation set (§1.6): 14 forecast origins per task (4 for M4 and ILI), with combination weights never fitted on the fine-tuning validation windows. GPU time is measured on an RTX A5000.

| Strategy | Score | vs default | GPU s per forecast origin (all 11 tasks) | One-time GPU min |
|---|---|---|---|---|
| TimesFM default (context 1,024) | 0.655 | – | 5.8 | – |
| TimesFM + Chronos-2 at 2,048, task-level weights | 0.633 | −3.5% | 10.5 | – |
| TimesFM + Chronos-2 at 8,192, task-level weights | 0.629 | −4.0% | 23.0 | – |
| TimesFM, context 8,192 | 0.633 | −3.4% | 37.4 | – |
| TimesFM, context 16,256 | 0.630 | −3.8% | 72.7 | – |
| TimesFM LoRA, forecasting at 1,024 | 0.640 | −2.4% | 5.9 | 63 |
| TimesFM LoRA, forecasting at 16,256 (3 seeds) | 0.623–0.626 | −4.5% to −5.0% | 72.7 | 63 |
| TimesFM LoRA at 16,256 + Chronos-2 at 8,192 | 0.617 | −5.8% | 89.8 | 63 |

1. **Zero-shot foundation models make largely the same errors** (85–94% of error shared on 10 of 11 tasks). This limits what combining these models can achieve, as classical combination theory predicts (§1.2–1.3). It does not establish an irreducible error floor.
2. **Context length is the largest single-model lever:** −3.8% at 16,256 (CI 0.940–0.989), mostly on Traffic and Weather. Choosing the context per task from past windows shows no clear advantage over one fixed long context (§1.4).
3. **LoRA fine-tuning of TimesFM adds a small gain on top of the long context:** −0.9% averaged over 3 seeds (CI 0.981–0.999). Single seeds range from −0.7% to −1.2%, and two of the three single-seed intervals touch zero. The gain comes from ECL (−4 to −5% in every seed) and M4 (−2.5 to −3%) (§1.5–1.6).
4. **Fine-tuning Chronos-2 the same way changes nothing** (ratio 0.998, CI 0.988–1.009), although it beat the zero-shot model on validation for 6 tasks (§1.6).
5. **The better TimesFM is, the less combination with Chronos-2 adds:**

   | TimesFM in the combination | Gain from adding Chronos-2 |
   |---|---|
   | default | 4.0% |
   | long context | 1.8% |
   | long context + LoRA | 1.25% |

   The first drop is clear: the gain is 2.3 points smaller (CI 0.3–4.0). The second drop is borderline (0.55 points; CI −0.02 to 1.15 with series and windows resampled, 0.1 to 1.06 with window blocks).
6. **Cost reverses the expected ranking.** Combining default TimesFM with Chronos-2 reaches about the accuracy of the 16,256 context, with no clear difference (CI of the ratio 0.976–1.028), at a seventh to a third of the recurring inference cost and with no training. Fine-tuning at 1,024 is the cheapest per forecast but less accurate (§1.6).
7. **Default-setting benchmarks partly confound model and configuration.** With a long context, TimesFM overtakes Chronos-2 on three tasks (§1.4).

Still open:
- literature re-check;
- writing.

---

## 1. What we already have

### 1.1 Benchmark (thesis Experiments 1–3, already written up)

- 9 models on 11 tasks: 10 datasets (M4 Hourly; ETTh1/h2/m1/m2, ECL, Traffic, Weather, Exchange, ILI) at horizon 48, plus ETTh1 at horizon 96; 5 rolling origins.
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

### 1.3 Why combination gains so little — **Established, squared-error view**

These numbers use squared scaled error, which the theory works with; the benchmark metric is absolute error (MASE). They come from `outputs/combination/analysis/{shared_error,combination_bound,pairwise_theory}.csv`.

- **The three models make largely the same mistakes.** Splitting each model's error into the part shared by all three (their average error) and a model-specific remainder:

  | Task | Shared share of error |
  |---|---|
  | Weather | 0.94 |
  | ETTh1 (horizon 48) | 0.94 |
  | ETTh1 (horizon 96) | 0.94 |
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
- **Classical combination theory accounts for it.** Take two forecasts with error standard deviations σ₁ ≤ σ₂ and error correlation ρ. The best non-negative weights reduce squared error if and only if ρ < k = σ₁/σ₂, by the factor (1 − ρ²)/(1 + k² − 2ρk) (Bates & Granger, 1969).
  - High correlation alone does not rule out a gain: ρ = 0.95 with k = 0.99 still gives about 1.6%.
  - What limits the gains here is that ρ is close to k: correlations 0.75–0.97 against accuracy ratios 0.82–1.00, so most pairs sit near the ρ = k boundary where the gain vanishes.
  - The exception is TimesFM–Chronos-2 on ILI (ρ ≈ 0, k = 0.37). Its theoretical gain is the largest of all pairs (~12%), but it comes mostly from down-weighting the much worse member.
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
| Context chosen per task from past windows | 0.640 (contexts only) and 0.642 (with flip on/off): **no clear evidence of an advantage over always using 16,256** (ratio 1.001, CI 0.980–1.021) |
| Ex-post best configuration per task (unattainable oracle) | 0.629 |
| Flip invariance off | Worse at every context in aggregate; at 16,256 it is worse on 8 of 11 tasks (3 significant after Holm). It halves inference time at long contexts |
| Chronos-2 context | Hardly matters: 0.715 (1,024) → 0.702 (8,192) |
| TimesFM vs Chronos-2 | Chronos-2 is 1.059× TimesFM (default) and 1.098× TimesFM (16,256). At 16,256, TimesFM overtakes Chronos-2 on Weather and Exchange and pulls ahead on Traffic. Chronos-2 stays better on ECL and the four ETT tasks |
| Combination after configuration | TimesFM 16,256 + Chronos-2, task-level weights: 0.624, ratio **0.976** (CI 0.958–0.991) to TimesFM 16,256. Adding TabPFN-TS: 0.623 |
| Original combination vs long context alone | 0.634 vs 0.639: ratio 0.992 (CI 0.967–1.014): **no clear difference**; the interval rules out differences larger than about 3% |
| Best overall | TimesFM 16,256 + Chronos-2 (+ TabPFN-TS), task-level weights: 0.623, ratio **0.940** (CI 0.916–0.962) to TimesFM default |
| Inference cost, all 11 tasks × 20 origins, A5000 | TimesFM 1,024: 5.6 min, 8,192: 21.7 min, 16,256: 42.3 min (no flip: 22.8 min); Chronos-2 8,192: 8.8 min. Logged per-model times include data handling (SeasonalNaive: 4.4 min) |

Per-task DM tests have little power with 17 evaluated windows: no single-task context gain survives the Holm correction; the closest is Traffic (p = 0.007 raw, 0.08 Holm). The aggregate gains are significant under all three bootstrap schemes.

Interpretation:
- **Per-task tuning shows no clear advantage over one fixed setting.** Context length is a real lever, but the right per-task setting could not be identified reliably from past windows. Window-to-window noise in the ranking of configurations outweighs their differences.
  - ETTm1 is the one task where selection helps: −10% from choosing 512.
  - On Weather and ETTh1 selection loses part of the long-context gain.
- **The pilot overstated the configuration gains.** The pilot below (Weather −28%, ETTm1 −13%) used 4 windows of the 5-origin benchmark; on 20 origins the gains are −10% (Weather, fixed long context) and −10% (ETTm1, selected).
- **Configuration and combination give similar gains.** A longer context alone gives about the same gain as the combination study reported (no clear difference). Combining still adds 2.4% on top of the long-context model, and the combination keeps its insurance role.
- **Longer context does not make TimesFM less redundant with Chronos-2.** The shared error share of the pair is unchanged (0.94–0.99 on the long-series tasks), consistent with §1.3.
- **The benchmark understated TimesFM.** Default-setting benchmarks partly confound model and configuration: three task rankings change, and the aggregate gap to Chronos-2 widens.

### 1.5 Fine-tuning (step 3, 2026-10-27) — **Established**; final comparison in §1.6

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
| Combination with Chronos-2 (task-level weights) | Provisional 0.621 here; superseded by the leakage-free comparison in §1.6 (0.617 on the common set) |
| Chronos-2 LoRA pilot (context 8,192, lr 1e-5, effective batch 256), corrected 2026-10-27 | ETTm1: fine-tuned model worse on validation, zero-shot kept (0%). Traffic −0.7% vs zero-shot Chronos-2. ~80 min training per task |

Caveats:
- In these first combination runs, the weights for windows 14–16 were fitted partly on the early-stopping windows 17–19, where the LoRA forecasts are in-sample for model selection. §1.6 fixes this by dropping windows 17–19 from all fitting and scoring.
- **Chronos-2 pilot correction.** The first pilot forecasts came from the model returned by `Chronos2Pipeline.fit`. That model is still in training mode (dropout on), runs its forward pass under a bf16 autocast wrapper left by Accelerate, and `fit` leaves TF32 matmuls enabled. Those forecasts were therefore noisy and reduced-precision; the earlier ETTm1 −1.2% was an artifact. `finetune/chronos2_ft.py` now forecasts from the saved adapter merged into a fresh fp32 base model, and uses the fine-tuned model only if it beats the zero-shot model on the validation windows. The corrected numbers are in the row above. A smoke test confirms that the reloaded model and the zero-shot forecasts reproduce exactly (difference 0.0).

Interpretation:
- **Fine-tuning and long context help on different tasks.** LoRA helps where the long context does not (ECL and M4 in every seed; ETTh1 in seed 0 only), and vice versa (Traffic, Weather). An adapter trained at 1,024 transfers to inference at 16,256 without loss.
- **Each improvement of the TimesFM member shrinks the gain from combination with Chronos-2.** Final numbers with intervals are in §1.6.
- **Fine-tuning is cheap for TimesFM** (6 min per task), but its gains are small (~1–4%) and limited to some tasks.

### 1.6 Three-way comparison (step 4, 2026-10-28) — **Established**

Setup (`scripts/run_strategy_comparison.py`; tables in `results/strategy_comparison/`):
- **Common evaluation set.** Every strategy is scored on the same windows.
  - Each task's 3 oldest windows are dropped everywhere: they are the fine-tuning validation windows, so forecasts there are not out of sample.
  - Combination weights are refitted under the rolling protocol on the remaining windows.
  - That leaves windows 0–13 (0–3 for M4 and ILI) for every strategy, so numbers differ slightly from §1.2–1.5.
- **Uncertainty.** Bootstrap with five schemes: series; windows; both; moving blocks of 3 consecutive windows; series plus window blocks.
  - Intervals below resample series and windows. The block schemes give very similar intervals.
  - Combination gains and their differences get their own paired intervals.
- **Seeds.** TimesFM LoRA was trained with 3 seeds (0, 1, 2).
- **Chronos-2 LoRA.** Context 2,048, learning rate 1e-5, effective batch 256, the library's default LoRA (rank 8, attention and output layers). It forecasts at 2,048 and 8,192 and is used only where it beats zero-shot on validation (selected on 6 of 11 tasks). Training took 2.9 GPU h.
- **Timing.** `finetune/measure_inference.py`, with the GPU otherwise idle.
  - Same implementation for all TimesFM variants (HF) and all Chronos-2 variants, with adapters merged into the weights.
  - TimesFM's context is capped at the longest history. Padding beyond it changes forecasts by ~1e-5 relative but costs up to 20× more time on M4.
  - TabPFN-TS was not re-measured (progress log: ~4,100 s per origin).

| Comparison | Ratio (95% CI) |
|---|---|
| TimesFM 16,256 vs default | 0.962 (0.940–0.989) |
| Per-task context selection vs 16,256 | 0.999 (0.980–1.017): no clear advantage |
| TimesFM LoRA at 16,256 vs zero-shot 16,256: seed 0 / 1 / 2 | 0.992 (0.982–1.002) / 0.993 (0.983–1.002) / 0.988 (0.978–0.997) |
| Same, geometric mean of the 3 seeds | 0.991 (0.981–0.999) |
| TimesFM LoRA at 1,024 vs zero-shot 16,256 | 1.015 (0.989–1.038) |
| Chronos-2 LoRA vs zero-shot Chronos-2, at 2,048 / at 8,192 | 0.998 (0.988–1.009) / 1.000 (0.990–1.010) |
| Combination gain (TimesFM + Chronos-2 at 8,192, task weights, vs its TimesFM member): default / 16,256 / LoRA at 16,256 | 0.960 (0.938–0.982) / 0.982 (0.962–1.001) / 0.988 (0.968–1.005) |
| Difference in gain, 16,256 vs default | 1.023 (1.003–1.040) |
| Difference in gain, LoRA at 16,256 vs 16,256 | 1.006 (1.000–1.012); with window blocks 1.001–1.011 |
| Fine-tuning Chronos-2 too, change in combination gain | 1.001 (0.996–1.006): none |
| TimesFM + Chronos-2 at 2,048 vs TimesFM 16,256 | 1.004 (0.976–1.028): no clear difference |
| Best overall, LoRA at 16,256 + Chronos-2 at 8,192, vs default | 0.942 (0.914–0.969) |
| Worst-case regret vs best single strategy per task | LoRA at 16,256 + Chronos-2: 1.049; TimesFM LoRA at 16,256 alone: 1.065 |

Per task, TimesFM LoRA at 16,256 vs zero-shot 16,256 (seeds 0 / 1 / 2):
- ECL 0.956 / 0.958 / 0.951;
- M4 0.975 / 0.982 / 0.969;
- ETTh1-h96 0.996 / 0.977 / 0.974;
- ETTh1-h48 0.960 / 1.000 / 1.000 (seeds 1 and 2 kept the zero-shot model);
- Weather 1.011 / 1.010 / 1.006 (slightly worse);
- the rest within ±1.5%.

Cost, all 11 tasks, RTX A5000:

| Strategy | Score | GPU s per origin | One-time GPU min |
|---|---|---|---|
| TimesFM default | 0.655 | 5.8 | – |
| TimesFM LoRA at 1,024 | 0.640 | 5.9 | 63 |
| TimesFM + Chronos-2 at 2,048 | 0.633 | 10.5 | – |
| TimesFM + Chronos-2 at 8,192 | 0.629 | 23.0 | – |
| TimesFM 8,192 | 0.633 | 37.4 | – |
| TimesFM 16,256 | 0.630 | 72.7 | – |
| TimesFM LoRA at 16,256 | 0.625 | 72.7 | 63 |
| TimesFM LoRA at 16,256 + Chronos-2 at 8,192 | 0.617 | 89.8 | 63 |
| … + Chronos-2 LoRA | 0.618 | 90.9 | 239 |
| Per-task context selection (all contexts run every origin; an upper bound) | 0.630 | 197 | – |

Interpretation:
- **Cheapest route to a given accuracy.**
  - Up to about 0.63: combine the default TimesFM with a zero-shot Chronos-2. It needs no training and its recurring cost is a seventh to a third of TimesFM's long context.
  - Beyond that: long context, then LoRA, then combination, each adding a little at a rising cost.
- **The earlier hypothesis "configuration > fine-tuning > combination per GPU hour" is not supported.** Long context is the most expensive at inference.
  - LoRA at 1,024 vs zero-shot 16,256: LoRA pays back its 63 training minutes after about 57 forecast origins, but is 1.5% less accurate.
  - The ranking depends on how often forecasts are produced.
- **Fine-tuning helps TimesFM a little and Chronos-2 not at all.** Chronos-2's validation gains (6 tasks) did not carry over to the evaluated windows, so the TimesFM result does not transfer to the second model.
  - Possible reasons, not tested: Chronos-2's smaller default adapter, its lower learning rate, or a model already well matched to these datasets.
- **Combination stays useful as insurance.** The combinations have the lowest worst-case regret (1.04–1.05, vs 1.065 for the best single model).

### 1.7 Pilots behind the direction decision (`results/pilots_2026-10/`)

| Pilot | Result | Label |
|---|---|---|
| Choosing TimesFM's context per task from past windows (Experiment 3 sweep, 4 tasks, windows 0–3) | Weather 0.231 → **0.167 (−28%)**, ETTm1 0.527 → **0.460 (−13%)**, ETTh1 and M4 unchanged. On those two tasks TimesFM nearly matches Chronos-2 (0.169, 0.441). **Superseded by §1.4** | Pilot |
| Chronos-2 forecasting all variables jointly (multivariate / cross-learning), 20 origins (rerun today, identical to the first run) | Weather −8.3%, ETTm1 −5.9%, ETTh1 −3.1%, ETTm2 0%, ETTh2 +2.2%, ECL **+14%** (unrelated series grouped together) | Pilot |
| Probabilistic combination | Mixing the three predictive distributions (1.043× best single model's quantile loss) beats averaging their quantiles (1.082×), the method used by leaderboard ensembles | Pilot |
| Damaged inputs (20% missing, missing block, outliers) | The models do **not** become more complementary: the equal mean stays 0.995–1.235× the best member in every condition (deterministic-seed rerun of 2026-10-26). TabPFN-TS degrades most (up to +43% with missing values on ETTm1, +61% with outliers on Weather) | Pilot |

### 1.8 Takeaways

1. Zero-shot foundation models are **largely redundant forecasters**: most of their error is shared. That limits what linear combinations of them can achieve, consistent with classical combination theory. It is not evidence of an irreducible error floor.
2. In practice, combining them amounts to **choosing the right model per task**. That has real value as insurance (worst case 1.03 vs up to 2.9), but no accuracy gain beyond the best model.
3. The levers that remain must reduce the **shared** error, by changing what the model does on the target data: its configuration, its weights, or the information it receives.
   - Configuration (§1.4) gives about −3.5% to −3.8% from one fixed long context. Per-task tuning shows no clear advantage over that.
4. Part of the model ranking in the benchmark reflects default settings rather than the models themselves. With a long context, TimesFM overtakes Chronos-2 on Weather and Exchange.
5. Fine-tuning (§1.5–1.6) adds a small gain on top of the long context for TimesFM, on different tasks, and nothing for Chronos-2. The more TimesFM itself is improved, the less combination with Chronos-2 adds; the first step of that pattern is statistically clear, the second borderline.
6. Per GPU second, combining two cheap zero-shot models is the most efficient route to the accuracy of TimesFM's long context. Which strategy is cheapest overall depends on how often forecasts are produced (§1.6).

### 1.9 Infrastructure

- **Code:** `src/timestransfer/combination.py` (combiners, rolling protocol) and `combination_analysis.py` (diversity, shared-error split, ideal-weight bound, DM tests, bootstrap, regret, learning curves, failure modes), with tests (all passing).
- **Archive:** all expensive outputs are on GitHub (`results/`, branch `ensemble_research`), verified to restore exactly.

---

## 2. Plans and prospects

### 2.1 Steps

| # | Step | Content | Compute | When |
|---|---|---|---|---|
| 1 | Close the combination study | **Done 2026-10-26:** shared-error split, ideal-weight bound, two-model theory, pools with AutoETS/AutoTheta. The figure of tasks on the correlation/accuracy plane moves to step 6 (thesis artifact pipeline) | CPU, minutes | done |
| 2 | TimesFM configuration study | **Done 2026-10-26** (§1.4): grid 3.1 h on the A5000; selection, combination after configuration, DM tests and bootstrap | 3.1 GPU h | done |
| 3 | Fine-tuning | **Done 2026-10-28** (§1.5–1.6): TimesFM LoRA with 3 seeds, Chronos-2 LoRA on all tasks (context 2,048). Long-context TimesFM LoRA skipped (the 1,024 adapter transfers to 16,256) | TimesFM 3 × 1.9 h, Chronos-2 2.9 h, pilots ~3 h | done |
| 4 | Three-way comparison | **Done 2026-10-28** (§1.6): common evaluation set without validation windows, block bootstrap, gain contrasts, seeds, measured inference cost, break-even | CPU minutes + 1 GPU h timing | done |
| 5 | Literature re-check | Error correlation of foundation models, context selection, fine-tuning vs zero-shot, multi-strategy comparisons; follow-ups to ECHO and Chronos-2 | – | before writing |
| 6 | Writing | Benchmark chapters as motivation → combination (redundancy) → configuration → fine-tuning → comparison → discussion; plus the deferred introduction and summary | – | November, until ~Nov 28 |

**Compute used (RTX A5000):**
- step 2 grid: 3.1 h;
- TimesFM LoRA: 3 seeds × ~1.9 h, including inference at both contexts;
- Chronos-2 LoRA: 3.1 h, including inference;
- timing runs: ~1 h;
- pilots: ~3 h.

No further GPU work is planned.

### 2.2 Cut first if time runs short

1. Chronos-2 related-series (multivariate) study: keep as a short section from the pilot.
2. External-validity datasets (GIFT-Eval subset).
3. Most of the probabilistic section: keep the mixture vs quantile-averaging result.

### 2.3 Risks

- **Configuration gains don't generalise beyond Weather and ETTm1.** Partly realised (§1.4): the gains are real but moderate (−3.5% aggregate) and come from one fixed long context, not per-task tuning. Fine-tuning must be compared against TimesFM at 16,256, the stronger zero-shot baseline, not only at 1,024.
- ~~The TimesFM LoRA setup fails~~ Resolved: the HF checkpoint reproduces the benchmark forecasts, and LoRA runs in an isolated environment (`finetune/`).
- **Possible pretraining contamination** (M4 and ECL are in some models' training data). Report it as a limitation; relative comparisons are less affected than absolute scores.
- **New related papers** in a fast-moving field. The step 5 re-check guards against this.
- **Writing time** is the most likely bottleneck, not compute.

### 2.4 Contribution and novelty (literature checks of 2026-10-15; repeat before writing)

1. **Measurement of shared vs model-specific error** across zero-shot foundation models, linked to the combination bound and classical theory. No prior measurement found.
2. **Weighting granularity under a leakage-free rolling protocol.** This contradicts the per-series / per-window weighting used by recent foundation-model ensembles (Toto 2.0, TimeRouter, ZooCast), at least on this benchmark.
3. **Backtested configuration of TimesFM.** The method itself is incremental (context sensitivity is documented; GITCO, arXiv 2606.05332).
   - What's new is the result under a leakage-free protocol: per-task selection shows no clear advantage over one fixed long context.
   - The long context alone gives about the same gain as combining foundation models, at a higher inference cost.
   - It changes three task-level rankings against Chronos-2.
4. **Combination vs configuration vs fine-tuning under one protocol.** No single source found. Fine-tuning vs zero-shot alone is known (e.g. Laglil et al., arXiv 2607.23146: LoRA +2–3% on medium/large datasets). Our result:
   - the levers are partly complementary: context and LoRA help on different tasks;
   - combination's gain shrinks as the single model improves;
   - fine-tuning helps TimesFM slightly and Chronos-2 not at all;
   - per GPU second, combining cheap zero-shot models beats long-context inference.

---

## 3. Expected results

| Arm | Expected result | Basis | What it would mean |
|---|---|---|---|
| Combination | Combining ≈ per-task model selection; ~4.5% better than a single fixed model, ≈0% better than the best model per task; ideal fixed weights ≤ ~3% (median) | **Established** | Redundancy, not the weighting method, limits foundation-model ensembles |
| Configuration | Fixed long context −3.5% to −3.8% (Traffic −11.5%, Weather −10%), none on short series (M4, ILI). Per-task selection from past windows shows no clear advantage over the fixed long context. Rankings vs Chronos-2 change on Weather and Exchange | **Established** (§1.4, §1.6) | Default-setting benchmarks partly confound model and configuration. The best form of configuration is a single good default, not per-task tuning |
| Fine-tuning | TimesFM LoRA: −2.4% at 1,024; with the adapter at 16,256, −0.9% beyond zero-shot 16,256 averaged over 3 seeds (CI 0.981–0.999), mainly ECL and M4. Chronos-2 LoRA: no effect (0.998, CI 0.988–1.009) | **Established** (§1.5–1.6) | Weight adaptation is a small lever for TimesFM and none for Chronos-2 here; the TimesFM result does not transfer to the second model |
| Combined strategies | Gain from combining with Chronos-2: 4.0% (default) → 1.8% (long context) → 1.25% (long context + LoRA). First drop clear (CI 0.3–4.0 points), second borderline. Fine-tuning Chronos-2 as well leaves the gain unchanged | **Established** (§1.6) | Model-side adaptation absorbs part of what combination provides; the redundancy result (§1.3) limits the rest |
| Cost-effectiveness | Not "configuration > fine-tuning > combination". TimesFM + zero-shot Chronos-2 reaches long-context accuracy at 1/7–1/3 of its inference cost with no training. LoRA at 1,024 is cheapest per forecast but 1.5% less accurate than long context (break-even with long context after ~57 origins) | **Established** (§1.6; A5000 timings) | Combining cheap zero-shot models is the efficient default; long context, then LoRA, buy the last 1–2% at a rising cost. The ranking of LoRA vs long context depends on forecast frequency |