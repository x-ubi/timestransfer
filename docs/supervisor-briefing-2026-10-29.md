# Briefing: what we did, what we found, and how to defend it (2026-10-29)

This briefing prepares you to present the October 2026 work to your supervisor and answer questions on it.

- §1 is the version to say out loud.
- §2 defines every term you will use.
- §3 walks through the work in order.
- §4–6 give the final numbers, the contributions and the limitations.
- §7 lists the questions you are likely to get, with answers.
- §8 covers what went wrong and how it was caught.

Numbers come from the archived runs in `results/`. The detailed record, with every table, is `docs/thesis-status-2026-10-26.md`. Section numbers like "status §1.6" refer to that file.

---

## 1. The one-minute version

> I compared three ways of transferring a pretrained time-series foundation model, TimesFM 2.5, to new datasets without training a new model:
> 1. combining it with other zero-shot foundation models;
> 2. changing how it is run (its inference configuration, mainly how much history it sees);
> 3. LoRA fine-tuning.
>
> All three were evaluated under one leakage-free rolling-origin protocol on 11 tasks, with bootstrap intervals, and with measured GPU cost.
>
> The main finding is that zero-shot foundation models make largely the same errors: 85–94% of the error is shared on 10 of 11 tasks. This limits how much combining them can help, and classical forecast-combination theory explains why.
>
> Each strategy helps a little:
> - a longer context gives about −4%;
> - combining with Chronos-2 gives about −4% on top of the default model;
> - fine-tuning adds about −1% on top of the long context for TimesFM, and nothing for Chronos-2.
>
> The improvements partly overlap: the better TimesFM gets, the less combination adds (4.0% → 1.8% → 1.25%). Counting compute cost reverses the intuitive ranking: combining TimesFM with a cheap zero-shot Chronos-2 matches the long-context accuracy at about a seventh of the inference cost and with no training.

**Five findings to remember:**
1. **Redundancy.** Foundation models share most of their error, so combining them is close to choosing the best model per task.
2. **Context matters; per-task tuning doesn't add to it.** One fixed long context (16,256 points) is the biggest single-model gain (−3.8%). Choosing the context per task from past data shows no clear advantage over that fixed setting.
3. **Fine-tuning is a small lever.** For TimesFM it gives −0.9% beyond the long context (3 seeds, CI 0.981–0.999). For Chronos-2 it gives nothing.
4. **The more TimesFM improves, the less combination adds.**
5. **Cost changes the recommendation.** Cheap zero-shot combination is the efficient default. The long context, then LoRA, buy the last 1–2% at a rising cost.

---

## 2. Concepts you need to explain

### 2.1 The models

| Model | What it is | Role here |
|---|---|---|
| **TimesFM 2.5** (Google) | Decoder-only transformer, ~200M parameters, pretrained on a large time-series corpus. Splits the history into patches of 32 points and forecasts quantiles. Accepts up to ~16k points of history | The main model the thesis is about |
| **Chronos-2** (Amazon) | Encoder-only transformer, ~120M parameters, patches of 16 points. Can attend across related series, but we used it on one series at a time. Context up to 8,192 | Second foundation model: combination partner and a second fine-tuning target |
| **TabPFN-TS** | Tabular foundation model applied to time series via engineered features | Third combination member. Weakest of the three and by far the slowest (~4,100 GPU s per forecast round) |
| SeasonalNaive | Repeats the last season | Baseline every score is measured against |
| AutoETS, AutoTheta, AutoARIMA | Classical statistical models | Benchmark comparison and extra combination pools |

All foundation models were used **zero-shot**, i.e. without training on our data, except in the fine-tuning arm.

### 2.2 The tasks

10 datasets, 11 tasks: ETTh1 is evaluated at two horizons (48 and 96).

| Task | Series | Frequency | Notes |
|---|---|---|---|
| M4 Hourly | 414 | hourly | short series (~700–1,000 points) |
| ETTh1 (h=48 and h=96), ETTh2 | 7 each | hourly | electricity transformer data |
| ETTm1, ETTm2 | 7 each | 15-min | |
| ECL | 321 | hourly | electricity consumption |
| Traffic | 862 | hourly | road occupancy |
| Weather | 21 | 10-min | |
| Exchange | 8 | daily | exchange rates, nearly a random walk |
| ILI | 7 | weekly | influenza-like illness, short |

Forecast horizon: 48 steps (96 for the second ETTh1 task).

### 2.3 Rolling-origin evaluation

Each task is forecast from **20 consecutive forecast origins** (10 for M4 and ILI), each one horizon apart. A forecast origin is the moment the forecast is made, also called a "window".

- Every forecast sees only data before its origin.
- Window 0 is the most recent; higher numbers are older.
- Rolling origins measure how a method performs repeatedly over time, not once. The variation across windows is real uncertainty, and the statistics account for it.

**Leakage-free protocol (the "rolling protocol").** Anything learned from data, such as combination weights or the context chosen per task, is fitted for window *k* only on windows older than *k*. A window is evaluated only if at least 3 older windows exist. That's why the combination study scores windows 0–16 (0–6 for M4 and ILI).

**The common evaluation set (step 4).** Fine-tuning uses the 3 oldest windows to decide when to stop training (early stopping), so forecasts there are not truly out of sample. The final comparison drops those 3 windows everywhere and refits every combiner without them. That leaves windows 0–13 (0–3 for M4 and ILI), identical for every strategy.

### 2.4 The metric

- **MASE** (mean absolute scaled error): the forecast's mean absolute error divided by the in-sample error of a seasonal-naive forecast. This makes errors comparable across series of different scales.
- **Relative MASE:** a model's task-level MASE divided by SeasonalNaive's MASE on the same rows. 1.0 means as good as SeasonalNaive; 0.63 means 37% lower error.
- **Aggregate score:** the geometric mean of relative MASE over the 11 tasks. Every task counts equally.
- **"Ratio to X":** score divided by X's score; below 1 means better than X. Example: 0.962 = 3.8% lower error. All the "−x%" figures come from these ratios.

### 2.5 Statistics

- **Bootstrap confidence intervals (95%).** We resample the data many times (2,000 replicates), recompute the aggregate each time, and take the middle 95% of the results. "CI 0.940–0.989" means the ratio is plausibly anywhere in that range. If the range excludes 1, the difference is statistically clear.
- **Resampling schemes:** what gets resampled within each task.
  - *series*: treats the observed windows as fixed;
  - *windows*: forecast origins are the uncertain part;
  - *both*: series and windows together (the default for the headline numbers);
  - *moving blocks of 3 consecutive windows*: allows for correlation between neighbouring windows. The external reviewer asked for this one;
  - *series + window blocks*.

  Results are very similar across schemes.
- **Paired contrasts.** To claim "combination adds less after fine-tuning", we bootstrap the *difference* between two gains directly, not just each gain separately.
- **Diebold–Mariano (DM) test:** a per-task test of whether two forecasters' losses differ across windows.
  - Uses a Newey–West variance (allows for autocorrelation) and the Harvey–Leybourne–Newbold small-sample correction.
  - Holm correction across the 11 tasks controls false positives from testing many tasks.
  - With 14–17 windows per task these tests have little power, so few per-task differences are significant even when the aggregate is.
- **Regret:** a strategy's score divided by the best single strategy for that task, chosen after the fact. Worst-case regret measures how badly a strategy can go wrong on some task.
- **Wording discipline.** "No clear difference" means the interval includes 1. It does **not** mean "equal". Where useful we say what the interval rules out, e.g. "differences larger than ~3%".

### 2.6 The three transfer strategies

**1. Combination.** A weighted average of the models' point forecasts.
- Main method: **task-level non-negative least squares (NNLS)**, one set of non-negative weights per task, fitted on past windows.
- Also tested:
  - equal weights;
  - the median;
  - picking the best model per task (`select_task`);
  - per-series weights, horizon-specific weights;
  - a learned meta-model in the style of FFORMA (a model that predicts the weights from features of each series).

**2. Configuration.** Changing how TimesFM is run without changing its weights:
- **context length**, i.e. how much history it sees: 512 to 16,256 points;
- **flip invariance**: also forecasts the sign-flipped series and averages, which makes the model symmetric to sign.

**3. Fine-tuning with LoRA (Low-Rank Adaptation).**
- The pretrained weights stay frozen. Each linear layer gets a small trainable correction ΔW = B·A of low rank (rank 4 for TimesFM).
- That is ~1.4M trainable parameters instead of 200M, so it is cheap and less prone to overfitting small datasets.
- B starts at zero, so training starts exactly from the zero-shot model.
- One adapter is trained per task, only on that task's data.

### 2.7 How this is "transfer learning"

A foundation model is trained once on a broad source corpus and transferred to a target dataset. The thesis compares three increasingly invasive ways to transfer it:

| Strategy | What it changes |
|---|---|
| Zero-shot use | nothing, plus combining several such models |
| Configuration | inference-time adaptation |
| LoRA | parameter-efficient adaptation of the weights |

The title "Transfer learning in time series modelling" covers all three.

---

## 3. What we did, step by step

### 3.1 Starting point: the benchmark (thesis Experiments 1–3, already written)

Setup: 9 models, 11 tasks, 5 rolling origins.

| Model | Aggregate score |
|---|---|
| TimesFM | 0.748 |
| Chronos-2 | 0.785 |
| TabPFN-TS | 0.840 |
| AutoARIMA (8 tasks) | 0.926 |

- TimesFM's lead rested largely on ILI. Without ILI, Chronos-2 led.
- Rankings changed with task, metric and forecast origin: the per-task winner was best in only 25 of 55 single-window runs.
- TimesFM was sensitive to context length (Weather improved with more history; ETTm1 with less). This seeded the configuration arm.

### 3.2 Choosing the direction (early October)

Your professor suggested four directions: improve a foundation model, ensembling, fine-tuning, and data issues (noise, missing values).

- Pilots showed that damaged inputs do not make the models complementary, so the data-issues direction stayed a negative pilot.
- The confirmed plan combines the other three into one comparison centred on TimesFM.

### 3.3 Combination study (status §1.2)

**Question:** can combining TimesFM, Chronos-2 and TabPFN-TS beat the best single model?

**Method:**
- 20 rolling origins;
- 15 combination methods, all fitted only on past windows;
- DM tests, bootstrap intervals and regret.

**Results:**
- **Best combiner** (task-level NNLS weights): 0.633 vs TimesFM 0.663. Ratio 0.955, CI 0.934–0.975: a real 4.5% gain over the single default model.
- **But it only matches the best model per task chosen after the fact:** mean regret 1.002, max 1.025. Choosing the past-best model per task gives 0.640.
- **The plain average is worse than TimesFM alone:** 1.029×, CI 1.007–1.060.
- **Finer weighting adds nothing:** per series 0.638, per horizon 0.634–0.635, the feature-based meta-learner 0.661.
- **One past window is enough to learn the weights:** the learning curve is flat.
- **Value as insurance:** worst-case regret is 1.03 for the combiner vs 1.10 for TimesFM and ~2.9 for Chronos-2 and TabPFN-TS.

**Interpretation:** combining foundation models amounts to per-task model selection with a little smoothing. Its value is robustness, not accuracy beyond the best model.

### 3.4 Why combination is limited (status §1.3)

**Shared error.** Each model's error splits into the part common to all three (their average error) and a model-specific remainder.
- The common part is 85–94% of the error on 10 of 11 tasks (60% on ILI).
- The first principal component of the error covariance tells the same story.

**Ideal-weight bound.** Even weights chosen after the fact, with full knowledge, reduce squared error vs the best single model by only a median of 3%.

**Classical theory (Bates & Granger, 1969).**
- For two forecasts with error standard deviations σ₁ ≤ σ₂ and error correlation ρ, non-negative combination helps if and only if ρ < k = σ₁/σ₂.
- The gain factor is (1 − ρ²)/(1 + k² − 2ρk).
- Our pairs have ρ = 0.75–0.97 and k = 0.82–1.00, so they sit near the ρ = k line where the gain vanishes.
- High correlation by itself does not rule out a gain (ρ = 0.95, k = 0.99 still gives ~1.6%). What matters is that ρ is close to k.
- ILI is the exception (ρ ≈ 0, k = 0.37): the largest theoretical gain (~12%), but it comes mostly from down-weighting the much worse model.

**Statistical models.** Adding AutoETS/AutoTheta increases diversity on paper (the ideal bound improves to a median of 0.938). But weights learned from past windows do *worse* with them at every amount of past data: the diversity is not recoverable out of sample.

**Bias correction.** Correcting each series' recurring bias from past windows makes things worse (pilot). The shared error is not a stable offset.

### 3.5 Configuration study (step 2, status §1.4)

**Question:** how much does TimesFM's inference configuration matter, and can it be chosen per task from past windows?

**Method:**
- TimesFM at 6 context lengths × flip invariance on/off, on the same 20 origins.
- Chronos-2 at 4 contexts as a fairness control.
- The default variants reproduce the earlier forecasts exactly.
- 3.1 GPU hours.

**Results (17-window numbers):**
- **One long context for all tasks (16,256):** 0.639 vs 0.663, ratio 0.965 (CI 0.944–0.987). Traffic −11.5%, Weather −10.2%; smaller gains elsewhere; no change on M4 and ILI (their series are shorter than 1,024).
- **Choosing the context per task from past windows:** 0.640, no clear advantage over always using 16,256. The ranking of settings changes too much from window to window. ETTm1 is the one task where it helps (−10%, by choosing 512).
- **Flip invariance off:** worse at every context, though about twice as fast at long contexts.
- **Chronos-2's context barely matters:** 0.715 at 1,024 vs 0.702 at 8,192.
- **Rankings change.** With the long context, TimesFM overtakes Chronos-2 on Weather, Exchange and Traffic. Chronos-2 stays ahead on ECL and the four ETT tasks.
- **Long context alone gives about the same gain as the original combination:** 0.634 vs 0.639, no clear difference.

**Interpretation:** configuration is a real lever, best used as one good default rather than tuned per task. Benchmarks run at default settings partly compare configurations, not models.

### 3.6 Fine-tuning (step 3, status §1.5)

**Implementation checks before any training:**
- The Hugging Face version of TimesFM reproduces our benchmark forecasts (differences ≤ 3·10⁻⁵ MASE units).
- Its built-in training loss is buggy: it pairs quantile outputs with the wrong levels. We wrote our own quantile (pinball) loss.

**TimesFM LoRA setup:**

| Setting | Value |
|---|---|
| Adapter | rank 4, on all linear layers |
| Optimiser | AdamW, learning rate 1e-4, cosine schedule |
| Training length | 1,000 steps, batch 32, context 1,024 |
| Training data | history before the 3 oldest windows |
| Early stopping | on the 3 oldest windows, with the untrained model (step 0) included as a candidate |
| Cost | ~6 GPU min per task (63 min for all 11) |
| Seeds | 3 (0, 1, 2) |

The learning rate was chosen on a pilot (Traffic and ETTm1; 1e-5 and 3e-5 also tried).

**Key design result:** the adapter trained at context 1,024 also works when forecasting at 16,256. That gives fine-tuned + long context without long-context training.

**Chronos-2 LoRA setup:**
- The library's own `fit()` with its default adapter (rank 8, attention and output layers).
- Learning rate 1e-5, effective batch 256, context 2,048, forecasting at 2,048 and 8,192.
- Same validation rule: the fine-tuned model is used only if it beats the untrained one on the validation windows.
- 2.9 GPU h of training.

**Where fine-tuning did nothing.** On several tasks (ETTm1, ETTm2, ILI for TimesFM) validation loss got worse from the first checkpoint. Early stopping then kept the untrained model, so those forecasts equal zero-shot.

### 3.7 Three-way comparison and cost (step 4, status §1.6)

All results below are on the common evaluation set (§2.3).

| Strategy | Score | vs default | GPU s per forecast round (all 11 tasks) | One-time GPU min |
|---|---|---|---|---|
| TimesFM default (context 1,024) | 0.655 | – | 5.8 | – |
| TimesFM + Chronos-2 at 2,048 | 0.633 | −3.5% | 10.5 | – |
| TimesFM + Chronos-2 at 8,192 | 0.629 | −4.0% | 23.0 | – |
| TimesFM, context 8,192 | 0.633 | −3.4% | 37.4 | – |
| TimesFM, context 16,256 | 0.630 | −3.8% | 72.7 | – |
| TimesFM LoRA, forecasting at 1,024 | 0.640 | −2.4% | 5.9 | 63 |
| TimesFM LoRA at 16,256 (3 seeds) | 0.623–0.626 | −4.5% to −5.0% | 72.7 | 63 |
| TimesFM LoRA at 16,256 + Chronos-2 at 8,192 | **0.617** | **−5.8%** (CI 0.914–0.969) | 89.8 | 63 |
| Chronos-2 zero-shot (8,192) | 0.702 | +7.2% | 17.1 | – |
| TabPFN-TS | 0.758 | +15.7% | ~4,100 | – |

All combinations above use task-level NNLS weights. "Forecast round" means one forecast origin across all 11 tasks.

**Key contrasts** (95% bootstrap CIs, series and windows resampled):

| Comparison | Ratio (CI) | Reading |
|---|---|---|
| TimesFM 16,256 vs default | 0.962 (0.940–0.989) | clear gain |
| Per-task context selection vs 16,256 | 0.999 (0.980–1.017) | no clear advantage |
| TimesFM LoRA at 16,256 vs zero-shot 16,256, mean of 3 seeds | 0.991 (0.981–0.999) | small gain |
| Same, per seed | 0.992 / 0.993 / 0.988 | two of three intervals touch 1 |
| Chronos-2 LoRA vs zero-shot Chronos-2 | 0.998 (0.988–1.009) | no effect |
| Combination gain with default / long / LoRA+long TimesFM | 0.960 / 0.982 / 0.988 | gain shrinks |
| Difference in gain, long vs default | 1.023 (1.003–1.040) | clearly smaller |
| Difference in gain, LoRA+long vs long | 1.006 (1.000–1.012) | borderline; 1.001–1.011 with window blocks |
| Fine-tuning Chronos-2 as well: change in combination gain | 1.001 (0.996–1.006) | none |
| TimesFM + Chronos-2 at 2,048 vs TimesFM 16,256 | 1.004 (0.976–1.028) | no clear difference |

**Where TimesFM fine-tuning helps** (vs zero-shot 16,256, seeds 0/1/2):

| Task | Ratios | Reading |
|---|---|---|
| ECL | 0.956 / 0.958 / 0.951 | consistent gain |
| M4 | 0.975 / 0.982 / 0.969 | consistent gain |
| ETTh1-h96 | 0.996 / 0.977 / 0.974 | gain in 2 of 3 seeds |
| ETTh1-h48 | 0.960 / 1.000 / 1.000 | gain in seed 0 only |
| Weather | ~1.01 | slightly worse |
| All other tasks | within ±1.5% | no clear effect |

**Cost interpretation:**
- Inference cost is per forecast round; training is paid once.
- Fine-tuning at 1,024 pays back its 63 training minutes vs the long context after ~57 forecast rounds, but is 1.5% less accurate.
- Combining TimesFM with a zero-shot Chronos-2 reaches long-context accuracy at 1/7–1/3 of the inference cost with no training. It is the efficient default.

### 3.8 External review and the corrections it triggered

An independent reviewer read the results and raised several issues. All were addressed:
1. **Early-stopping windows leaked into combination fitting.** Fixed by the common evaluation set; the earlier 0.621 became 0.617 under the clean protocol.
2. **The cost ranking was unsupported.** We measured inference cost and separated one-time from recurring cost. The intuitive "configuration > fine-tuning > combination" ranking turned out wrong.
3. **"Not significant" isn't "equal".** Wording corrected throughout, and the theory sentence was made precise (the ρ < k condition).
4. **Window dependence in the bootstrap.** Added the moving-block bootstrap. The results hold.
5. **One training seed.** Added two seeds. The gain holds on average but is borderline per seed.

---

## 4. What is established, borderline, or negative

| Status | Findings |
|---|---|
| **Clearly established** | Redundancy of foundation-model errors. Long context helps (−3.8%). Combination helps the default model (−4.0%). The first drop in combination gain after the long context. TimesFM beats Chronos-2 in aggregate. Combination lowers worst-case regret. Per-task selection and finer weighting add nothing over simple alternatives |
| **Borderline** | TimesFM fine-tuning gain beyond long context: clear when averaged over seeds, borderline per seed. The second drop in combination gain (after fine-tuning) |
| **Negative, but clean** | Per-task context selection; per-series, per-horizon and meta-learned combination weights; statistical models as extra combination members; bias correction; Chronos-2 fine-tuning; damaged inputs making models complementary (pilot) |

A negative result is a result. Several of them contradict practices in recent papers, e.g. per-series weighting in recent foundation-model ensembles.

---

## 5. Contributions and novelty

From literature checks of 2026-10-15; step 5 will re-check them before writing.

1. **Measuring shared vs model-specific error across zero-shot foundation models**, and linking it to the combination bound and classical theory. No prior measurement found.
2. **Weighting granularity under a leakage-free rolling protocol.** Coarse, task-level weights are best. This goes against the per-series/per-window weighting used by recent foundation-model ensembles (e.g. Toto 2.0, TimeRouter, ZooCast), at least on this benchmark.
3. **Backtested configuration of TimesFM.** Context sensitivity itself is known (e.g. GITCO, arXiv 2606.05332). New here:
   - per-task selection shows no clear advantage over one fixed long context;
   - the long context alone matches the combination gain;
   - it changes model rankings.
4. **Combination vs configuration vs fine-tuning under one protocol, with measured cost.** No single source found; fine-tuning vs zero-shot alone is known (e.g. Laglil et al., arXiv 2607.23146, reporting LoRA gains of 2–3%). New here:
   - the levers partly overlap and partly complement each other;
   - combination's gain shrinks as the single model improves;
   - fine-tuning helps TimesFM slightly and Chronos-2 not at all;
   - cost reverses the intuitive ranking.
5. **Side contributions:** two library pitfalls found and documented (§8).

**Honest framing:** this is not "a method that improves TimesFM by X%". It is a careful account of what helps when transferring time-series foundation models, why combination is limited, and what each option costs.

---

## 6. Limitations and threats to validity

**Breadth**
- 10 datasets, 11 tasks, univariate forecasting, point forecasts as the main metric.
- Equal task weighting in the geometric mean.

**Statistical power**
- 14–17 windows per task (4–7 for M4 and ILI) make per-task tests weak.
- The aggregate is more reliable than any single task.

**Pretraining contamination**
- Some datasets (e.g. M4, ECL) may be in the foundation models' pretraining corpora.
- This affects absolute scores more than comparisons between strategies of the same model.

**Hyperparameters**
- The TimesFM learning rate was chosen on a pilot after Traffic's test score had been seen. Validation loss pointed the same way, so the effect is likely negligible, but it should be stated.
- Chronos-2 fine-tuning used library defaults. "Fine-tuning doesn't help Chronos-2" means "as configured". A tuned run (~3 GPU h) would strengthen this.

**Implementation choices**
- TimesFM fine-tuning goes through the Hugging Face implementation, verified equivalent to the benchmark.
- Timings use one GPU type (RTX A5000) and capped contexts.
- TabPFN-TS timing comes from logs, not a clean measurement.

**Seeds**
- 3 seeds for TimesFM fine-tuning, 1 for Chronos-2.

**Scope**
- Probabilistic combination and Chronos-2's multivariate mode are pilots only.

---

## 7. Questions your supervisor may ask, with answers

**Q: Why TimesFM as the central model?**
It was the original thesis focus and the strongest model in the benchmark. It also has a usable fine-tuning path: the HF implementation, checked against the benchmark forecasts, plus `peft`. Chronos-2 serves as a second model, to test whether conclusions generalise.

**Q: Why not train a model from scratch?**
That is outside the scope agreed with you and the compute budget. The thesis is about transferring existing pretrained models.

**Q: Is there any leakage?**
- Combination weights and context selection for window *k* are fitted only on older windows.
- Fine-tuning trains on history before the 3 oldest windows and early-stops on those windows.
- The final comparison excludes those windows from all fitting and scoring.
- The earlier provisional combination number (0.621) used them and was replaced by 0.617.

**Q: Why are the gains so small?**
- Gains of a few percent are typical in forecasting at this level: Chronos-2 and TimesFM themselves differ by ~7%.
- The redundancy result explains the ceiling for combination.
- Fine-tuning has little target data per task, and the models are already strong zero-shot.

**Q: Is the fine-tuning gain real or noise?**
- Averaged over three seeds it is clear (CI 0.981–0.999). Per seed it is borderline.
- The gain comes consistently from ECL and M4. On ETTh1 it appears in one seed only.
- So: small, and consistent on some tasks.

**Q: Why did per-task context selection not work, when context matters so much?**
- The best context for a task changes from window to window; picks depended on recent windows.
- With 3–16 past windows, choosing among 12 settings is noisy.
- One long context gets most of what perfect per-task choice could give: 0.639 vs 0.629 for the after-the-fact best setting per task, which no method can know in advance (17-window set).

**Q: Why does an adapter trained at context 1,024 work at 16,256?**
- Empirically it does: −0.9% vs zero-shot 16,256.
- Plausible reason: the adapter changes how patches are processed in each layer, not anything specific to input length. This is a hypothesis we have not tested.

**Q: Why didn't fine-tuning help Chronos-2?**
- Its validation gains on 6 tasks did not carry over to the evaluation windows.
- Possible reasons, all untested: the library's smaller default adapter (attention and output layers only), a lower learning rate (1e-5), or a model already well suited to these datasets.
- It shows the TimesFM result does not automatically transfer to other models.

**Q: Why is combination limited if the models are different architectures?**
- Their errors are 75–97% correlated and their accuracies similar, so they sit near the ρ = k boundary of classical combination theory.
- Different architectures trained on overlapping data make similar mistakes. The shared error is mostly what is hard to predict in the data itself, at least for these models.

**Q: Couldn't a smarter combiner (per series, meta-learning) do better?**
We tested per-series, per-horizon and feature-based meta-learned weights. None beat simple task-level weights out of sample, because the winner per series changes about as often as chance.

**Q: Isn't "no significant difference" just low power?**
- Partly, which is why we report intervals, not just p-values.
- Example: long context vs the original combination has CI 0.967–1.014, ruling out differences beyond ~3%. That's informative even if not "significant".

**Q: How did you handle multiple testing?**
Holm correction across tasks for DM tests. The headline claims rest on aggregate bootstrap intervals, which are single comparisons per question.

**Q: What does the cost analysis actually show?**
- Combining default TimesFM with Chronos-2 at 2,048 costs 10.5 GPU s per forecast round across all tasks; TimesFM at 16,256 costs 72.7 s; the accuracy shows no clear difference.
- LoRA training is a one-time 63 GPU min. It pays back vs the long context after ~57 rounds, but is less accurate.
- The best strategy therefore depends on the accuracy target and the forecasting frequency.

**Q: How is this "transfer learning"?**
- The pretrained model is the source knowledge.
- Zero-shot use and combination transfer it unchanged.
- Configuration adapts how it is used.
- LoRA adapts its weights to the target data with few parameters.
- The thesis compares these transfer modes.

**Q: What is novel compared to the literature?**
See §5. Be careful to say "no prior measurement found as of our checks". Step 5 re-verifies this before writing.

**Q: What about probabilistic forecasts?**
- Point forecasts are the main metric.
- A pilot shows mixing predictive distributions beats averaging quantiles (1.043× vs 1.082× the best single model's quantile loss).
- This can be a short section, or cut if time runs short.

**Q: What would you do with more time?**
- A tuned Chronos-2 fine-tuning run.
- More datasets (e.g. a GIFT-Eval subset).
- Probabilistic evaluation.
- Chronos-2's multivariate mode (pilot: helps on physically coupled variables, hurts on unrelated series).
- Error-correction methods for the shared error (e.g. ECHO, arXiv 2610.16433).

**Q: Are the runs reproducible?**
- All expensive outputs are archived in `results/` with restore scripts.
- TimesFM fine-tuning is deterministic (a rerun reproduces forecasts exactly).
- Chronos-2 adapters are archived because its training is not bit-reproducible.
- Every check (implementation equivalence, reload equivalence) is scripted.

---

## 8. What went wrong, and how it was caught

These show the verification was careful. They are also good material for the method chapter.

1. **The configuration pilot overstated the gains.** A 4-window pilot suggested −28% on Weather and −13% on ETTm1; on 20 origins the gains are ~−10%. Lesson: small pilots on few origins are unreliable. The status document marks the pilot as superseded.
2. **The Hugging Face TimesFM loss is misaligned.** Its quantile loss pairs outputs with the wrong levels (shifted by one), and it fits the median with squared error. We use our own pinball loss.
3. **Chronos-2's `fit()` returns a model that is unsafe to forecast with:**
   - the model is still in training mode, so dropout is active and forecasts are random;
   - its forward pass carries a bf16 mixed-precision wrapper from the training library;
   - TF32 low-precision matrix multiplication stays on for the rest of the process.

   The first Chronos-2 pilot numbers were invalid because of this; for example, the ETTm1 "−1.2%" was an artifact. A smoke test caught it.
   - **Fix:** forecast only from the saved adapter, reloaded into a fresh full-precision model.
   - **Verification:** reload and zero-shot reproduction match exactly (difference 0.0).
4. **Timing artefact.** The TimesFM implementation pads inputs to the full context, inflating long-context cost ~20× on short-series tasks with identical forecasts. Timings cap the context at the longest history.
5. **CPU contention.** A CPU test running alongside GPU training starved it. Later analyses were run on capped cores.

---

## 9. What is left

- **Step 5 — literature re-check (~1 day):** confirm the novelty claims still hold and verify every citation.
- **Step 6 — writing (until ~Nov 28).**
  - Chapters: combination and redundancy → configuration → fine-tuning → three-way comparison with cost → discussion. Plus the introduction and summary.
  - Figures and tables come from the thesis artifact pipeline.
  - Use one window convention per chapter (17-window numbers for steps 1–3, the common 14-window set for the comparison) and state it.
- **Optional:** a tuned Chronos-2 fine-tuning run; reporting the two library pitfalls upstream (your decision, since it's public).

---

## 10. Where everything is

| What | Where |
|---|---|
| Full results record with every table | `docs/thesis-status-2026-10-26.md` |
| Confirmed direction and plan | `docs/research-direction-2026-10.md` |
| Archived outputs and restore instructions | `results/README.md` |
| Combination code and analysis | `src/timestransfer/combination.py`, `combination_analysis.py`, `scripts/run_combination*.py` |
| Configuration study | `configs/timesfm_config.yaml`, `scripts/run_timesfm_config_selection.sh`, `scripts/run_config_analysis.py` |
| Fine-tuning | `finetune/train_lora.py` (TimesFM), `finetune/chronos2_ft.py` (Chronos-2), `finetune/timesfm_hf.py`, `finetune/check_equivalence.py` |
| Final comparison and cost | `scripts/run_strategy_comparison.py`, `finetune/measure_inference.py`, tables in `results/strategy_comparison/` |
