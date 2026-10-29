# Results briefing: TSFM transfer strategies (2026-10-29)

Full tables: `docs/thesis-status-2026-10-26.md`; archived outputs: `results/`.

**Notation.** Unless stated otherwise:
- **Score:** geometric mean over 11 tasks of task-level MASE relative to SeasonalNaive.
- **Ratios:** score / reference score.
- **%:** (ratio − 1) × 100. Negative means lower error, i.e. better than the reference.
- **CIs:** 95% task-stratified bootstrap, 2,000 replicates, resampling series × windows.

## 1. Summary

**Scope.** TimesFM 2.5 transferred to 10 datasets / 11 tasks via three routes:
1. zero-shot combination with Chronos-2 and TabPFN-TS;
2. inference configuration (context length, flip invariance);
3. LoRA.

All three run under one rolling-origin protocol with no look-ahead.

**Final comparison** (common evaluation set, §2):

| Strategy | Score | Ratio vs TimesFM default (CI) | % vs default (CI) | GPU s / origin (11 tasks) | One-time GPU min |
|---|---|---|---|---|---|
| TimesFM, ctx 1024 (default) | 0.655 | – | – | 5.8 | – |
| TimesFM + Chronos-2 ctx 2048, NNLS per task | 0.633 | 0.966 (0.944–0.985) | −3.5% (−5.6 to −1.5) | 10.5 | – |
| TimesFM + Chronos-2 ctx 8192, NNLS per task | 0.629 | 0.960 (0.938–0.982) | −4.0% (−6.2 to −1.8) | 23.0 | – |
| TimesFM, ctx 16256 | 0.630 | 0.962 (0.940–0.989) | −3.8% (−6.0 to −1.1) | 72.7 | – |
| TimesFM LoRA, inference at 1024 | 0.640 | 0.976 (0.960–0.989) | −2.4% (−4.0 to −1.2) | 5.9 | 63 |
| TimesFM LoRA, inference at 16256 (seeds 0/1/2) | 0.625 / 0.626 / 0.623 | 0.954 / 0.955 / 0.951 | −4.6% / −4.5% / −5.0% | 72.7 | 63 |
| TimesFM LoRA@16256 + Chronos-2 ctx 8192 | 0.617 | 0.942 (0.914–0.969) | −5.8% (−8.6 to −3.1) | 89.8 | 63 |

**Findings:**
1. **Error redundancy.** Shared component (the members' mean error) carries 85–94% of the error on 10/11 tasks (ILI 0.60). Ex-post optimal simplex weights cut MSE vs the best member by a median of 3%. Combination ≈ per-task model selection.
2. **Context is the main single-model lever:** 16256 vs 1024 is −3.8%. Per-task context selection from past windows shows no clear advantage over fixed 16256: 0.999, i.e. −0.1% (CI −2.0 to +1.7%).
3. **LoRA:** adds −0.9% beyond zero-shot 16256 for TimesFM (geomean over 3 seeds, CI −1.9 to −0.1%), but nothing for Chronos-2 (−0.2%, CI −1.2 to +0.9%).
4. **The combination gain shrinks as the TimesFM member improves:**

   | TimesFM member | Gain from adding Chronos-2 |
   |---|---|
   | default | 4.0% |
   | long context | 1.8% |
   | long context + LoRA | 1.25% |

   The first drop is clear (CI 0.3–4.0 points). The second is borderline (CI −0.02 to 1.15 points).
5. **Cost reverses the intuitive ordering.** TimesFM + zero-shot Chronos-2 at ctx 2048 matches ctx 16256 (+0.4%, CI −2.4 to +2.8%) at 1/7 of the inference cost, with no training.

## 2. Protocol

**Rolling origin**
- 20 origins with stride = horizon (10 for M4 and ILI); H = 48, plus ETTh1 at H = 96.
- MASE is scaled by the per-window in-sample seasonal-naive MAE.

**Look-ahead control**
- Combination weights and selected configurations for window k are fitted on windows > k only (older windows), with at least 3 such windows.
- Steps 1–3 evaluate windows 0–16 (0–6 for M4 and ILI).

**Fine-tuning split**
- Training on history before window K−1; early stopping on windows K−3..K−1, with step 0 (the untrained model) as a candidate.

**Common evaluation set (step 4)**
- Windows K−3..K−1 are dropped from all fitting and scoring, because they are in-sample for early stopping.
- Combiners are refitted without them, leaving windows 0–13 (0–3 for M4 and ILI) for every strategy.

**Inference**
- Per-task DM tests: Newey–West long-run variance, HLN correction, Holm across tasks.
- Bootstrap schemes: series, windows, both, circular moving blocks of 3 windows, series × blocks. The schemes agree closely; headline CIs use series × windows.
- Paired contrasts of gains are bootstrapped directly as ratios of ratios.

## 3. Results by arm

### 3.1 Combination (20 origins, steps-1–3 window set)

**Pool and methods**
- Members: TimesFM ctx1024, Chronos-2 ctx8192, TabPFN-TS.
- Combiners:
  - equal mean, median;
  - select_task, select_series;
  - inverse-MASE (p = 1, 2, 4, top-2);
  - NNLS: global, per task, per task × horizon bins, per series, shrunk;
  - FFORMA-style leave-one-dataset-out (LODO) meta-learner.

**Results**
- NNLS per task: 0.633 vs TimesFM 0.663, ratio 0.955, i.e. −4.5% (CI −6.6 to −2.5%).
- Against the ex-post best member per task: mean regret 1.002 (+0.2%), max 1.025 (+2.5%). select_task: 0.640.
- Equal mean: 1.029× TimesFM, i.e. +2.9% (CI +0.7 to +6.0%).
- Finer granularity does not help: per horizon 0.634–0.635, per series 0.638, LODO meta-learner 0.661.
- Learning curve flat: one past window suffices.
- Per-series winner persistence ≈ chance.
- Worst-case regret: combiner 1.03 (+3%), TimesFM 1.10 (+10%), Chronos-2 2.90 (+190%), TabPFN-TS 2.92 (+192%).

### 3.2 Redundancy diagnosis (squared-error view)

**Shared error.** Shared share is 0.85–0.94 except ILI (0.60); the first-PC share agrees.

**Ex-post simplex-weight bound** (exact support enumeration on the second-moment matrix of scaled errors)
- Median 0.969 vs best member (−3.1% MSE); ILI 0.878 (−12.2%), ETTm1 0.884 (−11.6%).

**Two-member Bates–Granger analysis**
- With non-negative weights, a gain exists iff ρ < k = σ₁/σ₂. The optimal ratio is (1−ρ²)/(1+k²−2ρk), with uncentred ρ.
- Observed ρ = 0.75–0.97 and k = 0.82–1.00: pairs sit near ρ = k, so gains are small.
- ILI pair: ρ ≈ 0, k = 0.37, bound 0.88 (−12% MSE). The gain comes from down-weighting the weak member.

**Statistical members** (AutoETS, AutoTheta, SeasonalNaive)
- The shared share drops (e.g. M4 0.43) and the bound improves (median 0.938, −6.2% MSE).
- But out-of-sample NNLS is worse at every past-window budget: 0.625 → 0.629 → 0.632.
- The diversity is not exploitable out of sample.

**Per-series bias correction from past windows:** worse on most tasks. A series' mean error correlates only −0.16 to 0.37 between consecutive windows.

### 3.3 Configuration (step 2, steps-1–3 window set)

Grid: TimesFM ctx {512 … 16256} × force_flip_invariance on/off; Chronos-2 ctx {1024 … 8192}. The defaults reproduce R1 bit-exactly.

| Result | Number |
|---|---|
| ctx 16256 vs 1024 | 0.965 (0.944–0.987): −3.5% (−5.6 to −1.3%) |
| Biggest per-task gains | Traffic −11.5%, Weather −10.2%, ETTh2 −4.6%, ECL −3.9%, Exchange −3.9% |
| M4, ILI | unchanged (series shorter than 1024) |
| select_task over contexts | 0.640, vs fixed 16256 at 0.639; ex-post oracle 0.629 |
| Where selection helps | only ETTm1 (picks 512, −10%) |
| Where selection loses | part of the gain on Weather and ETTh1 |
| Flip off | worse at every context; at 16256, worse on 8/11 tasks (3 significant after Holm); ~2× faster |
| Chronos-2 context | nearly irrelevant: 0.715 (1024) → 0.702 (8192) |
| Rankings | at 16256, TimesFM overtakes Chronos-2 on Weather, Exchange, Traffic; Chronos-2's aggregate deficit widens from +5.9% to +9.8% |
| Original combination vs ctx 16256 | 0.992 (0.967–1.014): −0.8% (−3.3 to +1.4%) |
| Shared error | the TimesFM–Chronos-2 shared share is unchanged by the longer context |

### 3.4 LoRA (step 3)

**TimesFM**
- HF `TimesFm2_5ModelForPrediction` with PEFT. Verified against the benchmark runner at ≤3.3·10⁻⁵ MASE after replicating timesfm's quantile-crossing fix.
- Adapter: r = 4, α = 8, dropout 0.05, all-linear (~1.4M trainable parameters).
- Optimiser: AdamW, lr 1e-4, wd 0.01, cosine schedule, clip 1.0; 1000 steps, batch 32, ctx 1024.
- Loss: own pinball loss over q0.1–q0.9, per-sample normalised by context mean and std.
- The HF built-in loss is not used: its quantile channels are misaligned by one level, and it applies MSE to the median.
- lr selected on Traffic/ETTm1 pilots (1e-4 vs 3e-5 vs 1e-5).
- Training is deterministic.
- An adapter trained at ctx 1024 transfers to inference at 16256.

**Chronos-2**
- `Chronos2Pipeline.fit` with the default LoRA: r = 8, attention q/k/v/o + output layer.
- lr 1e-5, effective batch 256, ctx 2048; inference at 2048 and 8192.
- Selection: the fine-tuned model replaces zero-shot only if its validation pinball loss is lower (true on 6/11 tasks).

**Per task, TimesFM LoRA@16256 vs zero-shot 16256 (seeds 0/1/2):**

| Task | Ratios | % |
|---|---|---|
| ECL | 0.956 / 0.958 / 0.951 | −4.4 / −4.2 / −4.9 |
| M4 | 0.975 / 0.982 / 0.969 | −2.5 / −1.8 / −3.1 |
| ETTh1-h96 | 0.996 / 0.977 / 0.974 | −0.4 / −2.3 / −2.6 |
| ETTh1-h48 | 0.960 / 1.000 / 1.000 | −4.0 / 0.0 / 0.0 |
| Weather | ≈1.01 | ≈ +1 |
| Others | within 0.985–1.015 | within ±1.5 |

ETTm1, ETTm2 and ILI mostly keep step 0 (no adaptation).

### 3.5 Joint comparison (step 4, common set)

**Contrasts**

| Contrast | Ratio (CI) | % (CI) |
|---|---|---|
| LoRA@16256 vs zero-shot 16256, seed 0 | 0.992 (0.982–1.002) | −0.8 (−1.8 to +0.2) |
| LoRA@16256 vs zero-shot 16256, seed 1 | 0.993 (0.983–1.002) | −0.7 (−1.7 to +0.2) |
| LoRA@16256 vs zero-shot 16256, seed 2 | 0.988 (0.978–0.997) | −1.2 (−2.2 to −0.3) |
| LoRA@16256 vs zero-shot 16256, geomean of seeds | 0.991 (0.981–0.999) | −0.9 (−1.9 to −0.1) |
| LoRA@1024 vs zero-shot 16256 | 1.015 (0.989–1.038) | +1.5 (−1.1 to +3.8) |
| Chronos-2 LoRA vs zero-shot, ctx 2048 | 0.998 (0.988–1.009) | −0.2 (−1.2 to +0.9) |
| Chronos-2 LoRA vs zero-shot, ctx 8192 | 1.000 (0.990–1.010) | 0.0 (−1.0 to +1.0) |
| Combination gain vs its member, TimesFM default | 0.960 (0.938–0.982) | −4.0 (−6.2 to −1.8) |
| Combination gain vs its member, TimesFM 16256 | 0.982 (0.962–1.001) | −1.8 (−3.8 to +0.1) |
| Combination gain vs its member, TimesFM LoRA@16256 | 0.988 (0.968–1.005) | −1.25 (−3.2 to +0.5) |
| Gain difference, 16256 vs default | 1.023 (1.003–1.040) | gain 2.3 points smaller (0.3 to 4.0) |
| Gain difference, LoRA@16256 vs 16256 | 1.006 (1.000–1.012) | gain 0.55 points smaller (−0.02 to 1.15; block bootstrap 0.1 to 1.1) |
| Fine-tuning Chronos-2 as well, change in gain | 1.001 (0.996–1.006) | +0.1 (−0.4 to +0.6): none |
| Worst-case regret vs best single strategy per task | combinations 1.04–1.05; TimesFM LoRA@16256 1.065 | +4–5%; +6.5% |

**Cost**
- Measured on an RTX A5000 with merged adapters, an idle GPU, CUDA-synchronised timing.
- The same implementation is used per model family.
- TimesFM ctx is capped at the longest history: HF pads to the requested context, which inflates M4 ~20× for identical forecasts.
- TabPFN-TS timing is taken from logs (~4,100 s/origin).

**Break-even, LoRA@1024 vs zero-shot 16256:** ~57 origins (63 one-time GPU min vs a 67 s/origin saving), at +1.5% error.

## 4. Evidence status

| Status | Claims |
|---|---|
| **Firm** | Redundancy and the theory account; long-context gain; combination gain on the default member and its first reduction; no benefit from finer combination granularity or per-task configuration selection; regret reduction from combination; cost ordering |
| **Borderline** | TimesFM LoRA gain beyond long context (firm on the seed average, borderline per seed); second reduction in combination gain |
| **Null** | Chronos-2 LoRA (default hyperparameters); statistical members out of sample; bias correction; corruption-induced complementarity (pilot) |

## 5. Contributions

Novelty is as of the 2026-10-15 literature checks; step 5 re-checks it before writing.

1. **Measurement of shared vs model-specific error across zero-shot TSFMs, tied to the combination bound and the Bates–Granger condition.** No prior measurement found.
2. **Task-level weights dominate per-series, per-horizon and meta-learned weighting under a no-look-ahead protocol.** This contradicts per-series weighting in recent TSFM ensembles (Toto 2.0, TimeRouter, ZooCast).
3. **Backtested configuration.** Context sensitivity itself is known (e.g. GITCO, arXiv 2606.05332). New:
   - per-task selection shows no clear advantage over a fixed long context;
   - configuration alone matches the combination gain;
   - it changes model rankings.
4. **Combination vs configuration vs LoRA under one protocol, with measured cost.**
   - LoRA vs zero-shot alone is known (e.g. Laglil et al., arXiv 2607.23146: LoRA gains of 2–3%).
   - New: partial substitutability and complementarity, the shrinking combination gain, LoRA helping TimesFM but not Chronos-2, and the cost-based ordering.
5. **Two library pitfalls (§8).**

## 6. Limitations

- **Breadth:** 11 univariate tasks; point-forecast MASE as the primary metric; equal task weighting.
- **Power:** 14–17 evaluated windows per task (4–7 for M4 and ILI) leave per-task DM tests underpowered; the conclusions rest on aggregate intervals.
- **Pretraining contamination:** possible for M4 and ECL. It affects absolute scores more than within-model contrasts.
- **TimesFM learning rate:** chosen after Traffic's test score had been seen; the validation ranking agreed.
- **Chronos-2 LoRA:** untuned library defaults. The null result is conditional on that configuration.
- **Compute:** single GPU type; TimesFM via the HF implementation (verified equivalent).
- **Seeds:** 3 for TimesFM LoRA, 1 for Chronos-2.
- **Pilots only:** probabilistic combination (mixture pooling beats quantile averaging: 1.043× vs 1.082× the best member's quantile loss) and Chronos-2 multivariate mode.

## 7. Anticipated questions

**Q: Leakage?**
No look-ahead in weights or selection. Early-stopping windows are excluded from the final comparison. The provisional 0.621 that used them was replaced by 0.617.

**Q: Is the LoRA effect real?**
- The seed-averaged CI excludes 1, but two of the three per-seed CIs touch 1.
- Consistent on ECL and M4; seed-dependent on ETTh1.
- Report it as small and task-specific.

**Q: Why does per-task selection fail when context matters?**
The ranking of configurations is unstable across windows; selecting among 12 variants from 3–16 windows is noisy. Fixed 16256 already captures most of the oracle gain: 0.639 vs 0.629, so the oracle is only 1.6% better.

**Q: Why does a ctx-1024 adapter work at 16256?**
Observed, not explained. Hypothesis: the low-rank updates act per patch, not tied to context length. Untested.

**Q: Why no Chronos-2 LoRA gain?**
- Validation gains on 6 tasks did not carry over to the test windows.
- Candidates: the narrower default adapter (attention + output only), lr 1e-5, or the model already being well fitted to these domains.
- A tuned run (~3 GPU h) would settle whether this is configuration-specific.

**Q: Couldn't a better combiner do more?**
- Per-series, per-horizon and LODO meta-learned weights all lose out of sample to per-task NNLS.
- The ex-post simplex bound itself is only ~3% (median).

**Q: Does "no clear difference" mean equivalence?**
No. We report what the CIs exclude instead: e.g. original combination vs ctx 16256 is −0.8% (CI −3.3 to +1.4%), which excludes differences beyond ~3%.

**Q: Is the cost comparison fair?**
- Same implementation per family, capped contexts, merged adapters, synchronised GPU timing.
- Combination cost = sum of the members' costs; weight fitting is negligible CPU.
- Per-task selection cost (197 s/origin) assumes all variants run every origin, so it is an upper bound.

**Q: What is the transfer-learning framing?**
A spectrum of adaptation: zero-shot reuse and model pooling, then inference-time adaptation, then parameter-efficient adaptation of the weights. All are compared under one protocol.

## 8. Implementation issues found

1. **HF TimesFM training loss:** pinball channels misaligned by one level (the mean head and q0.1–0.4 are paired with shifted levels), plus MSE on the median. Replaced by our own loss.
2. **`Chronos2Pipeline.fit` returns a model that is unsafe to forecast with:**
   - still in training mode (dropout 0.1 active);
   - Accelerate's bf16 autocast wrapper still on `forward`;
   - TF32 matmuls left enabled for the process (restoring `fp32_precision="none"` defers to the legacy flag, so it must be set explicitly to `"ieee"`).

   The first Chronos-2 pilot forecasts were invalid because of this. Fix: forecast from the saved adapter merged into a fresh fp32 base. Verified: reload difference 0.0, and post-fit zero-shot reproduces the grid exactly.
3. **The 4-window configuration pilot overstated gains** (Weather −28%, ETTm1 −13% vs ~−10% on 20 origins).

## 9. Remaining

1. **Literature re-check** (novelty claims, citation verification).
2. **Writing** (until ~Nov 28): one window convention per chapter, stated explicitly.
3. **Optional:**
   - a tuned Chronos-2 LoRA run;
   - reporting the two library issues upstream.
