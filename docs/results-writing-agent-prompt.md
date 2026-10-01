# Task: update the thesis for the revised benchmark and write up its results

You are updating a master's thesis on time-series forecasting and transfer learning (Warsaw University of Technology, Faculty of Electronics and Information Technology; written in English). The benchmark behind its experimental chapters has been substantially revised and rerun. Your job:

1. Regenerate the thesis tables and figures from the new benchmark outputs.
2. Update Section 3 (Experiments) only where the benchmark has changed.
3. Rewrite Section 4 (Results) and the appendix around the new results.
4. Mark everything you change in blue.

Treat the current outputs as the final, correct results of the study. Write about the final design, not the history of how it changed.

## Repositories and ground rules

- **Code:** `/home/ubuntu/timestransfer`. **Thesis** (Overleaf checkout): `/home/ubuntu/thesis`. Read the `AGENTS.md` in both before starting, and follow them.
- **Do not rerun any forecasting model.** All results already exist. You may run the cheap post-processing scripts named below.
- **Do not commit, push, or rewrite Git history** in either repository. The author reviews and commits.
- **This task runs unattended.** Do not stop to ask questions. Make the most defensible reversible choice, mark it with `\todo{}` in the thesis where it affects the text, and list it in the final report.
- **Do not modify** `src/timestransfer/` or `configs/`. You may adapt the post-processing scripts `scripts/build_thesis_artifacts.py`, `scripts/verify_thesis_package.py` and `scripts/analyze_series.py`, and you may add new scripts under `scripts/`.
- **Do not edit** the literature chapter (`tex/2-literature.tex`) or `tex/timesfm.tex`. If a statement there contradicts the new setup, list it in your final report instead.
- **Previous plan is superseded.** `docs/experiment-writing-agent-plan.md` describes the earlier benchmark; use it only as background on the writing approach.

## Source of truth (read-only)

All results are in `/home/ubuntu/timestransfer/outputs/`.

- `outputs/metrics/metrics.csv` has columns `dataset, model, scope, horizon, unique_id, window, forecast_horizon, mae, rmse, smape, mase, n_obs, status, error`. The `scope` values mean:
  - `overall`, `horizon`, `series`: pooled over all 5 windows.
  - `window`: one rolling origin.
  - `model_status`: skipped or failed runs.
- `outputs/metrics/relative_to_seasonal_naive.csv`: per-task MASE and MAE divided by SeasonalNaive's.
- `outputs/metrics/aggregate_relative_scores.csv`: per model, the geometric mean of those ratios across tasks, plus `n_tasks`.
- `outputs/metrics/window_spread.csv`: mean, standard deviation, min and max of window-level MASE and sMAPE per task and model.
- `outputs/metrics/mase_scales.csv`: MASE denominators per (dataset, window, series).
- `outputs/forecasts/all_datasets_all_models.parquet`: 3,609,600 point forecasts with columns `dataset, model, unique_id, horizon, ds, y_true, y_pred, window, forecast_horizon`.
- `outputs/metadata/environment.json`: config, package versions, per-task model statuses and details (checkpoint files, revisions, context lengths, window counts), dataset summaries and git state.
- `outputs/timesfm_tuning/`: the same structure for Experiment 3 (11 TimesFM variants × 4 tasks, 5 windows, all OK).
- `/home/ubuntu/timestransfer_runs/code_snapshot_20260929T013934Z/`: the exact code that produced these results. The run used an uncommitted working tree on top of commit `24e9e585`.
- `outputs_archive_pre_rolling_origin/`: the **old** results the current thesis text describes. Use them only to see which statements are now outdated. Never report them as results.

**Exclude `tabpfn_ts_v3p5` everywhere.** It is an optional extra configuration that was not evaluated (license pending). Do not mention it in the thesis; Experiment 1 has nine models.

## What changed since the thesis text was written

The current `tex/experiments.tex`, `tex/results.tex` and `tex/experiment-appendix.tex` describe the old benchmark. The changes below are the complete list of what is now different. Verify every concrete value against the outputs, metadata or config before writing it.

1. **Evaluation protocol: rolling origins.**
   - Each series is evaluated on `n_windows = 5` non-overlapping test windows of length H. The stride equals H; for ETTh1/96 it is 96.
   - Window 0 is the final holdout, identical to the old single holdout. Window k tests on the H observations ending k·H steps before the series end.
   - For each window, the model receives only the observations before that window. Later observations are discarded, not merely hidden.
   - On M4 Hourly, window 0 is the official M4 test set; windows 1–4 lie inside the M4 training data.
   - A model is scored on a task only if all 5 windows succeed.
   - The old text says "no validation set, repeated seeds, or rolling origins are used" and "each series is tested on exactly one forecast window". Both are now false.
2. **Metrics and aggregation.**
   - MASE scales are computed per window from that window's own training history.
   - The `overall`, `series` and `horizon` scopes pool all windows. A new `window` scope scores each origin separately.
   - The headline cross-task summary is new: the geometric mean over tasks of per-task MASE (and MAE) divided by SeasonalNaive's. This follows the convention of the Chronos and TabPFN-TS papers; verify this in the primary sources before citing it.
   - Window-to-window spread is now reported; the median coefficient of variation of window-level MASE is about 0.31.
3. **TabPFN → TabPFN-TS.**
   - The pooled lag-feature `TabPFNRegressor` adaptation is gone. `tabpfn_ts` is the published TabPFN-TS pipeline (`tabpfn-time-series` 1.3.0, local GPU inference, `tabpfn` 9.0.0).
   - It uses the TabPFN-v2 regressor checkpoint `tabpfn-v2-regressor.ckpt` (SHA-256 in the metadata) and the most recent 4,096 observations. That is the configuration reported in Hoo et al., arXiv:2501.02945 v4.
   - It fits each series separately, using running-index, calendar (sine/cosine) and FFT-detected seasonal features, with no lags. The point forecast is the predictive median.
   - **Caveats:**
     - The pipeline code is the 1.3.0 release, not the paper's code version. Its auto-seasonal feature keeps up to 12 periods; the paper reports k = 5.
     - The downloaded v2 checkpoint may not be byte-identical to the paper's `tabpfn-v2-regression-2noar4o2.ckpt`.
     - M4's integer indices are mapped onto synthetic hourly timestamps, as for Prophet, so its calendar features encode only the 24-hour cycle.
   - Telemetry is disabled.
   - The bibliography entry `tabpfn_ts` points to v1; update it to the version actually followed (v4) and verify its title, authors and date.
4. **Chronos-Bolt → Chronos-2.**
   - The model is `amazon/chronos-2` (paper arXiv:2510.15821; verify the title and authors), at revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, with `chronos-forecasting` 2.3.2.
   - It runs zero-shot and univariate, with cross-learning disabled, so each series is forecast independently of the batch. It uses the full 8,192-observation context and the median as the point forecast.
   - Its native prediction length (1,024) covers every horizon here, so the old remarks about Bolt forecasting beyond 64 steps by feeding back its own predictions no longer apply.
   - Add a bibliography entry. Its training-data disclosure (subsets of the Chronos datasets and of GIFT-Eval Pretrain, plus synthetic data) must be checked in the model card and paper before you use it in the contamination discussion.
5. **AutoARIMA: restricted → full seasonal search.**
   - It now uses StatsForecast's default stepwise seasonal AutoARIMA with season length s and the last 5,000 observations. The old constraints are all removed: nonseasonal only, p, q ≤ 1, d ≤ 1, maximum order 2, five models, approximation, no drift.
   - It is not run when s > 52. That skips ETTm1, ETTm2 (s = 96) and Weather (s = 144); these are recorded as `skipped`.
   - **Why:** in a timing test, one seasonal fit at s = 96 on a single ETTm1 series did not finish within about 35 minutes, while fits at s = 24 took about 2–6 minutes per 5,000-point series. Seasonal ARIMA is also generally ill-suited to long seasonal periods. If you cite a source for that, such as Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, verify the exact section.
   - AutoARIMA's aggregate covers 8 tasks (`n_tasks = 8`). State this wherever it is compared.
6. **Linear regression.**
   - It is now the only pooled tabular model; TabPFN no longer shares its table.
   - The row cap is 200,000 per task and window, up from 10,000. Check in the metadata (`lag_feature_train_rows_per_window`) which tasks reach the cap.
   - The features are unchanged: lags, h, h/H, log scale and seasonal phase, with each series scaled by its training mean absolute value and median imputation.
   - **Report its results as observed, without calling them a defect.** Its very high relative MASE on Weather (about 11) and M4 (about 7.6) arises in series that are nearly constant at a high level, such as air pressure around 978 mbar. There, mean-absolute scaling leaves almost no relative variation, so small errors on the scaled values become large errors in real units. You may explain this as a property of the formulation.
7. **Reproducibility.**
   - The Hugging Face checkpoints are pinned: TimesFM 2.5 at `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`, and Chronos-2 as above.
   - The TabPFN checkpoint file name and SHA-256 are recorded in the metadata.
   - The silent fallback in the TimesFM inference configuration was removed, so recorded flags are the flags actually applied.
   - Statistical models and Prophet use 16 parallel workers instead of 4.
   - **Remove** the old claims about "unpinned checkpoints" and "package-default TabPFN".
   - **Provenance:** describe it from the metadata, i.e. the uncommitted tree on `24e9e585` plus the archived code snapshot. If the author commits before submission, a commit hash will replace this; leave a `\todo{}` for it.
8. **Experiment 3** (TimesFM sensitivity) was rerun with the same 5-window protocol and the pinned revision. Same 11 variants, 4 tasks, 44 combinations.
9. **Experiment 2** (feature–error analysis) must be recomputed on the new outputs; see Step 1. The features come from the training part of window 0, the final holdout's history. The per-series metrics pool all 5 windows. Describe exactly that.
10. **Unchanged:** datasets, loaders, ETT truncation, horizons, seasonal lags, MASE definition, sMAPE definition, nonnegativity clipping, Prophet settings (except worker count), AutoETS, AutoTheta, SeasonalNaive and TimesFM's 1,024 context and flags.

For orientation, the headline aggregate is in `aggregate_relative_scores.csv`. Verify these values; don't copy them.

| Model | Geometric mean of relative MASE | Tasks |
|---|---|---|
| TimesFM | 0.748 | 11 |
| Chronos-2 | 0.785 | 11 |
| TabPFN-TS | 0.840 | 11 |
| AutoARIMA | 0.926 | 8 |
| SeasonalNaive | 1.000 | 11 |
| AutoTheta | 1.163 | 11 |
| AutoETS | 1.212 | 11 |
| Prophet | 1.451 | 11 |
| Linear | 1.757 | 11 |

## Steps

### 1. Regenerate the analysis outputs and thesis artifacts (no model runs)

- **Experiment 2 analysis:** run `scripts/analyze_series.py --config configs/experiment.yaml`. It writes `outputs/analysis/`, which does not exist yet for the new run.
- **Artifact builder:** adapt `scripts/build_thesis_artifacts.py` to the multi-window outputs. At present it assumes one window and 9 models with 99 complete combinations. Concretely:
  - Forecasts are unique per `(dataset, model, window, unique_id, horizon)`.
  - MASE scales merge on `(dataset, window, unique_id)`.
  - Exclude `tabpfn_ts_v3p5`.
  - Handle the three skipped AutoARIMA tasks (show "—" in tables).
  - Use the new model keys and labels: `tabpfn_ts` → "TabPFN-TS", `chronos2` → "Chronos-2".
  - Keep its recomputation checks, adapted to the new structure.
- **New tables:** the aggregate relative scores, per-task relative MASE, and window spread (e.g. mean ± standard deviation of window-level MASE).
- **Forecast examples:** draw them from window 0, so the history shown is that window's training data, and say so in the captions.
- **Per-step figures** pool all windows; say so.
- **Verification script:** update `scripts/verify_thesis_package.py` accordingly and run it.
- **Where to write:** generated tables go to `/home/ubuntu/thesis/tables/experiments/` and figures to `/home/ubuntu/thesis/img/experiments/`. You may overwrite the stale ones there. The thesis repo is under Git, so the old versions are recoverable.

### 2. Update Section 3 (`tex/experiments.tex`): only what changed

Section 3 is otherwise considered finished. Change only the statements affected by the list above, and keep the author's structure, wording and labels everywhere else. That includes at least:
- the objectives (model list);
- the horizon-interpretation paragraph;
- the forecasting procedure (formalise the window index k and the cutoff T_{i,k} = N_i − kH − H);
- Table `tab:modelsettings`;
- the pooled tabular formulation (linear regression only, 200k cap);
- the pretrained-models paragraph (TabPFN-TS and Chronos-2);
- the statistical-baselines paragraph (seasonal AutoARIMA and the skip rule, replacing the existing `\todo` about why it was constrained);
- information budgets and reproducibility;
- metrics and aggregation (per-window scales, window scope, geometric-mean relative score, window spread);
- Table `tab:papers` (Chronos-2 and TabPFN-TS v4 rows);
- the Experiment 2 description (features from window 0; metrics pooled over windows);
- the Experiment 3 description (5 windows; "same test set" now means the same 5 test windows).

Resolve existing `\todo`s only where the change resolves them; leave the others.

### 3. Rewrite Section 4 (`tex/results.tex`) around the new results

Keep the existing subsection structure and labels unless a change is necessary:
- coverage and quantitative comparison;
- error across the horizon;
- qualitative examples;
- Experiment 2;
- Experiment 3;
- discussion and limitations.

Provide genuine analysis, not a recitation of tables:
- **Headline comparison:** use the aggregate first, then the per-task picture. Cover where each family wins and where SeasonalNaive is not beaten.
- **Close foundation models:** TimesFM, Chronos-2 and TabPFN-TS are close in aggregate and the winner changes by task. Discuss which tasks separate them, e.g. ILI, Weather, ECL and ETTm1. Do not state rankings as significant: no significance tests were run.
- **Window variability:** explain what the window spread implies about single-origin conclusions.
- **Metric disagreements:** discuss where sMAPE and MASE disagree, and why.
- **Full seasonal AutoARIMA:** report its standing, and that its aggregate covers 8 tasks.
- **Linear regression:** explain its behaviour, as in change 6.
- **Experiment 2:** report the new correlations, with the same descriptive caveats as before.
- **Experiment 3:** report the context and flag effects under the 5-window protocol.
- **Contamination and "zero-shot":** update for Chronos-2's disclosed training data and for TabPFN-TS-v2's synthetic-only pretraining. Use verified sources only.
- **Limitations:** remove the ones that no longer apply (single origin, restricted AutoARIMA, custom TabPFN adaptation, unpinned checkpoints). Keep or add the ones that still hold:
  - only 5 windows, which are not independent;
  - dependent variables within panels;
  - unequal context budgets;
  - AutoARIMA coverage limited to 8 tasks;
  - automatic Prophet settings;
  - TabPFN-TS code and checkpoint version caveats;
  - no significance testing;
  - no validation-based tuning.
- **Numbers:** every number in the text must be traceable to an output file. Recompute it; don't eyeball. Use consistent precision: as in the existing tables, sMAPE to 2 decimals and MASE to 3.

### 4. Update the appendix (`tex/experiment-appendix.tex`)

- The tabular lag inventory: the 200,000-row cap, and linear regression only.
- The raw MAE and RMSE tables.
- The horizon MASE figure.
- The context grid.
- The software table, using the versions in `environment.json`:
  - tabpfn 9.0.0 and tabpfn-time-series 1.3.0;
  - chronos-forecasting 2.3.2 (check how Chronos-2 is loaded, its dtype and batch size, in `src/timestransfer/models.py` and the config);
  - 16 workers;
  - checkpoint revisions.
- The dataset boundaries, now covering the earliest window as well. The training lengths for windows 1–4 are shorter by k·H.

### 5. Bibliography

- Add or update the entries for Chronos-2 and TabPFN-TS v4, and for any other source you cite. Prefer primary sources, and verify that each one exists and supports the claim.
- Follow the existing bibliography style, and mark new references in the same way the existing `authoredref` mechanism does (a parallel blue list; see below).

## Blue change marking (required)

`authorship.tex` already defines a red, switchable mechanism for agent-authored text: `\authored{}`, the `authoredblock` environment, a float hook and `\showauthorshiptrue`. Sections 3 and 4 sit inside `authoredblock`s, so they render red. Add a parallel **blue** mechanism for this revision:

- `\newif\ifshowrevisions` plus `\showrevisionstrue`, so the author can switch all blue off at once.
- A `\revisedstyle` that applies `\color{blue}` and sets the hyperref link colours to blue.
- `\DeclareRobustCommand{\revised}[1]{{\revisedstyle #1}}`, made safe for PDF strings (bookmarks) like `\authored`.
- A `revisedblock` environment, plus a float-reset hook so that tables and figures inside a `revisedblock` keep their blue captions and content. The existing red hook is the model; check that blue wins inside red blocks.
- Blue colouring for new or changed bibliography entries, analogous to `authoredref@`.

Marking rules:
- Every new or changed sentence, table row or cell, caption, equation and footnote is blue.
- Unchanged text keeps its current colour.
- Regenerated tables and figures: wrap their `\input`/`\includegraphics` and captions in the blue mechanism.
- Do not leave deleted text in the document. List substantive deletions in your final report instead.

## Writing standard

- **Register:** formal scientific English suitable for a Polish master's thesis (praca magisterska) at WUT EiTI.
  - Use impersonal or passive constructions, or "this thesis"/"this study"; no first person singular.
  - Use the present tense for what the results show and the past tense for procedures that were carried out.
  - Avoid promotional or informal language.
- **Figures, tables, abbreviations:**
  - Every table and figure is referenced and discussed in the text before or near where it appears.
  - Captions are self-contained.
  - Abbreviations are defined at first use.
  - Terminology is consistent with the rest of the thesis: task, series, window, horizon step, context.
- **Claims:**
  - Distinguish observed results from interpretations and hypotheses.
  - State uncertainty.
  - Make no claims of significance, causality or generality that the design does not support.
  - Cite primary sources for every external claim.
- **Prose:** write paragraphs, not bullet lists. Follow the existing number formatting: thousands separators with commas and the stated decimal precision.

## Proofreading iteration

Once the content is complete and compiles, proofread every changed passage with the `mcpmarket-me:professional-proofreader` skill.

- **Use its inline text mode.** Its file mode supports only .docx/.txt/.pdf and writes `UPDATED_` copies, which must not be created for `.tex` files.
- **Work in units.** Proofread section by section, one subsection or paragraph group at a time.
- **Apply corrections in place** in the `.tex` sources.

Constraints on the proofreading edits:
- Do not change numbers, labels, citations, LaTeX commands, math, defined terms, or the meaning or hedging of a claim.
- Keep the scientific register required above. Where the skill's "avoid unnecessary formalization" conflicts with it, the thesis register wins.
- Text corrected during proofreading stays (or becomes) blue.
- Proofread only blue text, not the author's unchanged prose.

After proofreading, run another pass: recompile, re-run the number cross-check, and confirm the skill introduced no factual changes. Summarise its modification log in the final report.

## Verification before you finish

1. **Compile.** From `/home/ubuntu/thesis`, run `latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build/experiments main.tex`. There must be no undefined references or citations and no new overfull boxes in the changed sections.
2. **Check every number.** Cross-check every number you wrote against the CSVs/metadata with a script, and report the check.
3. **Check colours.** Confirm that `\showrevisionsfalse` removes all blue and `\showauthorshipfalse` still removes all red.
4. **Final report** (as your reply, not in the thesis):
   - the files changed;
   - a summary of the changes per section;
   - the substantive deletions;
   - the scripts adapted or added;
   - any statements elsewhere in the thesis that now conflict with the new setup;
   - open questions or `\todo`s left for the author.
