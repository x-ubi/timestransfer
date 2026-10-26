# Evidence ledger

Current-state review: 2026-09-08. Original benchmark outputs are read-only inputs; no models are retrained. Numerical transformations and SHA-256 inputs are in artifact_manifest.json.

| Deliverable/claim | Primary evidence | Transformation | Limit |
|---|---|---|---|
| Protocol, settings, scaling, clipping | configs/*.yaml; src/timestransfer/{data,features,models,runner,metrics}.py; both environment.json files | Trace dispatch and requested settings | Historical dirty source tree; Prophet clipping uses capped history |
| 99 main / 44 sensitivity combinations | Both metrics.csv, combined parquet forecasts and status metadata | Recompute all four metrics at overall, series and horizon scopes | One final origin; repeated ETTh1 tasks |
| Dataset inventory and boundaries | Cached M4 and LH2 data, exact loader | Count, finite-value, duplicate, interval, split and MASE-scale checks | Archive dates can differ from source observation dates |
| Tables and six diagnostic forecasts | Saved metrics/forecasts and loaded training tails | See generator and selection JSON | Test-selected TimesFM best/median/worst; four displayed models |
| Features and correlations | outputs/analysis/*.csv; series_features.py; analysis.py | Recheck join/original Spearman; additional MASE Spearman | Capped training tails, dependent series, no significance analysis |
| TimesFM contexts and flags | outputs/timesfm_tuning/{metrics,metadata,forecasts} | Same-run baseline differences | Requested config is saved, compiled object is not; clipping remains |
| Comparator rationale | TimesFM arXiv 2310.10688v4 sec.6/App.A; Chronos 2403.07815v2 Tables 1/2; TabPFN-TS 2501.02945v1 sec.4/App.A; Prophet 2017 sec.3/Table 3 | Qualitative protocol/baseline mapping | Different versions, tasks and metrics; no literature score ranking |
| Pretraining overlap | Original papers above; Google TimesFM 2.5 card; Amazon Bolt card | Separate corpus inclusion, author exclusion claims, exact-target exposure | No checkpoint hashes or independent training-corpus audit |
| Dataset semantics | ETDataset README; M4-methods; Lai multivariate data repo; UCI Electricity; Autoformer v4 sec.4; Jena Weatherstation.pdf Table 3; CDC overview | Compare sources with exact archive headers/ranges | Anonymous/renamed variables and altered dates explicitly qualified |
| Metrics/features | Hyndman & Koehler mase.pdf; FPP3 STL features; SciPy periodogram and statsmodels ACF docs | Definitions checked against local code | Local epsilon, finite-sample and fallback conventions stated |

Preflight: thesis create/read/delete probe succeeded under approved escalation. Primary papers were retrieved using web tools. Benchmark files and cached data are readable. XeLaTeX/Biber/latexmk installation was authorized; template font dependencies are being resolved before the first successful compile.

## Verified source register

All URLs below were retrieved with the web tool in this execution. Dates on undated documentation entries indicate consultation year, not an asserted release date.

- `timesfm_paper`: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/html/2310.10688v4); arXiv:2310.10688, version 4.
- `chronos_paper`: [Chronos: Learning the Language of Time Series](https://arxiv.org/html/2403.07815v2); arXiv:2403.07815, version 2.
- `tabpfn_ts`: [The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features](https://arxiv.org/html/2501.02945v1); arXiv:2501.02945, version 1.
- `prophet_paper`: [Forecasting at Scale](https://facebook.github.io/prophet/static/prophet_paper_20170113.pdf); Preprint, 13 January 2017.
- `autoformer_paper`: [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/html/2106.13008v4); arXiv:2106.13008, version 4.
- `mase_paper`: [Another look at measures of forecast accuracy](https://robjhyndman.com/papers/mase.pdf); Author manuscript, 2 November 2005; subsequently published in 2006.
- `timesfm_card`: [TimesFM 2.5 200M PyTorch model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch); Checkpoint-specific documentation.
- `timesfm_repo`: [TimesFM official repository](https://github.com/google-research/timesfm); TimesFM 2.5 update and LoRA example, April 2026.
- `chronos_bolt_card`: [Chronos-Bolt Base model card](https://huggingface.co/amazon/chronos-bolt-base); Checkpoint-specific architecture and evaluation documentation.
- `prophet_seasonality`: [Seasonality, Holiday Effects, and Regressors](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html); Official documentation.
- `datasetsforecast_lh2`: [LongHorizon2 dataset loader source](https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/long_horizon2.py); Evaluation uses installed datasetsforecast 1.0.1; current source documentation.
- `m4_data`: [M4-methods: Data, benchmarks, and methods submitted to the M4 forecasting competition](https://github.com/Mcompetitions/M4-methods); Official data repository.
- `ett_data`: [Electricity Transformer Dataset (ETDataset)](https://github.com/zhouhaoyi/ETDataset); ETT-small release and variable definitions.
- `electricity_data`: [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014); UCI Machine Learning Repository; DOI: 10.24432/C58C86.
- `lai_data`: [Multivariate time-series data](https://github.com/laiguokun/multivariate-time-series-data); Electricity, Traffic and Exchange source descriptions.
- `jena_weather`: [Documentation of the weather station](https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf); Table 3: variables, symbols and units.
- `cdc_ili`: [U.S. Influenza Surveillance: Purpose and Methods](https://www.cdc.gov/fluview/overview/index.html); Official surveillance definitions.
- `stl_features`: [Forecasting: Principles and Practice, third edition: STL Features](https://otexts.com/fpp3/stlfeatures.html); Section 4.3.
- `periodogram_docs`: [scipy.signal.periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html); Official API documentation.
- `acf_docs`: [statsmodels.tsa.stattools.acf](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html); Official API documentation.

Preflight completed: original thesis compiled to 33 pages using XeLaTeX, Biber and latexmk after installing the template font dependencies. Baseline warnings include overfull paragraphs and missing inequality characters in the old pretraining recipe.
