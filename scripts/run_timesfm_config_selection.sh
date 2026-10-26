#!/bin/bash
# Step 2: per-task inference-configuration selection on the configuration grid
# (configs/timesfm_config.yaml), with the combination study's rolling protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=outputs/timesfm_config
COMMON=(--forecasts "$RUN/forecasts/all_datasets_all_models.parquet"
        --mase-scales "$RUN/metrics/mase_scales.csv"
        --outputs-root "$RUN/selection" --no-meta-learner
        --methods equal_mean median select_task select_series nnls_task)
TFM_FLIP=(timesfm_2p5_ctx{512,1024,2048,4096,8192,16256})
TFM_NOFLIP=(timesfm_2p5_ctx{512,1024,2048,4096,8192,16256}_noflip)
CHR=(chronos2_ctx{1024,2048,4096,8192})
run() { local name=$1; shift; echo "== $name"; uv run python scripts/run_combination.py "${COMMON[@]}" --name "$name" --members "$@"; }
run tfm_ctx "${TFM_FLIP[@]}"
run tfm_cfg "${TFM_FLIP[@]}" "${TFM_NOFLIP[@]}"
run chr_ctx "${CHR[@]}"
run default_pair timesfm_2p5_ctx1024 chronos2_ctx8192
run all_cfg "${TFM_FLIP[@]}" "${TFM_NOFLIP[@]}" "${CHR[@]}"
# Does combination still help once TimesFM uses its long context? TabPFN-TS comes from the
# 20-origin combination run (same windows, same data).
TABPFN=(outputs/combination/forecasts/*_tabpfn_ts.parquet)
run_tab() { local name=$1; shift; echo "== $name"; uv run python scripts/run_combination.py "${COMMON[@]}" --forecasts "$RUN/forecasts/all_datasets_all_models.parquet" "${TABPFN[@]}" --name "$name" --members "$@"; }
run long_pair timesfm_2p5_ctx16256 chronos2_ctx8192
run_tab default_trio timesfm_2p5_ctx1024 chronos2_ctx8192 tabpfn_ts
run_tab long_trio timesfm_2p5_ctx16256 chronos2_ctx8192 tabpfn_ts
