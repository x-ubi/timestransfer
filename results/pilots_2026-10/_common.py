"""Shared paths for the October 2026 pilot scripts (run from the repository root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
COMBINATION = ROOT / "outputs" / "combination"
FORECASTS = COMBINATION / "forecasts" / "all_datasets_all_models.parquet"
SCALES = COMBINATION / "metrics" / "mase_scales.csv"
TUNING = ROOT / "outputs" / "timesfm_tuning"
FM = ["timesfm_2p5", "chronos2", "tabpfn_ts"]
