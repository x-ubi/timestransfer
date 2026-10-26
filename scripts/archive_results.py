"""Copy expensive run outputs from the git-ignored ``outputs/`` tree into the tracked
``results/`` tree, or restore them from there.

Forecast parquets are split per (dataset, model) and recompressed with zstd so that no
file comes near GitHub's 100 MB limit; ``restore`` concatenates them back into
``outputs/<run>/forecasts/all_datasets_all_models.parquet``. Everything that is cheap to
regenerate (per-dataset forecast duplicates, combined forecasts, figures) is left out.

    uv run python scripts/archive_results.py archive
    uv run python scripts/archive_results.py restore
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
FORECASTS = "forecasts/all_datasets_all_models.parquet"

# Output directories (relative to outputs/) whose forecasts, metrics and metadata are kept.
# "." is the main 5-window benchmark that the thesis chapters cite.
RUNS = [".", "timesfm_tuning", "combination", "combination_stats"]

# Extra files and folders (relative to outputs/) copied as they are.
EXTRAS = [
    "analysis",
    "combination/analysis",
    "combination/combiners/series_features.parquet",
    "combination/combiners/fm/weights",
    "combination/combiners/fm/metadata",
    "combination/combiners/fm/metrics/aggregate_relative_scores.csv",
    "combination/combiners/fm/metrics/overall_by_dataset.csv",
    "combination/combiners/fm/metrics/relative_to_seasonal_naive.csv",
    "combination/combiners/fm/metrics/model_comparison_wide.csv",
    "combination/combiners/fm/metrics/window_spread.csv",
]

# Run logs hold the wall-clock timings; *.log is git-ignored, so they are stored as .txt.
LOGS = ["combination/run.log", "combination_stats/run.log", "combination/analysis_run.log"]


def _run_dir(base: Path, run: str) -> Path:
    return base if run == "." else base / run


def _results_run_dir(run: str) -> Path:
    return RESULTS / ("benchmark" if run == "." else run)


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*"))
    else:
        shutil.copy2(source, target)


def archive() -> None:
    for run in RUNS:
        source = _run_dir(OUTPUTS, run)
        target = _results_run_dir(run)
        table = pq.read_table(source / FORECASTS)
        split_dir = target / "forecasts"
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        for dataset in sorted(pc.unique(table["dataset"]).to_pylist()):
            in_dataset = table.filter(pc.equal(table["dataset"], dataset))
            for model in sorted(pc.unique(in_dataset["model"]).to_pylist()):
                part = in_dataset.filter(pc.equal(in_dataset["model"], model))
                pq.write_table(
                    part,
                    split_dir / f"{dataset}__{model}.parquet",
                    compression="zstd",
                    compression_level=19,
                )
        _verify(table, split_dir)
        for folder in ("metrics", "metadata"):
            if (source / folder).exists():
                _copy(source / folder, target / folder)
        print(f"archived {source} -> {target} ({table.num_rows} forecast rows)")
    for extra in EXTRAS:
        _copy(OUTPUTS / extra, RESULTS / "extras" / extra)
    for log in LOGS:
        if (OUTPUTS / log).exists():
            _copy(OUTPUTS / log, RESULTS / "logs" / (log.replace("/", "__") + ".txt"))


def _read_split(split_dir: Path) -> pa.Table:
    parts = [pq.read_table(path) for path in sorted(split_dir.glob("*.parquet"))]
    return pa.concat_tables(parts)


def _verify(original: pa.Table, split_dir: Path) -> None:
    keys = [("dataset", "ascending"), ("model", "ascending"), ("unique_id", "ascending"),
            ("window", "ascending"), ("horizon", "ascending")]
    restored = _read_split(split_dir).select(original.column_names)
    if not restored.sort_by(keys).equals(original.sort_by(keys)):
        raise RuntimeError(f"Round trip of {split_dir} does not reproduce the forecasts.")


def restore() -> None:
    for run in RUNS:
        source = _results_run_dir(run)
        target = _run_dir(OUTPUTS, run)
        (target / "forecasts").mkdir(parents=True, exist_ok=True)
        pq.write_table(_read_split(source / "forecasts"), target / FORECASTS)
        for folder in ("metrics", "metadata"):
            if (source / folder).exists():
                _copy(source / folder, target / folder)
        print(f"restored {source} -> {target}")
    for extra in EXTRAS:
        if (RESULTS / "extras" / extra).exists():
            _copy(RESULTS / "extras" / extra, OUTPUTS / extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["archive", "restore"])
    action = parser.parse_args().action
    archive() if action == "archive" else restore()


if __name__ == "__main__":
    main()
