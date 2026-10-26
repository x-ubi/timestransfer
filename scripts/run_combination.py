"""Combine saved member forecasts post hoc under the rolling-origin protocol (Phase B).

Writes combined forecasts (with the members and the baseline on the same windows) in the
benchmark's long format, the fitted weights, metrics computed with the benchmark's metric
and reporting functions, and a run record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

from timestransfer.combination import (
    RollingProtocol,
    build_combination_table,
    default_combiners,
    load_forecasts,
    load_mase_scales,
    rolling_combine,
    rolling_series_features,
)
from timestransfer.metadata import _git_metadata
from timestransfer.metrics import compute_metric_rows
from timestransfer.reporting import write_reporting_outputs

DEFAULT_MEMBERS = ["timesfm_2p5", "chronos2", "tabpfn_ts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasts",
        nargs="+",
        default=["outputs/combination/forecasts/all_datasets_all_models.parquet"],
        help="Long forecast parquet file(s) holding the members and the baseline.",
    )
    parser.add_argument(
        "--mase-scales",
        nargs="+",
        default=["outputs/combination/metrics/mase_scales.csv"],
        help="mase_scales.csv file(s) matching the forecasts.",
    )
    parser.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS, help="Pool to combine.")
    parser.add_argument(
        "--baseline",
        default="seasonal_naive",
        help="Model reported alongside (relative scores are computed against it).",
    )
    parser.add_argument("--methods", nargs="+", help="Subset of combiner names to run.")
    parser.add_argument("--datasets", nargs="+", help="Subset of datasets.")
    parser.add_argument("--min-past-windows", type=int, default=3)
    parser.add_argument("--max-past-windows", type=int, default=None)
    parser.add_argument(
        "--name",
        default="fm",
        help="Run name; outputs go to <outputs-root>/<name>.",
    )
    parser.add_argument("--outputs-root", default="outputs/combination/combiners")
    parser.add_argument(
        "--config",
        default="configs/combination.yaml",
        help="Benchmark config, used to recompute series features for the meta-learner.",
    )
    parser.add_argument(
        "--features-cache",
        default="outputs/combination/combiners/series_features.parquet",
        help="Series-feature cache for the meta-learner (computed if missing).",
    )
    parser.add_argument("--feature-max-points", type=int, default=2048)
    parser.add_argument("--n-jobs", type=int, default=16, help="Processes for series features.")
    parser.add_argument(
        "--no-meta-learner", action="store_true", help="Skip the FFORMA-style meta-learner."
    )
    return parser.parse_args()


def load_or_compute_features(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.features_cache)
    if path.exists():
        features = pd.read_parquet(path)
        if set(features["max_points"].unique()) == {args.feature_max_points}:
            return features
        print(f"Recomputing {path}: cached with a different max_points.")
    start = time.monotonic()
    features = rolling_series_features(
        args.config, max_points=args.feature_max_points, n_jobs=args.n_jobs
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    print(f"Series features: {len(features)} rows in {time.monotonic() - start:.0f}s -> {path}")
    return features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputs_root) / args.name
    for sub in ("forecasts", "weights", "metadata"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    forecasts = load_forecasts(args.forecasts)
    scales = load_mase_scales(args.mase_scales)
    if args.datasets:
        forecasts = forecasts[forecasts["dataset"].isin(args.datasets)]
        scales = scales[scales["dataset"].isin(args.datasets)]
    table = build_combination_table(forecasts, scales, args.members)
    protocol = RollingProtocol(args.min_past_windows, args.max_past_windows)

    features = None if args.no_meta_learner else load_or_compute_features(args)
    combiners = default_combiners(table.members, features)
    if args.methods:
        unknown = sorted(set(args.methods) - {combiner.name for combiner in combiners})
        if unknown:
            raise SystemExit(f"Unknown method(s): {unknown}")
        combiners = [combiner for combiner in combiners if combiner.name in args.methods]

    evaluated = protocol.evaluation_mask(table)
    pairs = table.frame.loc[evaluated, ["dataset", "window"]].drop_duplicates()
    reference_models = list(dict.fromkeys([*table.members, args.baseline]))
    reference = forecasts[forecasts["model"].isin(reference_models)].merge(
        pairs, on=["dataset", "window"], how="inner"
    )

    frames = [reference]
    weight_frames = []
    timings = {}
    for combiner in combiners:
        start = time.monotonic()
        result = rolling_combine(table, combiner, protocol)
        timings[combiner.name] = round(time.monotonic() - start, 1)
        print(f"{combiner.name}: {timings[combiner.name]}s")
        frames.append(result.forecasts)
        if not result.weights.empty:
            weight_frames.append(result.weights)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_dir / "forecasts" / "combined_forecasts.parquet", index=False)
    if weight_frames:
        pd.concat(weight_frames, ignore_index=True).to_parquet(
            output_dir / "weights" / "weights.parquet", index=False
        )

    metrics = compute_metric_rows(combined, mase_scales=scales)
    (output_dir / "metrics").mkdir(exist_ok=True)
    metrics.to_csv(output_dir / "metrics" / "metrics.csv", index=False)
    write_reporting_outputs(metrics, output_dir)

    record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "members": list(table.members),
        "protocol": {
            "min_past_windows": protocol.min_past_windows,
            "max_past_windows": protocol.max_past_windows,
        },
        "evaluated_windows": {
            dataset: protocol.evaluation_windows(n_windows)
            for dataset, n_windows in table.n_windows.items()
        },
        "methods": {combiner.name: combiner.describe() for combiner in combiners},
        "runtime_seconds": timings,
        "git": _git_metadata(),
    }
    with (output_dir / "metadata" / "combination_run.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)

    aggregate = pd.read_csv(output_dir / "metrics" / "aggregate_relative_scores.csv")
    print(aggregate.to_string(index=False))
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
