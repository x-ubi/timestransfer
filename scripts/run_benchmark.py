from __future__ import annotations

import argparse
from pathlib import Path

from timestransfer.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the time-series benchmark.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of configured model entry names to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional subset of configured dataset names to run.",
    )
    parser.add_argument(
        "--series-limit",
        type=int,
        default=None,
        help="Override config dataset.series_limit. Use config null for all series.",
    )
    parser.add_argument(
        "--allow-timesfm-failure",
        action="store_true",
        help="Accepted for compatibility; TimesFM failures are always recorded and non-fatal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_benchmark(
        config_path=Path(args.config),
        model_names=args.models,
        series_limit=args.series_limit,
        dataset_names=args.datasets,
    )
    print(f"metrics: {paths['metrics']}")
    print(f"metadata: {paths['metadata']}")
    print(f"forecasts: {paths['forecasts']}")


if __name__ == "__main__":
    main()
