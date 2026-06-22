from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


PACKAGE_NAMES = [
    "timestransfer",
    "datasetsforecast",
    "numpy",
    "pandas",
    "pyarrow",
    "pyyaml",
    "scikit-learn",
    "statsforecast",
    "tabpfn",
    "timesfm",
    "torch",
    "matplotlib",
]


TIMESFM_M4_CONTAMINATION = {
    "status": "possibly_or_likely_contaminated",
    "summary": (
        "M4 Hourly should be treated as a pipeline/comparability benchmark, not as a clean "
        "zero-shot generalization benchmark for TimesFM. Public TimesFM documentation for the "
        "2.5 checkpoint lists GiftEvalPretrain, Wikimedia Pageviews, Google Trends, and synthetic "
        "data; the original TimesFM paper explicitly says all M4 granularities were included in a "
        "TimesFM pretraining corpus. Exact overlap for the 2.5 checkpoint cannot be fully audited "
        "from public metadata."
    ),
    "sources": [
        "https://huggingface.co/google/timesfm-2.5-200m-pytorch",
        "https://github.com/google-research/timesfm",
        "https://arxiv.org/abs/2310.10688",
    ],
}

DATASET_CONTAMINATION_NOTES = {
    "m4_hourly": TIMESFM_M4_CONTAMINATION,
    "ett_h1_h48": {
        "status": "unknown_pretraining_overlap",
        "summary": (
            "ETTh1 reduces dependence on the known M4/TimesFM contamination concern, but public "
            "metadata is not sufficient to prove exclusion from TimesFM 2.5 pretraining."
        ),
        "sources": ["https://github.com/zhouhaoyi/ETDataset"],
    },
}


def build_environment_metadata(
    *,
    config: dict[str, Any],
    dataset_summary: dict[str, Any] | list[dict[str, Any]],
    model_statuses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dataset_summaries = dataset_summary if isinstance(dataset_summary, list) else [dataset_summary]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": _package_versions(),
        "gpu": _gpu_metadata(),
        "git": _git_metadata(),
        "config": config,
        "dataset": dataset_summary,
        "datasets": dataset_summaries,
        "model_statuses": model_statuses,
        "timesfm_m4_contamination": TIMESFM_M4_CONTAMINATION,
        "dataset_contamination_notes": DATASET_CONTAMINATION_NOTES,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_metadata() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nvidia_smi_available": False,
        "nvidia_smi_ok": False,
        "nvidia_smi_output": "",
        "torch_cuda_available": None,
        "torch_cuda_device_count": None,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload["nvidia_smi_available"] = True
        payload["nvidia_smi_ok"] = result.returncode == 0
        payload["nvidia_smi_output"] = (result.stdout or result.stderr).strip()
    except FileNotFoundError:
        payload["nvidia_smi_output"] = "nvidia-smi not found"
    except Exception as exc:  # pragma: no cover - hardware/environment dependent.
        payload["nvidia_smi_output"] = f"{type(exc).__name__}: {exc}"

    try:
        import torch

        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        payload["torch_cuda_available"] = None
        payload["torch_cuda_device_count"] = None
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"

    return payload


def _git_metadata() -> dict[str, str | None]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "status_short": run_git("status", "--short"),
    }
