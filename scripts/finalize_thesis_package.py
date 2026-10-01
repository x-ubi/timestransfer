"""Record delivery hashes and provenance after artifact validation and integration."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "thesis-package"
THESIS = Path("/home/ubuntu/thesis")
EV = PACKAGE / "evidence"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


validation = json.loads((EV / "package_validation.json").read_text())
assert validation["integrated_files_match_package"]
assert validation["build_pdf_exists_and_no_unresolved_references_or_missing_glyphs"]
artifacts = json.loads((EV / "artifact_manifest.json").read_text())
by_output = {item["output"]: item for item in artifacts["artifacts"]}
records = []
for path in sorted((PACKAGE / "thesis").rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(PACKAGE)
    integrated = THESIS / path.relative_to(PACKAGE / "thesis")
    assert path.read_bytes() == integrated.read_bytes(), relative
    origin = by_output.get(str(relative))
    if origin:
        inputs = origin["inputs"]
        method = origin["transformation"]
        command = "MPLCONFIGDIR=/tmp/thesis-mpl .venv/bin/python scripts/build_thesis_artifacts.py"
    else:
        inputs = ["docs/experiment-writing-agent-plan.md", "evidence/ledger.md"]
        original = EV / "pre-integration" / path.relative_to(PACKAGE / "thesis")
        if original.exists():
            inputs.append(str(original.relative_to(PACKAGE)))
        method = "Authored/integrated evidence-backed prose or static source/configuration table; see ledger and acceptance audit."
        command = "Editorial writing; python3 scripts/integrate_thesis_package.py"
    records.append(
        {
            "output": str(relative),
            "sha256": sha(path),
            "inputs": inputs,
            "transformation": method,
            "command": command,
        }
    )
assert len(records) == 46
# Preserve build evidence outside the thesis's rendered input tree.
for source, name in [
    (THESIS / "build/experiments/main.log", "integrated-latex.log"),
    (THESIS / "build/experiments/main.blg", "integrated-biber.log"),
]:
    shutil.copyfile(source, EV / name)
pdf = THESIS / "build/experiments/main.pdf"
info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
(EV / "integrated-pdfinfo.txt").write_text(info)
assert "Pages:           63" in info
sources = {
    "dataset_inventory.csv": (
        "cached data, project loader, configurations",
        "Counts and intervals after exact loading",
    ),
    "series_inventory.csv": (
        "cached data, project loader, MASE scales",
        "Per-series split boundaries, missingness, extrema and denominator checks",
    ),
    "timestamp_gaps.json": ("cached data timestamps", "Differences from configured interval"),
    "forecast_selections.json": (
        "saved main per-series metrics and loaded train tails",
        "Test-sMAPE rank with string-ID tie handling",
    ),
    "correlations_smape.csv": (
        "outputs/analysis/series_features_with_metrics.csv",
        "Within-task and pooled Spearman; original analysis reproduced",
    ),
    "correlations_mase.csv": (
        "outputs/analysis/series_features_with_metrics.csv",
        "Additional descriptive MASE Spearman; original outputs preserved",
    ),
    "flag_deltas.csv": (
        "outputs/timesfm_tuning/metrics/metrics.csv",
        "Ablation minus common same-run context-1024 baseline",
    ),
    "validation.json": (
        "both forecast/metric runs, cached data and configurations",
        "Generator assertions; see artifact manifest and generator source",
    ),
    "package_validation.json": (
        "generated LaTeX, saved CSVs, current source hashes, integrated build",
        "scripts/verify_thesis_package.py assertions",
    ),
}
evidence = []
for name, (inputs, method) in sources.items():
    evidence.append(
        {
            "output": f"evidence/{name}",
            "sha256": sha(EV / name),
            "inputs": inputs,
            "transformation": method,
        }
    )
code = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "scripts").glob("*thesis*.py"))}
dependencies = {
    str(p.relative_to(THESIS)): sha(p)
    for p in [THESIS / "eiti/eiti-thesis.cls", THESIS / "img/timesfm.jpg"]
}
manifest = {
    "thesis_files": records,
    "derived_evidence": evidence,
    "generation_and_validation_code": code,
    "original_thesis_dependencies": dependencies,
    "source_hashes": "evidence/artifact_manifest.json",
    "build": {
        "command": "latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build/experiments main.tex",
        "cwd": str(THESIS),
        "pdf": str(pdf),
        "pdf_sha256": sha(pdf),
        "pages": 63,
        "latex_log_sha256": sha(EV / "integrated-latex.log"),
        "biber_log_sha256": sha(EV / "integrated-biber.log"),
    },
    "validation_command": "MPLCONFIGDIR=/tmp/thesis-mpl .venv/bin/python scripts/verify_thesis_package.py --integrated /home/ubuntu/thesis --build-dir /home/ubuntu/thesis/build/experiments",
    "finalize_command": "python3 scripts/finalize_thesis_package.py",
    "notes": "Local integration only; no commits, pushes, benchmark reruns or Overleaf synchronization. Unknown archival provenance is documented, not inferred.",
}
(EV / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    f"Manifest covers {len(records)} integrated thesis files, {len(evidence)} derived evidence files, and the {manifest['build']['pages']}-page PDF."
)
