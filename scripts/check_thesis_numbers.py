"""Cross-check every number in the blue (revised) thesis prose against the saved outputs.

Run after build_thesis_artifacts.py and thesis_derived_stats.py:
    .venv/bin/python scripts/check_thesis_numbers.py
Generated tables are verified separately by verify_thesis_package.py. Here, every numeric
token inside \\revised{...} or a revisedblock of the Section 3/4/appendix sources must
(a) match a registry entry whose value is recomputed from outputs/, configs or recorded
metadata and which appears in a sentence containing the entry's anchor words, or
(b) be a documented constant (definitions, literature values, labels). Any other token,
and any registry entry not found in its sentence, is reported as a mismatch.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FILES = ["tex/experiments.tex", "tex/results.tex", "tex/experiment-appendix.tex"]
NUMBER = re.compile(
    r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,])-?\d+(?:\.\d+)?(?![\w])"
)


def balanced(text, start):
    """Return the content of the brace group opening at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i
    raise ValueError("unbalanced braces")


def blue_segments(text):
    text = "\n".join(re.split(r"(?<!\\)%", line)[0] for line in text.splitlines())
    spans = []
    for m in re.finditer(r"\\revised\{", text):
        body, end = balanced(text, m.end() - 1)
        spans.append((m.start(), end, body))
    for m in re.finditer(r"\\begin\{revisedblock\}(.*?)\\end\{revisedblock\}", text, re.S):
        spans.append((m.start(), m.end(), m.group(1)))
    spans.sort()
    merged = []
    for s, e, b in spans:
        if merged and s < merged[-1][1]:
            continue  # nested inside an earlier blue span
        merged.append((s, e, b))
    # Short inline revisions take their sentence context from the surrounding source text.
    return [(b, text[max(0, s - 250) : e + 250] if len(b) < 60 else b) for s, e, b in merged]


def clean(segment):
    s = segment
    s = re.sub(
        r"\\(?:input|includegraphics|label|ref|eqref|pageref|nolinkurl|texttt|todo)(?:\[[^]]*\])?\{[^}]*\}",
        " ",
        s,
    )
    s = re.sub(r"\\cite\w*(?:\[[^]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", s)
    s = re.sub(r"\d+\.\d+\.\d+", " ", s)  # package versions: checked against metadata
    s = re.sub(r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?", " ", s)  # dates: checked against evidence
    s = s.replace("--", " – ").replace("~", " ").replace("\\,", "")
    # Names and identifiers that contain digits but are not measurements.
    for pat in [
        r"TimesFM[ -]2\.[05]",
        r"TimesFM~2\.5",
        r"Chronos-2",
        r"TabPFN-v2",
        r"Moirai-2",
        r"Toto-1\.0",
        r"TimesFM-2\.[05]",
        r"ETT[hm][12](?:/\d+)?",
        r"ETTh1/96",
        r"\bM4\b",
        r"\bH\d+\b",
        r"\bv\d\b",
        r"tabpfn-time-series 1\.3\.0",
        r"\b1\.3\.0\b",
        r"SHA-256",
        r"float32",
        r"bfloat16",
        r"GIFT-Eval",
        r"Benchmark II",
        r"\bID \d+",
        r"\\[a-zA-Z]+",
    ]:
        s = re.sub(pat, " ", s)
    return s


def sentences(segment):
    return [x for x in re.split(r"(?<=[.;:])\s+(?=[A-Z\\(])", segment) if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis", type=Path, default=Path("/home/ubuntu/thesis"))
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "thesis-package/evidence_rolling_origin"
    )
    args = parser.parse_args()
    from thesis_number_registry import registry, constants

    reg, claims = registry(ROOT, args.evidence.resolve())
    consts = constants(ROOT)
    env = json.loads((ROOT / "outputs/metadata/environment.json").read_text())
    sel = json.loads((args.evidence.resolve() / "forecast_selections.json").read_text())
    inv = pd.read_csv(args.evidence.resolve() / "series_inventory.csv", dtype={"unique_id": str})
    versions = set(env["packages"].values()) | {env["python"]["version"].split()[0]}
    hexids = {
        env["git"]["commit"][:8],
        env["config"]["models"]["timesfm_2p5"]["revision"],
        env["config"]["models"]["chronos2"]["revision"],
    }
    hexids |= {v["tabpfn_ts"]["checkpoint_sha256"] for v in env["model_statuses"].values()}
    assert {v["timesfm_2p5"]["revision"] for v in env["model_statuses"].values()} | {
        v["chronos2"]["revision"] for v in env["model_statuses"].values()
    } <= hexids
    dates = {s["origin"][:16] for s in sel} | {s["origin"][:10] for s in sel}
    mismatches = []
    matched = 0
    tokens_total = 0
    used = set()
    idchecks = 0
    for rel in FILES:
        text = (args.thesis / rel).read_text()
        segs = blue_segments(text)
        for seg, _ in segs:
            for v in re.findall(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])", seg):
                idchecks += 1
                if v not in versions:
                    mismatches.append({"file": rel, "version_not_in_metadata": v})
            for h in re.findall(r"\\(?:nolinkurl|texttt)\{([0-9a-f]{8,64})\}", seg):
                idchecks += 1
                if h not in hexids:
                    mismatches.append({"file": rel, "identifier_not_in_metadata": h})
            for dt in re.findall(r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?", seg):
                idchecks += 1
                if dt not in dates:
                    mismatches.append({"file": rel, "date_not_in_evidence": dt})
        file_reg = [r for r in reg if r["file"] == rel]
        for seg, context in segs:
            for sent in sentences(seg):
                ctx = sent if context is seg else context
                cleaned = clean(sent)
                for tok in NUMBER.findall(cleaned):
                    tokens_total += 1
                    hits = [
                        i
                        for i, r in enumerate(file_reg)
                        if r["text"] == tok and all(a in ctx for a in r["anchors"])
                    ]
                    if hits:
                        matched += 1
                        used.update((rel, i) for i in hits)
                        continue
                    if tok in consts or tok.lstrip("-") in consts:
                        matched += 1
                        continue
                    mismatches.append({"file": rel, "token": tok, "sentence": sent.strip()[:220]})
        for i, r in enumerate(file_reg):
            if (rel, i) not in used:
                mismatches.append(
                    {
                        "file": rel,
                        "registry_entry_not_found": r["text"],
                        "anchors": r["anchors"],
                        "source": r["source"],
                    }
                )
    # Every registry value must equal its own recomputation (formatting is done in the registry).
    for r in reg:
        if r["text"] != r["expected"]:
            mismatches.append(
                {
                    "file": r["file"],
                    "registry_value_mismatch": r["text"],
                    "recomputed": r["expected"],
                    "source": r["source"],
                }
            )
    for c in claims:
        if not c["ok"]:
            mismatches.append({"file": c["file"], "false_claim": c["claim"]})
    report = {
        "blue_numeric_tokens": tokens_total,
        "verified_tokens": matched,
        "registry_entries": len(reg),
        "verbal_claims_checked": len(claims),
        "versions_identifiers_dates_checked": idchecks,
        "documented_constants": len(consts),
        "mismatches": len(mismatches),
    }
    out = args.evidence.resolve() / "number_check.json"
    out.write_text(
        json.dumps({"summary": report, "mismatches": mismatches}, indent=1, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps(report, indent=1))
    for m in mismatches:
        print("MISMATCH", json.dumps(m, ensure_ascii=False))
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
