"""Refresh saved pixel confusion counts and benchmark hashes without inference.

Requires the retained native-resolution test_per_image.csv files. This does not
retrain or reevaluate a model, or change the saved overlap headline scores.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.core.paths import ROOT


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_counts(directory: Path) -> None:
    source = directory / "test_per_image.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 860 or len({r["filename"] for r in rows}) != 860:
        raise ValueError(f"Expected 860 unique tumor slices: {source}")
    counts = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    path = directory / "test_metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["overall"].update(
        confusion_pixels=counts,
        micro_IoU=tp / (tp + fp + fn),
        micro_Dice=2 * tp / (2 * tp + fp + fn),
    )
    report["confusion_provenance"] = {
        "source": source.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(rows),
        "note": "Pixel counts summed from saved rows; no new model inference. Original overlap headline values retained.",
    }
    write_json(path, report)
    print(f"{source.relative_to(ROOT)}: {counts}")


def main() -> None:
    for run in ("cadx_segmenter_epoch159", "swin_hybrid_512_unfrozen_seed2026/epoch159_test"):
        refresh_counts(ROOT / "outputs/runs" / run)
    comparison_path = ROOT / "docs/benchmarks/comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    for record in comparison:
        record["report_sha256"] = hashlib.sha256((ROOT / record["report"]).read_bytes()).hexdigest()
    write_json(comparison_path, comparison)
    with comparison_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


if __name__ == "__main__":
    main()
