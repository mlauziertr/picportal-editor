"""Corpus -> dataset.

    python3 -m scripts.style.extract --corpus <dir> [--corpus <dir> ...] --out <dataset.npz>

An image is a training sample when it has an edited `.rrdata` sidecar or a Lightroom XMP with
develop settings. Features always come from the unedited source file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .features import features_for_path, is_supported
from .schema import FEATURE_COUNT, FEATURE_VERSION, OUTPUT_KEYS
from .targets import read_target


def find_images(corpus_dirs: list[Path]) -> list[tuple[Path, Path]]:
    found = []
    for root in corpus_dirs:
        if not root.is_dir():
            raise SystemExit(f"corpus directory not found: {root}")
        for directory, subdirectories, files in os.walk(root):
            subdirectories[:] = sorted(d for d in subdirectories if not d.startswith("."))
            for name in sorted(files):
                path = Path(directory) / name
                if is_supported(path):
                    found.append((root, path))
    return found


def _sample(job: tuple[Path, Path]) -> dict:
    root, path = job
    relative = path.relative_to(root).as_posix()
    target = read_target(path)
    if target is None:
        return {"id": relative, "skip": "no_edit"}
    try:
        features = features_for_path(path)
    except Exception as error:  # noqa: BLE001 - report and continue with the rest of the corpus
        return {"id": relative, "skip": "unreadable", "error": f"{type(error).__name__}: {error}"}
    if not np.all(np.isfinite(features)):
        return {"id": relative, "skip": "non_finite_features"}
    adjustments, source = target
    return {
        "id": relative,
        "source": source,
        "features": features,
        "target": np.array([adjustments[key] for key in OUTPUT_KEYS], dtype=np.float64),
    }


def build_dataset(corpus_dirs: list[Path], workers: int | None = None) -> dict:
    jobs = find_images(corpus_dirs)
    if workers == 1 or len(jobs) < 8:
        samples = [_sample(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            samples = list(pool.map(_sample, jobs, chunksize=4))
    kept = [sample for sample in samples if "skip" not in sample]
    skipped: dict[str, int] = {}
    for sample in samples:
        if "skip" in sample:
            skipped[sample["skip"]] = skipped.get(sample["skip"], 0) + 1
    errors = [f"{s['id']}: {s['error']}" for s in samples if s.get("skip") == "unreadable"]
    features = np.stack([s["features"] for s in kept]) if kept else np.zeros((0, FEATURE_COUNT))
    targets = np.stack([s["target"] for s in kept]) if kept else np.zeros((0, len(OUTPUT_KEYS)))
    return {
        "features": features,
        "targets": targets,
        "ids": [s["id"] for s in kept],
        "sources": [s["source"] for s in kept],
        "summary": {
            "featureVersion": FEATURE_VERSION,
            "imagesScanned": len(samples),
            "samples": len(kept),
            "bySource": {src: sum(1 for s in kept if s["source"] == src) for src in ("rrdata", "xmp")},
            "skipped": skipped,
            "unreadable": errors[:20],
        },
    }


def save_dataset(dataset: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=dataset["features"],
        targets=dataset["targets"],
        ids=np.array(dataset["ids"], dtype=str),
        sources=np.array(dataset["sources"], dtype=str),
        summary=np.array(json.dumps(dataset["summary"])),
    )


def load_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        summary = json.loads(str(data["summary"]))
        if summary.get("featureVersion") != FEATURE_VERSION:
            raise SystemExit(f"dataset {path} uses {summary.get('featureVersion')}, expected {FEATURE_VERSION}")
        return {
            "features": data["features"],
            "targets": data["targets"],
            "ids": [str(value) for value in data["ids"]],
            "sources": [str(value) for value in data["sources"]],
            "summary": summary,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    dataset = build_dataset(args.corpus, args.workers)
    save_dataset(dataset, args.out)
    print(json.dumps(dataset["summary"], indent=2))
    return 0 if dataset["summary"]["samples"] else 1


if __name__ == "__main__":
    sys.exit(main())
