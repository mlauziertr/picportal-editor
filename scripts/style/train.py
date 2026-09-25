"""Train, evaluate and export the style model.

    python3 -m scripts.style.train --corpus <dir> [--corpus <dir> ...] [--out models/style]
    python3 -m scripts.style.train --dataset <dataset.npz> [--out models/style]

Writes `<out>/style-model.onnx`, `<out>/manifest.json` (SHA-256 of the model) and
`<out>/report.json` (metrics). No file path of the corpus is written to the artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

from . import extract
from .models import INPUT_NAME, IR_VERSION, OPSET, OUTPUT_NAME, ClusterMean, MeanBaseline, Model, candidates
from .schema import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_VERSION,
    MANIFEST_VERSION,
    MIN_TRAINING_IMAGES,
    OUTPUT_KEYS,
    OUTPUT_RANGES,
)

MODEL_FILENAME = "style-model.onnx"


def clamp(predictions: np.ndarray) -> np.ndarray:
    low = np.array([OUTPUT_RANGES[key][0] for key in OUTPUT_KEYS])
    high = np.array([OUTPUT_RANGES[key][1] for key in OUTPUT_KEYS])
    return np.clip(predictions, low, high)


def mae(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.abs(clamp(predictions) - targets).mean(axis=0)


def relative_score(model_mae: np.ndarray, baseline_mae: np.ndarray) -> float:
    """Mean over keys of MAE / baseline MAE (keys the baseline already gets exactly are ignored)."""
    mask = baseline_mae > 1e-9
    return float((model_mae[mask] / baseline_mae[mask]).mean()) if mask.any() else 1.0


def folds(count: int, fold_count: int, seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(count)
    return [order[index::fold_count] for index in range(fold_count)]


def cross_validate(factory, features, targets, seed: int, fold_count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions of `factory()` and of the mean baseline."""
    predictions = np.zeros_like(targets)
    baseline = np.zeros_like(targets)
    for held_out in folds(len(features), min(fold_count, len(features)), seed):
        train = np.setdiff1d(np.arange(len(features)), held_out)
        predictions[held_out] = factory().fit(features[train], targets[train], seed).predict(features[held_out])
        baseline[held_out] = MeanBaseline().fit(features[train], targets[train]).predict(features[held_out])
    return predictions, baseline


def select_model(features, targets, seed: int) -> tuple[Model, list[dict]]:
    """Pick the unfitted candidate with the best cross-validated MAE relative to the baseline."""
    scored = []
    for candidate in candidates(len(features)):
        predictions, baseline = cross_validate(lambda c=candidate: copy.deepcopy(c), features, targets, seed)
        score = relative_score(mae(predictions, targets), mae(baseline, targets))
        scored.append((score, candidate))
    scored.sort(key=lambda pair: pair[0])
    table = [{"kind": c.kind, "params": c.params(), "cvRelativeMae": round(s, 4)} for s, c in scored]
    return scored[0][1], table


def per_key(values: np.ndarray) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in zip(OUTPUT_KEYS, values)}


def preset_recovery(predictions: np.ndarray, truth_path: Path | None, ids: list[str]) -> dict | None:
    """Synthetic corpora only: does the prediction land nearest to the preset that generated it?"""
    if truth_path is None or not truth_path.is_file():
        return None
    truth = json.loads(truth_path.read_text())
    presets = truth["presets"]
    style_keys = truth["styleOnlyKeys"]
    columns = [OUTPUT_KEYS.index(key) for key in style_keys]
    names = sorted(presets)
    matrix = np.array([[presets[name][key] for key in style_keys] for name in names])
    hits = total = 0
    for row, image_id in zip(predictions, ids):
        expected = truth["images"].get(image_id)
        if expected is None:
            continue
        nearest = names[int(np.abs(matrix - row[columns]).sum(axis=1).argmin())]
        hits += nearest == expected["preset"]
        total += 1
    return {"accuracy": round(hits / total, 4) if total else None, "images": total, "styleOnlyKeys": style_keys}


def evaluate(
    dataset: dict, seed: int, truth_path: Path | None, holdout: dict | None = None
) -> tuple[Model, dict]:
    """`holdout`, if given, receives the evaluated ids, targets, model and baseline predictions
    (clamped), for inspection; they are not written to the artifacts."""
    features, targets, ids = dataset["features"], dataset["targets"], dataset["ids"]
    count = len(features)
    if count < 2:
        raise SystemExit(f"need at least 2 edited images, found {count}")
    report: dict = {"samples": count, "minimumForLearnedModel": MIN_TRAINING_IMAGES}
    if count < MIN_TRAINING_IMAGES:
        report["mode"] = "fallback"
        predictions, baseline = cross_validate(ClusterMean, features, targets, seed)
        report["evaluation"] = {"protocol": "5-fold cross-validation (fallback, whole corpus)"}
        final = ClusterMean().fit(features, targets, seed)
        eval_ids = ids
    else:
        report["mode"] = "learned"
        order = np.random.default_rng(seed).permutation(count)
        test_count = max(10, count // 5)
        test, train = np.sort(order[:test_count]), np.sort(order[test_count:])
        chosen, scores = select_model(features[train], targets[train], seed)
        report["candidates"] = scores
        model = copy.deepcopy(chosen).fit(features[train], targets[train], seed)
        predictions = model.predict(features[test])
        baseline = MeanBaseline().fit(features[train], targets[train]).predict(features[test])
        targets = targets[test]
        eval_ids = [ids[index] for index in test]
        report["evaluation"] = {
            "protocol": f"holdout {len(test)} images (seed {seed}); model chosen by 5-fold CV on the other {len(train)}"
        }
        final = copy.deepcopy(chosen).fit(features, dataset["targets"], seed)
    model_mae, baseline_mae = mae(predictions, targets), mae(baseline, targets)
    report["model"] = {"kind": final.kind, "params": final.params()}
    report["evaluation"].update(
        {
            "maeModel": per_key(model_mae),
            "maeBaselineMean": per_key(baseline_mae),
            "relativeMae": round(relative_score(model_mae, baseline_mae), 4),
            "keysBeatingBaseline": [k for k, m, b in zip(OUTPUT_KEYS, model_mae, baseline_mae) if m < b],
            "beatsBaseline": bool(relative_score(model_mae, baseline_mae) < 1.0),
        }
    )
    if holdout is not None:
        holdout.update(ids=eval_ids, targets=targets, model=clamp(predictions), baseline=clamp(baseline))
    recovery = preset_recovery(predictions, truth_path, eval_ids)
    if recovery is not None:
        report["evaluation"]["presetRecovery"] = recovery
    return final, report


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_fingerprint(dataset: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(dataset["features"], dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(dataset["targets"], dtype=np.float64).tobytes())
    return digest.hexdigest()


def export(model: Model, report: dict, dataset: dict, out: Path, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / MODEL_FILENAME
    proto = model.to_onnx()
    onnx.save_model(proto, model_path)
    digest = sha256(model_path)
    manifest = {
        "version": MANIFEST_VERSION,
        "modelId": f"style-v1-{model.kind}-{digest[:12]}",
        "featureVersion": FEATURE_VERSION,
        "featureCount": FEATURE_COUNT,
        "featureNames": list(FEATURE_NAMES),
        "outputKeys": list(OUTPUT_KEYS),
        "outputRanges": {key: list(OUTPUT_RANGES[key]) for key in OUTPUT_KEYS},
        "input": INPUT_NAME,
        "output": OUTPUT_NAME,
        "kind": model.kind,
        "fallback": report["mode"] == "fallback",
        "trainingSamples": int(len(dataset["features"])),
        "minimumForLearnedModel": MIN_TRAINING_IMAGES,
        "corpusFingerprint": corpus_fingerprint(dataset),
        "createdAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": seed,
        "onnx": {"opset": OPSET, "irVersion": IR_VERSION},
        "trainer": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
        },
        "evaluation": {
            "relativeMae": report["evaluation"]["relativeMae"],
            "beatsBaseline": report["evaluation"]["beatsBaseline"],
        },
        "artifacts": [
            {"role": "regressor", "filename": MODEL_FILENAME, "sha256": digest, "bytes": model_path.stat().st_size}
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps({**report, "dataset": dataset["summary"]}, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, action="append", help="folder of edited photos (repeatable)")
    source.add_argument("--dataset", type=Path, help="dataset produced by scripts.style.extract")
    parser.add_argument("--out", type=Path, default=Path("models/style"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--save-dataset", type=Path, help="also save the extracted dataset (.npz)")
    args = parser.parse_args(argv)

    if args.dataset:
        dataset = extract.load_dataset(args.dataset)
    else:
        dataset = extract.build_dataset(args.corpus, args.workers)
        if args.save_dataset:
            extract.save_dataset(dataset, args.save_dataset)
    print(json.dumps(dataset["summary"], indent=2), file=sys.stderr)
    truth = None
    if args.corpus and len(args.corpus) == 1:
        truth = args.corpus[0] / "synthetic-truth.json"
    model, report = evaluate(dataset, args.seed, truth)
    manifest = export(model, report, dataset, args.out, args.seed)
    print(json.dumps({"manifest": manifest["modelId"], "sha256": manifest["artifacts"][0]["sha256"], **report}, indent=2))
    if report["mode"] == "fallback":
        print(
            f"WARNING: only {report['samples']} edited images (< {MIN_TRAINING_IMAGES}); "
            "exported the cluster-mean fallback.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
