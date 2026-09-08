"""Evaluate a saved model on the held-out split recorded by train_model.py.

Run:  python evaluate_model.py [--split test|val] [--model-path fabric_model.keras]

Without ``artifacts/split_manifest.json`` the whole dataset is evaluated, which
includes training images and therefore overstates real-world performance.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from fabric_utils import ARTIFACTS_DIR, DEFAULT_THRESHOLD, DatasetIndex, index_dataset, load_detector, source_id
from train_model import evaluate_split, make_dataset, print_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=None, help="defaults to fabric_model.keras, then fabric_model.h5")
    p.add_argument("--manifest", default=os.path.join(ARTIFACTS_DIR, "split_manifest.json"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--tta", action="store_true", help="average predictions over 8 flips/rotations")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="defect sensitivity for the defect-vs-good metrics")
    p.add_argument("--output", default=None, help="optional path to write metrics JSON")
    return p.parse_args()


def load_split(manifest_path: str, split: str, data_dir: str) -> tuple[DatasetIndex, str]:
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = manifest[split]
        paths = entry["paths"]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} files from the manifest are missing, e.g. {missing[0]}")
        index = DatasetIndex(paths=paths, labels=np.asarray(entry["labels"], dtype=np.int64), groups=[source_id(p) for p in paths])
        return index, f"{split} split from {manifest_path}"
    print(f"⚠️  No split manifest at '{manifest_path}'; evaluating on the full dataset (includes training images).")
    return index_dataset(data_dir), f"full dataset in {data_dir}"


def main() -> None:
    args = parse_args()
    model = load_detector(args.model_path)
    img_size = model.input_shape[1]
    index, description = load_split(args.manifest, args.split, args.data_dir)
    print(f"✅ Evaluating {len(index)} images ({description}); class counts {index.class_counts()}")

    ds = make_dataset(index, img_size, args.batch_size, training=False, seed=0)
    metrics = evaluate_split(model, ds, index.labels, tta=args.tta, threshold=args.threshold)
    print_metrics(description + (" [TTA]" if args.tta else ""), metrics)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Metrics written to {args.output}")


if __name__ == "__main__":
    main()
