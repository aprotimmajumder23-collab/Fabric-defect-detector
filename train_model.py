"""STEP 1 — Train the fabric defect classifier.

Run:  python train_model.py [--epochs 15 --fine-tune-epochs 6 --max-per-class 4000]

Images are streamed from disk with tf.data, the dataset is split by *source
image* (so patches from the same photo never leak between train/val/test) and
the resulting model plus metrics are written to ``fabric_model.keras`` and the
``artifacts/`` folder.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from fabric_utils import (
    ARTIFACTS_DIR,
    CLASSES,
    DEFAULT_THRESHOLD,
    GOOD_INDEX,
    IMG_SIZE,
    DatasetIndex,
    dihedral_variants,
    grouped_split,
    index_dataset,
)

AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--model-path", default="fabric_model.keras")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12, help="epochs with the backbone frozen")
    p.add_argument("--fine-tune-epochs", type=int, default=10, help="extra epochs with the top of the backbone unfrozen (0 to skip)")
    p.add_argument("--fine-tune-from", default="block_6_expand", help="backbone layer from which weights are unfrozen")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    p.add_argument("--max-per-class", type=int, default=4000, help="cap on images per class (the Good class has ~23k patches)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0, help="limit TensorFlow CPU threads (0 = let TF decide)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data pipeline
# --------------------------------------------------------------------------- #
def _decode(path: tf.Tensor, label: tf.Tensor, img_size: int):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    return tf.cast(img, tf.float32), label


def _augment(img: tf.Tensor, label: tf.Tensor):
    """Fabric texture is orientation-agnostic, so flips and 90° rotations are label-preserving."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, max_delta=20.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 255.0), label


def make_dataset(index: DatasetIndex, img_size: int, batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((index.paths, index.labels))
    if training:
        ds = ds.shuffle(len(index), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda path, label: _decode(path, label, img_size), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Square-root-damped inverse-frequency weights: lifts rare defect classes without swamping 'Good'."""
    counts = np.bincount(labels, minlength=len(CLASSES))
    total = counts.sum()
    return {i: float(np.sqrt(total / (len(CLASSES) * c))) if c > 0 else 0.0 for i, c in enumerate(counts)}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(img_size: int, seed: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    backbone = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3), pooling="avg")
    backbone.trainable = False
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(img_size, img_size, 3)),
            tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0),
            backbone,
            tf.keras.layers.Dropout(0.3, seed=seed),
            tf.keras.layers.Dense(256, activation="relu", kernel_initializer=initializer),
            tf.keras.layers.Dropout(0.3, seed=seed),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax", kernel_initializer=initializer),
        ],
        name="fabric_defect_detector",
    )
    return model, backbone


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top(backbone: tf.keras.Model, from_layer: str) -> int:
    """Unfreeze layers from ``from_layer`` onwards, keeping BatchNorm statistics frozen."""
    backbone.trainable = True
    names = [layer.name for layer in backbone.layers]
    start = names.index(from_layer) if from_layer in names else 0
    trainable = 0
    for i, layer in enumerate(backbone.layers):
        if i < start or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            trainable += 1
    return trainable


def callbacks_for(model_path: str, patience: int, best_so_far: float | None = None) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, initial_value_threshold=best_so_far),
    ]


# --------------------------------------------------------------------------- #
# Evaluation helpers (shared with evaluate_model.py)
# --------------------------------------------------------------------------- #
def predict_dataset(model: tf.keras.Model, ds: tf.data.Dataset, tta: bool = False) -> np.ndarray:
    if not tta:
        return model.predict(ds, verbose=0)
    outputs = []
    for images, _ in ds:
        batch = images.numpy()
        variants = dihedral_variants(batch)
        probs = model.predict(np.concatenate(variants, axis=0), verbose=0)
        outputs.append(probs.reshape(len(variants), len(batch), -1).mean(axis=0))
    return np.concatenate(outputs, axis=0)


def evaluate_split(model: tf.keras.Model, ds: tf.data.Dataset, labels: np.ndarray, tta: bool = False, threshold: float = DEFAULT_THRESHOLD) -> dict:
    probs = predict_dataset(model, ds, tta=tta)
    preds = probs.argmax(axis=1)
    report = classification_report(labels, preds, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))

    # Defect-vs-good uses the same rule as the app: flag when 1 - P(Good) >= threshold.
    defect_score = 1.0 - probs[:, GOOD_INDEX]
    is_defect_true = labels != GOOD_INDEX
    is_defect_pred = defect_score >= threshold
    tp = int(np.sum(is_defect_true & is_defect_pred))
    fn = int(np.sum(is_defect_true & ~is_defect_pred))
    fp = int(np.sum(~is_defect_true & is_defect_pred))
    auc = float(roc_auc_score(is_defect_true, defect_score)) if 0 < is_defect_true.sum() < len(labels) else float("nan")
    return {
        "n_samples": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class": {cls: report[cls] for cls in CLASSES},
        "confusion_matrix": cm.tolist(),
        "defect_detection": {
            "threshold": threshold,
            "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "false_alarm_rate": fp / max(int(np.sum(~is_defect_true)), 1),
            "roc_auc": auc,
            "missed_defects": fn,
            "false_alarms": fp,
        },
    }


def format_confusion(cm) -> str:
    width = max(8, max(len(c) for c in CLASSES))
    header = " " * width + " " + " ".join(f"{c[:width]:>{width}}" for c in CLASSES)
    rows = [f"{name[:width]:>{width}} " + " ".join(f"{v:{width}d}" for v in row) for name, row in zip(CLASSES, cm)]
    return "\n".join([header, *rows])


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n📊 {title}: accuracy {metrics['accuracy']:.1%}, macro-F1 {metrics['macro_f1']:.3f} (n={metrics['n_samples']})")
    print(f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for cls, m in metrics["per_class"].items():
        print(f"{cls:<14}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1-score']:>10.3f}{int(m['support']):>10d}")
    dd = metrics["defect_detection"]
    print(
        f"Defect vs Good @ sensitivity {dd['threshold']:.2f} — recall {dd['recall']:.1%}, precision {dd['precision']:.1%}, "
        f"false-alarm rate {dd['false_alarm_rate']:.1%}, ROC-AUC {dd['roc_auc']:.3f} (missed {dd['missed_defects']}, false alarms {dd['false_alarms']})"
    )
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(format_confusion(metrics["confusion_matrix"]))


def history_to_dict(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for h in histories:
        if h is None:
            continue
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(float(x) for x in v)
    return merged


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    if args.threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    index = index_dataset(args.data_dir, max_per_class=args.max_per_class, seed=args.seed)
    splits = grouped_split(index, args.val_fraction, args.test_fraction, args.seed)
    train_idx, val_idx, test_idx = (index.subset(splits[k]) for k in ("train", "val", "test"))
    print(f"✅ {len(index)} images from {len(set(index.groups))} source images")
    for name, sub in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        print(f"   {name:<5} {len(sub):>6} images, {len(set(sub.groups)):>4} sources, {sub.class_counts()}")

    with open(os.path.join(ARTIFACTS_DIR, "split_manifest.json"), "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "max_per_class": args.max_per_class,
                "classes": CLASSES,
                **{k: {"paths": sub.paths, "labels": sub.labels.tolist()} for k, sub in (("train", train_idx), ("val", val_idx), ("test", test_idx))},
            },
            f,
        )

    train_ds = make_dataset(train_idx, args.img_size, args.batch_size, training=True, seed=args.seed)
    val_ds = make_dataset(val_idx, args.img_size, args.batch_size, training=False, seed=args.seed)
    test_ds = make_dataset(test_idx, args.img_size, args.batch_size, training=False, seed=args.seed)
    class_weights = balanced_class_weights(train_idx.labels)
    print("   class weights:", {CLASSES[i]: round(w, 2) for i, w in class_weights.items()})

    model, backbone = build_model(args.img_size, args.seed)
    compile_model(model, args.learning_rate)
    model.summary()

    started = time.time()
    print("\n🚀 Phase 1 — training classifier head (backbone frozen)")
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weights, callbacks=callbacks_for(args.model_path, patience=4))

    hist2 = None
    if args.fine_tune_epochs > 0:
        n = unfreeze_top(backbone, args.fine_tune_from)
        compile_model(model, args.fine_tune_learning_rate)
        best_val_loss = float(min(hist1.history["val_loss"]))
        print(f"\n🔧 Phase 2 — fine-tuning {n} backbone layers from '{args.fine_tune_from}' (checkpoint only if val_loss < {best_val_loss:.4f})")
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks_for(args.model_path, patience=3, best_so_far=best_val_loss),
        )
    print(f"\n⏱️  Training took {(time.time() - started) / 60:.1f} min")

    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.save(args.model_path, include_optimizer=False)
    val_metrics = evaluate_split(model, val_ds, val_idx.labels)
    test_metrics = evaluate_split(model, test_ds, test_idx.labels)
    print_metrics("Validation", val_metrics)
    print_metrics("Held-out test (unseen source images)", test_metrics)

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics, "args": vars(args)}, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "training_history.json"), "w") as f:
        json.dump(history_to_dict(hist1, hist2), f, indent=2)
    print(f"\n💾 Model saved to '{args.model_path}', metrics to '{ARTIFACTS_DIR}/'")


if __name__ == "__main__":
    main()
