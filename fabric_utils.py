"""Shared helpers: dataset indexing, leakage-free splitting, preprocessing,
tiled whole-image inspection and Grad-CAM explanations."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
PATCH_SIZE = 64
CLASSES = ["Good", "Hole", "Objects", "Oil Spot", "Thread Error"]
GOOD_CLASS = "Good"
GOOD_INDEX = CLASSES.index(GOOD_CLASS)
DEFAULT_THRESHOLD = 0.5
FOLDER_TO_CLASS = {
    "good": "Good",
    "hole": "Hole",
    "objects": "Objects",
    "oil spot": "Oil Spot",
    "thread error": "Thread Error",
}
CLASS_TO_FOLDER = {label: folder for folder, label in FOLDER_TO_CLASS.items()}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MODEL_CANDIDATES = ["fabric_model.keras", "fabric_model.h5"]
ARTIFACTS_DIR = "artifacts"


# --------------------------------------------------------------------------- #
# Dataset indexing and splitting
# --------------------------------------------------------------------------- #
@dataclass
class DatasetIndex:
    paths: list[str]
    labels: np.ndarray
    groups: list[str]

    def __len__(self) -> int:
        return len(self.paths)

    def subset(self, indices) -> DatasetIndex:
        indices = list(indices)
        return DatasetIndex(
            paths=[self.paths[i] for i in indices],
            labels=self.labels[indices],
            groups=[self.groups[i] for i in indices],
        )

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.labels, minlength=len(CLASSES))
        return {cls: int(n) for cls, n in zip(CLASSES, counts)}


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def source_id(filename: str) -> str:
    """Identify the source fabric image a patch was cut from.

    Dataset patches are named ``<image>_patch<row>-<col>.png``; every patch of
    the same ``<image>`` shares texture, lighting and often the same defect, so
    they must never be split across train and validation.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_patch")[0]


def list_class_files(data_dir: str, class_name: str) -> list[str]:
    folder = os.path.join(data_dir, CLASS_TO_FOLDER[class_name])
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if is_image_file(f))


def index_dataset(data_dir: str = "dataset", max_per_class: int | None = None, seed: int = 42) -> DatasetIndex:
    """Collect every labelled image, optionally capping each class at ``max_per_class``."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found")

    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        files = list_class_files(data_dir, cls)
        if not files:
            raise ValueError(f"No images found for class '{cls}' in '{data_dir}/{CLASS_TO_FOLDER[cls]}'")
        if max_per_class is not None and len(files) > max_per_class:
            files = sorted(rng.choice(files, size=max_per_class, replace=False).tolist())
        paths.extend(files)
        labels.extend([idx] * len(files))

    return DatasetIndex(paths=paths, labels=np.asarray(labels, dtype=np.int64), groups=[source_id(p) for p in paths])


def grouped_split(index: DatasetIndex, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42) -> dict[str, list[int]]:
    """Split sample indices into train/val/test so that no source image appears in two splits."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1")

    unique_groups = sorted(set(index.groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_test = int(round(n_groups * test_fraction))
    n_val = int(round(n_groups * val_fraction))
    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, group in enumerate(index.groups):
        if group in test_groups:
            splits["test"].append(i)
        elif group in val_groups:
            splits["val"].append(i)
        else:
            splits["train"].append(i)
    return splits


# --------------------------------------------------------------------------- #
# Image loading and preprocessing
# --------------------------------------------------------------------------- #
def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def preprocess_batch(images: list[np.ndarray] | np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Resize RGB uint8 images to the model input size; normalisation happens inside the model."""
    batch = np.empty((len(images), img_size, img_size, 3), dtype=np.float32)
    for i, img in enumerate(images):
        img = to_rgb(img)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        batch[i] = img
    return batch


def preprocess_for_model(rgb_img: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    return preprocess_batch([rgb_img], img_size)


def get_real_dataset_sample(class_name: str | None = None, data_dir: str = "dataset", rng: random.Random | None = None):
    """Return ``(rgb_image, label, path)`` for a random labelled image."""
    rng = rng or random
    if class_name is None:
        class_name = rng.choice(CLASSES)
    files = list_class_files(data_dir, class_name)
    if not files:
        raise FileNotFoundError(f"No images found for class '{class_name}' in '{data_dir}'")
    path = rng.choice(files)
    return load_image_rgb(path), class_name, path


# --------------------------------------------------------------------------- #
# Model loading and prediction
# --------------------------------------------------------------------------- #
def find_model_path(candidates=MODEL_CANDIDATES) -> str | None:
    return next((p for p in candidates if os.path.exists(p)), None)


def load_detector(path: str | None = None) -> tf.keras.Model:
    path = path or find_model_path()
    if path is None:
        raise FileNotFoundError(f"No model found (looked for {', '.join(MODEL_CANDIDATES)}). Run `python train_model.py` first.")
    model = tf.keras.models.load_model(path, compile=False)
    n_out = model.output_shape[-1]
    if n_out != len(CLASSES):
        raise ValueError(f"Model at '{path}' has {n_out} outputs but {len(CLASSES)} classes are expected. Retrain with `python train_model.py`.")
    return model


def dihedral_variants(batch: np.ndarray) -> list[np.ndarray]:
    """The 8 flips/rotations of a ``(N, H, W, C)`` batch; fabric labels are invariant to all of them."""
    variants = []
    for k in range(4):
        rotated = np.rot90(batch, k=k, axes=(1, 2))
        variants.append(rotated)
        variants.append(rotated[:, :, ::-1])
    return variants


def predict_probabilities(model: tf.keras.Model, images: list[np.ndarray], batch_size: int = 64, tta: bool = False) -> np.ndarray:
    """Class probabilities for a list of RGB images. ``tta`` averages over the 8 dihedral transforms."""
    if len(images) == 0:
        return np.zeros((0, len(CLASSES)), dtype=np.float32)
    img_size = model.input_shape[1] or IMG_SIZE
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = preprocess_batch(images[start : start + batch_size], img_size)
        if tta:
            variants = dihedral_variants(batch)
            probs = model.predict(np.concatenate(variants, axis=0), verbose=0)
            outputs.append(probs.reshape(len(variants), len(batch), -1).mean(axis=0))
        else:
            outputs.append(model.predict(batch, verbose=0))
    return np.concatenate(outputs, axis=0)


def defect_score(probs: np.ndarray) -> np.ndarray:
    """Probability that a patch is defective (any non-good class)."""
    return 1.0 - probs[..., GOOD_INDEX]


def summarise_prediction(probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Turn class probabilities into a verdict.

    A patch is flagged defective when the summed defect probability reaches
    ``threshold``; the reported class is the most likely defect in that case.
    """
    probs = np.asarray(probs, dtype=np.float64)
    score = float(defect_score(probs))
    order = np.argsort(probs)[::-1]
    if score >= threshold:
        defect_probs = probs.copy()
        defect_probs[GOOD_INDEX] = -1
        label = CLASSES[int(np.argmax(defect_probs))]
    else:
        label = GOOD_CLASS
    return {
        "label": label,
        "is_defective": label != GOOD_CLASS,
        "defect_score": score,
        "confidence": float(probs[CLASSES.index(label)]),
        "top_class": CLASSES[int(order[0])],
        "runner_up": CLASSES[int(order[1])],
        "runner_up_confidence": float(probs[order[1]]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


# --------------------------------------------------------------------------- #
# Tiled inspection for whole fabric images
# --------------------------------------------------------------------------- #
def tile_image(rgb_img: np.ndarray, tile_size: int = PATCH_SIZE, stride: int | None = None):
    """Cut an image into square tiles. Returns ``(tiles, boxes)`` where each box is ``(x0, y0, x1, y1)``."""
    stride = stride or tile_size
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    h, w = rgb_img.shape[:2]
    if h < tile_size or w < tile_size:
        return [rgb_img], [(0, 0, w, h)]

    ys = list(range(0, h - tile_size + 1, stride))
    xs = list(range(0, w - tile_size + 1, stride))
    if ys[-1] != h - tile_size:
        ys.append(h - tile_size)
    if xs[-1] != w - tile_size:
        xs.append(w - tile_size)

    tiles, boxes = [], []
    for y in ys:
        for x in xs:
            tiles.append(rgb_img[y : y + tile_size, x : x + tile_size])
            boxes.append((x, y, x + tile_size, y + tile_size))
    return tiles, boxes


def inspect_image(
    model: tf.keras.Model, rgb_img: np.ndarray, tile_size: int = PATCH_SIZE, stride: int | None = None, threshold: float = DEFAULT_THRESHOLD, tta: bool = False
) -> dict:
    """Run the patch classifier over every tile of an image and aggregate the results."""
    rgb_img = to_rgb(rgb_img)
    tiles, boxes = tile_image(rgb_img, tile_size, stride)
    probs = predict_probabilities(model, tiles, tta=tta)
    scores = defect_score(probs)

    h, w = rgb_img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    flagged = []
    for (x0, y0, x1, y1), p, s in zip(boxes, probs, scores):
        heat[y0:y1, x0:x1] += s
        weight[y0:y1, x0:x1] += 1
        summary = summarise_prediction(p, threshold)
        if summary["is_defective"]:
            flagged.append({"box": (x0, y0, x1, y1), **summary})
    heat = np.divide(heat, weight, out=np.zeros_like(heat), where=weight > 0)

    class_votes = {cls: 0 for cls in CLASSES if cls != GOOD_CLASS}
    for f in flagged:
        class_votes[f["label"]] += 1
    dominant = max(class_votes, key=class_votes.get) if flagged else GOOD_CLASS

    return {
        "n_tiles": len(tiles),
        "boxes": boxes,
        "probabilities": probs,
        "scores": scores,
        "heatmap": heat,
        "flagged": flagged,
        "label": dominant,
        "is_defective": bool(flagged),
        "max_defect_score": float(scores.max()) if len(scores) else 0.0,
        "defect_area_fraction": float(len(flagged) / max(len(tiles), 1)),
        "class_votes": class_votes,
    }


def colourise_heatmap(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a ``[0, 1]`` heatmap over an RGB image using the JET colour map."""
    heatmap = np.clip(heatmap, 0.0, 1.0)
    if heatmap.shape != rgb_img.shape[:2]:
        heatmap = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    coloured = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(to_rgb(rgb_img), 1 - alpha, coloured, alpha, 0)


def draw_flagged_tiles(rgb_img: np.ndarray, flagged: list[dict], colour=(220, 38, 38)) -> np.ndarray:
    out = to_rgb(rgb_img).copy()
    thickness = max(1, min(out.shape[:2]) // 300)
    for f in flagged:
        x0, y0, x1, y1 = f["box"]
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), colour, thickness)
    return out


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("Model does not contain a nested convolutional backbone")


def _last_conv_layer(backbone: tf.keras.Model):
    for layer in reversed(backbone.layers):
        if len(layer.output.shape) == 4:
            return layer
    raise ValueError("No 4-D convolutional feature map found in backbone")


def grad_cam(model: tf.keras.Model, rgb_img: np.ndarray, class_index: int | None = None) -> np.ndarray:
    """Return a ``[0, 1]`` Grad-CAM heatmap (at input resolution) for ``class_index``."""
    backbone = _find_backbone(model)
    layers = model.layers
    b_idx = layers.index(backbone)
    pre_layers, post_layers = layers[:b_idx], layers[b_idx + 1 :]
    conv_layer = _last_conv_layer(backbone)
    conv_model = tf.keras.Model(backbone.inputs, [conv_layer.output, backbone.output])

    img_size = model.input_shape[1] or IMG_SIZE
    x = tf.convert_to_tensor(preprocess_for_model(rgb_img, img_size))
    with tf.GradientTape() as tape:
        for layer in pre_layers:
            x = layer(x, training=False)
        conv_out, features = conv_model(x, training=False)
        tape.watch(conv_out)
        for layer in post_layers:
            features = layer(features, training=False)
        preds = features
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cv2.resize(cam.astype(np.float32), (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)
