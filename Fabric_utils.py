"""Shared helpers: synthetic fabric generation + image preprocessing."""

import numpy as np
import cv2

IMG_SIZE = 128
CLASSES = ["Normal", "Hole", "Stain"]


def make_fabric_base(size=IMG_SIZE, seed=None):
    """Creates a regular woven-thread pattern (the 'normal' fabric look)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 205, dtype=np.float32)
    for i in range(0, size, 8):
        img[i:i + 3, :] = 155   # horizontal threads
        img[:, i:i + 3] = 155   # vertical threads
    img += rng.normal(0, 8, (size, size))  # realistic noise
    return np.clip(img, 0, 255)


def add_hole(img, rng):
    """Dark irregular blob = hole in the fabric."""
    size = img.shape[0]
    cx, cy = rng.integers(25, size - 25, 2)
    r = rng.integers(8, 16)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[mask] = np.clip(img[mask] * 0.15 + rng.normal(0, 6, int(mask.sum())), 0, 255)
    return img


def add_stain(img, rng):
    """Larger dark patch = stain on the fabric."""
    size = img.shape[0]
    cx, cy = rng.integers(35, size - 35, 2)
    r = rng.integers(18, 30)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[mask] = np.clip(img[mask] * 0.55, 0, 255)
    return img


def generate_sample(class_name=None, seed=None):
    """Generates one synthetic fabric image. Returns (rgb_image, class_name)."""
    rng = np.random.default_rng(seed)
    if class_name is None:
        class_name = CLASSES[rng.integers(0, len(CLASSES))]
    img = make_fabric_base(seed=seed)
    if class_name == "Hole":
        img = add_hole(img, rng)
    elif class_name == "Stain":
        img = add_stain(img, rng)
    img8 = img.astype(np.uint8)
    rgb = np.stack([img8] * 3, axis=-1)  # grayscale -> RGB
    return rgb, class_name


def preprocess_for_model(rgb_img):
    """Resize + normalize a single RGB image for prediction. Returns (1, H, W, 3)."""
    resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)