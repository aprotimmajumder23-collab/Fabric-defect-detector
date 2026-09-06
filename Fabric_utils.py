"""Shared helpers for real fabric dataset loading and preprocessing."""

import os
import random
import numpy as np
import cv2

IMG_SIZE = 96
CLASSES = ["Good", "Hole", "Objects", "Oil Spot", "Thread Error"]
FOLDER_TO_CLASS = {
    "good": "Good",
    "hole": "Hole",
    "objects": "Objects",
    "thread error": "Thread Error",
    "oil spot": "Oil Spot",
}


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


def add_wrinkle(image):
    img = image.copy()
    h, w = img.shape[:2]
    x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
    x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
    cv2.line(img, (x1, y1), (x2, y2), 40, thickness=np.random.randint(2, 5))
    return cv2.GaussianBlur(img, (5, 5), 0)


def add_oil_spot(image):
    img = image.copy()
    h, w = img.shape[:2]
    center = (np.random.randint(20, w - 20), np.random.randint(20, h - 20))
    radius = np.random.randint(15, 35)

    mask = np.ones_like(img, dtype=np.float32)
    cv2.circle(mask, center, radius, 0.5, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    img_filtered = (img.astype(np.float32) * mask).astype(np.uint8)
    return img_filtered


def get_real_dataset_sample(class_name=None, data_dir="dataset"):
    """Return a random real image and its label from the supplied dataset folders."""
    if class_name is None:
        class_name = random.choice(CLASSES)

    class_dir = None
    for folder_name, label in FOLDER_TO_CLASS.items():
        if label == class_name:
            class_dir = os.path.join(data_dir, folder_name)
            break

    if class_dir is None or not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Dataset folder for class '{class_name}' not found in '{data_dir}'")

    files = [
        os.path.join(class_dir, fname)
        for fname in os.listdir(class_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in '{class_dir}'")

    path = random.choice(files)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, class_name


def preprocess_for_model(rgb_img):
    """Resize a single RGB image; model layers perform final normalization."""
    if rgb_img.ndim == 2:
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32")
    return np.expand_dims(arr, axis=0)
