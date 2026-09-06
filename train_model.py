"""
STEP 1 — Train the fabric defect model
Run:  python train_model.py
Uses only real images from the dataset folders.
"""

import os
import zipfile
import numpy as np
import cv2
import tensorflow as tf
from Fabric_utils import IMG_SIZE, CLASSES, get_real_dataset_sample, preprocess_for_model

DATA_DIRS = ["dataset", "data"]
ARCHIVE_PATH = "archive.zip"
MODEL_PATH = "fabric_model.h5"
MAX_REAL_IMAGES_PER_CLASS = 1000
TRAIN_EPOCHS = 16
BATCH_SIZE = 8
SEED = 42

FOLDER_TO_CLASS = {
    "good": "Good",
    "hole": "Hole",
    "objects": "Objects",
    "thread error": "Thread Error",
    "oil spot": "Oil Spot",
}

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def ensure_dataset_ready():
    """Unzip the bundled archive if the dataset folders are missing."""
    if any(os.path.isdir(d) and os.listdir(d) for d in DATA_DIRS):
        return

    if os.path.exists(ARCHIVE_PATH) and zipfile.is_zipfile(ARCHIVE_PATH):
        print(f"📦 Extracting archive '{ARCHIVE_PATH}' into workspace...")
        with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
            zf.extractall(".")


def load_real_or_synthetic_data():
    """Load unique real dataset images without creating synthetic duplicates."""
    ensure_dataset_ready()
    data_dir = next((d for d in DATA_DIRS if os.path.isdir(d)), "dataset")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found. Please unzip '{ARCHIVE_PATH}' or add the folder.")

    X, y = [], []
    class_images = {cls: [] for cls in CLASSES}
    for folder_name, cls in FOLDER_TO_CLASS.items():
        folder = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder):
            continue

        files = [
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        rng = np.random.default_rng(SEED + len(y))
        if len(files) > MAX_REAL_IMAGES_PER_CLASS:
            files = rng.choice(files, size=MAX_REAL_IMAGES_PER_CLASS, replace=False).tolist()

        for path in files:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            class_images[cls].append(img)

    for idx, cls in enumerate(CLASSES):
        images = class_images[cls]
        if not images:
            raise ValueError(f"No real images found for class '{cls}' in '{data_dir}'")
        for img in images:
            X.append(img)
            y.append(idx)

    print(f"✅ Using {len(X)} real images from '{data_dir}/' across {len(CLASSES)} classes")
    return np.array(X), np.array(y)


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomContrast(0.1, seed=SEED),
        tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0),
        base_model,
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    X, y = load_real_or_synthetic_data()
    X = X.astype("float32")
    y = np.array(y)

    # Split each class independently so validation contains real, unseen images.
    rng = np.random.default_rng(SEED)
    train_indices, val_indices = [], []
    for class_id in range(len(CLASSES)):
        indices = np.flatnonzero(y == class_id)
        rng.shuffle(indices)
        split = max(1, int(len(indices) * 0.8))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    class_counts = np.bincount(y_train, minlength=len(CLASSES))
    class_weights = {
        i: float(np.sqrt(len(y_train) / (len(CLASSES) * count)))
        for i, count in enumerate(class_counts)
        if count > 0
    }
    print(f"✅ Train/validation split: {len(X_train)}/{len(X_val)} real images")
    print("   Training class counts:", dict(zip(CLASSES, class_counts.tolist())))

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True
        ),
    ]

    print("\n🚀 Training started...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weights, callbacks=callbacks)

    # Keep the best frozen-backbone checkpoint for reliable CPU inference.
    # Full MobileNet fine-tuning is too memory-intensive for this environment.
    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n📊 Validation accuracy: {acc:.1%}")

    predictions = np.argmax(model.predict(X_val, batch_size=BATCH_SIZE, verbose=0), axis=1)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int32)
    for actual, predicted in zip(y_val, predictions):
        confusion[actual, predicted] += 1
    print("\n📋 Confusion matrix (rows=actual, columns=predicted):")
    print("   " + " ".join(f"{name[:8]:>8}" for name in CLASSES))
    for name, row in zip(CLASSES, confusion):
        print(f"{name[:8]:>8} " + " ".join(f"{value:8d}" for value in row))

    print(f"💾 Best model already saved to '{MODEL_PATH}'")

    print("\n🧪 Real-dataset validation check:")
    for cls in CLASSES:
        img, true_cls = get_real_dataset_sample(class_name=cls)
        pred = model.predict(preprocess_for_model(img), verbose=0)
        guess = CLASSES[int(np.argmax(pred))]
        status = "✅" if guess == true_cls else "❌"
        print(f"   True: {true_cls:<7} → Predicted: {guess} {status}")


if __name__ == "__main__":
    main()
