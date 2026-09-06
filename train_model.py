"""
STEP 1 — Train the fabric defect model
Run:  python train_model.py
Uses real images from data/ folders if available, otherwise generates
synthetic fabric images so it works out of the box.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from fabric_utils import IMG_SIZE, CLASSES, generate_sample

DATA_DIR = "data"           # optional: data/Normal, data/Hole, data/Stain, data/Wrinkle, data/Oil Spot
MODEL_PATH = "fabric_model.h5"
SAMPLES_PER_CLASS = 200


def load_real_or_synthetic_data():
    X, y, real_count = [], [], 0
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(os.path.join(folder, fname))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(idx)
                    real_count += 1
    if real_count >= 30:
        print(f"✅ Using {real_count} real images from '{DATA_DIR}/'")
        return np.array(X), np.array(y)

    print("⚙️  No real dataset found — generating synthetic fabric images (demo mode)...")
    for idx, cls in enumerate(CLASSES):
        for _ in range(SAMPLES_PER_CLASS):
            img, _ = generate_sample(class_name=cls)
            X.append(img)
            y.append(idx)
    print(f"✅ Generated {len(X)} synthetic images ({SAMPLES_PER_CLASS} per class)")
    return np.array(X), np.array(y)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3) activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3) activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    X, y = load_real_or_synthetic_data()
    X = X.astype("float32") / 255.0
    y = np.array(y)

    # Shuffle + 80/20 train/validation split
    rng = np.random.default_rng(42)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model()
    model.summary()

    print("\n🚀 Training started...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=12, batch_size=32)

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n📊 Validation accuracy: {acc:.1%}")

    model.save(MODEL_PATH)
    print(f"💾 Model saved to '{MODEL_PATH}'")

    # 🧪 Self-test: predict on 3 fresh samples it has never seen
    print("\n🧪 Self-test on fresh samples:")
    for cls in CLASSES:
        img, true_cls = generate_sample(class_name=cls, seed=int(rng.integers(1e6)))
        pred = model.predict(np.expand_dims(img.astype("float32") / 255.0, 0), verbose=0)
        guess = CLASSES[int(np.argmax(pred))]
        status = "✅" if guess == true_cls else "❌"
        print(f"   True: {true_cls:<7} → Predicted: {guess} {status}")


if __name__ == "__main__":
    main()
