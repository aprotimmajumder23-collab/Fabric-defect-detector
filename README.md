# Fabric Defect Detector

A MobileNetV2 patch classifier for woven fabric inspection with a Streamlit UI.
It recognises five classes taken from the TILDA-style 64×64 patch dataset in
`dataset/`: **Good, Hole, Objects, Oil Spot, Thread Error**.

## Quick start

```bash
pip install -r requirements.txt
python train_model.py          # ~10 min on a 2-core CPU, writes fabric_model.keras + artifacts/
streamlit run app.py
```

A trained model is committed, so `streamlit run app.py` works out of the box.

## What the app does

| Feature | Notes |
| --- | --- |
| Single-image inspection | Small crops are classified directly with a Grad-CAM overlay showing what drove the decision. |
| Tiled scan | Photos larger than two patches are scanned tile-by-tile (configurable tile size/overlap) and rendered as a defect heatmap with flagged tiles and per-tile details. |
| Batch inspection | Upload many images, get a pass/fail table and a CSV report. |
| Defect sensitivity | Slider that trades missed defects against false alarms; a patch is flagged when `1 − P(Good)` reaches the threshold. |
| Test-time augmentation | Optional toggle that averages predictions over the 8 flips/rotations of each patch for steadier scores (~8× slower). |
| Labelled sample browser | Pull a random labelled patch per class and compare the prediction with ground truth. |
| Live metrics | The sidebar shows held-out test accuracy, macro-F1 and defect recall/precision from `artifacts/metrics.json`. |

## Training pipeline (`train_model.py`)

* **Leak-free splits.** Patches are named `<image>_patch<r>-<c>.png`; all patches
  from one source image go to the *same* split (train/val/test = 70/15/15 by
  default). Random per-patch splits — the previous behaviour — inflate validation
  accuracy because neighbouring patches of the same photo end up on both sides.
* **Streaming `tf.data` pipeline** with orientation-preserving augmentation
  (flips, 90° rotations, brightness/contrast), so the full dataset no longer has
  to fit in RAM.
* **Two-phase transfer learning:** train the head on a frozen ImageNet
  MobileNetV2, then fine-tune the top blocks at a low learning rate with
  BatchNorm frozen. The checkpoint is only overwritten when validation loss improves.
* **Honest metrics:** per-class precision/recall/F1, a confusion matrix and
  binary *defect-vs-good* recall / false-alarm rate / ROC-AUC (using the same
  `1 − P(Good) ≥ sensitivity` rule as the app) on the held-out test split, saved to
  `artifacts/metrics.json`, with the split manifest and training history alongside.

Useful flags:

```bash
python train_model.py --max-per-class 8000 --epochs 20 --fine-tune-epochs 8
python train_model.py --fine-tune-epochs 0 --threads 2      # fastest CPU run
python evaluate_model.py --split test                        # re-score the saved model
python evaluate_model.py --split val --model-path other.keras
python evaluate_model.py --split test --tta                  # 8-way flip/rotation averaging
```

## Shipped model

`fabric_model.keras` was trained with the defaults above (128 px input, backbone
fine-tuned from `block_6_expand`). On the held-out test split of 964 patches from
60 source images never seen in training:

| Metric | Value |
| --- | --- |
| Accuracy (5-class) | 78.4% (80.0% with TTA) |
| Macro-F1 | 0.66 |
| Defect-vs-good ROC-AUC | 0.92 |
| Defect recall / false-alarm rate at sensitivity 0.5 | 77% / 9% |
| Defect recall / false-alarm rate at sensitivity 0.3 | 91% / 25% |

`Oil Spot` and `Objects` are the weak classes (often confused with `Good`);
`Hole` and `Thread Error` are reliable. Raise the sensitivity slider when missing
a defect is costlier than a false alarm. The legacy `fabric_model.h5` was removed:
it cannot be loaded by Keras 3 and was trained with leaky per-patch splits.

## Project layout

```
app.py              Streamlit UI
train_model.py      training + evaluation, writes fabric_model.keras and artifacts/
evaluate_model.py   evaluate a saved model on the recorded split
fabric_utils.py     dataset indexing, grouped split, preprocessing, tiling, Grad-CAM
tests/              pytest unit tests for the utilities
dataset/            labelled 64×64 patches (one folder per class)
PUBLIC_DATASETS.md  data-sourcing policy
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

## Notes on the data

`dataset/good` holds ~23k patches versus 340–840 per defect class. Training caps
each class at `--max-per-class` (default 4000) and applies square-root inverse
frequency class weights. Only add images with a clear licence and matching
labels — see `PUBLIC_DATASETS.md`.
