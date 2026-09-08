import os

import numpy as np
import pytest

import fabric_utils as fu


def test_source_id_strips_patch_suffix():
    assert fu.source_id("dataset/good/000_patch0-0.png") == "000"
    assert fu.source_id("/x/y/347_patch7-2.PNG") == "347"
    assert fu.source_id("photo.jpg") == "photo"


def _make_dataset(tmp_path, per_class):
    import cv2

    for cls, n in per_class.items():
        folder = tmp_path / fu.CLASS_TO_FOLDER[cls]
        folder.mkdir()
        for i in range(n):
            img = np.full((64, 64, 3), i * 5 % 255, dtype=np.uint8)
            cv2.imwrite(str(folder / f"{i % 4:03d}_patch{i}-0.png"), img)
    return str(tmp_path)


def test_index_dataset_and_grouped_split_keep_sources_apart(tmp_path):
    data_dir = _make_dataset(tmp_path, {cls: 12 for cls in fu.CLASSES})
    index = fu.index_dataset(data_dir, max_per_class=None, seed=0)
    assert len(index) == 12 * len(fu.CLASSES)
    assert index.class_counts() == {cls: 12 for cls in fu.CLASSES}
    assert set(index.groups) == {"000", "001", "002", "003"}

    splits = fu.grouped_split(index, val_fraction=0.25, test_fraction=0.25, seed=0)
    all_idx = sorted(splits["train"] + splits["val"] + splits["test"])
    assert all_idx == list(range(len(index)))
    groups = {k: {index.groups[i] for i in v} for k, v in splits.items()}
    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])


def test_index_dataset_caps_per_class(tmp_path):
    data_dir = _make_dataset(tmp_path, {cls: 10 for cls in fu.CLASSES})
    index = fu.index_dataset(data_dir, max_per_class=3, seed=1)
    assert index.class_counts() == {cls: 3 for cls in fu.CLASSES}
    assert all(os.path.exists(p) for p in index.paths)


def test_grouped_split_rejects_bad_fractions():
    index = fu.DatasetIndex(paths=["a"], labels=np.array([0]), groups=["a"])
    with pytest.raises(ValueError):
        fu.grouped_split(index, val_fraction=0.6, test_fraction=0.5)


def test_preprocess_resizes_and_handles_grayscale():
    gray = np.zeros((40, 50), dtype=np.uint8)
    batch = fu.preprocess_for_model(gray)
    assert batch.shape == (1, fu.IMG_SIZE, fu.IMG_SIZE, 3)
    assert batch.dtype == np.float32


def test_summarise_prediction_respects_threshold():
    probs = np.array([0.55, 0.30, 0.05, 0.05, 0.05])
    good = fu.summarise_prediction(probs, threshold=0.5)
    assert good["label"] == "Good" and not good["is_defective"]
    assert good["defect_score"] == pytest.approx(0.45)

    strict = fu.summarise_prediction(probs, threshold=0.4)
    assert strict["label"] == "Hole" and strict["is_defective"]
    assert strict["confidence"] == pytest.approx(0.30)


def test_tile_image_covers_edges_without_going_out_of_bounds():
    img = np.zeros((100, 130, 3), dtype=np.uint8)
    tiles, boxes = fu.tile_image(img, tile_size=64)
    assert len(tiles) == len(boxes) == 6
    for tile, (x0, y0, x1, y1) in zip(tiles, boxes):
        assert tile.shape == (64, 64, 3)
        assert 0 <= x0 < x1 <= 130 and 0 <= y0 < y1 <= 100
    assert (66, 36, 130, 100) in boxes

    small_tiles, small_boxes = fu.tile_image(np.zeros((30, 30, 3), dtype=np.uint8), tile_size=64)
    assert len(small_tiles) == 1 and small_boxes == [(0, 0, 30, 30)]


def test_colourise_heatmap_matches_image_shape():
    img = np.zeros((48, 72, 3), dtype=np.uint8)
    heat = np.random.default_rng(0).random((12, 18)).astype(np.float32)
    out = fu.colourise_heatmap(img, heat)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_dihedral_variants_are_distinct_and_shape_preserving():
    batch = np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
    variants = fu.dihedral_variants(batch)
    assert len(variants) == 8
    assert all(v.shape == batch.shape for v in variants)
    assert len({v.tobytes() for v in variants}) == 8
    assert np.array_equal(variants[0], batch)
