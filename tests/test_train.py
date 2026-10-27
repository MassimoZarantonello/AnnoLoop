"""Training set, train / test split and model records, with a fake Cellpose training (no GPU, no weights)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel

from macroseg import train as train_module
from macroseg.annotation import export_for_gui, export_test, finish
from macroseg.config import SegmentationStep
from macroseg.train import split_tiles, tile_mosaic, training_set


@pytest.fixture
def annotated(tmp_path):
    """A 64 x 64 image with 16 square cells, exported and corrected: the cells of two tiles redrawn by hand."""
    image = np.arange(3 * 64 * 64, dtype=np.uint16).reshape(3, 64, 64)
    labels = np.zeros((64, 64), dtype=np.int32)
    for k, (y, x) in enumerate((y, x) for y in range(4, 64, 16) for x in range(4, 64, 16)):
        labels[y : y + 8, x : x + 8] = k + 1
    sdata = SpatialData(
        images={"img": Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=["NaK", "DAPI", "F480"])},
        labels={"cells": Labels2DModel.parse(labels, dims=("y", "x"))},
    )
    zarr = tmp_path / "img.zarr"
    sdata.write(zarr)
    cfg = SimpleNamespace(
        name="img",
        image=SimpleNamespace(element="img"),
        zarr=str(zarr),
        segmentation=(SegmentationStep(labels="cells", channels=("NaK", "DAPI")),),
    )
    root = tmp_path / "annotations"
    folder = export_for_gui(sdata, cfg, "cells", ["F480", "NaK", "DAPI"], root, round_number=1)
    seg_file = folder / "img_seg.npy"
    dat = np.load(seg_file, allow_pickle=True).item()
    ismanual = np.zeros(int(dat["masks"].max()), dtype=bool)
    ismanual[[0, 15]] = True  # first cell (tile at 0, 0) and last cell (tile at 32, 32) redrawn
    dat["ismanual"] = ismanual
    np.save(seg_file, dat)
    finish(folder)
    return root, image


def test_training_set_uses_the_start_channels_and_tiles(annotated):
    root, image = annotated
    data = training_set(root, tile_px=32, test_fraction=0.25)
    assert data.channels == ["NaK", "DAPI"]  # from the starting segmentation, not the GUI channels
    assert len(data.tiles) == 4 and data.images[0].shape == (2, 32, 32)
    np.testing.assert_array_equal(data.images[0], image[:2, :32, :32])
    assert list(data.tiles["cells"]) == [4, 4, 4, 4] and data.tiles["hand_drawn"].sum() == 2
    assert all(m.max() == 4 for m in data.masks)  # renumbered per tile
    assert data.channels == training_set(root, channels=["NaK", "DAPI"], tile_px=32).channels


def test_split_puts_hand_drawn_cells_on_both_sides():
    tiles = pd.DataFrame({"hand_drawn": [0, 3, 0, 0, 1, 0, 0, 0, 0, 0]})
    for seed in range(20):
        split = split_tiles(tiles, test_fraction=0.2, seed=seed)
        assert (split == "test").sum() == 2
        drawn = tiles["hand_drawn"] > 0
        assert (drawn & (split == "test")).any() and (drawn & (split == "train")).any()


def test_split_needs_two_drawn_tiles():
    with pytest.raises(ValueError, match="at least 2 tiles"):
        split_tiles(pd.DataFrame({"hand_drawn": [1, 0, 0]}))


def test_training_set_leaves_out_test_patches_and_tiles_near_them(annotated, tmp_path):
    root, _ = annotated
    sdata = read_zarr(tmp_path / "img.zarr")
    sdata.attrs.update(source="slide.tif", level=0, crop=[1000, 1064, 2000, 2064])
    record_file = root / "round_01" / "img" / "annotation.json"
    record = json.loads(record_file.read_text())
    record.update(source="slide.tif", level=0, crop=[1000, 1064, 2000, 2064])
    record_file.write_text(json.dumps(record))
    cfg = SimpleNamespace(name="other", image=SimpleNamespace(element="img"), zarr="x.zarr")
    with pytest.raises(ValueError, match="within 60 px"):
        export_test(sdata, cfg, [0, 32, 0, 32], root)  # inside the annotated crop
    # A patch up and right of the crop: 10 px above it, 20 px to its right; only the top-right tile is within 30 px.
    sdata.attrs.update(crop=[960, 1024, 2084, 2148])
    export_test(sdata, cfg, [0, 30, 0, 30], root, margin_px=10)
    data = training_set(root, tile_px=32, test_fraction=0.5, margin_px=30)
    assert list(zip(data.tiles["y0"], data.tiles["x0"], strict=True)) == [(0, 0), (32, 0), (32, 32)]
    assert data.near_test == 1
    assert training_set(root, tile_px=32, test_fraction=0.5, margin_px=10).near_test == 0
    assert "other_y960_x2084" not in set(data.tiles["image"])


def test_train_saves_weights_tiles_and_record(annotated, tmp_path, monkeypatch):
    root, _ = annotated
    calls = {}

    def fake_train_seg(net, train_data, train_labels, test_data, test_labels, save_path, model_name, **kwargs):
        calls.update(n_train=len(train_data), n_test=len(test_data), channel_axis=kwargs["channel_axis"])
        weights = save_path / "models" / model_name
        weights.parent.mkdir()
        weights.write_text("weights")
        return weights, np.array([1.0, 0.5]), np.array([0.0, 0.7])

    import cellpose.models
    import cellpose.train

    monkeypatch.setattr(cellpose.models, "CellposeModel", lambda **kwargs: SimpleNamespace(net=None))
    monkeypatch.setattr(cellpose.train, "train_seg", fake_train_seg)
    data = training_set(root, tile_px=32, test_fraction=0.25)
    folder = train_module.train(data, tmp_path / "models", "m1", n_epochs=2)

    assert calls == {"n_train": 3, "n_test": 1, "channel_axis": 0}
    assert (folder / "m1").read_text() == "weights" and not (folder / "models").exists()
    assert len(pd.read_csv(folder / "tiles.csv")) == 4
    record = json.loads((folder / "model.json").read_text())
    assert record["channels"] == ["NaK", "DAPI"] and record["tiles"] == {"train": 3, "test": 1}
    assert record["hand_drawn_cells"]["test"] == 1 and record["train_losses"] == [1.0, 0.5]
    assert record["test_losses"] == {"1": 0.7}  # epochs without a test loss are left out
    assert record["images"] == [{"round": 1, "image": "img"}]
    with pytest.raises(FileExistsError):
        train_module.train(data, tmp_path / "models", "m1")


def test_tile_mosaic_keeps_tiles_apart():
    labels = np.zeros((4, 8), dtype=np.int32)
    labels[0:2, 0:2] = 7
    labels[0:2, 4:6] = 9
    tiles = pd.DataFrame({"y0": [0, 0], "x0": [0, 4]})
    mosaic = tile_mosaic(labels, tiles, tile_px=4)
    assert mosaic.shape == (4, 8)
    assert sorted(np.unique(mosaic)) == [0, 1, 2]  # one cell per tile, different IDs
