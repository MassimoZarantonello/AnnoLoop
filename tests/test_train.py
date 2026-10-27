"""Training set and model records, with a fake Cellpose training (no GPU, no weights)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from annocell import folder as af
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel

from macroseg import train as train_module
from macroseg.annotation import export_round, export_test
from macroseg.config import SegmentationStep, TrainingConfig
from macroseg.train import training_set


def _complete(folder, edited=()):
    """What AnnoCell does: the cells ``edited`` lose a pixel (edited), then every cell is checked."""
    start = af.read_start(folder)
    masks = start.copy()
    for y, x in edited:
        masks[y, x] = 0
    cells = af.sync_cells({}, masks, start)
    for cell in cells.values():
        cell.state = af.OK
    af.save(folder, masks, cells)


@pytest.fixture
def annotated(tmp_path):
    """A 64 x 64 image with 16 square cells, exported for alice and completed: the cells of two tiles edited."""
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
    folder = export_round(sdata, cfg, "cells", root, 1, "alice")
    _complete(folder, edited=[(4, 4), (52, 52)])  # first cell (tile at 0, 0) and last cell (tile at 32, 32)
    return root, image


def test_training_set_uses_the_start_channels_and_tiles(annotated):
    root, image = annotated
    data = training_set(root, tile_px=32)
    assert data.channels == ["NaK", "DAPI"]  # from the starting segmentation, not every channel of the folder
    assert len(data.tiles) == 4 and data.images[0].shape == (2, 32, 32)
    np.testing.assert_array_equal(data.images[0], image[:2, :32, :32])
    assert list(data.tiles["cells"]) == [4, 4, 4, 4] and data.tiles["hand_drawn"].sum() == 2
    assert all(m.max() == 4 for m in data.masks)  # renumbered per tile
    assert data.channels == training_set(root, channels=["NaK", "DAPI"], tile_px=32).channels


def test_training_set_takes_complete_rois_of_one_annotator_only(annotated, tmp_path):
    root, _ = annotated
    sdata = read_zarr(tmp_path / "img.zarr")
    cfg = SimpleNamespace(
        name="img",
        image=SimpleNamespace(element="img"),
        zarr=str(tmp_path / "img.zarr"),
        segmentation=(SegmentationStep(labels="cells", channels=("NaK", "DAPI")),),
    )
    folder = export_round(sdata, cfg, "cells", root, 2, "bob")
    assert len(training_set(root, tile_px=32).tiles) == 4  # bob has not finished: left out
    _complete(folder)
    with pytest.raises(ValueError, match="more than one annotator"):
        training_set(root, tile_px=32)


def test_training_set_leaves_out_test_patches_and_tiles_near_them(annotated, tmp_path):
    root, _ = annotated
    sdata = read_zarr(tmp_path / "img.zarr")
    sdata.attrs.update(source="slide.tif", level=0, crop=[1000, 1064, 2000, 2064])
    record_file = root / "alice" / "round_01" / "img" / "annotation.json"
    record = json.loads(record_file.read_text())
    record.update(source="slide.tif", level=0, region=[1000, 1064, 2000, 2064])
    record_file.write_text(json.dumps(record))
    cfg = SimpleNamespace(name="other", image=SimpleNamespace(element="img"), zarr="x.zarr")
    with pytest.raises(ValueError, match="within 60 px"):
        export_test(sdata, cfg, [0, 32, 0, 32], root, "carla")  # inside the annotated crop
    # A patch up and right of the crop: 10 px above it, 20 px to its right; only the top-right tile is within 30 px.
    sdata.attrs.update(crop=[960, 1024, 2084, 2148])
    export_test(sdata, cfg, [0, 30, 0, 30], root, "carla", margin_px=10)
    data = training_set(root, tile_px=32, margin_px=30)
    assert list(zip(data.tiles["y0"], data.tiles["x0"], strict=True)) == [(0, 0), (32, 0), (32, 32)]
    assert data.near_test == 1
    assert training_set(root, tile_px=32, margin_px=10).near_test == 0
    assert "other_y960_x2084" not in set(data.tiles["image"])


def test_train_saves_weights_tiles_and_record(annotated, tmp_path, monkeypatch):
    root, _ = annotated
    calls = {}

    def fake_train_seg(net, train_data, train_labels, save_path, model_name, **kwargs):
        calls.update(n_train=len(train_data), test_data=kwargs.get("test_data"), n_epochs=kwargs["n_epochs"])
        weights = save_path / "models" / model_name
        weights.parent.mkdir()
        weights.write_text("weights")
        return weights, np.array([1.0, 0.5]), np.array([0.0, 0.0])

    import cellpose.models
    import cellpose.train

    monkeypatch.setattr(cellpose.models, "CellposeModel", lambda **kwargs: SimpleNamespace(net=None))
    monkeypatch.setattr(cellpose.train, "train_seg", fake_train_seg)
    data = training_set(root, tile_px=32)
    settings = TrainingConfig(n_epochs=2)
    folder = train_module.train(data, tmp_path / "models", "m1", settings)

    assert calls == {"n_train": 4, "test_data": None, "n_epochs": 2}  # every tile, no validation
    assert (folder / "m1").read_text() == "weights" and not (folder / "models").exists()
    assert len(pd.read_csv(folder / "tiles.csv")) == 4
    record = json.loads((folder / "model.json").read_text())
    assert record["channels"] == ["NaK", "DAPI"] and record["tiles"] == 4
    assert record["hand_drawn_cells"] == 2 and record["train_losses"] == [1.0, 0.5]
    assert record["parameters"] == {"learning_rate": 1e-5, "weight_decay": 0.1, "n_epochs": 2, "batch_size": 1}
    assert record["images"] == [{"round": 1, "annotator": "alice", "image": "img"}]
    with pytest.raises(FileExistsError):
        train_module.train(data, tmp_path / "models", "m1", settings)
