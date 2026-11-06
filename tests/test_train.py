"""Training set and model records, with a fake Cellpose training (no GPU, no weights)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from annocell import folder as af
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

from annoloop import train as train_module
from annoloop.annotation import export_round, export_test
from annoloop.segmentation import Segmentation
from annoloop.settings import TrainingSettings
from annoloop.train import training_set

PROPOSAL = Segmentation(labels="cells", channels=("NaK", "DAPI"))


def _complete(folder, edited=()):
    """What AnnoCell does: the cells ``edited`` lose a pixel (edited), then every cell is checked."""
    start = af.read_start(folder)
    masks = start.copy()
    for y, x in edited:
        masks[y, x] = 0
    cells = af.sync_cells({}, masks, start)
    for cell in cells.values():
        cell.ok = True
    af.save(folder, masks, cells, done=True)  # Done pressed


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
    root = tmp_path / "annotations"
    folder = export_round(sdata, "img", "img", PROPOSAL, root, 1, "alice")
    _complete(folder, edited=[(4, 4), (52, 52)])  # first cell (tile at 0, 0) and last cell (tile at 32, 32)
    return root, image, sdata


def test_training_set_uses_the_start_channels_and_tiles(annotated):
    root, image, _ = annotated
    data = training_set(root, tile_px=32)
    assert data.channels == ["NaK", "DAPI"]  # from the starting segmentation, not every channel of the folder
    assert len(data.tiles) == 4 and data.images[0].shape == (2, 32, 32)
    np.testing.assert_array_equal(data.images[0], image[:2, :32, :32])
    assert list(data.tiles["cells"]) == [4, 4, 4, 4] and data.tiles["hand_drawn"].sum() == 2
    assert all(m.max() == 4 for m in data.masks)  # renumbered per tile
    assert data.channels == training_set(root, channels=["NaK", "DAPI"], tile_px=32).channels


def test_an_image_is_annotated_in_one_round_only(annotated):
    root, _, sdata = annotated
    with pytest.raises(ValueError, match=r"already given in round 1 \(alice\): every image is annotated once"):
        export_round(sdata, "img", "img", PROPOSAL, root, 2, "bob")
    # A second copy made by hand (outside export_round) is still refused by training, not used twice.
    other = export_round(sdata, "other", "img", PROPOSAL, root, 2, "bob")
    assert len(training_set(root, tile_px=32).tiles) == 4  # bob has not finished: left out
    _complete(other)
    record = af.read_record(other)
    af.write_record(other, {**record, "image": "img"})
    with pytest.raises(ValueError, match="more than one annotator"):
        training_set(root, tile_px=32)


def test_training_set_leaves_out_test_images_and_tiles_near_them(annotated):
    root, _, sdata = annotated
    sdata.attrs.update(source="slide.tif", level=0, crop=[1000, 1064, 2000, 2064])
    record_file = root / "alice" / "round_01" / "img" / "annotation.json"
    record = json.loads(record_file.read_text())
    record.update(source="slide.tif", level=0, region=[1000, 1064, 2000, 2064])
    record_file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="within 60 px"):
        export_test(sdata, "other", "img", [0, 32, 0, 32], root, "carla")  # inside the annotated crop
    # A test image 10 px above the crop and 20 px to its right: only the top-right tile is within 30 px.
    sdata.attrs.update(crop=[960, 1024, 2084, 2148])
    export_test(sdata, "other", "img", [0, 30, 0, 30], root, "carla", margin_px=10)
    data = training_set(root, tile_px=32, margin_px=30)
    assert list(zip(data.tiles["y0"], data.tiles["x0"], strict=True)) == [(0, 0), (32, 0), (32, 32)]
    assert data.near_test == 1
    assert training_set(root, tile_px=32, margin_px=10).near_test == 0
    assert "other_y960_x2084" not in set(data.tiles["image"])


def test_train_saves_weights_tiles_and_record(annotated, tmp_path, monkeypatch):
    root, _, sdata = annotated
    calls = {}

    def fake_train_seg(net, train_data, train_labels, save_path, model_name, **kwargs):
        calls.update(n_train=len(train_data), test_data=kwargs.get("test_data"), n_epochs=kwargs["n_epochs"])
        weights = save_path / "models" / model_name
        weights.parent.mkdir()
        weights.write_text("weights")
        return weights, np.array([1.0, 0.5]), np.array([0.0, 0.0])

    import cellpose.models

    from annoloop import trainer

    monkeypatch.setattr(cellpose.models, "CellposeModel", lambda **kwargs: SimpleNamespace(net=None))
    monkeypatch.setattr(trainer, "train_seg", fake_train_seg)  # the default loop, ours
    data = training_set(root, tile_px=32)
    settings = TrainingSettings(n_epochs=2)
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


def test_score_test_images_writes_one_score_per_model(annotated):
    root, _, sdata = annotated
    sdata.attrs.update(source="slide.tif", level=0, crop=[5000, 5064, 5000, 5064])  # far from the image
    folder = export_test(sdata, "other", "img", [0, 32, 0, 32], root, "carla")
    with pytest.raises(ValueError, match="No complete test image"):
        train_module.score_test_images(root, "cpsam", "cpsam", 0, ["NaK", "DAPI"], segment=lambda image: image[0])
    truth = np.zeros((32, 32), dtype=np.uint32)
    truth[2:10, 2:10], truth[20:28, 20:28] = 1, 2
    cells = af.sync_cells({}, truth, np.zeros_like(truth))
    af.save(folder, truth, cells)
    with pytest.raises(ValueError, match="No complete test image"):  # drawn, but Done not pressed
        train_module.score_test_images(root, "cpsam", "cpsam", 0, ["NaK", "DAPI"], segment=lambda image: image[0])
    for cell in cells.values():
        cell.mark(True)
    af.save(folder, truth, cells, done=True)
    seen = []

    def half_right(image):  # finds the first cell only
        seen.append(image.shape)
        found = np.zeros(image.shape[1:], dtype=np.uint32)
        found[2:10, 2:10] = 1
        return found

    record = train_module.score_test_images(root, "cpsam", "cpsam", 0, ["NaK", "DAPI"], segment=half_right)
    assert seen == [(2, 32, 32)]  # the model's channels only
    [test] = record["images"]
    assert test["image"] == folder.name and test["AP@0.5"] == 0.5 and test["annotator"] == "carla"
    assert [s["model"] for s in af.read_scores(root)] == ["cpsam"]
