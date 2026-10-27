"""Annotation rounds in the AnnoCell layout: export (training images, test patches) and finish."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from annocell import folder as af
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

from macroseg.annotation import corrected_masks, export_round, export_test, finish
from macroseg.config import SegmentationStep


@pytest.fixture
def sdata_and_cfg():
    image = np.arange(4 * 20 * 30, dtype=np.uint16).reshape(4, 20, 30)
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[0:10, 0:10] = 40
    labels[0:10, 10:20] = 50
    labels[12:20, 0:10] = 60
    other = np.zeros((20, 30), dtype=np.int32)
    other[0:10, 0:20] = 7
    sdata = SpatialData(
        images={"img": Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=["DAPI", "NaK", "F480", "AF"])},
        labels={
            "cells": Labels2DModel.parse(labels, dims=("y", "x")),
            "model_a": Labels2DModel.parse(other, dims=("y", "x")),
        },
    )
    sdata.attrs.update(source="slide.tif", level=0, crop=[100, 120, 200, 230], pixel_size_um=0.17)
    cfg = SimpleNamespace(
        name="test_image",
        image=SimpleNamespace(element="img"),
        zarr="/somewhere/test_image.zarr",
        segmentation=(SegmentationStep(labels="cells", channels=("NaK", "DAPI")),),
    )
    return sdata, cfg


@pytest.fixture
def exported(sdata_and_cfg, tmp_path):
    sdata, cfg = sdata_and_cfg
    models = tmp_path / "models"
    (models / "model_a").mkdir(parents=True)
    (models / "model_a" / "model.json").write_text(json.dumps({"images": [{"round": 1, "image": "test_image"}]}))
    root = tmp_path / "annotations"
    return export_round(sdata, cfg, "cells", root, 2, compare=["model_a"], models_root=models)


def test_export_round_writes_a_self_contained_folder(exported, sdata_and_cfg):
    sdata, _ = sdata_and_cfg
    folder = exported
    assert folder.name == "test_image" and folder.parent.name == "round_02"
    image, names = af.read_image(folder)
    assert names == ["DAPI", "NaK", "F480", "AF"]
    np.testing.assert_array_equal(image, sdata.images["img"].values)
    start = af.read_start(folder)
    assert sorted(np.unique(start)) == [0, 1, 2, 3]  # renumbered
    np.testing.assert_array_equal(af.read_masks(folder), start)  # nothing saved yet
    record = af.read_record(folder)
    assert record["mode"] == "train" and record["region"] == [100, 120, 200, 230]
    assert record["format_version"] == af.FORMAT_VERSION
    assert record["visible_channels"] == ["NaK", "DAPI"]
    assert record["compare"] == [{"labels": "model_a", "trained_on_image": True}] and record["colours"] == {}
    [(name, masks, trained)] = af.compare_layers(folder)
    assert name == "model_a" and trained and masks.max() == 1


def test_export_round_keeps_corrections_unless_overwrite(exported, sdata_and_cfg):
    sdata, cfg = sdata_and_cfg
    root = exported.parents[1]
    af.save(exported, af.read_start(exported), {})
    assert export_round(sdata, cfg, "cells", root, 2) == exported and (exported / af.MASKS).exists()
    export_round(sdata, cfg, "cells", root, 2, overwrite=True)
    assert not (exported / af.MASKS).exists() and not (exported / af.COMPARE).exists()


def test_save_and_finish(exported):
    folder = exported
    masks = af.read_start(folder)
    masks[masks == 3] = 0  # deleted
    masks[12:20, 20:30] = 4  # drawn
    masks[0:2, 0:10] = 0  # cell 1 trimmed
    cells = af.sync_cells({}, masks, af.read_start(folder))
    af.save(folder, masks, cells)
    assert af.read_record(folder)["review"][af.UNREVIEWED] == 1  # cell 2, untouched
    record = finish(folder)
    assert record["status"] == "partial" and record["counts"]["drawn"] == 2  # drawn + edited
    final, by_hand = corrected_masks(folder)
    assert list(by_hand) == [1, 4] and (final == 4).sum() == 80
    cells[2].state = af.OK
    af.save(folder, masks, cells)
    assert finish(folder)["status"] == "complete"
    assert af.read_cells(folder)[1] == af.Cell(af.OK, af.EDITED, False)


def test_export_test_starts_empty_and_refuses_annotated_regions(exported, sdata_and_cfg):
    sdata, cfg = sdata_and_cfg
    root = exported.parents[1]
    with pytest.raises(ValueError, match="within"):
        export_test(sdata, cfg, [0, 10, 0, 10], root)
    sdata.attrs.update(crop=[100, 120, 400, 430])  # far right of the annotated crop
    folder = export_test(sdata, cfg, [5, 15, 10, 20], root)
    assert folder == root / "test" / "test_image_y105_x410"
    record = af.read_record(folder)
    assert record["mode"] == "test" and record["region"] == [105, 115, 410, 420]
    assert not (folder / af.START).exists() and af.read_image(folder)[0].shape == (4, 10, 10)
    assert af.read_masks(folder, (10, 10)).max() == 0
    with pytest.raises(FileNotFoundError):
        finish(folder)
    masks = np.zeros((10, 10), dtype=np.uint32)
    masks[2:6, 2:6] = 1
    af.save(folder, masks, {1: af.Cell(af.OK, af.DRAWN, True)})
    assert finish(folder)["counts"] == {"cells_final": 1, "macrophages": 1}
