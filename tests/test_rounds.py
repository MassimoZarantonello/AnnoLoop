"""Annotation rounds in the AnnoCell layout: export (training ROIs, test regions), close, finish, read back."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from annocell import folder as af
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

from macroseg.annotation import (
    change_counts,
    close_round,
    corrected_masks,
    export_round,
    export_test,
    finish,
    load_annotations,
    load_test_patches,
    summarize_rounds,
)
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
    return export_round(sdata, cfg, "cells", root, 2, "alice", compare=["model_a"], models_root=models)


def test_export_round_writes_a_self_contained_folder(exported, sdata_and_cfg):
    sdata, _ = sdata_and_cfg
    folder = exported
    assert folder.relative_to(folder.parents[2]).as_posix() == "alice/round_02/test_image"
    image, names = af.read_image(folder)
    assert names == ["DAPI", "NaK", "F480", "AF"]
    np.testing.assert_array_equal(image, sdata.images["img"].values)
    start = af.read_start(folder)
    assert sorted(np.unique(start)) == [0, 1, 2, 3]  # renumbered
    np.testing.assert_array_equal(af.read_masks(folder), start)  # nothing saved yet
    record = af.read_record(folder)
    assert record["mode"] == "train" and record["region"] == [100, 120, 200, 230]
    assert record["annotator"] == "alice" and record["round"] == 2
    assert record["format_version"] == af.FORMAT_VERSION
    assert record["visible_channels"] == ["NaK", "DAPI"]
    assert record["compare"] == [{"labels": "model_a", "trained_on_image": True}] and record["colours"] == {}
    [(name, masks, trained)] = af.compare_layers(folder)
    assert name == "model_a" and trained and masks.max() == 1


def test_export_round_keeps_corrections_unless_overwrite(exported, sdata_and_cfg):
    sdata, cfg = sdata_and_cfg
    root = exported.parents[2]
    af.save(exported, af.read_start(exported), {})
    assert export_round(sdata, cfg, "cells", root, 2, "alice") == exported and (exported / af.MASKS).exists()
    export_round(sdata, cfg, "cells", root, 2, "alice", overwrite=True)
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
    assert 0 < record["start_score"]["AP@0.5"] < 1
    final, by_hand = corrected_masks(folder)
    assert list(by_hand) == [1, 4] and (final == 4).sum() == 80
    cells[2].state = af.OK
    af.save(folder, masks, cells)
    assert finish(folder)["status"] == "complete"
    assert af.read_cells(folder)[1] == af.Cell(af.OK, af.EDITED, False)


def test_export_test_starts_empty_and_refuses_annotated_regions(exported, sdata_and_cfg):
    sdata, cfg = sdata_and_cfg
    root = exported.parents[2]
    with pytest.raises(ValueError, match="within"):
        export_test(sdata, cfg, [0, 10, 0, 10], root, "carla")
    sdata.attrs.update(crop=[100, 120, 400, 430])  # far right of the annotated crop
    folder = export_test(sdata, cfg, [5, 15, 10, 20], root, "carla")
    assert folder == root / "carla" / "test" / "test_image_y105_x410"
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


def test_one_annotator_per_roi_and_closed_rounds(exported, sdata_and_cfg):
    sdata, cfg = sdata_and_cfg
    root = exported.parents[2]
    with pytest.raises(ValueError, match="one annotator per ROI"):
        export_round(sdata, cfg, "cells", root, 2, "bob")
    with pytest.raises(ValueError, match="lowercase"):
        export_round(sdata, cfg, "cells", root, 3, "Bob")
    assert close_round(root, 2) == [exported]
    assert af.is_closed(af.read_record(exported))
    with pytest.raises(ValueError, match="closed"):
        export_round(sdata, cfg, "cells", root, 2, "alice", overwrite=True)
    with pytest.raises(ValueError, match="No ROI"):
        close_round(root, 5)


def test_load_annotations_and_summary(exported, sdata_and_cfg):
    sdata, cfg = sdata_and_cfg
    root = exported.parents[2]
    masks = af.read_start(exported)
    masks[masks == 3] = 0  # deleted
    cells = af.sync_cells({}, masks, masks)
    for cell in cells.values():
        cell.state = af.OK
    af.save(exported, masks, cells)
    finish(exported)
    export_round(sdata, SimpleNamespace(**{**vars(cfg), "name": "other"}), "cells", root, 2, "bob")
    sdata.attrs.update(crop=[100, 120, 400, 430])
    export_test(sdata, cfg, [5, 15, 10, 20], root, "carla")
    table = load_annotations(root)  # training ROIs only
    assert list(table["annotator"]) == ["alice", "bob"] and list(table["complete"]) == [True, False]
    assert list(table["status"]) == ["complete", "exported"] and not table["closed"].any()
    summary = summarize_rounds(table)
    assert summary.loc[2, "annotators"] == 2 and summary.loc[2, "rois"] == 2 and summary.loc[2, "complete"] == 1
    assert summary.loc[2, "cells_removed"] == 1 and "mean_start_AP@0.75" in summary.columns
    assert [r["annotator"] for r in load_test_patches(root)] == ["carla"]
    assert load_annotations(root / "empty").empty and summarize_rounds(load_annotations(root / "empty")).empty


def test_change_counts_without_hand_drawn_ids():
    start = np.zeros((10, 20), dtype=np.int32)
    start[:, :10] = 1
    final = start.copy()
    final[:, 10:] = 2  # a new cell
    assert change_counts(start, final) == {
        "cells_start": 1,
        "cells_final": 2,
        "unchanged": 1,
        "removed": 0,
        "drawn": 1,
        "changed_%": 0.0,
    }
