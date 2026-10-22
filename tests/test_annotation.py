"""Annotation rounds: export for the Cellpose GUI, a simulated correction, and the recorded changes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

from macroseg.annotation import (
    change_counts,
    corrected_masks,
    export_for_gui,
    finish,
    load_annotations,
    summarize_rounds,
)
from macroseg.config import SegmentationStep


@pytest.fixture
def sdata_and_cfg():
    image = np.arange(4 * 20 * 30, dtype=np.uint16).reshape(4, 20, 30)
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[0:10, 0:10] = 40  # IDs need not be consecutive
    labels[0:10, 10:20] = 50
    labels[12:20, 0:10] = 60
    sdata = SpatialData(
        images={"img": Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=["DAPI", "NaK", "F480", "AF"])},
        labels={"cells": Labels2DModel.parse(labels, dims=("y", "x"))},
    )
    sdata.attrs.update(source="slide.tif", level=0, crop=[0, 20, 0, 30], pixel_size_um=0.17)
    cfg = SimpleNamespace(
        name="test_image",
        image=SimpleNamespace(element="img"),
        zarr="/somewhere/test_image.zarr",
        segmentation=(SegmentationStep(labels="cells", channels=("NaK", "DAPI")),),
    )
    return sdata, cfg


def _correct_in_gui(folder):
    """What the GUI does: delete cell 3 (Cmd + click), draw a new cell, save with ismanual."""
    seg_file = folder / "test_image_seg.npy"
    dat = np.load(seg_file, allow_pickle=True).item()
    masks = dat["masks"].copy()
    masks[masks == 3] = 0
    masks[12:20, 20:30] = 3
    dat.update(masks=masks, ismanual=np.array([False, False, True]))
    np.save(seg_file, dat)


def test_export_writes_gui_files_and_record(sdata_and_cfg, tmp_path):
    sdata, cfg = sdata_and_cfg
    folder = export_for_gui(sdata, cfg, "cells", ["F480", "NaK", "DAPI"], tmp_path, round_number=1)
    assert folder == tmp_path / "round_01" / "test_image"
    image = tifffile.imread(folder / "test_image.tif")
    np.testing.assert_array_equal(image[0], sdata.images["img"].sel(c="F480").values)
    masks = np.load(folder / "test_image_seg.npy", allow_pickle=True).item()["masks"]
    assert sorted(np.unique(masks)) == [0, 1, 2, 3]  # renumbered for the GUI
    np.testing.assert_array_equal(tifffile.imread(folder / "start_masks.tif"), masks)
    record = json.loads((folder / "annotation.json").read_text())
    assert record["round"] == 1 and record["role"] == "train" and record["status"] == "exported"
    assert record["start_segmentation"] == {
        "labels": "cells",
        "model": "cpsam_v2",
        "channels": ["NaK", "DAPI"],
        "expand": None,
    }
    assert record["gui_channels"] == ["F480", "NaK", "DAPI"] and record["crop"] == [0, 20, 0, 30]


def test_export_keeps_corrections_unless_overwrite(sdata_and_cfg, tmp_path):
    sdata, cfg = sdata_and_cfg
    folder = export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=1)
    _correct_in_gui(folder)
    export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=1)
    assert 3 in np.load(folder / "test_image_seg.npy", allow_pickle=True).item()["ismanual"].nonzero()[0] + 1
    export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=1, overwrite=True)
    assert "ismanual" not in np.load(folder / "test_image_seg.npy", allow_pickle=True).item()


def test_export_rejects_bad_role_and_too_many_channels(sdata_and_cfg, tmp_path):
    sdata, cfg = sdata_and_cfg
    with pytest.raises(ValueError, match="role"):
        export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=1, role="validation")
    with pytest.raises(ValueError, match="at most 3"):
        export_for_gui(sdata, cfg, "cells", ["DAPI", "NaK", "F480", "AF"], tmp_path, round_number=1)


def test_finish_records_changes_and_start_score(sdata_and_cfg, tmp_path):
    sdata, cfg = sdata_and_cfg
    folder = export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=1)
    _correct_in_gui(folder)
    record = finish(folder)
    assert record["status"] == "complete"
    assert record["counts"] == {
        "cells_start": 3,
        "cells_final": 3,
        "unchanged": 2,
        "removed": 1,
        "drawn": 1,
        "changed_%": pytest.approx(33.33),
    }
    # Starting masks against the corrected ones: 2 of 3 cells right, 1 false, 1 missed -> AP 2 / 4.
    assert record["start_score"]["AP@0.5"] == pytest.approx(0.5)
    assert json.loads((folder / "annotation.json").read_text())["counts"]["drawn"] == 1
    assert finish(folder, complete=False)["status"] == "partial"


def test_corrected_masks_returns_the_hand_drawn_ids(sdata_and_cfg, tmp_path):
    sdata, cfg = sdata_and_cfg
    folder = export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=1)
    masks, hand_drawn = corrected_masks(folder)
    assert masks.max() == 3 and len(hand_drawn) == 0  # nothing drawn yet
    _correct_in_gui(folder)
    masks, hand_drawn = corrected_masks(folder)
    assert list(hand_drawn) == [3] and (masks[12:20, 20:30] == 3).all()


def test_change_counts_without_ismanual():
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


def test_load_annotations_and_summary(sdata_and_cfg, tmp_path):
    sdata, cfg = sdata_and_cfg
    done = export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=1)
    _correct_in_gui(done)
    finish(done)
    export_for_gui(sdata, cfg, "cells", ["DAPI"], tmp_path, round_number=2, role="test")
    table = load_annotations(tmp_path)
    assert list(table["round"]) == [1, 2] and list(table["status"]) == ["complete", "exported"]
    summary = summarize_rounds(table)
    assert summary.loc[1, "complete"] == 1 and summary.loc[1, "cells_drawn"] == 1
    assert summary.loc[1, "mean_start_AP@0.5"] == pytest.approx(0.5)
    assert "mean_start_AP@0.75" in summary.columns
    assert summary.loc[2, "test_images"] == 1 and np.isnan(summary.loc[2, "cells_drawn"])
    assert load_annotations(tmp_path / "empty").empty
    assert summarize_rounds(load_annotations(tmp_path / "empty")).empty
