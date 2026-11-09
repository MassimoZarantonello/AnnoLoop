"""Annotation rounds in the AnnoCell layout: export (images, test images), close, read back."""

from __future__ import annotations

import json

import numpy as np
import pytest
from annocell import folder as af
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

from annoloop.annotation import (
    OTHER,
    PRIMARY,
    close_round,
    corrected_masks,
    export_round,
    export_test,
    load_annotations,
    load_test_images,
    score_start,
    summarize_rounds,
)
from annoloop.segmentation import Segmentation

PROPOSAL = Segmentation(labels="cells", channels=("NaK", "DAPI"))


@pytest.fixture
def sdata():
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
    return sdata


@pytest.fixture
def exported(sdata, tmp_path):
    models = tmp_path / "models"
    (models / "model_a").mkdir(parents=True)
    (models / "model_a" / "model.json").write_text(json.dumps({"images": [{"round": 1, "image": "test_image"}]}))
    root = tmp_path / "annotations"
    folders = export_round(
        sdata, "test_image", "img", PROPOSAL, root, 2, "alice", compare=["model_a"], models_root=models
    )
    return folders["alice"]


def _complete(folder, masks=None):
    """What AnnoCell does when every cell is OK and the annotator presses Done."""
    masks = af.read_start(folder) if masks is None else masks
    cells = af.sync_cells({}, masks, af.read_start(folder))
    for cell in cells.values():
        cell.mark(True)
    af.save(folder, masks, cells, done=True)


def test_export_round_writes_a_self_contained_folder(exported, sdata):
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
    assert record["format_version"] == af.FORMAT_VERSION and record["role"] == PRIMARY and "status" not in record
    assert record["visible_channels"] == ["NaK", "DAPI"]
    assert record["start_segmentation"] == {"labels": "cells", "model": "cpsam_v2", "channels": ["NaK", "DAPI"]}
    assert record["compare"] == [{"labels": "model_a", "trained_on_image": True}] and record["colours"] == {}
    assert record["element"] == "img" and record["zarr"] is None and "code_commit" in record
    [(name, masks, trained)] = af.compare_layers(folder)
    assert name == "model_a" and trained and masks.max() == 1


def test_export_round_keeps_corrections_unless_overwrite(exported, sdata):
    root = exported.parents[2]
    af.save(exported, af.read_start(exported), {})
    assert export_round(sdata, "test_image", "img", PROPOSAL, root, 2, "alice") == {"alice": exported}
    assert (exported / af.MASKS).exists()
    export_round(sdata, "test_image", "img", PROPOSAL, root, 2, "alice", overwrite=True)
    assert not (exported / af.MASKS).exists() and not (exported / af.COMPARE).exists()


def test_corrections_and_start_score(exported):
    folder = exported
    masks = af.read_start(folder)
    masks[masks == 3] = 0  # deleted
    masks[12:20, 20:30] = 4  # drawn
    masks[0:2, 0:10] = 0  # cell 1 trimmed
    cells = af.sync_cells({}, masks, af.read_start(folder))
    af.save(folder, masks, cells)
    review = af.read_review(folder)  # AnnoCell's counts, the one definition
    assert review[af.UNREVIEWED] == 1 and (review[af.DRAWN], review[af.EDITED], review["deleted"]) == (1, 1, 1)
    final, by_hand = corrected_masks(folder)
    assert list(by_hand) == [1, 4] and (final == 4).sum() == 80
    result = score_start(folder)
    assert 0 < result["AP@0.5"] < 1 and af.read_record(folder)["start_score"] == result


def test_export_test_starts_empty_and_refuses_annotated_images(exported, sdata):
    root = exported.parents[2]
    with pytest.raises(ValueError, match="within"):
        export_test(sdata, "test_image", "img", [0, 10, 0, 10], root, "carla")
    sdata.attrs.update(crop=[100, 120, 400, 430])  # far right of the annotated crop
    folder = export_test(sdata, "test_image", "img", [5, 15, 10, 20], root, "carla")
    assert folder == root / "carla" / "test" / "test_image_y105_x410"
    record = af.read_record(folder)
    assert record["mode"] == "test" and record["region"] == [105, 115, 410, 420]
    assert not (folder / af.START).exists() and af.read_image(folder)[0].shape == (4, 10, 10)
    assert af.read_masks(folder, (10, 10)).max() == 0


def test_an_image_with_other_annotators_gets_one_folder_each_with_its_role(exported, sdata):
    root = exported.parents[2]
    folders = export_round(sdata, "test_image", "img", PROPOSAL, root, 2, "alice", other_annotators=["bob", "carla"])
    assert folders["alice"] == exported  # alice's folder was already there: kept as it was
    assert folders["bob"] == root / "bob" / "round_02" / "test_image"
    roles = {person: af.read_record(folder)["role"] for person, folder in folders.items()}
    assert roles == {"alice": PRIMARY, "bob": OTHER, "carla": OTHER}
    np.testing.assert_array_equal(af.read_start(folders["bob"]), af.read_start(exported))  # the same proposal
    # Someone not listed for the image is refused, before any folder is written.
    with pytest.raises(ValueError, match="already given to"):
        export_round(sdata, "test_image", "img", PROPOSAL, root, 2, "dan", other_annotators=["alice"])
    assert not (root / "dan").exists()


def test_an_image_goes_only_to_its_annotators_and_closed_rounds(exported, sdata):
    root = exported.parents[2]
    with pytest.raises(ValueError, match=r"already given to \['alice'\]"):
        export_round(sdata, "test_image", "img", PROPOSAL, root, 2, "bob")
    with pytest.raises(ValueError, match="lowercase"):
        export_round(sdata, "test_image", "img", PROPOSAL, root, 3, "Bob")
    assert close_round(root, 2) == [exported]
    assert af.is_closed(af.read_record(exported))
    with pytest.raises(ValueError, match="closed"):
        export_round(sdata, "test_image", "img", PROPOSAL, root, 2, "alice", overwrite=True)
    with pytest.raises(ValueError, match="No image"):
        close_round(root, 5)


def test_close_round_scores_the_complete_images_and_summary(exported, sdata):
    root = exported.parents[2]
    masks = af.read_start(exported)
    masks[masks == 3] = 0  # deleted
    _complete(exported, masks)
    other = export_round(sdata, "other", "img", PROPOSAL, root, 2, "bob")["bob"]
    sdata.attrs.update(crop=[100, 120, 400, 430])
    export_test(sdata, "test_image", "img", [5, 15, 10, 20], root, "carla")
    close_round(root, 2)
    assert "start_score" in af.read_record(exported) and "start_score" not in af.read_record(other)

    table = load_annotations(root)  # training images only
    assert list(table["annotator"]) == ["alice", "bob"] and list(table["complete"]) == [True, False]
    assert table["closed"].all()
    summary = summarize_rounds(table)
    assert summary.loc[2, "annotators"] == 2 and summary.loc[2, "images"] == 2 and summary.loc[2, "complete"] == 1
    assert summary.loc[2, "cells_deleted"] == 1 and summary.loc[2, "cells_kept"] == 2
    assert summary.loc[2, "mean_start_AP@0.5"] == pytest.approx(
        2 / 3, abs=1e-4
    )  # bob's image has no score: not counted
    per_annotator = summarize_rounds(table, by=("round", "annotator"))
    assert list(per_annotator.index) == [(2, "alice"), (2, "bob")]
    assert [r["annotator"] for r in load_test_images(root)] == ["carla"]
    assert load_annotations(root / "empty").empty and summarize_rounds(load_annotations(root / "empty")).empty
