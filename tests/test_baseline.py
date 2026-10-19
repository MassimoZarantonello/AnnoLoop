"""Ingest, histograms and the baseline pipeline on a small synthetic OME-TIFF.

Segmentation uses a fake model (thresholding), so these tests need neither Cellpose weights nor a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi

from macroseg.baseline import run
from macroseg.config import ConfigError, InputConfig, parse_config
from macroseg.histograms import channel_summary, plot_histograms, sample_pixels
from macroseg.ingest import ingest

FILE_NAMES = ["DAPI C1 (DAPI C1)", "Na_K_ATPase (APC C2)", "F4_80 (APC C4)"]
RENAME = {"DAPI C1 (DAPI C1)": "DAPI", "Na_K_ATPase (APC C2)": "NaK_ATPase"}


@pytest.fixture
def ome_tiff(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 100, size=(3, 128, 160), dtype=np.uint16)
    for y, x in [(30, 30), (30, 100), (90, 60)]:  # three bright "nuclei"
        image[:, y - 8 : y + 8, x - 8 : x + 8] += 1000
    path = tmp_path / "synthetic.ome.tif"
    tifffile.imwrite(
        path,
        image,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": FILE_NAMES}, "PhysicalSizeX": 170, "PhysicalSizeXUnit": "nm"},
    )
    return path, image


def threshold_model(img, **_):
    """Stand-in for Cellpose: (z, y, x, c) -> (z, y, x, 1) labels of the bright squares."""
    labels, _ = ndi.label(img[0, ..., 0] > 500)
    return labels[None, ..., None].astype(np.uint32)


def test_ingest_renames_crops_and_reads_pixel_size(ome_tiff):
    path, image = ome_tiff
    sdata = ingest(InputConfig(path=str(path), channels=RENAME, crop=(10, 74, 20, 140)), "img")
    element = sdata.images["img"]
    assert [str(c) for c in element.c.values] == ["DAPI", "NaK_ATPase"]
    assert element.shape == (2, 64, 120)
    np.testing.assert_array_equal(element.sel(c="DAPI").values, image[0, 10:74, 20:140])
    assert sdata.attrs["pixel_size_um"] == pytest.approx(0.17)


def test_ingest_shortens_file_names_without_mapping(ome_tiff):
    path, _ = ome_tiff
    sdata = ingest(InputConfig(path=str(path)), "img")
    assert [str(c) for c in sdata.images["img"].c.values] == ["DAPI", "Na_K_ATPase", "F4_80"]


def test_short_names():
    from macroseg.ingest import short_names

    names = ["CD68 REAL1346 (PE C6)", "FITC (FITC C0)", "F4/80", "DAPI"]
    assert short_names(names) == ["CD68", "FITC", "F4_80", "DAPI"]
    with pytest.raises(ValueError, match="duplicate short names"):
        short_names(["DAPI C1 (DAPI C1)", "DAPI C2 (DAPI C2)"])


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({"channels": {"missing": "X"}}, "not in the file"),
        ({"crop": (0, 999, 0, 10)}, "exceeds the image size"),
        ({"level": 3}, "level 3 requested"),
    ],
)
def test_ingest_errors(ome_tiff, cfg, message):
    path, _ = ome_tiff
    with pytest.raises(ValueError, match=message):
        ingest(InputConfig(path=str(path), **cfg), "img")


def test_histograms(ome_tiff):
    path, _ = ome_tiff
    sdata = ingest(InputConfig(path=str(path), channels=RENAME), "img")
    pixels = sample_pixels(sdata, "img", max_pixels=1000)
    assert set(pixels) == {"DAPI", "NaK_ATPase"}
    assert all(len(v) <= 1100 for v in pixels.values())
    summary = channel_summary(pixels)
    assert list(summary.columns) == ["p1", "p50", "p99", "p99.9"]
    assert plot_histograms(pixels).axes


def _config(path, zarr):
    return parse_config(
        {
            "name": "synthetic",
            "target": "macrophages",
            "input": {"path": str(path), "channels": RENAME},
            "zarr": str(zarr),
            "image": {"element": "img"},
            "segmentation": {
                "nuclei": {"channels": ["DAPI"], "labels": "nuclei"},
                "cells": {"channels": ["NaK_ATPase", "DAPI"], "labels": "cells", "diameter": 20},
            },
            "channels": {"cellpose": ["DAPI"], "candidate_marker": "DAPI"},
            "annotation": {"shapes_prefix": "manual", "class_label": "macrophage"},
        }
    )


def test_run_writes_masks_and_mean_sum_max(ome_tiff, tmp_path):
    path, _ = ome_tiff
    zarr = tmp_path / "out.zarr"
    sdata = run(_config(path, zarr), model=threshold_model)
    assert set(sdata.labels) == {"nuclei", "cells"}
    assert len(np.unique(sdata.labels["nuclei"].values)) - 1 == 3
    table = sdata.tables["nuclei_intensities"]
    assert table.n_obs == 3
    assert list(table.var_names) == ["DAPI", "NaK_ATPase"]
    assert {"sum_DAPI", "max_DAPI", "sum_NaK_ATPase", "max_NaK_ATPase"} <= set(table.obs.columns)
    assert (zarr / "labels" / "cells").exists()
    with pytest.raises(FileExistsError):
        run(_config(path, zarr), model=threshold_model)
    # Overwriting an existing store must work too (SpatialData cannot inspect the lazy TIFF reader).
    sdata = run(_config(path, zarr), overwrite=True, model=threshold_model)
    assert set(sdata.labels) == {"nuclei", "cells"}


def test_config_segmentation_and_input_validation():
    base = {
        "name": "x",
        "target": "macrophages",
        "image": {"element": "img"},
        "channels": {"cellpose": ["DAPI"], "candidate_marker": "DAPI"},
        "annotation": {"shapes_prefix": "m", "class_label": "macrophage"},
    }
    with pytest.raises(ConfigError, match="cannot contain '/'"):
        parse_config({**base, "input": {"path": "a.tif", "channels": {"F4_80": "F4/80"}}})
    with pytest.raises(ConfigError, match="y0 < y1"):
        parse_config({**base, "input": {"path": "a.tif", "crop": [10, 5, 0, 10]}})
    with pytest.raises(ConfigError, match="different labels"):
        parse_config(
            {
                **base,
                "segmentation": {
                    "nuclei": {"channels": ["DAPI"], "labels": "m"},
                    "cells": {"channels": ["DAPI"], "labels": "m"},
                },
            }
        )
    cfg = parse_config({**base, "segmentation": {"nuclei": {"channels": ["DAPI"], "labels": "n"}}})
    assert cfg.segmentation.cells is None
    assert cfg.input is None


def test_view_adds_channels_and_outlines(ome_tiff, tmp_path, monkeypatch):
    napari = pytest.importorskip("napari")
    from napari.components import ViewerModel

    from macroseg.view import open_viewer

    path, _ = ome_tiff
    zarr = tmp_path / "out.zarr"
    run(_config(path, zarr), model=threshold_model)
    monkeypatch.setattr(napari, "Viewer", lambda **kw: ViewerModel(**kw))  # no window needed
    viewer = open_viewer(zarr, channels=["NaK_ATPase"], run=False)
    layers = {layer.name: layer for layer in viewer.layers}
    assert set(layers) == {"DAPI", "NaK_ATPase", "nuclei", "cells"}
    assert layers["NaK_ATPase"].visible and not layers["DAPI"].visible
    assert layers["nuclei"].contour == 2
    with pytest.raises(ValueError, match="not in"):
        open_viewer(zarr, channels=["missing"], run=False)


def test_segmentation_model_defaults_to_cpsam_v2():
    from macroseg.config import SegmentationStep, _segmentation_step

    assert SegmentationStep(channels=("DAPI",), labels="n").model == "cpsam_v2"
    step = _segmentation_step({"channels": ["DAPI"], "labels": "n", "model": "/models/finetuned"}, "seg")
    assert step.model == "/models/finetuned"
