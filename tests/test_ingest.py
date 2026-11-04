"""Ingest of a raw image (OME-TIFF) into SpatialData, on small synthetic files."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from annoloop.ingest import ingest, short_names

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


def test_ingest_renames_crops_and_reads_pixel_size(ome_tiff):
    path, image = ome_tiff
    sdata = ingest(path, "img", channels=RENAME, crop=(10, 74, 20, 140))
    element = sdata.images["img"]
    assert [str(c) for c in element.c.values] == ["DAPI", "NaK_ATPase"]
    assert element.shape == (2, 64, 120)
    np.testing.assert_array_equal(element.sel(c="DAPI").values, image[0, 10:74, 20:140])
    assert sdata.attrs["pixel_size_um"] == pytest.approx(0.17)


def test_ingest_shortens_file_names_without_mapping(ome_tiff):
    path, _ = ome_tiff
    sdata = ingest(path, "img")
    assert [str(c) for c in sdata.images["img"].c.values] == ["DAPI", "Na_K_ATPase", "F4_80"]


def test_short_names():
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
        ingest(path, "img", **cfg)


def test_ingest_pixel_size_follows_pyramid_level(tmp_path):
    # MACSima-style pyramid: every level is 4x smaller than the previous one.
    image = np.zeros((2, 256, 320), dtype=np.uint16)
    path = tmp_path / "pyramid.ome.tif"
    metadata = {"axes": "CYX", "Channel": {"Name": ["DAPI", "M"]}, "PhysicalSizeX": 0.17, "PhysicalSizeXUnit": "µm"}
    with tifffile.TiffWriter(path, ome=True) as tif:
        tif.write(image, subifds=2, metadata=metadata, photometric="minisblack")
        for factor in (4, 16):
            tif.write(image[:, ::factor, ::factor], subfiletype=1, photometric="minisblack")
    sizes = {level: ingest(path, "img", level=level) for level in (0, 1, 2)}
    assert [sizes[lv].images["img"].shape for lv in (0, 1, 2)] == [(2, 256, 320), (2, 64, 80), (2, 16, 20)]
    assert [sizes[lv].attrs["pixel_size_um"] for lv in (0, 1, 2)] == pytest.approx([0.17, 0.68, 2.72])
