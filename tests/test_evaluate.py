"""Evaluation measures on small hand-made masks with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from macroseg.evaluate import border_ratio, marker_separation, nuclei_per_cell, shape_features


def test_nuclei_per_cell_counts_merged_empty_and_missed():
    cells = np.zeros((20, 30), dtype=np.int32)
    cells[0:10, 0:10] = 1  # one nucleus
    cells[0:10, 10:30] = 2  # two nuclei -> merged cells
    cells[12:20, 0:10] = 3  # no nucleus -> fragment or false cell
    nuclei = np.zeros_like(cells)
    nuclei[2:6, 2:6] = 10  # inside cell 1
    nuclei[2:6, 12:16] = 20  # inside cell 2
    nuclei[2:6, 22:26] = 21  # inside cell 2
    nuclei[14:18, 14:18] = 30  # in no cell -> missed cell
    nuclei[8:12, 5:9] = 40  # half in cell 1 (rows 8-9), half in background (rows 10-11)
    per_cell, per_nucleus = nuclei_per_cell(cells, nuclei)
    assert dict(zip(per_cell["cell_ID"], per_cell["n_nuclei"], strict=True)) == {1: 2, 2: 2, 3: 0}
    assigned = dict(zip(per_nucleus["nucleus_ID"], per_nucleus["cell_ID"], strict=True))
    assert assigned == {10: 1, 20: 2, 21: 2, 30: 0, 40: 1}  # 40: exactly half covered counts (>= 0.5)
    _, strict = nuclei_per_cell(cells, nuclei, min_fraction=0.6)
    assert strict.set_index("nucleus_ID").loc[40, "cell_ID"] == 0


def test_border_ratio_is_high_when_outline_sits_on_the_membrane():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:25, 5:25] = 1
    on_membrane = np.ones((30, 30))
    on_membrane[5:25, 5:25] = 1.0
    on_membrane[5:7, 5:25] = on_membrane[23:25, 5:25] = 10.0  # bright 2-px band on the outline
    on_membrane[5:25, 5:7] = on_membrane[5:25, 23:25] = 10.0
    result = border_ratio(labels, on_membrane, width=2)
    assert result.loc[0, "ratio"] == pytest.approx(10.0)
    flat = np.ones((30, 30))
    assert border_ratio(labels, flat, width=2).loc[0, "ratio"] == pytest.approx(1.0)


def test_border_ratio_nan_for_objects_without_inside():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1  # 3x3: no pixel has a full 5x5 neighbourhood inside it
    assert np.isnan(border_ratio(labels, np.ones((10, 10)), width=2).loc[0, "ratio"])


def test_marker_separation_clean_vs_mixed():
    rng = np.random.default_rng(0)
    clean = np.concatenate([rng.lognormal(4, 0.2, 900), rng.lognormal(7, 0.2, 100)])
    mixed = rng.lognormal(5, 0.6, 1000)  # one broad group: no clean split
    a, b = marker_separation(clean), marker_separation(mixed)
    assert a["ashman_d"] > 5 and a["fraction_ambiguous"] < 0.02
    assert a["fraction_positive"] == pytest.approx(0.1, abs=0.01)
    assert b["ashman_d"] < a["ashman_d"]


def test_shape_features():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[5:15, 5:15] = 1  # square: solid, round-ish
    labels[20:22, 5:35] = 2  # thin bar: elongated
    labels[25:35, 25:35] = 3
    labels[25:30, 30:35] = 0  # L shape: less solid
    shape = shape_features(labels).set_index("cell_ID")
    assert shape.loc[1, "area"] == 100 and shape.loc[1, "solidity"] == pytest.approx(1.0)
    assert shape.loc[2, "eccentricity"] > 0.99
    assert shape.loc[3, "solidity"] < 0.9


def test_evaluate_candidates_on_a_pipeline_store(tmp_path):
    from macroseg.baseline import run
    from macroseg.evaluate import evaluate_candidates
    from test_baseline import _config, threshold_model

    rng = np.random.default_rng(0)
    image = rng.integers(0, 100, size=(3, 128, 160), dtype=np.uint16)
    for y, x in [(30, 30), (30, 100), (90, 60)]:
        image[:, y - 8 : y + 8, x - 8 : x + 8] += 1000
    import tifffile

    path = tmp_path / "s.ome.tif"
    names = ["DAPI C1 (DAPI C1)", "Na_K_ATPase (APC C2)", "F4_80 (APC C4)"]
    tifffile.imwrite(
        path,
        image,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": names}, "PhysicalSizeX": 170, "PhysicalSizeXUnit": "nm"},
    )
    sdata = run(_config(path, tmp_path / "out.zarr"), model=threshold_model)
    summary, per_cell = evaluate_candidates(
        sdata, "img", ["membrane", "nuclei_expanded"], nuclei="nuclei", membrane="NaK_ATPase", marker="DAPI"
    )
    assert list(summary.index) == ["membrane", "nuclei_expanded"]
    # Grown nuclei contain exactly their own nucleus: the sanity check for this measure.
    assert summary.loc["nuclei_expanded", "cells_1_nucleus_%"] == 100
    assert summary.loc["nuclei_expanded", "nuclei_in_no_cell_%"] == 0
    assert set(per_cell["membrane"].columns) >= {"cell_ID", "n_nuclei", "border_ratio", "marker_mean", "area"}
    import matplotlib

    matplotlib.use("Agg")
    from macroseg.evaluate import plot_comparison

    fig = plot_comparison(per_cell, marker="DAPI", pixel_size_um=0.17)
    assert len(fig.axes) == 4
