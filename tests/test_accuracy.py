"""Ground-truth measures on small hand-made masks with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from macroseg.accuracy import load_cellpose_masks, match_cells, score, score_segmentations, worst_matches


def _truth() -> np.ndarray:
    truth = np.zeros((20, 40), dtype=np.int32)
    truth[0:10, 0:10] = 5  # IDs need not be consecutive
    truth[0:10, 10:20] = 7
    truth[0:10, 25:35] = 9
    return truth


def test_identical_segmentation_scores_one():
    truth = _truth()
    result = score(truth, truth * 3)  # same cells, other IDs
    assert result["AP@0.5"] == result["AP@0.9"] == pytest.approx(1.0)
    assert result["true_cells"] == result["segmented_cells"] == 3


def test_merged_missed_and_false_cells():
    truth = _truth()
    pred = np.zeros_like(truth)
    pred[0:10, 0:20] = 1  # cells 5 and 7 merged: IoU 0.5 with either, at most one can match
    pred[15:20, 0:5] = 2  # false cell; cell 9 missed
    matches = match_cells(truth, pred).set_index("truth_ID")
    assert matches.loc[9, "pred_ID"] == 0 and matches.loc[9, "iou"] == 0
    result = score(truth, pred, thresholds=[0.5, 0.75])
    # IoU 0.5: TP 1 (merged object with one cell), FP 1 (false cell), FN 2 (other cell, cell 9).
    assert result["AP@0.5"] == pytest.approx(1 / 4)
    assert result["precision@0.5"] == pytest.approx(1 / 2)
    assert result["recall@0.5"] == pytest.approx(1 / 3)
    assert result["AP@0.75"] == 0  # the merged object is not precise enough


def test_outline_precision_depends_on_threshold():
    truth = _truth()
    pred = truth.copy()
    pred[0:2, 25:35] = 0  # cell 9 loses 20 % of its pixels: IoU 0.8
    result = score(truth, pred, thresholds=[0.75, 0.9])
    assert result["recall@0.75"] == pytest.approx(1.0)
    assert result["recall@0.9"] == pytest.approx(2 / 3)


def test_subset_recall_counts_only_the_given_true_cells():
    truth = _truth()
    pred = truth.copy()
    pred[truth == 9] = 0
    result = score(truth, pred, thresholds=[0.5], truth_ids=[7, 9])
    assert result["subset_cells"] == 2
    assert result["subset_recall@0.5"] == pytest.approx(0.5)
    assert result["recall@0.5"] == pytest.approx(2 / 3)


def test_missed_cell_has_no_partner_even_when_other_cells_are_left():
    truth = _truth()
    pred = np.zeros_like(truth)
    pred[0:10, 0:10] = 1  # cell 5
    pred[15:20, 30:40] = 2  # false cell, overlaps nothing: must not be paired with cell 7 or 9
    matches = match_cells(truth, pred).set_index("truth_ID")
    assert matches.loc[5, "pred_ID"] == 1
    assert (matches.loc[[7, 9], "pred_ID"] == 0).all()


def test_empty_segmentation_and_table():
    truth = _truth()
    result = score(truth, np.zeros_like(truth), thresholds=[0.5])
    assert result["recall@0.5"] == 0 and np.isnan(result["precision@0.5"])
    table = score_segmentations(truth, {"same": truth, "empty": np.zeros_like(truth)}, thresholds=[0.5])
    assert list(table.index) == ["same", "empty"]
    assert table.loc["same", "AP@0.5"] == pytest.approx(1.0)


def test_rejects_misaligned_or_float_labels():
    truth = _truth()
    with pytest.raises(ValueError, match="aligned"):
        score(truth, truth[:10])
    with pytest.raises(ValueError, match="integers"):
        score(truth, truth.astype(float))


def test_load_cellpose_masks(tmp_path):
    truth = _truth()
    np.save(tmp_path / "img_seg.npy", {"masks": truth, "outlines": truth})
    np.testing.assert_array_equal(load_cellpose_masks(tmp_path / "img_seg.npy"), truth)


def test_worst_matches_lists_missed_cells_first_with_centres():
    truth = _truth()
    pred = truth.copy()
    pred[truth == 9] = 0  # missed
    pred[0:2, 0:10] = 0  # cell 5: IoU 0.8
    worst = worst_matches(truth, pred, n=2)
    assert list(worst["truth_ID"]) == [9, 5]
    assert worst.loc[0, "iou"] == 0 and (worst.loc[0, "y"], worst.loc[0, "x"]) == (4, 30)
