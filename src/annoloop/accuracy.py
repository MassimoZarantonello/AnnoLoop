"""Measures of a cell segmentation against hand-corrected masks.

Cells are matched one to one between the corrected masks and the segmentation (Cellpose's matching: Hungarian
assignment on the intersection over union, IoU). At each IoU threshold a matched pair counts as a true positive;
unmatched segmented cells are false positives, unmatched true cells false negatives.

- ``AP``: TP / (TP + FP + FN), Cellpose's definition, so values compare with the Cellpose papers. At IoU 0.5 it
  mostly says whether cells are found, at 0.75 or 0.9 whether their outlines are right.
- ``precision`` (TP / segmented cells) and ``recall`` (TP / true cells).

All functions take plain numpy arrays: 2D integer labels, 0 = background.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential

DEFAULT_THRESHOLDS = (0.5, 0.75, 0.9)


def match_cells(truth: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Best one-to-one match of every true cell to a segmented cell.

    Returns one row per true cell: ``truth_ID``, ``pred_ID`` (0 when unmatched) and ``iou``.
    """
    from cellpose.metrics import mask_ious

    _check_labels(truth, pred)
    truth_seq, _, truth_ids = relabel_sequential(truth)
    pred_seq, _, pred_ids = relabel_sequential(pred)
    if truth_seq.max() == 0:
        return pd.DataFrame({"truth_ID": [], "pred_ID": [], "iou": []})
    if pred_seq.max() == 0:
        ious, preds = np.zeros(truth_seq.max()), np.zeros(truth_seq.max(), dtype=np.int64)
    else:
        ious, preds = mask_ious(truth_seq, pred_seq)
        preds = np.where(ious > 0, preds, 0)  # the assignment also pairs cells that do not overlap
    return pd.DataFrame(
        {
            "truth_ID": np.asarray(truth_ids)[1:].astype(np.int64),
            "pred_ID": np.asarray(pred_ids)[preds].astype(np.int64),  # pred_ids[0] == 0: unmatched
            "iou": ious,
        }
    )


def score(truth: np.ndarray, pred: np.ndarray, thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> dict[str, float]:
    """Average precision, precision and recall of ``pred`` at each IoU threshold, and the numbers of cells."""
    matches = match_cells(truth, pred)
    n_truth, n_pred = len(matches), len(_ids(pred))
    row: dict[str, float] = {"true_cells": n_truth, "segmented_cells": n_pred}
    for t in thresholds:
        tp = int((matches["iou"] >= t).sum())
        fp, fn = n_pred - tp, n_truth - tp
        row[f"AP@{t}"] = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        row[f"precision@{t}"] = tp / n_pred if n_pred else float("nan")
        row[f"recall@{t}"] = tp / n_truth if n_truth else float("nan")
    return row


def _ids(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    return ids[ids > 0].astype(np.int64)


def _check_labels(labels: np.ndarray, other: np.ndarray) -> None:
    if labels.ndim != 2 or labels.shape != other.shape:
        raise ValueError(f"Expected two aligned 2D arrays, got shapes {labels.shape} and {other.shape}.")
    for array in (labels, other):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Labels must be integers, got {array.dtype}.")
