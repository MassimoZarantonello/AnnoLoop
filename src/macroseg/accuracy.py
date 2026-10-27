"""Measures of a cell segmentation against a hand-corrected ground truth.

Cells are matched one to one between ground truth and segmentation (Cellpose's matching: Hungarian
assignment on the intersection over union, IoU). At each IoU threshold a matched pair counts as a
true positive; unmatched segmented cells are false positives, unmatched true cells false negatives.

- ``average_precision``: TP / (TP + FP + FN), Cellpose's definition, so values compare with the
  Cellpose papers. At IoU 0.5 it mostly says whether cells are found, at 0.75 or 0.9 whether their
  outlines are right.
- ``precision`` (TP / segmented cells) and ``recall`` (TP / true cells).
- ``recall`` on a subset of true cells (e.g. macrophages) with ``truth_ids``.

Ground truth made by correcting one segmentation favours that segmentation: its untouched cells are identical.
``corrected_region`` and ``score_segmentations(..., region=...)`` score only where the annotator changed something.

All functions take plain numpy arrays: 2D integer labels, 0 = background.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential

DEFAULT_THRESHOLDS = (0.5, 0.75, 0.9)


def load_cellpose_masks(path: str | Path) -> np.ndarray:
    """Masks of a Cellpose GUI ``<image>_seg.npy`` file."""
    return np.asarray(np.load(path, allow_pickle=True).item()["masks"])


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


def score(
    truth: np.ndarray,
    pred: np.ndarray,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    truth_ids: Iterable[int] | None = None,
) -> dict[str, float]:
    """Average precision, precision and recall of ``pred`` at each IoU threshold.

    ``truth_ids`` (optional) adds ``subset_recall@<t>``: the recall on those true cells only.
    """
    matches = match_cells(truth, pred)
    n_truth, n_pred = len(matches), len(_ids(pred))
    row: dict[str, float] = {"true_cells": n_truth, "segmented_cells": n_pred}
    subset = None if truth_ids is None else matches["truth_ID"].isin(list(truth_ids))
    if subset is not None:
        row["subset_cells"] = int(subset.sum())
    for t in thresholds:
        tp = int((matches["iou"] >= t).sum())
        fp, fn = n_pred - tp, n_truth - tp
        row[f"AP@{t}"] = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        row[f"precision@{t}"] = tp / n_pred if n_pred else float("nan")
        row[f"recall@{t}"] = tp / n_truth if n_truth else float("nan")
        if subset is not None:
            row[f"subset_recall@{t}"] = (matches.loc[subset, "iou"] >= t).mean() if subset.any() else float("nan")
    return row


def score_segmentations(
    truth: np.ndarray,
    segmentations: Mapping[str, np.ndarray],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    truth_ids: Iterable[int] | None = None,
    region: np.ndarray | None = None,
) -> pd.DataFrame:
    """``score`` of each segmentation against the same ground truth, one row per segmentation.

    ``region`` (optional, boolean): score only the cells, true and segmented, mostly inside it (``in_region``).
    """
    thresholds = tuple(thresholds)
    truth_ids = None if truth_ids is None else list(truth_ids)
    if region is not None:
        truth = in_region(truth, region)
        segmentations = {name: in_region(labels, region) for name, labels in segmentations.items()}
    rows = {name: score(truth, labels, thresholds, truth_ids) for name, labels in segmentations.items()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("segmentation")


def corrected_region(start: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Pixels the annotator changed: the final cells that are not identical to a starting cell (drawn or
    modified) and the starting cells that are not identical to a final cell (removed or modified)."""
    matches = match_cells(final, start)
    unchanged = matches[matches["iou"] == 1]
    changed_final = np.isin(final, matches.loc[matches["iou"] < 1, "truth_ID"])
    changed_start = (start > 0) & ~np.isin(start, unchanged["pred_ID"])
    return changed_final | changed_start


def in_region(labels: np.ndarray, region: np.ndarray, min_fraction: float = 0.5) -> np.ndarray:
    """``labels`` keeping only the cells with more than ``min_fraction`` of their pixels in ``region``."""
    _check_labels(labels, region.astype(np.int8))
    ids, inside = np.unique(labels[region], return_counts=True)
    all_ids, sizes = np.unique(labels, return_counts=True)
    fraction = pd.Series(inside, index=ids).reindex(all_ids, fill_value=0) / sizes
    keep = fraction.index[(fraction > min_fraction) & (fraction.index > 0)]
    return np.where(np.isin(labels, keep), labels, 0)


def worst_matches(truth: np.ndarray, pred: np.ndarray, n: int = 15) -> pd.DataFrame:
    """The ``n`` true cells matched worst by ``pred`` (``iou`` 0 = missed), with their centre ``y``, ``x`` in pixels."""
    from skimage.measure import regionprops_table

    centres = pd.DataFrame(regionprops_table(truth, properties=("label", "centroid")))
    centres = centres.rename(columns={"label": "truth_ID", "centroid-0": "y", "centroid-1": "x"}).round().astype(int)
    return match_cells(truth, pred).merge(centres).sort_values("iou").head(n).reset_index(drop=True)


def plot_ap_curve(table: pd.DataFrame, prefix: str = "AP"):
    """``<prefix>@<t>`` against the IoU threshold ``t``, one line per segmentation (rows of a
    ``score_segmentations`` table). ``prefix`` can also be ``recall``, ``precision`` or ``subset_recall``."""
    import matplotlib.pyplot as plt

    from macroseg.evaluate import _colors

    columns = [c for c in table.columns if c.startswith(f"{prefix}@")]
    thresholds = [float(c.split("@")[1]) for c in columns]
    colors = _colors(list(table.index))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, row in table[columns].iterrows():
        ax.plot(thresholds, row.to_numpy(), color=colors[name] ,linewidth=2, marker="o", markersize=5, label=name)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel(prefix)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def _ids(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    return ids[ids > 0].astype(np.int64)


def _check_labels(labels: np.ndarray, other: np.ndarray) -> None:
    if labels.ndim != 2 or labels.shape != other.shape:
        raise ValueError(f"Expected two aligned 2D arrays, got shapes {labels.shape} and {other.shape}.")
    for array in (labels, other):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Labels must be integers, got {array.dtype}.")
