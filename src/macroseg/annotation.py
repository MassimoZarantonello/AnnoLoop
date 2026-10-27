"""Annotation rounds: export an ROI for correction, close a round, record what the annotator changed.

One self-contained folder per ROI, in the AnnoCell layout (``annocell.folder``), corrected in AnnoCell's napari
tool (``annocell <folder>``, a separate package). Each annotator has one folder with their rounds and test regions:
``<root>/<annotator>/round_<NN>/<roi>/`` and ``<root>/<annotator>/test/<roi>/``; one annotator per ROI.

- ``export_round``: a training ROI for one annotator: all channels, the proposed masks of one segmentation and
  read-only comparison masks of others;
- ``export_test``: a test region, drawn from scratch (no masks at all), never used for training; refused within
  ``margin_px`` of an annotated region of the same slide, and the reverse is enforced by ``macroseg.train``;
- ``close_round``: the round is over (decided by hand): AnnoCell opens its ROIs read-only;
- ``finish``: compare the corrected masks with the proposed ones and record it in ``annotation.json``;
- ``load_annotations`` / ``load_test_patches``: every ``annotation.json`` under a root, one row per ROI.

Which ROIs go into training is ``annocell.folder.is_complete`` (every cell checked), the same function overviews
(AnnOverview) use.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from annocell import folder as af
from skimage.segmentation import relabel_sequential

from macroseg.accuracy import match_cells, score

SCORE_THRESHOLDS = (0.5, 0.75)
TEST = "test"
MARGIN_PX = 60  # about one cell diameter at 0.17 µm / pixel


def export_round(
    sdata,
    cfg,
    labels: str,
    root: str | Path,
    round_number: int,
    annotator: str,
    compare: Sequence[str] = (),
    visible_channels: Sequence[str] | None = None,
    colours: Mapping[str, str] | None = None,
    models_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the folder of training ROI ``cfg.name`` for ``annotator`` in round ``round_number`` and return it:
    ``<root>/<annotator>/round_<NN>/<cfg.name>/``.

    ``labels``: the proposed segmentation (a labels element of ``sdata``); ``compare``: other labels elements
    shown read-only; one that is a model in ``models_root`` (``<models_root>/<name>/model.json``) is marked as
    trained on this image if it was. ``visible_channels``: shown when the tool opens (default: the channels of
    the proposed segmentation). ``colours``: napari colormap per channel (e.g. ``{"F480": "red", "NaK_ATPase":
    "green", "DAPI": "blue"}``); channels without one get red, green, blue in turn. An existing folder is left
    untouched (it may hold corrections) unless ``overwrite``, which also discards the corrections.

    Refused in a closed round, and if the ROI is already given to another annotator in this round (one annotator
    per ROI: blending several is not built).
    """
    folder = af.round_folder(root, annotator, round_number) / cfg.name
    same_round = [r for r in af.rois(root) if r.round == round_number]
    if any(af.is_closed(af.read_record(r.folder)) for r in same_round):
        raise ValueError(f"Round {round_number} is closed: export into a new round.")
    others = sorted(r.annotator for r in same_round if r.name == cfg.name and r.annotator != annotator)
    if others:
        raise ValueError(f"{cfg.name} is already given to {others} in round {round_number}: one annotator per ROI.")
    if (folder / af.RECORD).exists() and not overwrite:
        return folder
    _clear(folder)
    # Everything the tool shows is copied into the folder: the annotator's laptop has no store.
    image, channels = _image(sdata, cfg.image.element)
    # The config entry of the proposed segmentation: its channels are what a model trained on this image learns from.
    step = next((s for s in cfg.segmentation or () if s.labels == labels), None)
    start_channels = None if step is None or step.expand else list(step.channels)
    af.write_image(folder, image)
    af.write_masks(folder, _masks(sdata, labels), af.START)
    for name in compare:
        af.write_masks(folder / af.COMPARE, _masks(sdata, name), f"{name}.tif")
    record = {
        "annotator": annotator,
        "round": round_number,
        "image": cfg.name,
        "role": "train",
        "mode": "train",
        "format_version": af.FORMAT_VERSION,
        "status": "exported",
        "exported_on": date.today().isoformat(),
        **_provenance(sdata, cfg),
        "region": _slide_region(sdata, image.shape[1:]),
        "channels": channels,
        "visible_channels": list(visible_channels or start_channels or channels[:3]),
        "colours": dict(colours or {}),
        "start_segmentation": {
            "labels": labels,
            "model": None if step is None or step.expand else step.model,
            "channels": start_channels,
            "expand": None if step is None else step.expand,
        },
        "compare": [{"labels": name, "trained_on_image": _trained_on(models_root, name, cfg.name)} for name in compare],
    }
    af.write_record(folder, record)
    return folder


def export_test(
    sdata,
    cfg,
    region: Sequence[int],
    root: str | Path,
    annotator: str,
    name: str | None = None,
    visible_channels: Sequence[str] | None = None,
    colours: Mapping[str, str] | None = None,
    margin_px: int = MARGIN_PX,
    overwrite: bool = False,
) -> Path:
    """Write test region ``region`` (``[y0, y1, x0, x1]`` in pixels of the store) of ``sdata`` for ``annotator``
    (the expert who draws every test region) to ``<root>/<annotator>/test/<name>/`` and return the folder. It is
    drawn from scratch: no masks are exported.

    ``visible_channels`` and ``colours`` as in ``export_round``.
    Refused if the region lies within ``margin_px`` of an annotated ROI or another test region of the same slide.
    ``name`` defaults to ``<cfg.name>_y<y0>_x<x0>`` in slide pixels.
    """
    y0, y1, x0, x1 = (int(v) for v in region)
    # Overlap is checked in slide pixels: stores of different crops of the same slide share them.
    slide_region = _slide_region(sdata, (y1 - y0, x1 - x0), (y0, x0))
    name = name or f"{cfg.name}_y{slide_region[0]}_x{slide_region[2]}"
    folder = af.test_regions_folder(root, annotator) / name
    if (folder / af.RECORD).exists() and not overwrite:
        return folder
    source, level = sdata.attrs.get("source"), sdata.attrs.get("level")
    others = [r for r in _records(root) if Path(r["_folder"]) != folder]  # every other image and test patch
    near = [r["image"] for r in others if _near(r, source, level, slide_region, margin_px)]
    if near:
        raise ValueError(f"Test region {slide_region} is within {margin_px} px of annotated {near} on the same slide.")
    _clear(folder)
    image, channels = _image(sdata, cfg.image.element)
    af.write_image(folder, image[:, y0:y1, x0:x1])  # the image only: no start masks, no comparison masks
    record = {
        "annotator": annotator,
        "image": name,
        "role": TEST,
        "mode": TEST,
        "format_version": af.FORMAT_VERSION,
        "status": "exported",
        "exported_on": date.today().isoformat(),
        **_provenance(sdata, cfg),
        "region": slide_region,
        "channels": channels,
        "visible_channels": list(visible_channels or channels[:3]),
        "colours": dict(colours or {}),
    }
    af.write_record(folder, record)
    return folder


def marker_windows(sdata, labels: str, marker: str, threshold: float, window_px: int = 512) -> pd.DataFrame:
    """Windows of ``window_px`` on a grid of half a window, with the number of cells of ``labels`` and of cells
    whose mean ``marker`` exceeds ``threshold`` (from the ``<labels>_intensities`` table), most such cells first.
    To place test patches where a population (e.g. F4/80-high macrophages) is dense."""
    table = sdata.tables[f"{labels}_intensities"]
    centres = np.asarray(table.obsm["spatial"])  # (y, x)
    high = np.asarray(table[:, marker].X).ravel() > threshold
    height, width = sdata.labels[labels].shape[-2:]
    # A cell belongs to a window if its centre is inside; windows overlap by half, so a dense spot is not split.
    rows = []
    for y0 in range(0, height - window_px + 1, window_px // 2):
        for x0 in range(0, width - window_px + 1, window_px // 2):
            inside = (
                (centres[:, 0] >= y0)
                & (centres[:, 0] < y0 + window_px)
                & (centres[:, 1] >= x0)
                & (centres[:, 1] < x0 + window_px)
            )
            rows.append({"y0": y0, "x0": x0, "cells": int(inside.sum()), "high": int((inside & high).sum())})
    return pd.DataFrame(rows).sort_values("high", ascending=False).reset_index(drop=True)


def near_test_patch(record: dict, region: Sequence[int], patches: Sequence[dict], margin_px: int = MARGIN_PX) -> bool:
    """Whether ``region`` (slide pixels) of the image of ``record`` lies within ``margin_px`` of a test patch
    of the same slide."""
    return any(_near(p, record.get("source"), record.get("level"), region, margin_px) for p in patches)


def slide_region(record: dict) -> list[int] | None:
    """``[y0, y1, x0, x1]`` of an exported image in slide pixels (``region``, or ``crop`` in older records);
    None for a whole slide."""
    region = record.get("region") or record.get("crop")
    return None if region is None else [int(v) for v in region]


def close_round(root: str | Path, round_number: int) -> list[Path]:
    """Close round ``round_number`` for every annotator (``closed_on`` in each ROI): AnnoCell then opens its ROIs
    read-only, and no ROI can be exported into it any more. Unfinished ROIs are closed too and stay out of training
    (``annocell.folder.is_complete``). Returns the closed folders."""
    folders = [r.folder for r in af.rois(root) if r.round == round_number]
    if not folders:
        raise ValueError(f"No ROI in round {round_number} under {root}.")
    for folder in folders:
        af.close(folder)
    return folders


def finish(folder: str | Path) -> dict:
    """Compare the corrected masks with the proposed ones and record it in ``annotation.json`` (counts and the
    score of the proposal); ``status``: ``complete`` (``annocell.folder.is_complete``) or ``partial``. Can be run
    again after further corrections; it overwrites the previous result. Training does not need it."""
    folder = Path(folder)
    record = af.read_record(folder)
    # A test region starts empty: before the tool's first save there is nothing to record.
    if af.is_test(record) and not (folder / af.MASKS).exists():
        raise FileNotFoundError(f"{folder}: no cell drawn yet ({af.MASKS} is missing).")
    final, hand_drawn = corrected_masks(folder)
    review = af.summary(af.read_cells(folder))
    record["review"] = {**record.get("review", {}), **review}
    record["status"] = "complete" if af.is_complete(record) else "partial"
    if af.is_test(record):  # drawn from scratch: nothing to compare with
        record["counts"] = {"cells_final": review["cells"], "macrophages": review["macrophages"]}
    else:
        start = af.read_start(folder)
        record["counts"] = change_counts(start, final, hand_drawn)
        result = score(final, start, thresholds=SCORE_THRESHOLDS)
        record["start_score"] = {k: round(v, 4) for k, v in result.items() if "@" in k}
    af.write_record(folder, record)
    return record


def corrected_masks(folder: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """The corrected masks in ``folder`` (``masks.tif``, else the proposal) and the IDs of the cells the annotator
    drew or edited (``cells.csv``)."""
    cells = af.read_cells(folder)
    by_hand = sorted(i for i, c in cells.items() if c.origin in (af.DRAWN, af.EDITED))
    return af.read_masks(folder), np.array(by_hand, dtype=np.int64)


def change_counts(start: np.ndarray, final: np.ndarray, hand_drawn: Sequence[int] | None = None) -> dict:
    """How the masks changed: cells before and after, unchanged (identical pixels), removed, drawn by hand.

    Without ``hand_drawn`` (the IDs of the drawn and edited cells), every final cell that is not unchanged counts
    as drawn.
    """
    matches = match_cells(final, start)
    n_start, n_final = int(len(np.unique(start[start > 0]))), len(matches)
    unchanged = int((matches["iou"] == 1).sum())
    drawn = len(hand_drawn) if hand_drawn is not None else n_final - unchanged
    return {
        "cells_start": n_start,
        "cells_final": n_final,
        "unchanged": unchanged,
        "removed": n_start - unchanged,
        "drawn": drawn,
        "changed_%": round(100 * (1 - unchanged / n_start), 2) if n_start else 0.0,
    }


def load_annotations(root: str | Path) -> pd.DataFrame:
    """One row per training ROI of the rounds under ``root``, nested fields flattened (e.g. ``counts.drawn``), with
    ``folder``, ``annotator`` and ``round`` from the tree, ``complete`` (``annocell.folder.is_complete``) and
    ``closed``."""
    records = [
        {
            **(record := af.read_record(r.folder)),
            "folder": str(r.folder),
            "annotator": r.annotator,
            "round": r.round,
            "complete": af.is_complete(record),
            "closed": af.is_closed(record),
        }
        for r in af.rois(root)
        if r.round is not None
    ]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records).sort_values(["round", "annotator", "image"]).reset_index(drop=True)


def load_test_patches(root: str | Path) -> list[dict]:
    """The ``annotation.json`` of every test region under ``root``."""
    return [af.read_record(r.folder) for r in af.rois(root) if r.round is None]


def summarize_rounds(annotations: pd.DataFrame) -> pd.DataFrame:
    """Per round: annotators, ROIs (all, complete: going into training), cells changed (from ``finish``), and the
    mean score of the starting masks."""
    if annotations.empty:
        return pd.DataFrame()
    done = annotations[annotations["status"] != "exported"]
    per_round = annotations.groupby("round").agg(
        annotators=("annotator", "nunique"),
        rois=("image", "size"),
        complete=("complete", "sum"),
    )
    if done.empty:
        return per_round
    score_columns = [c for c in done.columns if c.startswith("start_score.AP@")]
    finished = done.groupby("round").agg(
        cells_drawn=("counts.drawn", "sum"),
        cells_removed=("counts.removed", "sum"),
        mean_changed_pct=("counts.changed_%", "mean"),
        **{f"mean_start_{c.removeprefix('start_score.')}": (c, "mean") for c in score_columns},
    )
    return per_round.join(finished)


def _image(sdata, element: str) -> tuple[np.ndarray, list[str]]:
    """Every channel of image ``element``, in memory, and the channel names."""
    image = sdata.images[element]
    return image.data.compute(), [str(c) for c in image.coords["c"].values]


def _masks(sdata, labels: str) -> np.ndarray:
    """Labels element ``labels`` in memory, cells renumbered 1, 2, ... (uint32, as the tool writes them)."""
    return relabel_sequential(sdata.labels[labels].data.compute())[0].astype(np.uint32)


def _provenance(sdata, cfg) -> dict:
    """Where an exported image comes from: slide, pyramid level, crop, store, code version."""
    return {
        "source": sdata.attrs.get("source"),
        "level": sdata.attrs.get("level"),
        "crop": sdata.attrs.get("crop"),
        "pixel_size_um": sdata.attrs.get("pixel_size_um"),
        "zarr": cfg.zarr,
        "element": cfg.image.element,
        "macroseg_commit": _git_commit(),
    }


def _slide_region(sdata, shape: Sequence[int], offset: Sequence[int] = (0, 0)) -> list[int]:
    """``[y0, y1, x0, x1]`` in slide pixels of a ``shape`` region at ``offset`` in the store."""
    crop = sdata.attrs.get("crop")  # where the store itself starts in the slide, [y0, y1, x0, x1]
    y0, x0 = (crop[0], crop[2]) if crop else (0, 0)
    y0, x0 = int(y0 + offset[0]), int(x0 + offset[1])
    return [y0, y0 + int(shape[0]), x0, x0 + int(shape[1])]


def _near(record: dict, source, level, region: Sequence[int], margin_px: int) -> bool:
    """Whether the image of ``record`` comes within ``margin_px`` of ``region`` on the same slide and level.
    A record without a region (a whole slide) is near everything on its slide."""
    if record.get("source") != source or record.get("level") != level:
        return False
    other = slide_region(record)
    if other is None:
        return True
    y0, y1, x0, x1 = region
    # Two rectangles are near if they overlap once ``region`` is grown by the margin on every side.
    return (
        y0 - margin_px < other[1]
        and other[0] < y1 + margin_px
        and x0 - margin_px < other[3]
        and other[2] < x1 + margin_px
    )


def _records(root: str | Path) -> list[dict]:
    """Every ``annotation.json`` under ``root`` (rounds and test regions), with its folder in ``_folder``."""
    return [{**af.read_record(r.folder), "_folder": str(r.folder)} for r in af.rois(root)]


def _trained_on(models_root: str | Path | None, name: str, image: str) -> bool | None:
    """Whether model ``name`` in ``models_root`` was trained on ``image``; None if ``name`` is not a model there."""
    record = None if models_root is None else Path(models_root) / name / "model.json"
    if record is None or not record.exists():
        return None
    return any(i["image"] == image for i in json.loads(record.read_text()).get("images", []))


def _clear(folder: Path) -> None:
    """Remove what an earlier export and the annotator wrote in ``folder``."""
    for name in (af.MASKS, af.CELLS, af.START, af.IMAGE):
        (folder / name).unlink(missing_ok=True)
    shutil.rmtree(folder / af.COMPARE, ignore_errors=True)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
