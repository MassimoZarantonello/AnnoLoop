"""Annotation rounds: export an image for correction, close a round, read the corrections back.

One self-contained folder per image, in the AnnoCell layout (``annocell.folder``), corrected in AnnoCell's napari
tool (a separate package). Each annotator has one folder with their rounds and test images:
``<root>/<annotator>/round_<NN>/<image>/`` and ``<root>/<annotator>/test/<region>/``. An image of a round has one
primary annotator, whose version goes into training, and may have other annotators, whose versions only measure
how annotators differ (``role`` in annotation.json; AnnoLoop.md, O1).

- ``export_round``: a whole image for its primary annotator and its other annotators, one folder each: all
  channels, the proposed masks and read-only comparison masks of other segmentations;
- ``export_test``: a test image, drawn from scratch (no masks at all), never used for training; refused within
  ``margin_px`` of an annotated image of the same slide, and the reverse is enforced by ``annoloop.train``;
- ``close_round``: the round is over (decided by hand): AnnoCell opens its images read-only, and the proposal of
  every complete image is scored against its correction (``score_start``);
- ``load_annotations`` / ``load_test_images``: every ``annotation.json`` under a root, with the review AnnoCell
  wrote (``review.json``); ``summarize_rounds`` sums them per round or per round and annotator.

How much of the proposal was kept, edited, deleted and drawn new, and how many cells were OK'd by hand or in bulk,
is AnnoCell's review (``annocell.folder.read_review``), the one definition of those counts: nothing here recounts
them. Which images go into training is ``annocell.folder.is_complete``, the same function AnnOverview uses.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from annocell import folder as af
from skimage.segmentation import relabel_sequential

from annoloop.accuracy import score
from annoloop.segmentation import Segmentation
from annoloop.version import code_commit

SCORE_THRESHOLDS = (0.5, 0.75)
MARGIN_PX = 60  # about one cell diameter at 0.17 µm / pixel
# An image's ``role`` in annotation.json: its one primary annotator, whose version goes into training, or one of its
# other annotators, whose versions only measure how annotators differ (AnnoLoop.md, O1). No role: primary.
PRIMARY, OTHER = "primary", "other"


def export_round(
    sdata,
    name: str,
    element: str,
    proposal: Segmentation,
    root: str | Path,
    round_number: int,
    annotator: str,
    other_annotators: Sequence[str] = (),
    compare: Sequence[str] = (),
    visible_channels: Sequence[str] | None = None,
    colours: Mapping[str, str] | None = None,
    models_root: str | Path | None = None,
    settings: Mapping | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write the folders of image ``name`` in round ``round_number`` for ``annotator`` (the primary) and for each of
    ``other_annotators``, ``<root>/<annotator>/round_<NN>/<name>/``, and return them by annotator. All get the same
    image and proposal; the record's ``role`` is ``PRIMARY`` or ``OTHER``.

    ``element``: the image element of ``sdata``; ``proposal``: the segmentation whose labels element (already in
    ``sdata``) becomes the proposed masks; its channels are what a model trained on this image learns from.
    ``compare``: other labels elements shown read-only; one that is a model in ``models_root``
    (``<models_root>/<name>/model.json``) is marked as trained on this image if it was. ``visible_channels``: shown
    when the tool opens (default: the proposal's channels). ``colours``: napari colormap per channel. ``settings``:
    the settings the export was made with, recorded as is. An existing folder is left untouched (it may hold
    corrections) unless ``overwrite``, which also discards the corrections.

    Refused, before any folder is written: in a closed round, if the image is already given in this round to
    someone who is neither ``annotator`` nor one of ``other_annotators``, and if it was given in an earlier round:
    every image is annotated in one round (AnnoLoop.md, D17).
    """
    all_annotators = annotator, *other_annotators
    round_dict = {}
    for a in all_annotators:
        role = PRIMARY if a == annotator else OTHER
        folder = af.round_folder(root, a, round_number) / name
        same_round = [r for r in af.images(root) if r.round == round_number]
        if any(af.is_closed(af.read_record(r.folder)) for r in same_round):
            raise ValueError(f"Round {round_number} is closed: export into a new round.")
        others = sorted(r.annotator for r in same_round if r.name == name and r.annotator not in all_annotators)
        if others:
            raise ValueError(f"{name} is already given to {others} in round {round_number}: one annotator per image.")
        before = given_in_other_rounds(root, name, round_number)
        if before:
            raise ValueError(f"{name} was already given in {before}: every image is annotated once.")
        if (folder / af.RECORD).exists() and not overwrite:
            round_dict[a] = folder
            continue
        _clear(folder)
        # Everything the tool shows is copied into the folder: the annotator's computer has no store.
        image, channels = _image(sdata, element)
        af.write_image(folder, image)
        af.write_masks(folder, _masks(sdata, proposal.labels), af.START)
        for other in compare:
            af.write_masks(folder / af.COMPARE, _masks(sdata, other), f"{other}.tif")
        record = {
            "annotator": a,
            "role": role,
            "round": round_number,
            "image": name,
            "mode": "train",
            "format_version": af.FORMAT_VERSION,
            "exported_on": date.today().isoformat(),
            **_provenance(sdata, element),
            "region": _slide_region(sdata, image.shape[1:]),
            "channels": channels,
            "visible_channels": list(visible_channels or proposal.channels),
            "colours": dict(colours or {}),
            "start_segmentation": {
                "labels": proposal.labels,
                "model": proposal.model,
                "channels": list(proposal.channels),
            },
            "compare": [
                {"labels": other, "trained_on_image": _trained_on(models_root, other, name)} for other in compare
            ],
        }
        if settings is not None:
            record["settings"] = dict(settings)
        af.write_record(folder, record)
        round_dict[a] = folder
    return round_dict


def given_in_other_rounds(root: str | Path, name: str, round_number: int) -> str | None:
    """Where image ``name`` was given in a round other than ``round_number`` (e.g. ``round 1 (anna)``), or None.
    An image is annotated in one round only: two corrections of the same pixels cannot both go into training."""
    given = sorted(
        (r.round, r.annotator) for r in af.images(root) if r.name == name and r.round not in (None, round_number)
    )
    return ", ".join(f"round {number} ({annotator})" for number, annotator in given) or None


def export_test(
    sdata,
    image: str,
    element: str,
    region: Sequence[int],
    root: str | Path,
    annotator: str,
    name: str | None = None,
    visible_channels: Sequence[str] | None = None,
    colours: Mapping[str, str] | None = None,
    margin_px: int = MARGIN_PX,
    overwrite: bool = False,
) -> Path:
    """Write test image ``region`` (``[y0, y1, x0, x1]`` in pixels of the store) of image ``element`` of
    ``sdata`` for ``annotator`` (the expert who draws every test image) to ``<root>/<annotator>/test/<name>/``
    and return the folder. It is drawn from scratch: no masks are exported.

    ``image``: the image's name, for the default ``name`` ``<image>_y<y0>_x<x0>`` (slide pixels).
    ``visible_channels`` and ``colours`` as in ``export_round``. Refused if the region lies within ``margin_px`` of
    an annotated image or another test image of the same slide.
    """
    y0, y1, x0, x1 = (int(v) for v in region)
    # Overlap is checked in slide pixels: stores of different crops of the same slide share them.
    slide_region = _slide_region(sdata, (y1 - y0, x1 - x0), (y0, x0))
    name = name or f"{image}_y{slide_region[0]}_x{slide_region[2]}"
    folder = af.test_images_folder(root, annotator) / name
    if (folder / af.RECORD).exists() and not overwrite:
        return folder
    source, level = sdata.attrs.get("source"), sdata.attrs.get("level")
    others = [r for r in _records(root) if Path(r["_folder"]) != folder]  # every other image and test image
    near = [r["image"] for r in others if _near(r, source, level, slide_region, margin_px)]
    if near:
        raise ValueError(f"Test image {slide_region} is within {margin_px} px of annotated {near} on the same slide.")
    _clear(folder)
    pixels, channels = _image(sdata, element)
    af.write_image(folder, pixels[:, y0:y1, x0:x1])  # the image only: no start masks, no comparison masks
    record = {
        "annotator": annotator,
        "image": name,
        "mode": af.TEST,
        "format_version": af.FORMAT_VERSION,
        "exported_on": date.today().isoformat(),
        **_provenance(sdata, element),
        "region": slide_region,
        "channels": channels,
        "visible_channels": list(visible_channels or channels[:3]),
        "colours": dict(colours or {}),
    }
    af.write_record(folder, record)
    return folder


def near_test_image(record: dict, region: Sequence[int], tests: Sequence[dict], margin_px: int = MARGIN_PX) -> bool:
    """Whether ``region`` (slide pixels) of the image of ``record`` lies within ``margin_px`` of a test image
    of the same slide."""
    return any(_near(t, record.get("source"), record.get("level"), region, margin_px) for t in tests)


def close_round(root: str | Path, round_number: int) -> list[Path]:
    """Close round ``round_number`` for every annotator (``closed_on`` in each image): AnnoCell then opens its
    images read-only, and no image can be exported into it any more. Unfinished images are closed too and stay
    out of training (``annocell.folder.is_complete``). The proposal of every complete image is then scored against
    its correction (``score_start``). Returns the closed folders."""
    folders = [r.folder for r in af.images(root) if r.round == round_number]
    if not folders:
        raise ValueError(f"No image in round {round_number} under {root}.")
    for folder in folders:
        af.close(folder)
        if af.is_complete(af.read_review(folder)):
            score_start(folder)
    return folders


def score_start(folder: str | Path) -> dict:
    """Score the proposal of a training image against its corrected masks and record it as ``start_score`` in
    ``annotation.json`` (only that file is written, so AnnoCell's files are never touched). Returns the score.
    Meaningful once the image is complete; ``close_round`` does it for every complete image."""
    folder = Path(folder)
    record = af.read_record(folder)
    result = score(af.read_masks(folder), af.read_start(folder), thresholds=SCORE_THRESHOLDS)
    record["start_score"] = {k: round(v, 4) for k, v in result.items() if "@" in k}
    af.write_record(folder, record)
    return record["start_score"]


def corrected_masks(folder: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """The corrected masks in ``folder`` (``masks.tif``, else the proposal) and the IDs of the cells the annotator
    drew or edited (``cells.csv``)."""
    cells = af.read_cells(folder)
    by_hand = sorted(i for i, c in cells.items() if c.origin in (af.DRAWN, af.EDITED))
    return af.read_masks(folder), np.array(by_hand, dtype=np.int64)


def load_annotations(root: str | Path) -> pd.DataFrame:
    """One row per training image of the rounds under ``root``, nested fields flattened (e.g. ``review.drawn``):
    its ``annotation.json``, AnnoCell's ``review`` (``annocell.folder.read_review``; empty if never saved),
    ``folder``, ``annotator`` and ``round`` from the tree, ``complete`` (``annocell.folder.is_complete``) and
    ``closed``."""
    records = [
        {
            **(record := af.read_record(r.folder)),
            "review": (review := af.read_review(r.folder)),
            "folder": str(r.folder),
            "annotator": r.annotator,
            "round": r.round,
            "complete": af.is_complete(review),
            "closed": af.is_closed(record),
        }
        for r in af.images(root)
        if r.round is not None
    ]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records).sort_values(["round", "annotator", "image"]).reset_index(drop=True)


def load_test_images(root: str | Path) -> list[dict]:
    """The ``annotation.json`` of every test image under ``root``."""
    return [af.read_record(r.folder) for r in af.images(root) if r.round is None]


def summarize_rounds(annotations: pd.DataFrame, by: Sequence[str] = ("round",)) -> pd.DataFrame:
    """Per round (or per ``by``, e.g. ``("round", "annotator")``): annotators, images (all, complete; an image with
    other annotators counts once per annotator, and only its primary version goes into training), the corrections
    in the images AnnoCell saved (its review: cells kept as proposed, edited, deleted, drawn new, OK'd by hand and
    in bulk), and the mean score of the proposal on the complete images (from
    ``score_start``, written when the round is closed)."""
    if annotations.empty:
        return pd.DataFrame()
    by = list(by)
    summary = annotations.groupby(by).agg(
        annotators=("annotator", "nunique"),
        images=("image", "size"),
        complete=("complete", "sum"),
    )
    saved = annotations[annotations["review.saved_at"].notna()] if "review.saved_at" in annotations else []
    if len(saved):
        counts = {"kept": af.PROPOSED, "edited": af.EDITED, "deleted": "deleted", "drawn": af.DRAWN}
        counts |= {"ok_hand": "ok_hand", "ok_bulk": "ok_bulk"}
        present = {k: f"review.{c}" for k, c in counts.items() if f"review.{c}" in saved.columns}
        summary = summary.join(saved.groupby(by).agg(**{f"cells_{k}": (c, "sum") for k, c in present.items()}))
    score_columns = [c for c in annotations.columns if c.startswith("start_score.AP@")]
    if score_columns:
        scores = {f"mean_start_{c.removeprefix('start_score.')}": (c, "mean") for c in score_columns}
        summary = summary.join(annotations.groupby(by).agg(**scores))
    return summary


def _image(sdata, element: str) -> tuple[np.ndarray, list[str]]:
    """Every channel of image ``element``, in memory, and the channel names."""
    image = sdata.images[element]
    return image.data.compute(), [str(c) for c in image.coords["c"].values]


def _masks(sdata, labels: str) -> np.ndarray:
    """Labels element ``labels`` in memory, cells renumbered 1, 2, ... (uint32, as the tool writes them)."""
    return relabel_sequential(sdata.labels[labels].data.compute())[0].astype(np.uint32)


def _provenance(sdata, element: str) -> dict:
    """Where an exported image comes from: slide, pyramid level, crop, store, code version."""
    return {
        "source": sdata.attrs.get("source"),
        "level": sdata.attrs.get("level"),
        "crop": sdata.attrs.get("crop"),
        "pixel_size_um": sdata.attrs.get("pixel_size_um"),
        "zarr": None if sdata.path is None else str(sdata.path),
        "element": element,
        "code_commit": code_commit(),
    }


def _slide_region(sdata, shape: Sequence[int], offset: Sequence[int] = (0, 0)) -> list[int]:
    """``[y0, y1, x0, x1]`` in slide pixels of a ``shape`` region at ``offset`` in the store."""
    crop = sdata.attrs.get("crop")  # where the store itself starts in the slide, [y0, y1, x0, x1]
    y0, x0 = (crop[0], crop[2]) if crop else (0, 0)
    y0, x0 = int(y0 + offset[0]), int(x0 + offset[1])
    return [y0, y0 + int(shape[0]), x0, x0 + int(shape[1])]


def _near(record: dict, source, level, region: Sequence[int], margin_px: int) -> bool:
    """Whether the image of ``record`` comes within ``margin_px`` of ``region`` on the same slide and level."""
    if record.get("source") != source or record.get("level") != level:
        return False
    y0, y1, x0, x1 = region
    other = record["region"]
    # Two rectangles are near if they overlap once ``region`` is grown by the margin on every side.
    return (
        y0 - margin_px < other[1]
        and other[0] < y1 + margin_px
        and x0 - margin_px < other[3]
        and other[2] < x1 + margin_px
    )


def _records(root: str | Path) -> list[dict]:
    """Every ``annotation.json`` under ``root`` (rounds and test images), with its folder in ``_folder``."""
    return [{**af.read_record(r.folder), "_folder": str(r.folder)} for r in af.images(root)]


def _trained_on(models_root: str | Path | None, name: str, image: str) -> bool | None:
    """Whether model ``name`` in ``models_root`` was trained on ``image``; None if ``name`` is not a model there."""
    record = None if models_root is None else Path(models_root) / name / "model.json"
    if record is None or not record.exists():
        return None
    return any(i["image"] == image for i in json.loads(record.read_text()).get("images", []))


def _clear(folder: Path) -> None:
    """Remove what an earlier export and the annotator wrote in ``folder``."""
    for file in (af.MASKS, af.CELLS, af.REVIEW, af.START, af.IMAGE):
        (folder / file).unlink(missing_ok=True)
    shutil.rmtree(folder / af.COMPARE, ignore_errors=True)
