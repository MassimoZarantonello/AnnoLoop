"""Fine-tuning Cellpose on the corrected masks of the annotation rounds.

1. ``training_set``: every complete ROI of the annotation rounds (``annocell.folder.is_complete``: every cell
   checked), read from its store with the channels of the segmentation it started from (e.g. NaK_ATPase + DAPI),
   together with its corrected masks, cut into square tiles. Every tile is used for training: there is no
   validation split (it cost a fifth of the corrected cells for a loss no decision used); models are scored on the
   test regions. Test regions (``macroseg.annotation.export_test``) never enter it, and tiles within ``margin_px``
   of a test region of the same slide are dropped. One annotator per ROI: an ROI complete for two is refused.
2. ``train``: fine-tune a pretrained Cellpose model on the tiles, with the settings of the config's ``training``
   section (the same every round: never changed after looking at a test score), and save it in
   ``<models>/<name>/``: the weights (``<name>``), ``tiles.csv`` (every tile) and ``model.json`` (start model,
   channels, ROIs, settings, losses, versions).
3. ``apply_model``: segment a store with the fine-tuned model, as one more labels element.
4. ``score_model`` / ``score_test_regions``: segment every test region with a model and score it against the
   expert's masks, written to ``<annotations>/scores/<model>.json`` (``annocell.folder.write_score``), where
   overviews (AnnOverview) plot it round after round.

Tiles matter because Cellpose-SAM trains on one random 256 x 256 patch per image and epoch: a single large image
would be seen through one small patch per epoch.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from annocell import folder as af
from skimage.segmentation import relabel_sequential

from macroseg.accuracy import score
from macroseg.annotation import (
    MARGIN_PX,
    SCORE_THRESHOLDS,
    _git_commit,
    corrected_masks,
    load_annotations,
    load_test_patches,
    near_test_patch,
    slide_region,
)
from macroseg.config import TrainingConfig

TILE_PX = 512


@dataclass
class TrainingSet:
    """Tiles of the annotated ROIs: ``tiles`` has one row per tile (round, annotator, image, y0, x0, cells,
    hand_drawn); ``images`` and ``masks`` hold the arrays in the same order, images as (channels, y, x).
    ``near_test``: tiles dropped because they lie near a test region."""

    channels: list[str]
    tile_px: int
    tiles: pd.DataFrame
    images: list[np.ndarray] = field(repr=False)
    masks: list[np.ndarray] = field(repr=False)
    near_test: int = 0


def training_set(
    root: str | Path,
    channels: Sequence[str] | None = None,
    tile_px: int = TILE_PX,
    margin_px: int = MARGIN_PX,
) -> TrainingSet:
    """Tiles of every complete training ROI under the annotations ``root``.

    ``channels``: image channels the model learns from; default the channels of the segmentation each image
    started from (they must be the same for all images).
    """
    from spatialdata import read_zarr

    annotations = load_annotations(root)
    # Only fully reviewed training ROIs; test regions (<annotator>/test) are never read here.
    done = annotations[annotations["complete"]] if not annotations.empty else annotations
    if done.empty:
        raise ValueError(f"No complete ROI under {root}: every cell of an ROI must be checked.")
    twice = done.loc[done["image"].duplicated(keep=False), ["image", "annotator"]]
    if not twice.empty:  # blending several annotators of one ROI is not built (docs/concurrent_annotations.md)
        raise ValueError(f"ROIs complete for more than one annotator: {twice.to_dict('records')}.")
    patches = load_test_patches(root)
    if channels is None:
        options = {tuple(c) for c in done["start_segmentation.channels"]}
        if len(options) != 1:
            raise ValueError(f"Images started from segmentations with different channels {options}; pass channels.")
        channels = list(options.pop())
    channels = list(channels)

    rows, images, masks, near_test = [], [], [], 0
    for record in done.to_dict("records"):
        # Where this image lies in its slide, to find the tiles near a test patch.
        region = slide_region({k: _value(v) for k, v in record.items()})
        sdata = read_zarr(record["zarr"])
        element = record.get("element")
        element = element if isinstance(element, str) else next(iter(sdata.images))
        image = sdata.images[element].sel(c=channels).data.compute()
        truth, hand_drawn = corrected_masks(record["folder"])
        if image.shape[1:] != truth.shape:
            raise ValueError(f"{record['image']}: image {image.shape[1:]} and masks {truth.shape} differ in size.")
        for y0, x0, tile_image, tile_masks, n_hand_drawn in _tiles(image, truth, hand_drawn, tile_px):
            # The tile in slide pixels: the image's offset in the slide plus the tile's offset in the image.
            oy, ox = (region[0], region[2]) if region else (0, 0)
            tile_region = [oy + y0, oy + y0 + tile_masks.shape[0], ox + x0, ox + x0 + tile_masks.shape[1]]
            if near_test_patch(record, tile_region, patches, margin_px):  # never train next to a test patch
                near_test += 1
                continue
            rows.append(
                {
                    "round": record["round"],
                    "annotator": record["annotator"],
                    "image": record["image"],
                    "y0": y0,
                    "x0": x0,
                    "cells": int(tile_masks.max()),
                    "hand_drawn": n_hand_drawn,
                }
            )
            images.append(tile_image)
            masks.append(tile_masks)
    return TrainingSet(channels, tile_px, pd.DataFrame(rows), images, masks, near_test)


def train(
    data: TrainingSet,
    models_root: str | Path,
    name: str,
    settings: TrainingConfig,
    start_model: str = "cpsam_v2",
    device=None,
) -> Path:
    """Fine-tune ``start_model`` on every tile with ``settings`` (the config's ``training`` section) and save it in
    ``<models_root>/<name>/``; returns that folder."""
    from cellpose import models
    from cellpose import train as cellpose_train

    folder = Path(models_root) / name
    if folder.exists():
        raise FileExistsError(f"{folder} exists; choose another name or delete it.")
    model = models.CellposeModel(gpu=True, pretrained_model=start_model, device=device)
    folder.mkdir(parents=True)
    weights, train_losses, _ = cellpose_train.train_seg(
        model.net,
        train_data=data.images,
        train_labels=data.masks,
        channel_axis=0,
        n_epochs=settings.n_epochs,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        batch_size=settings.batch_size,
        save_path=folder,
        model_name=name,
    )
    # Cellpose writes <save_path>/models/<name>; keep the weights next to their records instead.
    shutil.move(str(weights), folder / name)
    shutil.rmtree(folder / "models", ignore_errors=True)

    data.tiles.to_csv(folder / "tiles.csv", index=False)
    record = {
        "name": name,
        "trained_on": date.today().isoformat(),
        "start_model": start_model,
        "channels": data.channels,
        "tile_px": data.tile_px,
        "images": data.tiles[["round", "annotator", "image"]].drop_duplicates().to_dict("records"),
        "tiles": len(data.tiles),
        "tiles_dropped_near_test": data.near_test,
        "cells": int(data.tiles["cells"].sum()),
        "hand_drawn_cells": int(data.tiles["hand_drawn"].sum()),
        "parameters": asdict(settings),
        "train_losses": [round(float(v), 5) for v in np.ravel(train_losses)],
        "macroseg_commit": _git_commit(),
        "cellpose_version": version("cellpose"),
    }
    (folder / "model.json").write_text(json.dumps(record, indent=2) + "\n")
    return folder


def load_model_record(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "model.json").read_text())


def score_model(root: str | Path, folder: str | Path, **kwargs) -> dict:
    """``score_test_regions`` of the fine-tuned model in ``folder``: its weights, its channels, and the latest round
    of the ROIs it was trained on."""
    record = load_model_record(folder)
    round_number = max((int(i["round"]) for i in record.get("images", [])), default=0)
    return score_test_regions(
        root, str(Path(folder) / record["name"]), record["name"], round_number, record["channels"], **kwargs
    )


def score_test_regions(
    root: str | Path,
    model: str,
    name: str,
    round_number: int,
    channels: Sequence[str],
    thresholds: Sequence[float] = SCORE_THRESHOLDS,
    segment=None,
    **model_kwargs,
) -> dict:
    """Segment every test region drawn so far under the annotations ``root`` with Cellpose ``model`` (weights, or a
    pretrained name such as ``"cpsam"`` for the starting model, ``round_number`` 0) on ``channels``, score it
    against the expert's masks (AP, precision, recall at each IoU threshold), write it to
    ``<root>/scores/<name>.json`` and return it.

    ``round_number``: the round whose corrections the model was trained on. ``segment``: a function from an image
    (channels, y, x) to masks, instead of Cellpose (tests); ``model_kwargs`` go to ``CellposeModel``.
    """
    if segment is None:
        from cellpose import models

        net = models.CellposeModel(gpu=True, pretrained_model=model, **model_kwargs)

        def segment(image):
            return net.eval(image, channel_axis=0)[0]

    regions = []
    for roi in af.rois(root):
        if roi.round is not None or not (roi.folder / af.MASKS).exists():  # test regions with cells drawn
            continue
        image, names = af.read_image(roi.folder)
        missing = [c for c in channels if c not in names]
        if missing:
            raise ValueError(f"{roi.folder}: no channel {missing} for model {name}.")
        predicted = np.asarray(segment(image[[names.index(c) for c in channels]]))
        result = score(af.read_masks(roi.folder), predicted, thresholds)
        # NaN (nothing to score) is not JSON: written as null.
        values = {k: (round(float(v), 4) if np.isfinite(v) else None) for k, v in result.items()}
        regions.append({"region": roi.name, "annotator": roi.annotator, **values})
    if not regions:
        raise ValueError(f"No test region with cells drawn under {root}.")
    record = {
        "model": name,
        "round": round_number,
        "scored_on": date.today().isoformat(),
        "channels": list(channels),
        "regions": regions,
        "macroseg_commit": _git_commit(),
    }
    af.write_score(root, record)
    return record


def apply_model(sdata, element: str, folder: str | Path, labels: str | None = None, **model_kwargs):
    """Segment image ``element`` of ``sdata`` with the model in ``folder`` (its own channels), then measure.

    Writes labels element ``labels`` (default: the model name) and its ``<labels>_intensities`` table.
    """
    from macroseg.baseline import measure, segment
    from macroseg.config import SegmentationStep

    record = load_model_record(folder)
    step = SegmentationStep(
        labels=labels or record["name"], channels=tuple(record["channels"]), model=str(Path(folder) / record["name"])
    )
    sdata = segment(sdata, element, step, **model_kwargs)
    return measure(sdata, element, step.labels)


def _tiles(image: np.ndarray, masks: np.ndarray, hand_drawn: np.ndarray, tile_px: int):
    """Non-overlapping ``tile_px`` squares covering the image (the last row and column may be cut short:
    they are dropped if under half a tile)."""
    height, width = masks.shape
    drawn = set(np.asarray(hand_drawn).tolist())
    for y0 in range(0, height, tile_px):
        for x0 in range(0, width, tile_px):
            tile_masks = masks[y0 : y0 + tile_px, x0 : x0 + tile_px]
            if min(tile_masks.shape) < tile_px // 2:
                continue
            ids = np.unique(tile_masks[tile_masks > 0])
            n_hand_drawn = int(sum(1 for i in ids if int(i) in drawn))
            yield (
                y0,
                x0,
                image[:, y0 : y0 + tile_px, x0 : x0 + tile_px],
                relabel_sequential(tile_masks)[0].astype(np.uint32),
                n_hand_drawn,
            )


def _value(v):
    """A record value from a pandas row: NaN (missing in that record) as None."""
    return None if isinstance(v, float) and np.isnan(v) else v
