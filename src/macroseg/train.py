"""Fine-tuning Cellpose on the corrected masks of the annotation rounds.

1. ``training_set``: every ``complete`` image of the annotation rounds, read from its store with the channels of
   the segmentation it started from (e.g. NaK_ATPase + DAPI), together with its corrected masks, cut into square
   tiles; about ``test_fraction`` of the tiles are test tiles (the test loss), with hand-corrected cells on both
   sides. Test patches (``macroseg.annotation.export_test``) never enter it, and tiles within ``margin_px`` of a
   test patch of the same slide are dropped.
2. ``train``: fine-tune a pretrained Cellpose model on the train tiles (test tiles give the test loss) and save it in
   ``<models>/<name>/``: the weights (``<name>``), ``tiles.csv`` (every tile and its split) and ``model.json``
   (start model, channels, images, parameters, losses, versions).
3. ``apply_model``: segment a store with the fine-tuned model, as one more labels element.

Tiles matter because Cellpose-SAM trains on one random 256 x 256 patch per image and epoch: a single large image
would be seen through one small patch per epoch.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential

from macroseg.annotation import (
    MARGIN_PX,
    _git_commit,
    corrected_masks,
    image_folder,
    load_annotations,
    load_test_patches,
    near_test_patch,
    slide_region,
)

TILE_PX = 512
TEST_FRACTION = 0.2


@dataclass
class TrainingSet:
    """Tiles of the annotated images: ``tiles`` has one row per tile (round, image, y0, x0, cells,
    hand_drawn, split); ``images`` and ``masks`` hold the arrays in the same order, images as (channels, y, x).
    ``near_test``: tiles dropped because they lie near a test patch."""

    channels: list[str]
    tile_px: int
    tiles: pd.DataFrame
    images: list[np.ndarray] = field(repr=False)
    masks: list[np.ndarray] = field(repr=False)
    near_test: int = 0

    def split(self, name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        index = np.flatnonzero(self.tiles["split"].to_numpy() == name)
        return [self.images[i] for i in index], [self.masks[i] for i in index]


def training_set(
    root: str | Path,
    channels: Sequence[str] | None = None,
    tile_px: int = TILE_PX,
    test_fraction: float = TEST_FRACTION,
    seed: int = 0,
    margin_px: int = MARGIN_PX,
) -> TrainingSet:
    """Tiles of every ``complete`` training image under the annotations ``root``, split into train and test.

    ``channels``: image channels the model learns from; default the channels of the segmentation each image
    started from (they must be the same for all images).
    """
    from spatialdata import read_zarr

    annotations = load_annotations(root)
    if annotations.empty or not (annotations["status"] == "complete").any():
        raise ValueError(f"No complete annotation under {root}: run finish() after annotating.")
    # Only fully reviewed training images; test patches live in <root>/test and are never read here.
    done = annotations[(annotations["status"] == "complete") & (annotations["role"] != "test")]
    if done.empty:
        raise ValueError(f"No complete training image under {root}.")
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
        truth, hand_drawn = corrected_masks(image_folder(root, record["round"], record["image"]))
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
                    "image": record["image"],
                    "y0": y0,
                    "x0": x0,
                    "cells": int(tile_masks.max()),
                    "hand_drawn": n_hand_drawn,
                }
            )
            images.append(tile_image)
            masks.append(tile_masks)
    tiles = pd.DataFrame(rows)
    tiles["split"] = split_tiles(tiles, test_fraction, seed)
    return TrainingSet(channels, tile_px, tiles, images, masks, near_test)


def split_tiles(tiles: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = 0) -> np.ndarray:
    """``"train"`` or ``"test"`` per tile: about ``test_fraction`` go to test, at least one with hand-drawn cells
    and leaving one with hand-drawn cells in train."""
    split = np.full(len(tiles), "train", dtype=object)
    candidates = np.arange(len(tiles))
    with_drawn = [i for i in candidates if tiles["hand_drawn"].iloc[i] > 0]
    if len(with_drawn) < 2:
        raise ValueError("Need hand-drawn cells in at least 2 tiles, so that both train and test have some.")
    rng = np.random.default_rng(seed)
    n_test = min(max(1, round(test_fraction * len(candidates))), len(candidates) - 1)
    first = rng.choice(with_drawn)
    rest = rng.permutation([i for i in candidates if i != first])
    test = [first]
    for i in rest:
        if len(test) == n_test:
            break
        # Keep at least one tile with hand-drawn cells for training.
        drawn_left = [j for j in with_drawn if j not in test and j != i]
        if tiles["hand_drawn"].iloc[i] > 0 and not drawn_left:
            continue
        test.append(i)
    split[test] = "test"
    return split


def train(
    data: TrainingSet,
    models_root: str | Path,
    name: str,
    start_model: str = "cpsam_v2",
    n_epochs: int = 100,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
    batch_size: int = 1,
    device=None,
) -> Path:
    """Fine-tune ``start_model`` on the train tiles and save it in ``<models_root>/<name>/``; returns that folder."""
    from cellpose import models
    from cellpose import train as cellpose_train

    folder = Path(models_root) / name
    if folder.exists():
        raise FileExistsError(f"{folder} exists; choose another name or delete it.")
    train_images, train_masks = data.split("train")
    test_images, test_masks = data.split("test")
    model = models.CellposeModel(gpu=True, pretrained_model=start_model, device=device)
    folder.mkdir(parents=True)
    weights, train_losses, test_losses = cellpose_train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images,
        test_labels=test_masks,
        channel_axis=0,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        save_path=folder,
        model_name=name,
    )
    # Cellpose writes <save_path>/models/<name>; keep the weights next to their records instead.
    shutil.move(str(weights), folder / name)
    shutil.rmtree(folder / "models", ignore_errors=True)

    data.tiles.to_csv(folder / "tiles.csv", index=False)
    by_split = data.tiles.groupby("split")[["cells", "hand_drawn"]].sum()
    record = {
        "name": name,
        "trained_on": date.today().isoformat(),
        "start_model": start_model,
        "channels": data.channels,
        "tile_px": data.tile_px,
        "images": data.tiles[["round", "image"]].drop_duplicates().to_dict("records"),
        "tiles": data.tiles["split"].value_counts().to_dict(),
        "tiles_dropped_near_test": data.near_test,
        "cells": by_split["cells"].to_dict(),
        "hand_drawn_cells": by_split["hand_drawn"].to_dict(),
        "parameters": {
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
        "train_losses": [round(float(v), 5) for v in np.ravel(train_losses)],
        # Cellpose computes the test loss every few epochs only and leaves 0 in between: keep the computed ones.
        "test_losses": {str(i): round(float(v), 5) for i, v in enumerate(np.ravel(test_losses)) if v > 0},
        "macroseg_commit": _git_commit(),
        "cellpose_version": version("cellpose"),
    }
    (folder / "model.json").write_text(json.dumps(record, indent=2) + "\n")
    return folder


def load_model_record(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "model.json").read_text())


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


def tile_mosaic(labels: np.ndarray, tiles: pd.DataFrame, tile_px: int) -> np.ndarray:
    """The ``tiles`` (rows with ``y0``, ``x0``) of ``labels`` side by side in one array, cells renumbered per tile
    so that no two tiles share an ID; cells cut by a tile edge are cut the same way in every segmentation."""
    pieces, offset = [], 0
    for y0, x0 in tiles[["y0", "x0"]].itertuples(index=False):
        piece = relabel_sequential(labels[y0 : y0 + tile_px, x0 : x0 + tile_px])[0].astype(np.int64)
        piece[piece > 0] += offset
        offset = int(piece.max()) if piece.max() else offset
        pieces.append(piece)
    return np.concatenate(pieces, axis=1)


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
