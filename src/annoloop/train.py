"""Fine-tuning Cellpose on the corrected masks of the annotation rounds.

1. ``training_set``: every complete image of the annotation rounds (``annocell.folder.is_complete``: every cell
   checked and Done pressed), read from its AnnoCell folder with the channels of the segmentation it started from
   (e.g. NaK_ATPase + DAPI + F480), together with its corrected masks, cut into square tiles. Every tile is used
   for training: there is no validation split (it cost a fifth of the corrected cells for a loss no decision used);
   models are scored on the test images. Test images (``annoloop.annotation.export_test``) never enter it, and
   tiles within ``margin_px`` of a test image of the same slide are dropped. One annotator per image: an image
   complete for two is refused.
2. ``train``: fine-tune a Cellpose model on the tiles, with the ``training`` settings (the same every round: never
   changed after looking at a test score), and save it in ``<models>/<name>/``: the weights (``<name>``),
   ``tiles.csv`` (every tile) and ``model.json`` (start model, channels, images, settings, losses, versions).
3. ``score_model`` / ``score_test_images``: segment every test image with a model and score it against the
   expert's masks, written to ``<annotations>/scores/<model>.json`` (``annocell.folder.write_score``), where
   overviews (AnnOverview) plot it round after round.

Tiles matter because Cellpose-SAM trains on one random 256 x 256 patch per image and epoch: a single large image
would be seen through one small patch per epoch.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from annocell import folder as af
from skimage.segmentation import relabel_sequential

from annoloop.accuracy import score
from annoloop.annotation import (
    MARGIN_PX,
    SCORE_THRESHOLDS,
    corrected_masks,
    load_annotations,
    load_test_images,
    near_test_image,
)
from annoloop.settings import TrainingSettings
from annoloop.version import code_commit

TILE_PX = 512


@dataclass
class TrainingSet:
    """Tiles of the annotated images: ``tiles`` has one row per tile (round, annotator, image, y0, x0, cells,
    hand_drawn); ``images`` and ``masks`` hold the arrays in the same order, images as (channels, y, x).
    ``near_test``: tiles dropped because they lie near a test image."""

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
    rounds: Collection[int] | None = None,
) -> TrainingSet:
    """Tiles of every complete training image under the annotations ``root``.

    ``channels``: image channels the model learns from; default the channels of the segmentation each image
    started from (they must be the same for all images). ``rounds``: only the images of these rounds (default all).
    """
    annotations = load_annotations(root)
    # Only complete training images; test images (<annotator>/test) are never read here.
    done = annotations[annotations["complete"]] if not annotations.empty else annotations
    if rounds is not None and not done.empty:
        done = done[done["round"].isin(list(rounds))]
    if done.empty:
        where = "" if rounds is None else f" in rounds {sorted(rounds)}"
        raise ValueError(f"No complete image under {root}{where}: every cell must be OK and Done pressed.")
    twice = done.loc[done["image"].duplicated(keep=False), ["image", "annotator"]]
    if not twice.empty:  # blending several annotators of one image is not built (docs/concurrent_annotations.md)
        raise ValueError(f"Images complete for more than one annotator: {twice.to_dict('records')}.")
    tests = load_test_images(root)
    if channels is None:
        options = {tuple(c) for c in done["start_segmentation.channels"]}
        if len(options) != 1:
            raise ValueError(f"Images started from segmentations with different channels {options}; pass channels.")
        channels = list(options.pop())
    channels = list(channels)

    rows, images, masks, near_test = [], [], [], 0
    for record in done.to_dict("records"):
        # The AnnoCell folder holds the image with every channel: the store is not needed (nor where it is).
        image = _read_channels(record["folder"], channels)
        truth, hand_drawn = corrected_masks(record["folder"])
        if image.shape[1:] != truth.shape:
            raise ValueError(f"{record['image']}: image {image.shape[1:]} and masks {truth.shape} differ in size.")
        for y0, x0, tile_image, tile_masks, n_hand_drawn in _tiles(image, truth, hand_drawn, tile_px):
            # The tile in slide pixels: the image's offset in the slide plus the tile's offset in the image.
            oy, ox = record["region"][0], record["region"][2]
            tile_region = [oy + y0, oy + y0 + tile_masks.shape[0], ox + x0, ox + x0 + tile_masks.shape[1]]
            if near_test_image(record, tile_region, tests, margin_px):  # never train next to a test image
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
    settings: TrainingSettings,
    start_model: str = "cpsam_v2",
    device=None,
    extra: Mapping | None = None,
    trainer=None,
) -> Path:
    """Fine-tune ``start_model`` (a pretrained name or the weights of a fine-tuned model) on every tile with
    ``settings`` (the global settings' ``training`` section) and save it in ``<models_root>/<name>/``; returns that
    folder. ``extra``: more fields for ``model.json`` (e.g. the settings files, the model it continues).
    ``trainer``: the training loop, with the signature and return value of Cellpose's ``train.train_seg``; default
    our own, ``annoloop.trainer.train_seg`` (Cellpose's is ``cellpose.train.train_seg``)."""
    from cellpose import models

    folder = Path(models_root) / name
    if folder.exists():
        raise FileExistsError(f"{folder} exists; choose another name or delete it.")
    model = models.CellposeModel(gpu=True, pretrained_model=start_model, device=device)
    folder.mkdir(parents=True)
    if trainer is None:
        from annoloop.trainer import train_seg as trainer
    weights, train_losses, _ = trainer(
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
        "trainer": f"{trainer.__module__}.{trainer.__qualname__}",
        "code_commit": code_commit(),
        "cellpose_version": version("cellpose"),
        **dict(extra or {}),
    }
    (folder / "model.json").write_text(json.dumps(record, indent=2) + "\n")
    return folder


def load_model_record(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "model.json").read_text())


def score_model(root: str | Path, folder: str | Path, **kwargs) -> dict:
    """``score_test_images`` of the fine-tuned model in ``folder``: its weights, its channels, and the latest round
    of the images it was trained on."""
    record = load_model_record(folder)
    round_number = max((int(i["round"]) for i in record.get("images", [])), default=0)
    return score_test_images(
        root, str(Path(folder) / record["name"]), record["name"], round_number, record["channels"], **kwargs
    )


def score_test_images(
    root: str | Path,
    model: str,
    name: str,
    round_number: int,
    channels: Sequence[str],
    thresholds: Sequence[float] = SCORE_THRESHOLDS,
    segment=None,
    **model_kwargs,
) -> dict:
    """Segment every complete test image (the expert pressed Done) under the annotations ``root`` with Cellpose
    ``model`` (weights, or a pretrained name such as ``"cpsam"`` for the starting model, ``round_number`` 0) on
    ``channels``, score it against the expert's masks (AP, precision, recall at each IoU threshold), write it to
    ``<root>/scores/<name>.json`` and return it.

    ``round_number``: the round whose corrections the model was trained on. ``segment``: a function from an image
    (channels, y, x) to masks, instead of Cellpose (tests); ``model_kwargs`` go to ``CellposeModel``.
    """
    if segment is None:
        from cellpose import models

        net = models.CellposeModel(gpu=True, pretrained_model=model, **model_kwargs)

        def segment(image):
            return net.eval(image, channel_axis=0)[0]

    scored = []
    for test in af.images(root):
        # Test images only, and only once the expert pressed Done, as AnnOverview counts them.
        if test.round is not None or not af.is_complete(af.read_review(test.folder)):
            continue
        predicted = np.asarray(segment(_read_channels(test.folder, channels)))
        result = score(af.read_masks(test.folder), predicted, thresholds)
        # NaN (nothing to score) is not JSON: written as null.
        values = {k: (round(float(v), 4) if np.isfinite(v) else None) for k, v in result.items()}
        scored.append({"image": test.name, "annotator": test.annotator, **values})
    if not scored:
        raise ValueError(f"No complete test image under {root}.")
    record = {
        "model": name,
        "round": round_number,
        "scored_on": date.today().isoformat(),
        "channels": list(channels),
        "images": scored,
        "code_commit": code_commit(),
    }
    af.write_score(root, record)
    return record


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


def _read_channels(folder: str | Path, channels: Sequence[str]) -> np.ndarray:
    """The image of an AnnoCell folder, only ``channels``, in that order: (channels, y, x)."""
    pixels, names = af.read_image(folder)
    missing = [c for c in channels if c not in names]
    if missing:
        raise ValueError(f"{folder}: no channel {missing} (it has {names}).")
    return pixels[[names.index(c) for c in channels]]
