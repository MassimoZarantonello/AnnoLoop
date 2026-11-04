"""The segmentation that proposes the masks of a round, and the intensities per cell.

A ``Segmentation`` is one Cellpose run on up to three channels of an image, written to the store as a labels
element: the starting model (``start`` in the global settings) or a fine-tuned one (``from_model``). ``segment``
runs it chunk-wise through Harpy, one chunk at a time on the GPU; ``measure`` adds its intensity table.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEPTH_PX = 100


@dataclass(frozen=True)
class Segmentation:
    labels: str
    """Labels element written to the store."""
    channels: tuple[str, ...]
    """Cellpose input channels, in order (at most 3). The order is part of a trained model."""
    model: str = "cpsam_v2"
    """A pretrained Cellpose model name, or the path of a fine-tuned model's weights."""
    diameter: float | None = None
    """Expected cell diameter in pixels; None lets Cellpose-SAM use the image as it is."""


def from_model(folder: str | Path, diameter: float | None = None) -> Segmentation:
    """The segmentation of the fine-tuned model in ``folder`` (``<models>/<name>/``): its weights and its channels,
    from its ``model.json``; the labels element is named after the model."""
    folder = Path(folder)
    record = json.loads((folder / "model.json").read_text())
    return Segmentation(
        labels=record["name"], channels=tuple(record["channels"]), model=str(folder / record["name"]), diameter=diameter
    )


def segment(sdata, element: str, segmentation: Segmentation, model: Callable | None = None, **model_kwargs):
    """Run ``segmentation`` on image ``element`` chunk-wise with Harpy and add its labels element to ``sdata``.

    ``model`` defaults to Cellpose with ``segmentation.model``; extra keyword arguments go to the model
    (e.g. ``device="cuda:0"``, ``batch_size=16``)."""
    import dask
    import harpy as hp

    if model is None:
        model = hp.im.cellpose_callable
        model_kwargs = {"pretrained_model": segmentation.model, "diameter": segmentation.diameter, **model_kwargs}
    # Chunks overlap by `depth` pixels; Harpy recommends more than twice the object diameter.
    diameter = segmentation.diameter
    depth = DEFAULT_DEPTH_PX if diameter is None else max(DEFAULT_DEPTH_PX, int(2 * diameter) + 1)
    # One chunk at a time: every chunk loads its own Cellpose model on the GPU, so parallel chunks fill the GPU
    # memory (CUDA out of memory on a whole slide).
    with dask.config.set(scheduler="synchronous"):
        return hp.im.segment(
            sdata,
            image_name=element,
            image_channels=list(segmentation.channels),
            model=model,
            output_labels_name=segmentation.labels,
            output_shapes_name=None,
            depth=depth,
            overwrite=True,
            **model_kwargs,
        )


def measure(sdata, element: str, labels: str):
    """Intensity of every channel per cell, in table ``<labels>_intensities``: the mean per channel in ``.X``;
    ``sum_<channel>``, ``max_<channel>``, the area in pixels (``shapeSize``) and ``cell_ID`` in ``.obs``; each
    cell's centre in ``.obsm["spatial"]``."""
    import harpy as hp

    return hp.tb.aggregate_image(
        sdata,
        image_name=element,
        labels_name=labels,
        output_table_name=f"{labels}_intensities",
        mode="mean",
        obs_stats=["sum", "max", "count"],  # "count" is stored as shapeSize (area in pixels)
        overwrite=True,
    )
