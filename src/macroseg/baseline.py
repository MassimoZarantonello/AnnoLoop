"""Baseline segmentation with the stock Cellpose-SAM model (``cpsam_v2``), plus intensities per object.

Pipeline: ingest -> write ``.zarr`` -> every segmentation listed in the config (e.g. DAPI only,
membrane only, membrane + DAPI), each with its own labels element -> mean, sum and max intensity of
every channel per object.

Usage: python -m macroseg.baseline configs/<experiment>.yaml [--overwrite]
Input image and output store come from the config (input.path, zarr).
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable
from pathlib import Path

import dask
from spatialdata import SpatialData, read_zarr

from macroseg.config import ExperimentConfig, SegmentationStep, load_config
from macroseg.ingest import ingest

DEFAULT_DEPTH_PX = 100


def segment(
    sdata: SpatialData,
    element: str,
    step: SegmentationStep,
    model: Callable | None = None,
    **model_kwargs,
) -> SpatialData:
    """Run one segmentation step chunk-wise with Harpy and add ``step.labels`` to ``sdata``.

    ``model`` defaults to Cellpose with ``step.model`` (Cellpose-SAM v2); extra keyword arguments go to the model
    (e.g. ``device="cuda:0"``, ``batch_size=16``, ``flow_threshold=0.4``).
    """
    import harpy as hp

    if model is None:
        model = hp.im.cellpose_callable
        model_kwargs = {"pretrained_model": step.model, "diameter": step.diameter, **model_kwargs}
    # Chunks overlap by `depth` pixels; Harpy recommends more than twice the object diameter.
    depth = DEFAULT_DEPTH_PX if step.diameter is None else max(DEFAULT_DEPTH_PX, int(2 * step.diameter) + 1)
    # One chunk at a time: every chunk loads its own Cellpose model on the GPU, so parallel
    # chunks fill the GPU memory (CUDA out of memory on a whole slide).
    with dask.config.set(scheduler="synchronous"):
        return hp.im.segment(
            sdata,
            image_name=element,
            image_channels=list(step.channels),
            model=model,
            output_labels_name=step.labels,
            output_shapes_name=None,
            depth=depth,
            overwrite=True,
            **model_kwargs,
        )


def measure(sdata: SpatialData, element: str, labels: str) -> SpatialData:
    """Intensity of every channel per object, in table ``<labels>_intensities``.

    ``.X`` holds the mean per channel; ``.obs`` holds ``sum_<channel>``, ``max_<channel>``, the
    object area in pixels (``shapeSize``), ``cell_ID`` (the value in the labels element) and
    ``fov_labels`` (the labels element). ``.obsm["spatial"]`` holds each object's centre (y, x).
    Sum or max suit markers that cover only part of a cell, such as a membrane marker, whose mean
    is diluted by the cell interior.
    """
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


def run(
    cfg: ExperimentConfig,
    overwrite: bool = False,
    model: Callable | None = None,
    **model_kwargs,
) -> SpatialData:
    """Full baseline pipeline: reads ``cfg.input.path`` and writes ``cfg.zarr``."""
    if cfg.input is None or cfg.segmentation is None:
        raise ValueError(f"Config {cfg.name!r} needs 'input' and 'segmentation' sections for the baseline.")
    if cfg.zarr is None:
        raise ValueError(f"Config {cfg.name!r} needs a 'zarr' path for the output store.")
    zarr_path = Path(cfg.zarr)
    if zarr_path.exists() and not overwrite:
        raise FileExistsError(f"{zarr_path} exists; pass overwrite=True (--overwrite) to replace it.")

    if zarr_path.exists():
        # Remove the old store ourselves: SpatialData's own overwrite check fails on the lazy TIFF reader.
        if zarr_path.suffix != ".zarr" or not zarr_path.is_dir():
            raise ValueError(f"Refusing to overwrite {zarr_path}: not a .zarr directory.")
        shutil.rmtree(zarr_path)

    element = cfg.image.element
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ingest(cfg.input, element).write(zarr_path)
    # Reopen backed by the store, so Harpy writes masks and tables straight to disk.
    sdata = read_zarr(zarr_path)
    # Check every step first, so a typo in the last one does not waste the earlier GPU runs.
    available = [str(c) for c in sdata.images[element].c.values]
    for step in cfg.segmentation:
        missing = set(step.channels) - set(available)
        if missing:
            raise ValueError(
                f"Channels {sorted(missing)} of {step.labels!r} are not in image {element!r}. Available: {available}."
            )
    for step in cfg.segmentation:
        sdata = segment(sdata, element, step, model=model, **model_kwargs)
        sdata = measure(sdata, element, step.labels)
    return sdata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing .zarr store.")
    parser.add_argument("--device", help="Torch device, e.g. cuda:0, mps or cpu (default: automatic).")
    args = parser.parse_args()
    kwargs = {"device": args.device} if args.device else {}
    sdata = run(load_config(args.config), overwrite=args.overwrite, **kwargs)
    print(sdata)


if __name__ == "__main__":
    main()
