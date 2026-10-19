"""Baseline segmentation with the stock Cellpose-SAM model (``cpsam_v2``), plus intensities per object.

Pipeline: ingest -> write ``.zarr`` -> nuclei segmentation (always) -> optional second
segmentation with another marker -> mean, sum and max intensity of every channel per object.

Usage: python -m macroseg.baseline configs/<experiment>.yaml [--input FILE] [--zarr STORE] [--overwrite]
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

    ``.X`` holds the mean per channel; ``.obs`` holds ``<channel>_sum``, ``<channel>_max`` and the
    object size in pixels (``shapeSize``). Sum or max suit markers that cover only part of a cell,
    such as a membrane marker, whose mean is diluted by the cell interior.
    """
    import harpy as hp

    return hp.tb.aggregate_image(
        sdata,
        image_name=element,
        labels_name=labels,
        output_table_name=f"{labels}_intensities",
        mode="mean",
        obs_stats=["sum", "max"],
        overwrite=True,
    )


def run(
    cfg: ExperimentConfig,
    input_path: str | Path | None = None,
    zarr_path: str | Path | None = None,
    overwrite: bool = False,
    model: Callable | None = None,
    **model_kwargs,
) -> SpatialData:
    """Full baseline pipeline. ``input_path`` and ``zarr_path`` override the paths in the config."""
    if cfg.input is None or cfg.segmentation is None:
        raise ValueError(f"Config {cfg.name!r} needs 'input' and 'segmentation' sections for the baseline.")
    zarr_path = zarr_path or cfg.zarr
    if zarr_path is None:
        raise ValueError("Give a .zarr path, in the config ('zarr') or as zarr_path.")
    zarr_path = Path(zarr_path)
    if zarr_path.exists() and not overwrite:
        raise FileExistsError(f"{zarr_path} exists; pass overwrite=True (--overwrite) to replace it.")

    if zarr_path.exists():
        # Remove the old store ourselves: SpatialData's own overwrite check fails on the lazy TIFF reader.
        if zarr_path.suffix != ".zarr" or not zarr_path.is_dir():
            raise ValueError(f"Refusing to overwrite {zarr_path}: not a .zarr directory.")
        shutil.rmtree(zarr_path)

    element = cfg.image.element
    ingest(cfg.input, element, input_path).write(zarr_path)
    # Reopen backed by the store, so Harpy writes masks and tables straight to disk.
    sdata = read_zarr(zarr_path)
    for step in (cfg.segmentation.nuclei, cfg.segmentation.cells):
        if step is None:
            continue
        available = [str(c) for c in sdata.images[element].c.values]
        missing = set(step.channels) - set(available)
        if missing:
            raise ValueError(
                f"Channels {sorted(missing)} of {step.labels!r} are not in image {element!r}. Available: {available}."
            )
        sdata = segment(sdata, element, step, model=model, **model_kwargs)
        sdata = measure(sdata, element, step.labels)
    return sdata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--input", type=Path, help="Raw image; overrides input.path in the config.")
    parser.add_argument("--zarr", type=Path, help="Output store; overrides zarr in the config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing .zarr store.")
    parser.add_argument("--device", help="Torch device, e.g. cuda:0, mps or cpu (default: automatic).")
    args = parser.parse_args()
    kwargs = {"device": args.device} if args.device else {}
    sdata = run(load_config(args.config), args.input, args.zarr, overwrite=args.overwrite, **kwargs)
    print(sdata)


if __name__ == "__main__":
    main()
