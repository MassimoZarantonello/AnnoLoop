"""Open a macroseg ``.zarr`` store in napari: image channels plus segmentation outlines.

Usage: python -m macroseg.view <store.zarr> [--channels DAPI NaK_ATPase] [--element macsima_image]

Every channel is loaded (additive blending); only ``--channels`` (default: the first two) start
visible. Contrast limits come from the 1st and 99.9th percentiles of a pixel sample. Every labels
element is shown as outlines, one colour per object. Needs the ``viewer`` extra and a display (Mac).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from spatialdata import read_zarr

from macroseg.histograms import sample_pixels

COLORMAPS = ("blue", "green", "magenta", "yellow", "cyan", "red", "gray")


def open_viewer(
    zarr_path: str | Path,
    element: str | None = None,
    channels: Sequence[str] | None = None,
    run: bool = True,
):
    import napari

    sdata = read_zarr(zarr_path)
    element = element or next(iter(sdata.images))
    image = sdata.images[element]
    names = [str(c) for c in image.c.values]
    visible = list(channels) if channels else names[:2]
    unknown = set(visible) - set(names)
    if unknown:
        raise ValueError(f"Channels {sorted(unknown)} not in {element!r}. Available: {names}.")

    pixels = sample_pixels(sdata, element, max_pixels=1_000_000)
    viewer = napari.Viewer(title=f"{Path(zarr_path).name} — {element}")
    for i, name in enumerate(names):
        low, high = np.percentile(pixels[name], (1, 99.9))
        viewer.add_image(
            image.sel(c=name).data,
            name=name,
            colormap=COLORMAPS[i % len(COLORMAPS)],
            blending="additive",
            contrast_limits=(float(low), float(max(high, low + 1))),
            visible=name in visible,
        )
    for labels_name, labels in sdata.labels.items():
        layer = viewer.add_labels(labels.data, name=labels_name, opacity=0.8)
        layer.contour = 2  # outlines only, so the signal underneath stays visible
    if run:
        napari.run()
    return viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zarr", type=Path)
    parser.add_argument("--element", help="Image element (default: the first one).")
    parser.add_argument("--channels", nargs="+", help="Channels visible at start (default: the first two).")
    args = parser.parse_args()
    open_viewer(args.zarr, args.element, args.channels)


if __name__ == "__main__":
    main()
