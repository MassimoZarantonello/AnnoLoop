"""Per-pixel intensity histograms of each channel.

They show whether signal and background separate for a marker before segmenting with it.
Large images are subsampled on a regular grid, so the whole slide is never loaded at once.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from spatialdata import SpatialData

PERCENTILES = (1, 50, 99, 99.9)


def sample_pixels(sdata: SpatialData, element: str, max_pixels: int = 2_000_000) -> dict[str, np.ndarray]:
    """Return up to ``max_pixels`` pixels per channel, taken on a regular grid."""
    image = sdata.images[element]
    height, width = image.shape[-2:]
    step = max(1, int(np.ceil(np.sqrt(height * width / max_pixels))))
    sampled = image[:, ::step, ::step].compute()
    return {str(c): sampled.sel(c=c).values.ravel() for c in sampled.c.values}


def channel_summary(pixels: dict[str, np.ndarray]) -> pd.DataFrame:
    """Percentiles per channel, to choose display ranges and thresholds."""
    rows = {name: np.percentile(values, PERCENTILES) for name, values in pixels.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"p{p:g}" for p in PERCENTILES])


def plot_histograms(pixels: dict[str, np.ndarray], channels: Iterable[str] | None = None, bins: int = 200):
    """One panel per channel, log-spaced x axis (uint16 signal is squashed at the low end on a linear axis).

    Dashed lines mark the 1st, 50th and 99th percentiles.
    """
    import matplotlib.pyplot as plt

    names = list(channels) if channels is not None else list(pixels)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names, strict=False):
        values = pixels[name].astype(np.float64) + 1  # +1 so zeros fit on a log axis
        edges = np.logspace(0, np.log10(values.max() + 1), bins)
        ax.hist(values, bins=edges, color="0.3")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for p in (1, 50, 99):
            ax.axvline(np.percentile(values, p), color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("intensity + 1")
        ax.set_ylabel("pixels")
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
