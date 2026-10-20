"""Measures to compare cell segmentations without ground truth.

Each measure looks at one kind of error and has its own blind spot, so they are meant to be read
together (see the README of the evaluation notebook):

- ``nuclei_per_cell``: merged cells (2+ nuclei), fragments or false cells (0), missed cells
  (nuclei in no cell). Uses a nuclear segmentation as reference; meaningless for cells built by
  growing that same segmentation.
- ``border_ratio``: whether outlines lie on the membrane (membrane signal on the outline vs inside).
  Favours segmentations that used the membrane channel.
- ``marker_separation``: whether each cell holds the signal of one cell only; with clean cells a
  population marker (e.g. F4/80) splits into two well separated groups (Ruth's suggestion).
- ``shape_features``: area, solidity and eccentricity, to spot abnormal objects.

All functions take plain numpy arrays: 2D integer labels (0 = background) and 2D signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def nuclei_per_cell(cells: np.ndarray, nuclei: np.ndarray, min_fraction: float = 0.5):
    """Assign every nucleus to a cell and count nuclei per cell.

    A nucleus belongs to the cell that covers most of it, if that cell covers at least
    ``min_fraction`` of the nucleus area; otherwise it belongs to no cell (``cell_ID`` 0).

    Returns ``(per_cell, per_nucleus)``: ``per_cell`` has ``cell_ID`` and ``n_nuclei`` for every cell;
    ``per_nucleus`` has ``nucleus_ID``, ``cell_ID`` and ``covered_fraction`` for every nucleus.
    """
    _check_labels(cells, nuclei)
    inside = nuclei > 0
    nucleus, cell = nuclei[inside].astype(np.int64), cells[inside].astype(np.int64)
    # Pixel count of every (nucleus, cell) pair, background cell 0 included.
    pairs, overlap = np.unique(nucleus * (int(cells.max()) + 1) + cell, return_counts=True)
    pair_nucleus, pair_cell = np.divmod(pairs, int(cells.max()) + 1)
    table = pd.DataFrame({"nucleus_ID": pair_nucleus, "cell_ID": pair_cell, "overlap": overlap})
    table["nucleus_area"] = table.groupby("nucleus_ID")["overlap"].transform("sum")
    best = table[table["cell_ID"] > 0].sort_values("overlap").drop_duplicates("nucleus_ID", keep="last")

    per_nucleus = pd.DataFrame({"nucleus_ID": np.unique(nucleus)})
    per_nucleus = per_nucleus.merge(best[["nucleus_ID", "cell_ID", "overlap", "nucleus_area"]], how="left")
    per_nucleus["covered_fraction"] = (per_nucleus["overlap"] / per_nucleus["nucleus_area"]).fillna(0.0)
    assigned = per_nucleus["covered_fraction"] >= min_fraction
    per_nucleus["cell_ID"] = per_nucleus["cell_ID"].where(assigned, 0).fillna(0).astype(np.int64)
    per_nucleus = per_nucleus[["nucleus_ID", "cell_ID", "covered_fraction"]]

    counts = per_nucleus.loc[per_nucleus["cell_ID"] > 0, "cell_ID"].value_counts()
    per_cell = pd.DataFrame({"cell_ID": _ids(cells)})
    per_cell["n_nuclei"] = per_cell["cell_ID"].map(counts).fillna(0).astype(np.int64)
    return per_cell, per_nucleus


def border_ratio(labels: np.ndarray, signal: np.ndarray, width: int = 2) -> pd.DataFrame:
    """Mean ``signal`` on each object's outline divided by its mean inside.

    The outline is the band of ``width`` pixels inside the object along its edge (touching another
    object or background). Objects too small to have an inside get NaN.
    Returns ``cell_ID``, ``border_mean``, ``inside_mean`` and ``ratio``.
    """
    from scipy import ndimage as ndi

    _check_labels(labels, signal)
    size = 2 * width + 1
    # A pixel is inside if its whole (2w+1)^2 neighbourhood has the same label.
    same = (ndi.maximum_filter(labels, size=size) == labels) & (ndi.minimum_filter(labels, size=size) == labels)
    obj = labels > 0
    flat, values = labels.ravel(), signal.ravel().astype(np.float64)
    n = int(labels.max()) + 1
    border = (obj & ~same).ravel()
    inside = (obj & same).ravel()
    border_sum = np.bincount(flat[border], weights=values[border], minlength=n)
    border_n = np.bincount(flat[border], minlength=n)
    inside_sum = np.bincount(flat[inside], weights=values[inside], minlength=n)
    inside_n = np.bincount(flat[inside], minlength=n)
    ids = _ids(labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        border_mean = border_sum[ids] / border_n[ids]
        inside_mean = np.where(inside_n[ids] > 0, inside_sum[ids] / np.maximum(inside_n[ids], 1), np.nan)
        ratio = border_mean / inside_mean
    return pd.DataFrame({"cell_ID": ids, "border_mean": border_mean, "inside_mean": inside_mean, "ratio": ratio})


def marker_separation(values: np.ndarray, ambiguous: tuple[float, float] = (0.2, 0.8), seed: int = 0) -> dict:
    """How well per-cell marker values split into a negative and a positive group.

    Fits a two-component Gaussian mixture on ``log1p(values)`` and returns:

    - ``ashman_d``: distance between the two groups in units of their spread
      (``sqrt(2) |m1 - m2| / sqrt(s1^2 + s2^2)``); above ~2 the groups are cleanly separated;
    - ``fraction_ambiguous``: fraction of cells whose probability of being positive lies within
      ``ambiguous`` (default 0.2-0.8), i.e. cells that look "half positive";
    - ``fraction_positive``: fraction of cells assigned to the brighter group.
    """
    from sklearn.mixture import GaussianMixture

    x = np.log1p(np.asarray(values, dtype=np.float64)).reshape(-1, 1)
    x = x[np.isfinite(x[:, 0])]
    gmm = GaussianMixture(2, random_state=seed).fit(x)
    means = gmm.means_.ravel()
    variances = gmm.covariances_.ravel()
    positive = int(np.argmax(means))
    p_positive = gmm.predict_proba(x)[:, positive]
    low, high = ambiguous
    return {
        "ashman_d": float(np.sqrt(2) * abs(means[0] - means[1]) / np.sqrt(variances.sum())),
        "fraction_ambiguous": float(np.mean((p_positive > low) & (p_positive < high))),
        "fraction_positive": float(np.mean(p_positive >= 0.5)),
    }


def shape_features(labels: np.ndarray) -> pd.DataFrame:
    """Area (pixels), solidity (area / convex hull area; low = ragged, often merged cells) and
    eccentricity (0 = round, towards 1 = elongated) of every object."""
    from skimage.measure import regionprops_table

    table = regionprops_table(labels, properties=("label", "area", "solidity", "eccentricity"))
    return pd.DataFrame(table).rename(columns={"label": "cell_ID"})


def _ids(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    return ids[ids > 0].astype(np.int64)


def _check_labels(labels: np.ndarray, other: np.ndarray) -> None:
    if labels.ndim != 2 or labels.shape != other.shape:
        raise ValueError(f"Expected two aligned 2D arrays, got shapes {labels.shape} and {other.shape}.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Labels must be integers, got {labels.dtype}.")


def evaluate_candidates(
    sdata,
    image: str,
    candidates: list[str],
    nuclei: str,
    membrane: str,
    marker: str,
    border_width: int = 2,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run every measure on each candidate cell segmentation of a macroseg ``.zarr``.

    ``candidates`` are labels elements with cells; ``nuclei`` the reference nuclear labels element;
    ``membrane`` and ``marker`` channel names of ``image``. The marker means come from the
    ``<candidate>_intensities`` tables written by the baseline.

    Returns ``(summary, per_cell)``: one summary row per candidate, and for each candidate a table
    with one row per cell (n_nuclei, border ratio, marker mean, shape).
    """
    pixel_size = sdata.attrs.get("pixel_size_um") or float("nan")
    nuclei_labels = sdata.labels[nuclei].values
    membrane_signal = sdata.images[image].sel(c=membrane).values
    summary, per_cell = [], {}
    for name in candidates:
        cells = sdata.labels[name].values
        counts, per_nucleus = nuclei_per_cell(cells, nuclei_labels)
        border = border_ratio(cells, membrane_signal, width=border_width)
        table = sdata.tables[f"{name}_intensities"]
        marker_mean = pd.DataFrame({"cell_ID": table.obs["cell_ID"].to_numpy(), "marker_mean": table.to_df()[marker]})
        cells_table = (
            counts.merge(border[["cell_ID", "ratio"]].rename(columns={"ratio": "border_ratio"}))
            .merge(marker_mean, how="left")
            .merge(shape_features(cells))
        )
        separation = marker_separation(cells_table["marker_mean"].dropna())
        n = cells_table["n_nuclei"]
        summary.append(
            {
                "segmentation": name,
                "cells": len(cells_table),
                "median_area_um2": cells_table["area"].median() * pixel_size**2,
                "cells_0_nuclei_%": 100 * (n == 0).mean(),
                "cells_1_nucleus_%": 100 * (n == 1).mean(),
                "cells_2plus_nuclei_%": 100 * (n >= 2).mean(),
                "nuclei_in_no_cell_%": 100 * (per_nucleus["cell_ID"] == 0).mean(),
                "median_border_ratio": cells_table["border_ratio"].median(),
                f"{marker}_ashman_d": separation["ashman_d"],
                f"{marker}_ambiguous_%": 100 * separation["fraction_ambiguous"],
                f"{marker}_positive_%": 100 * separation["fraction_positive"],
                "median_solidity": cells_table["solidity"].median(),
            }
        )
        per_cell[name] = cells_table
    return pd.DataFrame(summary).set_index("segmentation"), per_cell


# Categorical slots 1-3 of the dataviz reference palette (blue, orange, aqua), in fixed order.
CANDIDATE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")


def plot_comparison(per_cell: dict[str, pd.DataFrame], marker: str, pixel_size_um: float):
    """Four small multiples, one line or bar group per candidate (same colour everywhere):
    nuclei per cell, border ratio, marker mean per cell (log axis) and cell area in µm² (log axis)."""
    import matplotlib.pyplot as plt

    names = list(per_cell)
    if len(names) > len(CANDIDATE_COLORS):
        raise ValueError(f"At most {len(CANDIDATE_COLORS)} candidates per figure, got {len(names)}.")
    colors = dict(zip(names, CANDIDATE_COLORS, strict=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        n = per_cell[name]["n_nuclei"].clip(upper=2)
        share = [100 * (n == k).mean() for k in (0, 1, 2)]
        ax.bar(np.arange(3) + (i - (len(names) - 1) / 2) * width, share, width * 0.9, color=colors[name], label=name)
    ax.set_xticks(range(3), ["0", "1", "2+"])
    ax.set_xlabel("nuclei in the cell")
    ax.set_ylabel("% of cells")
    ax.set_title("Nuclei per cell")
    panels = [
        (axes[0, 1], "border_ratio", "membrane on outline / inside", "Border ratio (1 = no edge)", False),
        (axes[1, 0], "marker_mean", f"mean {marker} per cell", f"{marker} per cell", True),
        (axes[1, 1], "area", "cell area (µm²)", "Cell area", True),
    ]
    for ax, column, xlabel, title, log in panels:
        values = {name: per_cell[name][column].dropna().to_numpy(dtype=float) for name in names}
        if column == "area":
            values = {name: v * pixel_size_um**2 for name, v in values.items()}
        pooled = np.concatenate(list(values.values()))
        low, high = np.nanpercentile(pooled, [0.5, 99.5])
        bins = np.geomspace(max(low, 1e-3), high, 60) if log else np.linspace(low, high, 60)
        for name in names:
            ax.hist(values[name], bins=bins, histtype="step", linewidth=2, color=colors[name], label=name, density=True)
        if log:
            ax.set_xscale("log")
        if column == "border_ratio":
            ax.axvline(1, color="0.5", linewidth=1, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(title)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(names), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
