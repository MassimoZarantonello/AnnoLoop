"""Annotation rounds in the Cellpose GUI: export an image and its starting masks, then record what was corrected.

One folder per image and round, ``<root>/round_<NN>/<image>/``:

- ``<image>.tif``: the channels shown in the GUI (red, green, blue);
- ``<image>_seg.npy``: the masks in the Cellpose GUI format; the GUI loads it and saves the corrections into it;
- ``start_masks.tif``: the masks as exported, to count what the annotator changed;
- ``annotation.json``: where the image and the starting masks come from, its role (``train`` or ``test``),
  and, after ``finish``, the status, what was changed and how well the starting masks scored.

``open_gui`` opens the GUI on a folder, ``corrected_masks`` reads its result, and ``load_annotations`` gathers
every ``annotation.json`` under a root into one table (one row per image and round).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import relabel_sequential

from macroseg.accuracy import load_cellpose_masks, match_cells, score

ROLES = ("train", "test")
SCORE_THRESHOLDS = (0.5, 0.75)


def image_folder(root: str | Path, round_number: int, name: str) -> Path:
    return Path(root) / f"round_{round_number:02d}" / name


def export_for_gui(
    sdata,
    cfg,
    labels: str,
    channels: Sequence[str],
    root: str | Path,
    round_number: int,
    role: str = "train",
    overwrite: bool = False,
) -> Path:
    """Write the folder of image ``cfg.name`` for round ``round_number`` and return it.

    ``labels`` is the starting segmentation (a labels element of ``sdata``), ``channels`` the image channels
    shown in the GUI. An existing folder is left untouched (it may hold corrections) unless ``overwrite``.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}.")
    if len(channels) > 3:
        raise ValueError(f"The Cellpose GUI shows at most 3 channels, got {list(channels)}.")
    folder = image_folder(root, round_number, cfg.name)
    seg_file = folder / f"{cfg.name}_seg.npy"
    if seg_file.exists() and not overwrite:
        return folder

    image = sdata.images[cfg.image.element].sel(c=list(channels)).data.compute()
    masks = relabel_sequential(sdata.labels[labels].data.compute())[0].astype(np.uint32)
    folder.mkdir(parents=True, exist_ok=True)
    image_file = folder / f"{cfg.name}.tif"
    tifffile.imwrite(image_file, image)
    tifffile.imwrite(folder / "start_masks.tif", masks)
    _write_seg(seg_file, masks, image_file)

    step = next((s for s in cfg.segmentation or () if s.labels == labels), None)
    record = {
        "round": round_number,
        "image": cfg.name,
        "role": role,
        "status": "exported",
        "exported_on": date.today().isoformat(),
        "source": sdata.attrs.get("source"),
        "level": sdata.attrs.get("level"),
        "crop": sdata.attrs.get("crop"),
        "pixel_size_um": sdata.attrs.get("pixel_size_um"),
        "zarr": cfg.zarr,
        "start_segmentation": {
            "labels": labels,
            "model": None if step is None or step.expand else step.model,
            "channels": None if step is None else list(step.channels),
            "expand": None if step is None else step.expand,
        },
        "gui_channels": list(channels),
        "macroseg_commit": _git_commit(),
        "cellpose_version": version("cellpose"),
    }
    _write_record(folder, record)
    return folder


def finish(folder: str | Path, complete: bool = True) -> dict:
    """Compare the corrected masks with the starting ones and record it in ``annotation.json``.

    ``complete``: the whole image was reviewed, so untouched cells count as correct. Can be run again
    after further corrections; it overwrites the previous result.
    """
    folder = Path(folder)
    record = json.loads((folder / "annotation.json").read_text())
    final, hand_drawn = corrected_masks(folder)
    start = tifffile.imread(folder / "start_masks.tif")
    record["status"] = "complete" if complete else "partial"
    record["counts"] = change_counts(start, final, hand_drawn)
    result = score(final, start, thresholds=SCORE_THRESHOLDS)
    record["start_score"] = {k: round(v, 4) for k, v in result.items() if "@" in k}
    _write_record(folder, record)
    return record


def corrected_masks(folder: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """The masks saved by the GUI in ``folder`` and the IDs of the cells drawn by hand."""
    folder = Path(folder)
    seg_file = folder / f"{folder.name}_seg.npy"
    masks = load_cellpose_masks(seg_file)
    ismanual = np.load(seg_file, allow_pickle=True).item().get("ismanual")
    hand_drawn = np.flatnonzero(ismanual) + 1 if ismanual is not None else np.array([], dtype=np.int64)
    return masks, hand_drawn


def open_gui(folder: str | Path) -> subprocess.Popen:
    """Open the Cellpose GUI on the image of ``folder``, as a separate process (closing it leaves Python running)."""
    image_file = Path(folder) / f"{Path(folder).name}.tif"
    code = f"from cellpose.gui import gui; gui.run(image={str(image_file)!r})"
    return subprocess.Popen([sys.executable, "-c", code])


def change_counts(start: np.ndarray, final: np.ndarray, hand_drawn: Sequence[int] | None = None) -> dict:
    """How the masks changed: cells before and after, unchanged (identical pixels), removed, drawn by hand.

    Without ``hand_drawn`` (the IDs the GUI marks as drawn), every final cell that is not unchanged counts as drawn.
    """
    matches = match_cells(final, start)
    n_start, n_final = int(len(np.unique(start[start > 0]))), len(matches)
    unchanged = int((matches["iou"] == 1).sum())
    drawn = len(hand_drawn) if hand_drawn is not None else n_final - unchanged
    return {
        "cells_start": n_start,
        "cells_final": n_final,
        "unchanged": unchanged,
        "removed": n_start - unchanged,
        "drawn": drawn,
        "changed_%": round(100 * (1 - unchanged / n_start), 2) if n_start else 0.0,
    }


def load_annotations(root: str | Path) -> pd.DataFrame:
    """One row per ``annotation.json`` under ``root``, nested fields flattened (e.g. ``counts.drawn``)."""
    records = [json.loads(p.read_text()) for p in sorted(Path(root).glob("round_*/*/annotation.json"))]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records).sort_values(["round", "image"]).reset_index(drop=True)


def summarize_rounds(annotations: pd.DataFrame) -> pd.DataFrame:
    """Per round: images (all, complete, test), cells changed, and the mean score of the starting masks."""
    if annotations.empty:
        return pd.DataFrame()
    done = annotations[annotations["status"] != "exported"]
    per_round = annotations.groupby("round").agg(
        images=("image", "size"),
        complete=("status", lambda s: int((s == "complete").sum())),
        test_images=("role", lambda s: int((s == "test").sum())),
    )
    if done.empty:
        return per_round
    score_columns = [c for c in done.columns if c.startswith("start_score.AP@")]
    finished = done.groupby("round").agg(
        cells_drawn=("counts.drawn", "sum"),
        cells_removed=("counts.removed", "sum"),
        mean_changed_pct=("counts.changed_%", "mean"),
        **{f"mean_start_{c.removeprefix('start_score.')}": (c, "mean") for c in score_columns},
    )
    return per_round.join(finished)


def _write_seg(seg_file: Path, masks: np.ndarray, image_file: Path) -> None:
    from cellpose import utils

    outlines = (masks * utils.masks_to_outlines(masks)).astype(np.uint32)
    np.save(seg_file, {"masks": masks, "outlines": outlines, "filename": str(image_file)})


def _write_record(folder: Path, record: dict) -> None:
    (folder / "annotation.json").write_text(json.dumps(record, indent=2) + "\n")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
