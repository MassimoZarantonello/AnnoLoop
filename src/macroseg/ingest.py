"""Read a raw image (OME-TIFF or CZI) into a SpatialData object.

Channels are renamed to the short names of the config, so the rest of the pipeline never sees
file-specific channel names. Coordinates stay in pixels (the ``global`` coordinate system);
the physical pixel size is kept in ``sdata.attrs["pixel_size_um"]``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import dask.array as da
import numpy as np
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

from macroseg.config import InputConfig

CHUNK_PX = 2048


def ingest(cfg: InputConfig, element: str) -> SpatialData:
    """Read ``cfg.path`` as image element ``element``."""
    path = Path(cfg.path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        image, names, pixel_size = _read_ome_tiff(path, cfg.level)
    elif suffix == ".czi":
        image, names, pixel_size = _read_czi(path)
    else:
        raise ValueError(f"{path}: unsupported format {suffix!r}; expected .tif, .tiff or .czi.")

    if cfg.crop is not None:
        y0, y1, x0, x1 = cfg.crop
        if y1 > image.shape[1] or x1 > image.shape[2]:
            raise ValueError(f"Crop {cfg.crop} exceeds the image size {image.shape[1:]} at level {cfg.level}.")
        image = image[:, y0:y1, x0:x1]

    image, names = _select_channels(image, names, cfg.channels)
    image = Image2DModel.parse(image.rechunk((1, CHUNK_PX, CHUNK_PX)), dims=("c", "y", "x"), c_coords=names)
    sdata = SpatialData(images={element: image})
    sdata.attrs.update(
        source=str(path),
        level=cfg.level,
        crop=list(cfg.crop) if cfg.crop else None,
        pixel_size_um=pixel_size,
    )
    return sdata


def _select_channels(
    image: da.Array, names: list[str], mapping: Mapping[str, str] | None
) -> tuple[da.Array, list[str]]:
    if mapping is None:
        return image, short_names(names)
    missing = [name for name in mapping if name not in names]
    if missing:
        raise ValueError(f"Channels {missing} not in the file. Available: {names}.")
    index = [names.index(name) for name in mapping]
    return image[index], list(mapping.values())


def short_names(names: list[str]) -> list[str]:
    """First word of each file channel name: ``"Na_K_ATPase (APC C2)"`` -> ``"Na_K_ATPase"``.

    Raises if two channels get the same short name; then list them under ``input.channels``.
    """
    short = [re.split(r"[\s(]", name.strip(), maxsplit=1)[0].replace("/", "_") or name for name in names]
    duplicates = sorted({name for name in short if short.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Channels {names} give duplicate short names {duplicates}; "
            "map them explicitly under input.channels in the config."
        )
    return short


def _read_ome_tiff(path: Path, level: int) -> tuple[da.Array, list[str], float | None]:
    """Lazy (channels, y, x) array of pyramid ``level``, channel names, pixel size in µm at that level."""
    import tifffile
    import zarr

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "CYX":
            raise ValueError(f"{path}: expected axes CYX, got {series.axes}.")
        if level >= len(series.levels):
            raise ValueError(f"{path}: level {level} requested, file has {len(series.levels)} level(s).")
        metadata = tif.ome_metadata or ""
        # Downsampling factor of the chosen level, from the real widths (MACSima pyramids use 4x per level).
        downsampling = series.levels[0].shape[-1] / series.levels[level].shape[-1]
    names = re.findall(r'<Channel[^>]*\bName="([^"]*)"', metadata)
    n_channels = series.shape[0]
    if len(names) != n_channels:
        names = [f"channel_{i}" for i in range(n_channels)]
    # Lazy: only the chunks that are used (e.g. a crop) are read from disk.
    store = tifffile.imread(path, aszarr=True, level=level)
    image = da.from_zarr(zarr.open(store, mode="r"))
    pixel_size = _pixel_size_um(metadata)
    return image, names, None if pixel_size is None else pixel_size * downsampling


def _pixel_size_um(ome_metadata: str) -> float | None:
    match = re.search(r'PhysicalSizeX="([^"]+)"', ome_metadata)
    if not match:
        return None
    unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome_metadata)
    factor = {"nm": 1e-3, "µm": 1.0, "um": 1.0, "mm": 1e3}.get(unit.group(1) if unit else "µm")
    return None if factor is None else float(match.group(1)) * factor


def _read_czi(path: Path) -> tuple[da.Array, list[str], float | None]:
    import xml.etree.ElementTree as ET

    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(str(path)) as czi:
        metadata = ET.fromstring(czi.raw_metadata)
        names = [c.get("Name") for c in metadata.findall(".//Information/Image/Dimensions/Channels/Channel")]
        planes = [czi.read(plane={"C": c, "Z": 0, "T": 0})[..., 0] for c in range(len(names))]
    scaling = metadata.find(".//Scaling/Items/Distance[@Id='X']/Value")
    pixel_size = None if scaling is None else float(scaling.text) * 1e6  # metres -> µm
    return da.from_array(np.stack(planes)), names, pixel_size
