"""Settings of the annotation loop: one global file and one file per round (AnnoLoop.md, D9).

- **Global** (``configs/annoloop.yaml``): what must not change between rounds: where the stores are, how the
  channels of the raw files are renamed, the starting segmentation (model and channels), what the annotators see
  first, and the training settings.
- **Round** (``configs/rounds/round_<NN>.yaml``): the round number, the model that proposes the masks (none: the
  starting model), and the images with the annotators each one goes to.

Both are strict: an unknown key, a missing key or a channel that is not defined is an
error before anything runs. Paths may use ``${ANNOLOOP_...}`` variables from ``.env`` (``annoloop.paths``).

Global file::

    paths:
      input: ${ANNOLOOP_INPUT}              # image paths in round files are relative to this
      runs: ${ANNOLOOP_RUNS}                # one .zarr store per image: <runs>/<image>.zarr
      annotations: ${ANNOLOOP_ANNOTATIONS}  # AnnoCell folders: <annotations>/<annotator>/round_<NN>/<image>/
      models: ${ANNOLOOP_MODELS}            # fine-tuned models: <models>/<name>/
    element: macsima_image    # image element in each store
    level: 0                  # pyramid level read from the raw files (optional, default 0)
    channels:                 # raw file channel name -> short name; unlisted channels are dropped
      "DAPI C1 (DAPI C1)": DAPI
    start:                    # the starting segmentation (round 1, D5)
      labels: membrane_dapi_f480   # its labels element in each store
      model: cpsam_v2
      channels: [NaK_ATPase, DAPI, F480]
      diameter: null          # optional
    display:                  # optional: what AnnoCell shows when an image opens
      visible_channels: [F480, NaK_ATPase, DAPI]
      colours: {F480: red, NaK_ATPase: green, DAPI: blue}
    training: {learning_rate: 1e-5, weight_decay: 0.1, n_epochs: 100, batch_size: 1}   # optional

Round file::

    round: 1
    model: null               # a fine-tuned model in paths.models; null = the starting segmentation
    images:
      exp201_spc331:          # the image's name: its folder for every annotator, and its store
        path: Exp201_SPC331_261014_AVH_SUBSET_SingleTiff.tif   # relative to paths.input
        crop: null            # optional [y0, y1, x0, x1] in pixels of the level
        annotators: [anna]    # more than one: a shared image (AnnoLoop.md, O1)
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from annoloop import paths
from annoloop.segmentation import Segmentation

MAX_CELLPOSE_CHANNELS = 3

# The same rule as ``annocell.folder.ANNOTATOR`` (a folder name on Mac and Windows alike); repeated here so the
# settings load without the imaging stack. A test keeps the two equal.
ANNOTATOR = re.compile(r"[a-z0-9-]+")
# An image name is a folder name (AnnoCell) and a store name (<runs>/<name>.zarr).
IMAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class ConfigError(ValueError):
    """A settings file is missing keys, has unknown keys or invalid values."""


@dataclass(frozen=True)
class TrainingSettings:
    """The same every round: never changed after looking at a test score."""

    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    n_epochs: int = 100
    batch_size: int = 1


@dataclass(frozen=True)
class Paths:
    input: str
    runs: str
    annotations: str
    models: str


@dataclass(frozen=True)
class Display:
    visible_channels: tuple[str, ...] | None = None
    """Channels shown when AnnoCell opens an image; ``None``: the channels of the proposed segmentation."""
    colours: Mapping[str, str] = field(default_factory=dict)
    """Napari colormap per channel; AnnOverview expects the marker red and the structure channels green and blue."""


@dataclass(frozen=True)
class GlobalSettings:
    paths: Paths
    element: str
    channels: Mapping[str, str]
    """Raw file channel name -> short name."""
    start: Segmentation
    """The starting segmentation of every image (round 1)."""
    level: int = 0
    display: Display = field(default_factory=Display)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    source: str = "<settings>"
    """The file these settings were read from."""


@dataclass(frozen=True)
class RoundImage:
    name: str
    path: str
    """The raw file, resolved against ``paths.input``."""
    annotators: tuple[str, ...]
    crop: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class RoundSettings:
    round: int
    images: tuple[RoundImage, ...]
    model: str | None = None
    """A fine-tuned model in ``paths.models`` that proposes the masks; ``None``: the starting segmentation."""
    source: str = "<round>"

    def by_annotator(self) -> dict[str, list[str]]:
        """Annotator -> the names of their images in this round."""
        out: dict[str, list[str]] = {}
        for image in self.images:
            for annotator in image.annotators:
                out.setdefault(annotator, []).append(image.name)
        return dict(sorted(out.items()))


def load_settings(path: str | Path) -> GlobalSettings:
    """Load and check the global settings file."""
    return parse_settings(_read_yaml(path), source=str(path))


def load_round(path: str | Path, settings: GlobalSettings) -> RoundSettings:
    """Load and check a round file; image paths are resolved against ``settings.paths.input``."""
    return parse_round(_read_yaml(path), settings, source=str(path))


def parse_settings(raw: Any, source: str = "<settings>") -> GlobalSettings:
    root = _mapping(raw, source)
    _check_keys(
        root,
        source,
        required={"paths", "element", "channels", "start"},
        optional={"level", "display", "training"},
    )

    section = _mapping(root["paths"], f"{source}: paths")
    _check_keys(section, f"{source}: paths", required={"input", "runs", "annotations", "models"}, optional=set())
    paths = Paths(**{k: _path(section[k], f"{source}: paths.{k}") for k in ("input", "runs", "annotations", "models")})

    channels = _mapping(root["channels"], f"{source}: channels")
    if not channels:
        raise ConfigError(f"{source}: channels: expected at least one channel.")
    channels = {_str(k, f"{source}: channels"): _str(v, f"{source}: channels") for k, v in channels.items()}
    short = list(channels.values())
    if len(set(short)) != len(short):
        raise ConfigError(f"{source}: channels: short names must be unique, got {short}.")
    if any("/" in name for name in short):
        raise ConfigError(f"{source}: channels: short names cannot contain '/', got {short}.")

    level = root.get("level", 0)
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ConfigError(f"{source}: level: expected a non-negative integer, got {level!r}.")

    where = f"{source}: start"
    section = _mapping(root["start"], where)
    _check_keys(section, where, required={"labels", "channels"}, optional={"model", "diameter"})
    diameter = section.get("diameter")
    start = Segmentation(
        labels=_name(section["labels"], f"{where}.labels"),
        channels=_known(_cellpose_channels(section["channels"], f"{where}.channels"), short, f"{where}.channels"),
        model=_str(section.get("model", Segmentation.model), f"{where}.model"),
        diameter=None if diameter is None else _positive_float(diameter, f"{where}.diameter"),
    )

    where = f"{source}: display"
    section = _mapping(root.get("display") or {}, where)
    _check_keys(section, where, required=set(), optional={"visible_channels", "colours"})
    visible = section.get("visible_channels")
    if visible is not None:
        if not isinstance(visible, list) or not visible:
            raise ConfigError(f"{where}.visible_channels: expected a non-empty list of channel names.")
        visible = _known(
            tuple(_str(v, f"{where}.visible_channels") for v in visible), short, f"{where}.visible_channels"
        )
    colours = _mapping(section.get("colours") or {}, f"{where}.colours")
    colours = {_str(k, f"{where}.colours"): _str(v, f"{where}.colours") for k, v in colours.items()}
    _known(tuple(colours), short, f"{where}.colours")

    where = f"{source}: training"
    section = _mapping(root.get("training") or {}, where)
    _check_keys(section, where, required=set(), optional={"learning_rate", "weight_decay", "n_epochs", "batch_size"})
    defaults = TrainingSettings()
    training = TrainingSettings(
        learning_rate=_positive_float(section.get("learning_rate", defaults.learning_rate), f"{where}.learning_rate"),
        weight_decay=_non_negative_float(section.get("weight_decay", defaults.weight_decay), f"{where}.weight_decay"),
        n_epochs=_positive_int(section.get("n_epochs", defaults.n_epochs), f"{where}.n_epochs"),
        batch_size=_positive_int(section.get("batch_size", defaults.batch_size), f"{where}.batch_size"),
    )

    return GlobalSettings(
        paths=paths,
        element=_name(root["element"], f"{source}: element"),
        channels=channels,
        start=start,
        level=level,
        display=Display(visible_channels=visible, colours=colours),
        training=training,
        source=source,
    )


def parse_round(raw: Any, settings: GlobalSettings, source: str = "<round>") -> RoundSettings:
    root = _mapping(raw, source)
    _check_keys(root, source, required={"round", "images"}, optional={"model"})
    number = _positive_int(root["round"], f"{source}: round")
    model = root.get("model")
    if model is not None:
        model = _name(model, f"{source}: model")
        if model == settings.start.labels:
            raise ConfigError(f"{source}: model: {model!r} is the starting segmentation's labels; rename the model.")

    images_section = _mapping(root["images"], f"{source}: images")
    if not images_section:
        raise ConfigError(f"{source}: images: expected at least one image.")
    images = []
    for name, value in images_section.items():
        where = f"{source}: images.{name}"
        name = _name(name, where)
        section = _mapping(value, where)
        _check_keys(section, where, required={"path", "annotators"}, optional={"crop"})
        annotators = section["annotators"]
        if not isinstance(annotators, list) or not annotators:
            raise ConfigError(f"{where}.annotators: expected a non-empty list of annotator names.")
        annotators = tuple(_annotator(a, f"{where}.annotators") for a in annotators)
        if len(set(annotators)) != len(annotators):
            raise ConfigError(f"{where}.annotators: an annotator is listed twice: {list(annotators)}.")
        path = Path(_path(section["path"], f"{where}.path"))
        path = path if path.is_absolute() else Path(settings.paths.input) / path
        images.append(
            RoundImage(
                name=name, path=str(path), annotators=annotators, crop=_crop(section.get("crop"), f"{where}.crop")
            )
        )
    return RoundSettings(round=number, images=tuple(images), model=model, source=source)


def _read_yaml(path: str | Path) -> Any:
    path = Path(path)
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error


def _name(value: Any, where: str) -> str:
    """A name that is also a folder or element name."""
    text = _str(value, where)
    if not IMAGE_NAME.fullmatch(text):
        raise ConfigError(f"{where}: {text!r}: use letters, digits, '_', '.' and '-' only.")
    return text


def _annotator(value: Any, where: str) -> str:
    text = _str(value, where)
    if not ANNOTATOR.fullmatch(text):
        raise ConfigError(f"{where}: {text!r}: use lowercase letters, digits and '-' only (e.g. 'alice').")
    return text


def _known(names: tuple[str, ...], channels: list[str], where: str) -> tuple[str, ...]:
    """``names`` if every one is a short channel name of ``channels``."""
    unknown = [n for n in names if n not in channels]
    if unknown:
        raise ConfigError(f"{where}: {unknown} not among the channels {channels}.")
    return names


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}.")
    return value


def _check_keys(section: Mapping[str, Any], where: str, required: set[str], optional: set[str]) -> None:
    missing = required - section.keys()
    if missing:
        raise ConfigError(f"{where}: missing required key(s): {', '.join(sorted(missing))}.")
    unknown = section.keys() - required - optional
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(sorted(map(str, unknown)))}.")


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: expected a non-empty string, got {value!r}.")
    return value


def _path(value: Any, where: str) -> str:
    """A string with its ``${ANNOLOOP_...}`` variables replaced (see ``annoloop.paths``)."""
    try:
        return paths.expand(_str(value, where))
    except KeyError as error:
        raise ConfigError(f"{where}: {error.args[0]}") from error


def _cellpose_channels(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: expected a non-empty list of channel names.")
    channels = tuple(_str(item, where) for item in value)
    if len(channels) > MAX_CELLPOSE_CHANNELS:
        raise ConfigError(f"{where}: Cellpose takes at most {MAX_CELLPOSE_CHANNELS} channels, got {len(channels)}.")
    if len(set(channels)) != len(channels):
        raise ConfigError(f"{where}: channel names must be unique, got {list(channels)}.")
    return channels


def _crop(value: Any, where: str) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value)
    ):
        raise ConfigError(f"{where}: expected [y0, y1, x0, x1] as non-negative integers, got {value!r}.")
    y0, y1, x0, x1 = value
    if y1 <= y0 or x1 <= x0:
        raise ConfigError(f"{where}: expected y0 < y1 and x0 < x1, got {value!r}.")
    return (y0, y1, x0, x1)


def _float(value: Any, where: str) -> float:
    # PyYAML reads `1e-5` (no dot) as a string, so numeric strings are accepted too.
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected a number, got {value!r}.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected a number, got {value!r}.") from None


def _positive_float(value: Any, where: str) -> float:
    number = _float(value, where)
    if number <= 0:
        raise ConfigError(f"{where}: must be > 0, got {number}.")
    return number


def _non_negative_float(value: Any, where: str) -> float:
    number = _float(value, where)
    if number < 0:
        raise ConfigError(f"{where}: must be >= 0, got {number}.")
    return number


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}: expected a positive integer, got {value!r}.")
    return value
