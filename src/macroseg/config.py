"""Experiment configuration.

Everything problem-specific (marker names, Cellpose channel order, element names, training
parameters) lives in a YAML file and is loaded into the frozen dataclasses below. Code reads
these objects and never hard-codes marker or channel names.

Unknown keys are rejected so that typos in a config fail loudly instead of being ignored.
Paths (``input.path``, ``zarr``) may use ``${MACROSEG_...}`` variables from ``.env`` (see ``macroseg.paths``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from macroseg import paths

MAX_CELLPOSE_CHANNELS = 3


class ConfigError(ValueError):
    """Raised when a config file is missing keys, has unknown keys or invalid values."""


@dataclass(frozen=True)
class ImageConfig:
    """SpatialData element names used by the pipeline."""

    element: str
    """Image element holding the MACSIMA channels."""
    baseline_labels: str = "baseline_cellpose"
    """Labels element written by the stock Cellpose model."""


@dataclass(frozen=True)
class ChannelsConfig:
    cellpose: tuple[str, ...]
    """Cellpose input channels, in order (at most 3). The order is part of the trained model."""
    candidate_marker: str
    """Marker used to pre-select candidate cells for annotation (e.g. F4/80)."""


@dataclass(frozen=True)
class CandidatesConfig:
    min_mean_intensity: float | None = None
    """Mean marker intensity above which a cell is a candidate. ``None`` means automatic (Otsu)."""


@dataclass(frozen=True)
class AnnotationConfig:
    shapes_prefix: str
    """Prefix of the shapes element each annotator saves polygons to (``<prefix>__<annotator>``)."""
    class_label: str
    """Class name biologists give to confirmed target cells."""


@dataclass(frozen=True)
class RoiConfig:
    size_px: int = 1024
    """Side length of square ROIs, in pixels."""


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    n_epochs: int = 100
    batch_size: int = 1


@dataclass(frozen=True)
class SplitConfig:
    group_by: str = "sample"
    """Column that defines independent groups; train/val/test never share a group."""
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True)
class InputConfig:
    """Raw image to ingest. Paths are machine-specific: override them when running elsewhere."""

    path: str
    """OME-TIFF (``.tif``/``.tiff``) or CZI file."""
    channels: Mapping[str, str] | None = None
    """File channel name -> short name used everywhere else. Unlisted channels are dropped.
    ``None`` keeps all channels, named by the first word of their file name
    (``"Na_K_ATPase (APC C2)"`` -> ``Na_K_ATPase``)."""
    level: int = 0
    """Pyramid level to read (0 = full resolution). Ignored for files without a pyramid."""
    crop: tuple[int, int, int, int] | None = None
    """``(y0, y1, x0, x1)`` in pixels of the chosen level; ``None`` reads the whole image."""


@dataclass(frozen=True)
class SegmentationStep:
    """One segmentation, written to labels element ``labels``. Either a Cellpose run on ``channels``,
    or (``expand`` set) an earlier segmentation grown by ``distance_um``, e.g. nuclei -> approximate cells."""

    labels: str
    """Labels element written to the ``.zarr`` store (the step's name in the config)."""
    channels: tuple[str, ...] = ()
    """Cellpose input channels, in order (at most 3). Empty for an expansion."""
    diameter: float | None = None
    """Expected object diameter in pixels. ``None`` lets Cellpose-SAM use the image as is."""
    model: str = "cpsam_v2"
    """Cellpose pretrained model name (e.g. ``cpsam_v2``, ``cpsam``) or path to a fine-tuned model."""
    expand: str | None = None
    """Name of an earlier step whose objects are grown, instead of running Cellpose."""
    distance_um: float | None = None
    """How far each object of ``expand`` grows, in µm (objects stop where they meet)."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    target: str
    """Cell type being segmented, e.g. ``macrophages``."""
    image: ImageConfig
    channels: ChannelsConfig
    annotation: AnnotationConfig
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    input: InputConfig | None = None
    segmentation: tuple[SegmentationStep, ...] | None = None
    """Segmentations to run, in config order; each writes its own labels element."""
    zarr: str | None = None
    """SpatialData store written by the pipeline."""


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    return parse_config(raw, source=str(path))


def parse_config(raw: Any, source: str = "<config>") -> ExperimentConfig:
    """Validate an already-parsed mapping and build an :class:`ExperimentConfig`."""
    root = _mapping(raw, source)
    _check_keys(
        root,
        source,
        required={"name", "target", "image", "channels", "annotation"},
        optional={"candidates", "roi", "training", "split", "input", "segmentation", "zarr"},
    )

    image = _section(root, "image", source, required={"element"}, optional={"baseline_labels"})
    channels = _section(root, "channels", source, required={"cellpose", "candidate_marker"})
    annotation = _section(root, "annotation", source, required={"shapes_prefix", "class_label"})
    candidates = _section(root, "candidates", source, optional={"min_mean_intensity"})
    roi = _section(root, "roi", source, optional={"size_px"})
    training = _section(root, "training", source, optional={"learning_rate", "weight_decay", "n_epochs", "batch_size"})
    split = _section(root, "split", source, optional={"group_by", "val_fraction", "test_fraction"})

    config = ExperimentConfig(
        name=_str(root["name"], f"{source}: name"),
        target=_str(root["target"], f"{source}: target"),
        image=ImageConfig(
            element=_str(image["element"], f"{source}: image.element"),
            baseline_labels=_str(
                image.get("baseline_labels", ImageConfig.baseline_labels), f"{source}: image.baseline_labels"
            ),
        ),
        channels=ChannelsConfig(
            cellpose=_cellpose_channels(channels["cellpose"], f"{source}: channels.cellpose"),
            candidate_marker=_str(channels["candidate_marker"], f"{source}: channels.candidate_marker"),
        ),
        annotation=AnnotationConfig(
            shapes_prefix=_str(annotation["shapes_prefix"], f"{source}: annotation.shapes_prefix"),
            class_label=_str(annotation["class_label"], f"{source}: annotation.class_label"),
        ),
        candidates=CandidatesConfig(
            min_mean_intensity=_optional_float(
                candidates.get("min_mean_intensity"), f"{source}: candidates.min_mean_intensity"
            ),
        ),
        roi=RoiConfig(size_px=_positive_int(roi.get("size_px", RoiConfig.size_px), f"{source}: roi.size_px")),
        training=TrainingConfig(
            learning_rate=_positive_float(
                training.get("learning_rate", TrainingConfig.learning_rate), f"{source}: training.learning_rate"
            ),
            weight_decay=_non_negative_float(
                training.get("weight_decay", TrainingConfig.weight_decay), f"{source}: training.weight_decay"
            ),
            n_epochs=_positive_int(training.get("n_epochs", TrainingConfig.n_epochs), f"{source}: training.n_epochs"),
            batch_size=_positive_int(
                training.get("batch_size", TrainingConfig.batch_size), f"{source}: training.batch_size"
            ),
        ),
        split=SplitConfig(
            group_by=_str(split.get("group_by", SplitConfig.group_by), f"{source}: split.group_by"),
            val_fraction=_fraction(
                split.get("val_fraction", SplitConfig.val_fraction), f"{source}: split.val_fraction"
            ),
            test_fraction=_fraction(
                split.get("test_fraction", SplitConfig.test_fraction), f"{source}: split.test_fraction"
            ),
        ),
    )

    config = replace(
        config,
        input=_input(root.get("input"), f"{source}: input"),
        segmentation=_segmentation(root.get("segmentation"), f"{source}: segmentation"),
        zarr=None if root.get("zarr") is None else _path(root["zarr"], f"{source}: zarr"),
    )

    if config.split.val_fraction + config.split.test_fraction >= 1:
        raise ConfigError(f"{source}: split.val_fraction + split.test_fraction must be < 1.")
    return config


def _input(value: Any, where: str) -> InputConfig | None:
    if value is None:
        return None
    section = _mapping(value, where)
    _check_keys(section, where, required={"path"}, optional={"channels", "level", "crop"})
    channels = section.get("channels")
    if channels is not None:
        channels = _mapping(channels, f"{where}.channels")
        if not channels:
            raise ConfigError(f"{where}.channels: expected at least one channel.")
        channels = {_str(k, f"{where}.channels"): _str(v, f"{where}.channels") for k, v in channels.items()}
        if len(set(channels.values())) != len(channels):
            raise ConfigError(f"{where}.channels: short names must be unique, got {list(channels.values())}.")
        if any("/" in name for name in channels.values()):
            # Short names become table column and zarr element names, where "/" is a path separator.
            raise ConfigError(f"{where}.channels: short names cannot contain '/', got {list(channels.values())}.")
    level = section.get("level", InputConfig.level)
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ConfigError(f"{where}.level: expected a non-negative integer, got {level!r}.")
    return InputConfig(
        path=_path(section["path"], f"{where}.path"),
        channels=channels,
        level=level,
        crop=_crop(section.get("crop"), f"{where}.crop"),
    )


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


def _segmentation(value: Any, where: str) -> tuple[SegmentationStep, ...] | None:
    """``{name: step, ...}``, run in order. A step is ``{channels, diameter?, model?}`` (Cellpose)
    or ``{expand: <earlier step>, distance_um}`` (grow an earlier segmentation)."""
    if value is None:
        return None
    section = _mapping(value, where)
    if not section:
        raise ConfigError(f"{where}: expected at least one segmentation.")
    steps = tuple(_segmentation_step(name, step, f"{where}.{name}") for name, step in section.items())
    for i, step in enumerate(steps):
        if step.expand is not None and step.expand not in {s.labels for s in steps[:i]}:
            raise ConfigError(f"{where}.{step.labels}.expand: {step.expand!r} must be an earlier segmentation.")
    return steps


def _segmentation_step(name: Any, value: Any, where: str) -> SegmentationStep:
    name = _str(name, where)
    if "/" in name:
        raise ConfigError(f"{where}: segmentation names cannot contain '/'.")
    section = _mapping(value, where)
    if "expand" in section:
        _check_keys(section, where, required={"expand", "distance_um"}, optional=set())
        return SegmentationStep(
            labels=name,
            expand=_str(section["expand"], f"{where}.expand"),
            distance_um=_positive_float(section["distance_um"], f"{where}.distance_um"),
        )
    _check_keys(section, where, required={"channels"}, optional={"diameter", "model"})
    diameter = section.get("diameter")
    return SegmentationStep(
        labels=name,
        channels=_cellpose_channels(section["channels"], f"{where}.channels"),
        diameter=None if diameter is None else _positive_float(diameter, f"{where}.diameter"),
        model=_str(section.get("model", SegmentationStep.model), f"{where}.model"),
    )


def _section(
    root: Mapping[str, Any],
    name: str,
    source: str,
    required: set[str] = frozenset(),
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    value = root.get(name)
    if value is None and not required:
        return {}
    section = _mapping(value, f"{source}: {name}")
    _check_keys(section, f"{source}: {name}", required=set(required), optional=set(optional))
    return section


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
    """A string with its ``${MACROSEG_...}`` variables replaced (see ``macroseg.paths``)."""
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


def _float(value: Any, where: str) -> float:
    # PyYAML reads `1e-5` (no dot) as a string, so accept numeric strings too.
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected a number, got {value!r}.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected a number, got {value!r}.") from None


def _optional_float(value: Any, where: str) -> float | None:
    return None if value is None else _float(value, where)


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


def _fraction(value: Any, where: str) -> float:
    number = _float(value, where)
    if not 0 <= number < 1:
        raise ConfigError(f"{where}: must be in [0, 1), got {number}.")
    return number


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where}: expected a positive integer, got {value!r}.")
    return value
