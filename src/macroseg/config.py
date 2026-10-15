"""Experiment configuration.

Everything problem-specific (marker names, Cellpose channel order, element names, training
parameters) lives in a YAML file and is loaded into the frozen dataclasses below. Code reads
these objects and never hard-codes marker or channel names.

Unknown keys are rejected so that typos in a config fail loudly instead of being ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
        optional={"candidates", "roi", "training", "split"},
    )

    image = _section(root, "image", source, required={"element"}, optional={"baseline_labels"})
    channels = _section(root, "channels", source, required={"cellpose", "candidate_marker"})
    annotation = _section(root, "annotation", source, required={"shapes_prefix", "class_label"})
    candidates = _section(root, "candidates", source, optional={"min_mean_intensity"})
    roi = _section(root, "roi", source, optional={"size_px"})
    training = _section(
        root, "training", source, optional={"learning_rate", "weight_decay", "n_epochs", "batch_size"}
    )
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
            val_fraction=_fraction(split.get("val_fraction", SplitConfig.val_fraction), f"{source}: split.val_fraction"),
            test_fraction=_fraction(
                split.get("test_fraction", SplitConfig.test_fraction), f"{source}: split.test_fraction"
            ),
        ),
    )

    if config.split.val_fraction + config.split.test_fraction >= 1:
        raise ConfigError(f"{source}: split.val_fraction + split.test_fraction must be < 1.")
    return config


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
