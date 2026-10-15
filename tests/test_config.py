from __future__ import annotations

import copy
from pathlib import Path

import pytest

from macroseg.config import ConfigError, ExperimentConfig, load_config, parse_config

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "example_macrophages.yaml"

MINIMAL = {
    "name": "exp",
    "target": "macrophages",
    "image": {"element": "img"},
    "channels": {"cellpose": ["F4/80", "DAPI"], "candidate_marker": "F4/80"},
    "annotation": {"shapes_prefix": "macrophages_manual", "class_label": "macrophage"},
}


def _with(**overrides):
    raw = copy.deepcopy(MINIMAL)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition("__")
        if key:
            raw.setdefault(section, {})[key] = value
        else:
            raw[section] = value
    return raw


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.channels.cellpose == ("F4/80", "DAPI")
    assert cfg.channels.candidate_marker == "F4/80"
    assert cfg.candidates.min_mean_intensity is None
    assert cfg.training.learning_rate == pytest.approx(1e-5)


def test_minimal_config_uses_defaults():
    cfg = parse_config(MINIMAL)
    assert cfg.image.baseline_labels == "baseline_cellpose"
    assert cfg.roi.size_px == 1024
    assert cfg.training.n_epochs == 100
    assert cfg.split.group_by == "sample"


def test_scientific_notation_without_dot_is_accepted():
    # PyYAML parses `1e-5` as a string; the loader must still read it as a float.
    cfg = parse_config(_with(training__learning_rate="1e-5"))
    assert cfg.training.learning_rate == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({k: v for k, v in MINIMAL.items() if k != "channels"}, "missing required key(s): channels"),
        (_with(chanels={}), "unknown key(s): chanels"),
        (_with(training__epochs=10), "unknown key(s): epochs"),
        (_with(channels__cellpose=["A", "B", "C", "D"]), "at most 3 channels"),
        (_with(channels__cellpose=["DAPI", "DAPI"]), "must be unique"),
        (_with(channels__cellpose=[]), "non-empty list"),
        (_with(roi__size_px=0), "positive integer"),
        (_with(training__n_epochs=True), "positive integer"),
        (_with(training__learning_rate="fast"), "expected a number"),
        (_with(split__val_fraction=0.5, split__test_fraction=0.5), "must be < 1"),
        (_with(name=""), "non-empty string"),
        ([], "expected a mapping"),
    ],
)
def test_invalid_configs_are_rejected(raw, message):
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_config(raw)


def test_config_is_immutable():
    cfg = parse_config(MINIMAL)
    with pytest.raises(AttributeError):
        cfg.name = "other"  # type: ignore[misc]


def test_invalid_yaml_reports_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
