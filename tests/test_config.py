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


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """A private .env and a private copy of the environment, so tests neither read the machine's own
    .env nor leave variables behind for other tests."""
    import os

    from macroseg import paths

    monkeypatch.setattr(paths, "ENV_FILE", tmp_path / ".env")
    environ = {k: v for k, v in os.environ.items() if not k.startswith("MACROSEG_")}
    monkeypatch.setattr(os, "environ", environ)
    return tmp_path / ".env"


def test_paths_come_from_env_file(env_file):
    from macroseg import paths

    env_file.write_text("MACROSEG_RUNS=/data/runs\nMACROSEG_INPUT=raw\n")
    cfg = parse_config(_with(zarr="${MACROSEG_RUNS}/exp.zarr", input={"path": "${MACROSEG_INPUT}/slide.tif"}))
    assert cfg.zarr == "/data/runs/exp.zarr"
    assert cfg.input.path == str(paths.REPO / "raw" / "slide.tif")  # relative values start at the repo root


def test_environment_wins_over_env_file(env_file, monkeypatch):
    env_file.write_text("MACROSEG_RUNS=/from/file\n")
    monkeypatch.setenv("MACROSEG_RUNS", "/from/environment")
    assert parse_config(_with(zarr="${MACROSEG_RUNS}/exp.zarr")).zarr == "/from/environment/exp.zarr"


def test_missing_path_variable_is_reported(env_file):
    with pytest.raises(ConfigError, match=r"zarr: MACROSEG_RUNS is not set"):
        parse_config(_with(zarr="${MACROSEG_RUNS}/exp.zarr"))


def test_paths_without_variables_are_kept(env_file):
    assert parse_config(_with(zarr="/plain/exp.zarr")).zarr == "/plain/exp.zarr"
