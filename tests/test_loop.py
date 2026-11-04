"""Settings files and the `annoloop` commands, on a small synthetic OME-TIFF with a fake segmentation model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml
from annocell import folder as af
from scipy import ndimage as ndi

from annoloop import settings as st
from annoloop.annotation import close_round
from annoloop.loop import check_round, main, prepare_round, proposal
from annoloop.settings import ConfigError

FILE_NAMES = ["DAPI C1 (DAPI C1)", "Na_K_ATPase (APC C2)", "F4_80 (APC C4)"]
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def raw(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 100, size=(3, 64, 80), dtype=np.uint16)
    for y, x in [(20, 20), (20, 60), (45, 40)]:
        image[:, y - 6 : y + 6, x - 6 : x + 6] += 1000
    folder = tmp_path / "input"
    folder.mkdir()
    for name in ("slide_a.ome.tif", "slide_b.ome.tif"):
        tifffile.imwrite(
            folder / name,
            image,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": FILE_NAMES}, "PhysicalSizeX": 0.17},
        )
    return folder


def _global(tmp_path, **changes):
    raw = {
        "paths": {k: str(tmp_path / k) for k in ("input", "runs", "annotations", "models")},
        "element": "img",
        "channels": {"DAPI C1 (DAPI C1)": "DAPI", "Na_K_ATPase (APC C2)": "NaK_ATPase", "F4_80 (APC C4)": "F480"},
        "start": {"labels": "start_cells", "channels": ["NaK_ATPase", "DAPI", "F480"]},
        "display": {"visible_channels": ["F480", "DAPI"], "colours": {"F480": "red", "DAPI": "blue"}},
    }
    return {**raw, **changes}


def _round(**changes):
    raw = {
        "round": 1,
        "images": {
            "img_a": {"path": "slide_a.ome.tif", "annotators": ["alice"]},
            "img_b": {"path": "slide_b.ome.tif", "crop": [0, 32, 0, 40], "annotators": ["bob"]},
        },
    }
    return {**raw, **changes}


def _load(tmp_path, global_raw=None, round_raw=None):
    settings = st.parse_settings(global_raw or _global(tmp_path), "annoloop.yaml")
    return settings, st.parse_round(round_raw or _round(), settings, "round_01.yaml")


class CountingModel:
    """Stand-in for Cellpose: (z, y, x, c) -> (z, y, x, 1) labels of the bright squares; counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, img, **_):
        self.calls += 1
        labels, _ = ndi.label(img[0, ..., 0] > 500)
        return labels[None, ..., None].astype(np.uint32)


def test_settings_load_and_resolve_image_paths(tmp_path):
    settings, round_settings = _load(tmp_path)
    assert settings.start.model == "cpsam_v2" and settings.start.channels == ("NaK_ATPase", "DAPI", "F480")
    assert settings.level == 0 and settings.training.n_epochs == 100
    assert settings.display.visible_channels == ("F480", "DAPI")
    [a, b] = round_settings.images
    assert a.path == str(tmp_path / "input" / "slide_a.ome.tif") and a.crop is None and b.crop == (0, 32, 0, 40)
    assert round_settings.model is None and round_settings.by_annotator() == {"alice": ["img_a"], "bob": ["img_b"]}
    assert st.ANNOTATOR.pattern == af.ANNOTATOR.pattern


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": 1}, "unknown key"),
        ({"start": {"labels": "s", "channels": ["CD68"]}}, r"\['CD68'\] not among the channels"),
        ({"start": {"labels": "s", "channels": ["DAPI", "F480", "NaK_ATPase", "DAPI"]}}, "at most 3"),
        ({"display": {"colours": {"CD68": "red"}}}, "not among the channels"),
        ({"element": "a/b"}, "letters, digits"),
        ({"channels": {"x": "DAPI", "y": "DAPI"}}, "must be unique"),
    ],
)
def test_global_settings_errors(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        st.parse_settings(_global(tmp_path, **change))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"images": {}}, "at least one image"),
        ({"images": {"img a": {"path": "x.tif", "annotators": ["alice"]}}}, "letters, digits"),
        ({"images": {"img": {"path": "x.tif", "annotators": ["Alice"]}}}, "lowercase"),
        ({"images": {"img": {"path": "x.tif", "annotators": ["al", "al"]}}}, "listed twice"),
        ({"images": {"img": {"path": "x.tif", "annotators": []}}}, "non-empty list"),
        ({"images": {"img": {"path": "x.tif"}}}, "missing required key"),
        ({"model": "start_cells"}, "starting segmentation"),
        ({"round": 0}, "positive integer"),
    ],
)
def test_round_settings_errors(tmp_path, change, message):
    settings = st.parse_settings(_global(tmp_path))
    with pytest.raises(ConfigError, match=message):
        st.parse_round(_round(**change), settings)


def test_example_settings_files_load(tmp_path, monkeypatch):
    for name in ("INPUT", "RUNS", "ANNOTATIONS", "MODELS"):
        monkeypatch.setenv(f"ANNOLOOP_{name}", str(tmp_path / name.lower()))
    settings = st.load_settings(CONFIGS / "annoloop.yaml")
    round_settings = st.load_round(CONFIGS / "rounds" / "round_01.yaml", settings)
    assert settings.start.labels == "membrane_dapi_f480" and round_settings.round == 1
    assert Path(round_settings.images[0].path).parent == tmp_path / "input"


def test_prepare_round_ingests_segments_and_exports(tmp_path, raw):
    settings, round_settings = _load(tmp_path)
    model = CountingModel()
    folders = prepare_round(settings, round_settings, model=model)
    root = Path(settings.paths.annotations)
    assert folders == [root / "alice" / "round_01" / "img_a", root / "bob" / "round_01" / "img_b"]
    assert (tmp_path / "runs" / "img_a.zarr").is_dir() and not (tmp_path / "runs" / "img_a.partial.zarr").exists()
    record = af.read_record(folders[0])
    assert record["annotator"] == "alice" and record["round"] == 1 and record["image"] == "img_a"
    assert record["channels"] == ["DAPI", "NaK_ATPase", "F480"]
    assert record["visible_channels"] == ["F480", "DAPI"] and record["colours"] == {"F480": "red", "DAPI": "blue"}
    assert record["start_segmentation"]["channels"] == ["NaK_ATPase", "DAPI", "F480"]
    assert (
        record["start_segmentation"]["model"] == "cpsam_v2" and record["start_segmentation"]["labels"] == "start_cells"
    )
    assert record["settings"] == {"global": "annoloop.yaml", "round": "round_01.yaml", "model": None}
    assert af.read_start(folders[0]).max() == 3
    assert af.read_image(folders[1])[0].shape == (3, 32, 40) and record["format_version"] == af.FORMAT_VERSION

    # Run again: nothing is ingested, segmented or exported again, and corrections are kept.
    calls = model.calls
    af.save(folders[0], af.read_start(folders[0]), {})
    assert prepare_round(settings, round_settings, model=model) == folders
    assert model.calls == calls and (folders[0] / af.MASKS).exists()


def test_dry_run_does_nothing(tmp_path, raw):
    settings, round_settings = _load(tmp_path)
    folders = prepare_round(settings, round_settings, dry_run=True, model=CountingModel())
    assert len(folders) == 2 and not (tmp_path / "runs").exists() and not (tmp_path / "annotations").exists()


def test_check_round_refuses_before_any_work(tmp_path, raw):
    settings, round_settings = _load(tmp_path)
    shared = _round(images={"img_a": {"path": "slide_a.ome.tif", "annotators": ["alice", "bob"]}})
    with pytest.raises(ValueError, match="shared by"):
        check_round(settings, st.parse_round(shared, settings))
    missing = _round(images={"img_c": {"path": "nowhere.tif", "annotators": ["alice"]}})
    with pytest.raises(ValueError, match="no raw file"):
        check_round(settings, st.parse_round(missing, settings))

    prepare_round(settings, round_settings, model=CountingModel())
    moved = _round(images={"img_a": {"path": "slide_a.ome.tif", "annotators": ["carla"]}})
    with pytest.raises(ValueError, match=r"already given to \['alice'\]"):
        check_round(settings, st.parse_round(moved, settings))
    again = _round(round=2, images={"img_a": {"path": "slide_a.ome.tif", "annotators": ["alice"]}})
    with pytest.raises(ValueError, match=r"img_a was already given in round 1 \(alice\)"):
        check_round(settings, st.parse_round(again, settings))
    close_round(settings.paths.annotations, 1)
    with pytest.raises(ValueError, match="round 1 is closed"):
        check_round(settings, round_settings)


def test_a_store_name_is_not_reused_for_another_image(tmp_path, raw):
    settings, round_settings = _load(tmp_path)
    prepare_round(settings, round_settings, model=CountingModel())
    shutil.rmtree(settings.paths.annotations)  # the store stays, e.g. from a run whose round file was then changed
    other = _round(round=2, images={"img_a": {"path": "slide_b.ome.tif", "annotators": ["alice"]}})
    with pytest.raises(ValueError, match="give the image another name"):
        prepare_round(settings, st.parse_round(other, settings), model=CountingModel())


def test_a_later_round_is_proposed_by_its_model(tmp_path, raw):
    settings, _ = _load(tmp_path)
    folder = tmp_path / "models" / "model_r1"
    folder.mkdir(parents=True)
    (folder / "model.json").write_text(json.dumps({"name": "model_r1", "channels": ["NaK_ATPase", "DAPI", "F480"]}))
    round_2 = st.parse_round(_round(round=2, model="model_r1"), settings, "round_02.yaml")
    step = proposal(settings, round_2)
    assert step.labels == "model_r1" and step.model == str(folder / "model_r1")
    [first, _] = prepare_round(settings, round_2, model=CountingModel())
    record = af.read_record(first)
    assert record["start_segmentation"]["labels"] == "model_r1" and record["settings"]["model"] == "model_r1"
    with pytest.raises(FileNotFoundError, match="model.json"):
        proposal(settings, st.parse_round(_round(model="nothing"), settings))


def test_command_line(tmp_path, raw, caplog):
    settings_file, round_file = tmp_path / "annoloop.yaml", tmp_path / "round_01.yaml"
    settings_file.write_text(yaml.safe_dump(_global(tmp_path)))
    round_file.write_text(yaml.safe_dump(_round()))
    with caplog.at_level("INFO", logger="annoloop"):
        main(["prepare", str(settings_file), str(round_file), "--dry-run"])
    assert "to export" in caplog.text and not (tmp_path / "annotations").exists()
    with pytest.raises(ValueError, match="No image"):
        main(["close", str(settings_file), str(round_file)])


def _complete_all(folders):
    """What AnnoCell does when an annotator checks every cell and presses Done."""
    for folder in folders:
        masks = af.read_start(folder)
        cells = af.sync_cells({}, masks, masks)
        for cell in cells.values():
            cell.mark(af.OK)
        af.save(folder, masks, cells, done=True)


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Cellpose training and evaluation without weights or GPU; records what each training was given."""
    import functools
    import types

    import cellpose.models
    import cellpose.train

    from annoloop import train as train_module

    trainings = []

    def train_seg(net, train_data, train_labels, save_path, model_name, **kwargs):
        trainings.append({"start": net, "tiles": len(train_data), "n_epochs": kwargs["n_epochs"]})
        weights = save_path / "models" / model_name
        weights.parent.mkdir()
        weights.write_text("weights")
        return weights, np.array([1.0]), np.array([0.0])

    def model(pretrained_model=None, **_):
        def evaluate(image, **__):
            labels, _ = ndi.label(image[0] > 500)
            return labels, None, None

        return types.SimpleNamespace(net=pretrained_model, eval=evaluate)

    monkeypatch.setattr(cellpose.models, "CellposeModel", model)
    monkeypatch.setattr(cellpose.train, "train_seg", train_seg)
    monkeypatch.setattr(train_module, "training_set", functools.partial(train_module.training_set, tile_px=32))
    return trainings


def test_train_from_the_start_model_on_every_round_or_sequentially(tmp_path, raw, fake_cellpose):
    from annoloop.loop import train_round

    settings, round_1 = _load(tmp_path)
    _complete_all(prepare_round(settings, round_1, model=CountingModel()))
    close_round(settings.paths.annotations, 1)
    first = train_round(settings, round_1)
    assert first == tmp_path / "models" / "model_r01" and fake_cellpose[-1]["start"] == "cpsam_v2"
    record = json.loads((first / "model.json").read_text())
    assert (
        record["rounds"] == [1]
        and record["from_model"] is None
        and record["channels"] == ["NaK_ATPase", "DAPI", "F480"]
    )
    assert train_round(settings, round_1) == first and len(fake_cellpose) == 1  # exists: not trained again

    round_2 = st.parse_round(
        _round(round=2, model="model_r01", images={"img_c": {"path": "slide_a.ome.tif", "annotators": ["carla"]}}),
        settings,
    )
    _complete_all(prepare_round(settings, round_2, model=CountingModel()))
    base = train_round(settings, round_2)  # from cpsam_v2, rounds 1 and 2
    assert base.name == "model_r02" and fake_cellpose[-1]["start"] == "cpsam_v2"
    assert json.loads((base / "model.json").read_text())["rounds"] == [1, 2]
    sequential = train_round(settings, round_2, from_model="model_r01")  # model_r01, continued on round 2 only
    assert sequential.name == "model_r02_seq" and fake_cellpose[-1]["start"] == str(first / "model_r01")
    record = json.loads((sequential / "model.json").read_text())
    assert record["rounds"] == [2] and record["from_model"] == "model_r01"
    assert fake_cellpose[-1]["tiles"] < fake_cellpose[-2]["tiles"]
    with pytest.raises(FileNotFoundError, match="--from nothing"):
        train_round(settings, round_2, from_model="nothing", name="other")


def test_score_and_stats(tmp_path, raw, fake_cellpose):
    from annoloop.annotation import export_test
    from annoloop.loop import score_on_test_images, stats, train_round

    settings, round_1 = _load(tmp_path)
    folders = prepare_round(settings, round_1, model=CountingModel())
    _complete_all(folders[:1])
    table = stats(settings, by_annotator=True)
    assert list(table.index) == [(1, "alice"), (1, "bob")] and list(table["complete"]) == [1, 0]
    assert stats(settings).loc[1, "images"] == 2

    from spatialdata import read_zarr

    sdata = read_zarr(tmp_path / "runs" / "img_a.zarr")
    sdata.attrs.update(source="another_slide.tif")
    region = export_test(sdata, "slide_c", "img", [0, 64, 0, 80], settings.paths.annotations, "expert")
    truth, _ = ndi.label(af.read_image(region)[0][0] > 500)
    cells = af.sync_cells({}, truth, np.zeros_like(truth))
    for cell in cells.values():
        cell.mark(af.OK)
    af.save(region, truth.astype(np.uint32), cells, done=True)  # the expert pressed Done

    start = score_on_test_images(settings)
    assert start["model"] == "cpsam_v2" and start["round"] == 0 and start["images"][0]["AP@0.5"] == 1.0
    train_round(settings, round_1)
    assert score_on_test_images(settings, "model_r01")["round"] == 1
    assert [s["model"] for s in af.read_scores(settings.paths.annotations)] == ["cpsam_v2", "model_r01"]
    with pytest.raises(FileNotFoundError, match="no "):
        score_on_test_images(settings, "nothing")
