"""The steps of the annotation loop as commands, for a cluster job or a terminal (AnnoLoop.md, "One round").

Every command takes the global settings file, and the round file when it acts on a round (``annoloop.settings``),
asks nothing, logs what it does, and can be run again: what is done is skipped, and what would be unsafe is refused
before any GPU work.

- ``prepare``: for each image of the round: ingest the raw file into ``<runs>/<image>.zarr`` (once), run the
  segmentation that proposes the masks (the starting one, or the round's fine-tuned model; once per store), and
  export one AnnoCell folder for the image's annotator and one for each of its other annotators
  (``annoloop.annotation.export_round``; a folder that exists is kept, with its corrections). ``--dry-run`` only
  says what would be done.
- ``close``: close the round for every annotator (``annoloop.annotation.close_round``).
- ``train``: fine-tune on the complete images of the round (``annoloop.train``), saved as ``<models>/model_r<NN>``,
  the name a later round file gives as its ``model``. By default from the starting model (``start.model``) on every
  round so far (AnnoLoop.md, D14); with ``--from <model>``, sequentially: that model, continued on this round's
  images only (``model_r<NN>_seq``). A model that exists is not trained again.
- ``score``: score a model (``--model <name>``) or the starting model (``--start``) on the test images, written to
  ``<annotations>/scores/<model>.json``.
- ``stats``: the corrections per round (``--by-annotator``: per round and annotator), from AnnoCell's reviews.

Usage::

    annoloop prepare configs/annoloop.yaml configs/rounds/round_01.yaml [--device cuda:0] [--dry-run]
    annoloop close configs/annoloop.yaml configs/rounds/round_01.yaml
    annoloop train configs/annoloop.yaml configs/rounds/round_01.yaml [--from model_r01] [--name N] [--device cuda:0]
    annoloop score configs/annoloop.yaml (--model model_r01 | --start) [--device cuda:0]
    annoloop stats configs/annoloop.yaml [--by-annotator] [--csv stats.csv]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from annoloop.segmentation import Segmentation, from_model
from annoloop.settings import GlobalSettings, RoundImage, RoundSettings, load_round, load_settings

log = logging.getLogger("annoloop")


def proposal(settings: GlobalSettings, round_settings: RoundSettings) -> Segmentation:
    """The segmentation that proposes the masks of the round: the starting one (``start``), or the round's
    fine-tuned ``model`` (``annoloop.segmentation.from_model``)."""
    if round_settings.model is None:
        return settings.start
    folder = Path(settings.paths.models) / round_settings.model
    if not (folder / "model.json").exists():
        raise FileNotFoundError(f"{round_settings.source}: model {round_settings.model!r}: no {folder}/model.json.")
    return from_model(folder, diameter=settings.start.diameter)


def store_path(settings: GlobalSettings, image: RoundImage) -> Path:
    return Path(settings.paths.runs) / f"{image.name}.zarr"


def check_round(settings: GlobalSettings, round_settings: RoundSettings) -> Segmentation:
    """Everything that can be checked before any work: returns the proposing segmentation, or raises.

    Refused: a closed round; an image already given in this round to someone who is neither its annotator nor one
    of its other annotators, or given in another round (each image is annotated in one round, D17); a missing raw
    file (unless its store exists); a missing model; a segmentation on channels that are not defined."""
    from annocell import folder as af

    from annoloop.annotation import given_in_other_rounds

    step = proposal(settings, round_settings)
    short = list(settings.channels.values())
    missing = [c for c in step.channels if c not in short]
    if missing:
        raise ValueError(f"{step.labels}: channels {missing} are not among the channels {short} of {settings.source}.")

    problems = []
    existing = [r for r in af.images(settings.paths.annotations) if r.round == round_settings.round]
    if any(af.is_closed(af.read_record(r.folder)) for r in existing):
        problems.append(f"round {round_settings.round} is closed: write a new round file")
    for image in round_settings.images:
        others = sorted(
            r.annotator
            for r in existing
            if r.name == image.name and r.annotator not in (image.annotator, *image.other_annotators)
        )
        if others:
            problems.append(f"{image.name} is already given to {others} in round {round_settings.round}")
        before = given_in_other_rounds(settings.paths.annotations, image.name, round_settings.round)
        if before:
            problems.append(f"{image.name} was already given in {before}: each image is annotated once")
        if not store_path(settings, image).exists() and not Path(image.path).is_file():
            problems.append(f"{image.name}: no raw file {image.path}")
    if problems:
        raise ValueError(f"{round_settings.source}: " + "; ".join(problems) + ".")
    return step


def prepare_round(
    settings: GlobalSettings,
    round_settings: RoundSettings,
    dry_run: bool = False,
    model: Callable | None = None,
    **model_kwargs,
) -> list[Path]:
    """Ingest, segment and export every image of the round (see the module docstring); returns the annotators'
    folders (all of them: exported now or kept from an earlier run). ``dry_run``: check and log only.
    ``model``: a segmentation function instead of Cellpose (tests); ``model_kwargs`` go to Cellpose
    (e.g. ``device="cuda:0"``)."""
    from annocell import folder as af

    step = check_round(settings, round_settings)
    log.info(
        "Round %d: %d image(s) for %d annotator(s), masks proposed by %s on %s.",
        round_settings.round,
        len(round_settings.images),
        len(round_settings.by_annotator()),
        step.labels,
        list(step.channels),
    )
    folders = []
    for image in round_settings.images:
        store = store_path(settings, image)
        root = settings.paths.annotations
        targets = {
            a: af.round_folder(root, a, round_settings.round) / image.name
            for a in (image.annotator, *image.other_annotators)
        }
        existed = {a: (folder / af.RECORD).exists() for a, folder in targets.items()}
        if dry_run:
            log.info("%s: store %s (%s)", image.name, store, "exists" if store.exists() else "to ingest")
            for annotator, folder in targets.items():
                log.info("%s: %s (%s)", image.name, folder, "exists, kept" if existed[annotator] else "to export")
            folders += targets.values()
            continue
        sdata = _store(settings, image)
        sdata = _segmentation(sdata, settings, image, step, model, **model_kwargs)
        exports = _export(sdata, image, settings, round_settings, step, image.annotator, image.other_annotators)
        for annotator, folder in exports.items():
            log.info("%s: %s %s", image.name, "kept" if existed[annotator] else "exported", folder)
            folders.append(folder)
    return folders


def close(settings: GlobalSettings, round_settings: RoundSettings) -> list[Path]:
    from annoloop.annotation import close_round

    folders = close_round(settings.paths.annotations, round_settings.round)
    log.info("Round %d closed: %d image folder(s).", round_settings.round, len(folders))
    return folders


def model_name(round_number: int, sequential: bool = False) -> str:
    """The name of the model trained on round ``round_number``: ``model_r<NN>``, ``model_r<NN>_seq`` if it continues
    an earlier model."""
    return f"model_r{round_number:02d}" + ("_seq" if sequential else "")


def train_round(
    settings: GlobalSettings,
    round_settings: RoundSettings,
    from_model: str | None = None,
    name: str | None = None,
    device=None,
) -> Path:
    """Fine-tune on the complete images of the round and return the model's folder (see the module docstring).

    ``from_model``: continue that model (in ``paths.models``) on this round's images only, instead of starting
    from ``start.model`` with every round so far. ``device``: a ``torch.device`` (default: automatic)."""
    from annocell import folder as af

    from annoloop.train import load_model_record, train, training_set

    models, root, number = Path(settings.paths.models), settings.paths.annotations, round_settings.round
    name = name or model_name(number, sequential=from_model is not None)
    folder = models / name
    if (folder / "model.json").exists():
        log.info("Model %s exists (%s): not trained again.", name, folder)
        return folder
    if folder.exists():  # no model.json: a training that did not finish
        log.warning("Removing %s: an unfinished training.", folder)
        shutil.rmtree(folder)

    if from_model is None:
        start_model, rounds = settings.start.model, list(range(1, number + 1))
    else:
        if not (models / from_model / "model.json").exists():
            raise FileNotFoundError(f"--from {from_model}: no {models / from_model}/model.json.")
        record = load_model_record(models / from_model)
        if list(record["channels"]) != list(settings.start.channels):
            raise ValueError(f"{from_model} learnt from {record['channels']}, not {list(settings.start.channels)}.")
        start_model, rounds = str(models / from_model / record["name"]), [number]
    still_open = sorted(
        {r.round for r in af.images(root) if r.round in rounds and not af.is_closed(af.read_record(r.folder))}
    )
    if still_open:
        log.warning("Round(s) %s not closed: training on the images complete now.", still_open)

    data = training_set(root, channels=settings.start.channels, rounds=rounds)
    if data.tiles.empty:
        raise ValueError(f"No training tile in round(s) {rounds}: the complete images are smaller than half a tile.")
    log.info(
        "Training %s from %s: %d image(s) of round(s) %s, %d tiles, %d cells.",
        name,
        from_model or start_model,
        data.tiles[["round", "annotator", "image"]].drop_duplicates().shape[0],
        sorted(set(data.tiles["round"])),
        len(data.tiles),
        int(data.tiles["cells"].sum()),
    )
    started = time.monotonic()
    folder = train(
        data,
        models,
        name,
        settings.training,
        start_model=start_model,
        device=device,
        extra={
            "from_model": from_model,
            "rounds": sorted(int(r) for r in set(data.tiles["round"])),
            "settings": {"global": settings.source, "round": round_settings.source},
        },
    )
    log.info("Trained %s in %.0f s: %s", name, time.monotonic() - started, folder)
    return folder


def score_on_test_images(settings: GlobalSettings, model: str | None = None, device=None) -> dict:
    """Score ``model`` (in ``paths.models``), or the starting model if None (round 0), on the test images."""
    from annoloop.train import score_model, score_test_images

    kwargs = {} if device is None else {"device": device}
    root, start = settings.paths.annotations, settings.start
    if model is None:
        record = score_test_images(root, start.model, start.model, 0, start.channels, **kwargs)
    else:
        folder = Path(settings.paths.models) / model
        if not (folder / "model.json").exists():
            raise FileNotFoundError(f"Model {model}: no {folder}/model.json.")
        record = score_model(root, folder, **kwargs)
    for test in record["images"]:
        values = ", ".join(f"{k} {v}" for k, v in test.items() if k.startswith("AP@"))
        log.info("%s on %s (%s): %s", record["model"], test["image"], test["annotator"], values)
    return record


def stats(settings: GlobalSettings, by_annotator: bool = False):
    """The corrections per round, or per round and annotator (``annoloop.annotation.summarize_rounds``)."""
    from annoloop.annotation import load_annotations, summarize_rounds

    by = ("round", "annotator") if by_annotator else ("round",)
    return summarize_rounds(load_annotations(settings.paths.annotations), by=by)


def _store(settings: GlobalSettings, image: RoundImage):
    """The image's store, ingested from its raw file if it does not exist yet. An existing store made from another
    file, level or crop is refused: the image name is already taken."""
    from spatialdata import read_zarr

    from annoloop.ingest import ingest

    store = store_path(settings, image)
    if store.exists():
        sdata = read_zarr(store)
        made_from = (sdata.attrs.get("source"), sdata.attrs.get("level"), sdata.attrs.get("crop"))
        wanted = (image.path, settings.level, list(image.crop) if image.crop else None)
        if made_from != wanted:
            raise ValueError(
                f"{store} was made from {made_from} (source, level, crop), not {wanted}: "
                "give the image another name, or remove the store."
            )
        log.info("%s: store exists, %s", image.name, store)
        return sdata
    started = time.monotonic()
    # Written under another name and renamed when complete, so a job killed halfway leaves no store to trust.
    partial = store.with_name(f"{image.name}.partial.zarr")
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    ingest(image.path, settings.element, settings.channels, settings.level, image.crop).write(partial)
    partial.rename(store)
    log.info("%s: ingested into %s in %.0f s", image.name, store, time.monotonic() - started)
    return read_zarr(store)


def _segmentation(sdata, settings: GlobalSettings, image: RoundImage, step: Segmentation, model, **model_kwargs):
    """``step`` in the store, run unless it is there already: its intensity table is written last, so a labels
    element without one is from an interrupted run and is redone."""
    from annoloop.segmentation import measure, segment

    if step.labels in sdata.labels and f"{step.labels}_intensities" in sdata.tables:
        log.info("%s: segmentation %s exists", image.name, step.labels)
        return sdata
    available = [str(c) for c in sdata.images[settings.element].c.values]
    missing = [c for c in step.channels if c not in available]
    if missing:
        raise ValueError(f"{image.name}: channels {missing} of {step.labels} are not in the store ({available}).")
    started = time.monotonic()
    sdata = segment(sdata, settings.element, step, model=model, **model_kwargs)
    sdata = measure(sdata, settings.element, step.labels)
    log.info("%s: segmented with %s in %.0f s", image.name, step.labels, time.monotonic() - started)
    return sdata


def _export(
    sdata,
    image: RoundImage,
    settings: GlobalSettings,
    round_settings: RoundSettings,
    step,
    annotator: str,
    other_annotators: tuple[str, ...],
):
    from annoloop.annotation import export_round

    display = settings.display
    return export_round(
        sdata,
        image.name,
        settings.element,
        step,
        settings.paths.annotations,
        round_settings.round,
        annotator,
        other_annotators,
        visible_channels=display.visible_channels,
        colours=display.colours,
        models_root=settings.paths.models,
        settings={"global": settings.source, "round": round_settings.source, "model": round_settings.model},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="annoloop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    texts = {
        "prepare": "Ingest, segment and export the images of a round.",
        "close": "Close a round.",
        "train": "Fine-tune a model on the complete images of a round.",
        "score": "Score a model on the test images.",
        "stats": "The corrections per round, from AnnoCell's reviews.",
    }
    parsers = {}
    for name, text in texts.items():
        parsers[name] = command = commands.add_parser(name, help=text, description=text)
        command.add_argument("settings", type=Path, help="The global settings file.")
        if name in ("prepare", "close", "train"):
            command.add_argument("round", type=Path, help="The round file.")
        if name in ("prepare", "train", "score"):
            command.add_argument("--device", help="Torch device, e.g. cuda:0, mps or cpu (default: automatic).")
    parsers["prepare"].add_argument(
        "--dry-run", action="store_true", help="Check the round and say what would be done."
    )
    parsers["train"].add_argument("--from", dest="from_model", help="Continue this model on the round's images only.")
    parsers["train"].add_argument("--name", help="The model's name (default model_r<NN>, or model_r<NN>_seq).")
    which = parsers["score"].add_mutually_exclusive_group(required=True)
    which.add_argument("--model", help="A model in paths.models.")
    which.add_argument("--start", action="store_true", help="The starting model (round 0).")
    parsers["stats"].add_argument("--by-annotator", action="store_true", help="Per round and annotator.")
    parsers["stats"].add_argument("--csv", type=Path, help="Also write the table to this file.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings(args.settings)
    round_settings = load_round(args.round, settings) if "round" in args else None
    if args.command == "prepare":
        kwargs = {"device": args.device} if args.device else {}
        prepare_round(settings, round_settings, dry_run=args.dry_run, **kwargs)
    elif args.command == "close":
        close(settings, round_settings)
    elif args.command == "train":
        train_round(settings, round_settings, args.from_model, args.name, _torch_device(args.device))
    elif args.command == "score":
        score_on_test_images(settings, None if args.start else args.model, _torch_device(args.device))
    else:
        table = stats(settings, by_annotator=args.by_annotator)
        print(table.to_string() if not table.empty else "No image exported yet.")
        if args.csv is not None:
            table.to_csv(args.csv)


def _torch_device(name: str | None):
    if name is None:
        return None
    import torch

    return torch.device(name)


if __name__ == "__main__":
    main()
