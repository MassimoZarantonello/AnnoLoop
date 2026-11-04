# AnnoLoop

The annotation loop for fine-tuning Cellpose to find **macrophages** in MACSima images. Annotators correct a
proposed segmentation in [AnnoCell](https://github.com/MassimoZarantonello/AnnoCell); AnnoLoop prepares what they
correct, collects their corrections, trains Cellpose on them, and proposes the next round with the new model. It is
built on [SpatialData](https://spatialdata.scverse.org/) and [Harpy](https://harpy.readthedocs.io/).

Scope, decisions and open questions are in `AnnoLoop.md`. AnnoLoop started as a copy of macroseg but is now
independent of it: it neither imports nor installs macroseg, and the two can live side by side.

## Install

AnnoLoop reads and writes the annotation folders through AnnoCell's folder format (`annocell.folder`). AnnoCell must
be cloned **next to this repository**, in the same parent folder (`../AnnoCell`):

```bash
git clone git@github.com:MassimoZarantonello/AnnoCell.git ../AnnoCell
```

Python 3.13 (`pylibczirw`, the CZI reader, has no 3.14 wheel yet):

```bash
uv sync --python 3.13 --extra imaging --extra dev
```

```bash
cp .env.example .env
```

`.env` holds the paths of this machine (not in git): `ANNOLOOP_INPUT` (raw images, read only), `ANNOLOOP_RUNS`
(`.zarr` stores), `ANNOLOOP_ANNOTATIONS` (the annotators' folders) and `ANNOLOOP_MODELS` (fine-tuned models). A
variable set in the environment wins over `.env`.

```bash
.venv/bin/python -m pytest
```

## The loop: `annoloop`

One command per step. Each takes the global settings file and, when it works on a round, the round file. Nothing is
interactive, so every step can run as a cluster job and leave a log. Every step can be run again: work already done
is skipped, and anything unsafe is refused before any GPU work starts.

| Step | Command | Where |
|---|---|---|
| 1. Draft the round file: which images go to which annotator | by hand, or AnnOverview's round planner | anywhere |
| 2. Ingest, segment, export the images | `annoloop prepare <settings> <round>` | cluster |
| 3. Annotate | AnnoCell, on the annotators' own computers | |
| 4. Watch progress | AnnOverview (read only) | |
| 5. Close the round | `annoloop close <settings> <round>` | anywhere |
| 6. Statistics | `annoloop stats <settings>` | anywhere |
| 7. Train | `annoloop train <settings> <round>` | cluster |
| 8. Score on the test images | `annoloop score <settings> --model model_r<NN>` | cluster |
| 9. Next round | a new round file with `model: model_r<NN>`, back to step 2 | |

`annoloop --help` and `annoloop <command> --help` list every option. Without the installed script, the same is
`.venv/bin/python -m annoloop.loop`.

### Settings files

**Global** ([configs/annoloop.yaml](configs/annoloop.yaml)) holds what must not change between rounds:

```yaml
paths:
  input: ${ANNOLOOP_INPUT}              # round files give image paths relative to this
  runs: ${ANNOLOOP_RUNS}                # one store per image: <runs>/<image>.zarr
  annotations: ${ANNOLOOP_ANNOTATIONS}  # <annotations>/<annotator>/round_<NN>/<image>/
  models: ${ANNOLOOP_MODELS}            # <models>/<model>/
element: macsima_image                  # the image element in each store
level: 0                                # pyramid level read from the raw files
channels:                               # raw channel name -> short name; unlisted channels are dropped
  "DAPI C1 (DAPI C1)": DAPI
  "Na_K_ATPase (APC C2)": NaK_ATPase
  "F4_80 (APC C4)": F480
start:                                  # the starting segmentation
  labels: membrane_dapi_f480            # its labels element in each store
  model: cpsam_v2
  channels: [NaK_ATPase, DAPI, F480]    # what every model learns from (at most 3)
display:                                # what AnnoCell shows first; annotators change it themselves
  visible_channels: [F480, NaK_ATPase, DAPI]
  colours: {F480: red, NaK_ATPase: green, DAPI: blue}
training: {learning_rate: 1.0e-5, weight_decay: 0.1, n_epochs: 100, batch_size: 1}
```

**Round** ([configs/rounds/round_01.yaml](configs/rounds/round_01.yaml)), one file per round:

```yaml
round: 1
model: null            # who proposes the masks: null = start; later a model in paths.models, e.g. model_r01
images:
  exp201_spc331:       # the image's name: its folder for the annotator, and its store
    path: Exp201_SPC331_261014_AVH_SUBSET_SingleTiff.tif   # relative to paths.input
    crop: null         # optional [y0, y1, x0, x1] in pixels
    annotators: [annotator-1]
```

Both files are checked strictly before anything runs. Unknown or missing keys are errors, and so are channels that
are not defined in `channels`. Annotator names are lowercase letters, digits and `-`. Image names are letters,
digits, `_`, `.` and `-`.

### `prepare`: ingest, segment, export

```bash
annoloop prepare configs/annoloop.yaml configs/rounds/round_01.yaml --dry-run      # what would be done
annoloop prepare configs/annoloop.yaml configs/rounds/round_01.yaml --device cuda:0
```

First the whole round is checked. The command refuses any of these:

- a closed round;
- an image given to more than one annotator (shared images are not built yet);
- an image already given to someone else in this round;
- an image already given in an earlier round: each image is annotated once;
- a raw file that is missing;
- a model that is missing.

Then, for each image:

1. **Ingest** the raw file (OME-TIFF or CZI) into `<runs>/<image>.zarr`, once. The store is written as
   `<image>.partial.zarr` and renamed when it is complete, so a job killed halfway leaves nothing behind. If a
   store with that name already exists but was made from another file, level or crop, the command refuses.
2. **Segment** with the round's proposing model, once per store:
   - in round 1, the `start` segmentation (Cellpose-SAM on the three channels);
   - afterwards, the round's `model`, written as a labels element named after it.

   The segmentation is followed by an intensity table. A segmentation without its table comes from a run that was
   interrupted, so it is redone.
3. **Export** one AnnoCell folder for the annotator: `<annotations>/<annotator>/round_<NN>/<image>/`, holding:
   - `image.tif`: all channels, raw;
   - `start_masks.tif`: the proposed masks;
   - `annotation.json`: where the image comes from, and the settings it was exported with.

   A folder that already exists is kept as it is, with the annotator's corrections.

### `close`

```bash
annoloop close configs/annoloop.yaml configs/rounds/round_01.yaml
```

Sets `closed_on` in every image of the round. AnnoCell then opens them read-only, even an image that is open at that
moment. Images not finished stay out of training. An image counts as complete only when every cell is OK and the
annotator pressed Done.

For every complete image, `close` also scores the proposed masks against the corrected ones and records the result
as `start_score` in `annotation.json`. This should rise from round to round as the model improves.

### `stats`

```bash
annoloop stats configs/annoloop.yaml --by-annotator --csv stats.csv
```

Gives, per round (or per round and annotator):

- the number of annotators, images, and complete images;
- the corrections from AnnoCell's `review.json`: cells kept, edited, deleted, drawn new, and OK'd by hand or in bulk;
- the mean `start_score` of the complete images (written by `close`).

### `train`

```bash
annoloop train configs/annoloop.yaml configs/rounds/round_02.yaml                    # model_r02
annoloop train configs/annoloop.yaml configs/rounds/round_02.yaml --from model_r01   # model_r02_seq
```

- **By default** it starts from the base model (`start.model`) and trains on every complete image of rounds 1 to N.
  The model is saved as `<models>/model_r<NN>/`.
- **Sequential training** (`--from <model>`) continues that model on round N's images only, saved as
  `model_r<NN>_seq`. It is there to compare the two ways once all the annotations exist.

The complete images are cut into 512 px tiles. Tiles within 60 px of a test image are dropped. The training
settings come from the global `training` section.

Each model folder holds:

- the weights;
- `tiles.csv`, every tile used;
- `model.json`, with:
  - the model it started from, its channels, its images and rounds;
  - the training settings, the losses, the training loop used;
  - the settings files and the code version.

A model that already exists is not trained again, and an unfinished training is redone. A round that is still open is
trained on with a warning.

### `score`

```bash
annoloop score configs/annoloop.yaml --start             # the starting model, as round 0
annoloop score configs/annoloop.yaml --model model_r02
```

Segments every complete test image (the expert pressed Done) and scores the result against the expert's masks:
AP, precision and recall at IoU 0.5 and 0.75. Results go to `<annotations>/scores/<model>.json`, where AnnOverview
plots them round after round. Test images are exported with `annoloop.annotation.export_test`; there is no command
for that yet.

## The folders

```
<annotations>/
  <annotator>/
    round_01/<image>/      image.tif, start_masks.tif, compare/, annotation.json   (AnnoLoop)
                           masks.tif, cells.csv, review.json                       (AnnoCell)
    test/<image>/          a test image, drawn from scratch, never trained on
  scores/<model>.json      each model's score on the test images
<runs>/<image>.zarr        the image, its segmentations and their intensity tables
<models>/<model>/          weights, tiles.csv, model.json
```

Each file has one writer, so a save in AnnoCell and a `close` here never overwrite each other. The annotator's
computer needs only AnnoCell, never this repository. A plan for blending several annotators of one image is in
[docs/concurrent_annotations.md](docs/concurrent_annotations.md); it is not built.

## Modules

| Module | Does |
|---|---|
| `settings` | the global and round settings files |
| `loop` | the `annoloop` commands |
| `ingest` | OME-TIFF (lazy, any pyramid level, optional crop) or CZI → SpatialData, channels renamed |
| `segmentation` | the proposing segmentation (start model or fine-tuned), run chunk-wise through Harpy; intensities per cell |
| `annotation` | export an image or a test image, close a round (and score the proposals), read the annotations back, summarise them |
| `train` | training tiles from the AnnoCell folders, fine-tuning (`trainer=` takes another training loop), scoring on the test images |
| `accuracy` | AP, precision and recall against corrected masks |
| `paths`, `version` | `${ANNOLOOP_...}` paths from `.env`; the code version recorded in every output |

Choosing the starting segmentation (comparing channel sets, measures without ground truth) is done in macroseg: its
choice goes into `start` in the global settings.

## Next

- Cluster job scripts (SLURM), once the cluster and the storage are decided.
- Our own training loop: [docs/training_loop.md](docs/training_loop.md).
- Shared images (several annotators per image) and the progress measure across rounds.
