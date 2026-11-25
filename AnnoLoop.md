# AnnoLoop

The pipeline of the annotation loop: it prepares what the annotators correct, collects what they corrected, trains
Cellpose on it, and proposes the next round. It runs as **scripts on a GPU cluster**; it has no notebooks (those stay in macroseg).

Status: **plan agreed; settings files and the `annoloop` commands (prepare, close, train, score, stats) built; cluster job scripts wait** (last revised 2026-11-04). This is the one
file for the scope, the decisions and what is open; update it whenever a decision is taken or reopened, so work can
continue in another chat from this file alone.

## Aim

Fine-tune Cellpose to find **macrophages** in MACSima images (mouse neuroblastoma and liver cancer; DAPI, NaK-ATPase,
F4/80 and others). Annotators correct a starting segmentation, a few whole images each per round; the corrected
images train the model, and the model proposes the next round. Stop when the corrections per round stop shrinking.

## The three tools

| Tool | Who and where | Does | Reads / writes |
|---|---|---|---|
| **AnnoCell** | the annotator, on their own computer (needs only AnnoCell, never AnnoLoop) | correct a proposed segmentation in napari, cell by cell, and record which cells were checked | writes `masks.tif`, `cells.csv`, `review.json` in the annotator's folder; never `annotation.json` |
| **AnnoLoop** (this repository) | the project lead; scripts on the cluster | prepare rounds, collect corrections, train, score, propose the next round | writes the image folders and closes rounds; reads the corrections through `annocell.folder` |
| **AnnOverview** | the project lead, read-only | which images each annotator has, their state and counts per round (no times, D8), how many images the next training takes, the model scores | only reads the shared folder, when the page opens and on Refresh |

They share one thing: the **folder layout** of AnnoCell (`annocell.folder`):
`<root>/<annotator>/round_<NN>/<image>/` with `image.tif` (all channels, raw), `start_masks.tif` (the proposal),
`compare/` (read-only masks of other segmentations), `annotation.json` (provenance, `closed_on`, `format_version`:
written by AnnoLoop only), and `masks.tif`, `cells.csv` and `review.json` (written by AnnoCell only). Test images go
in `<root>/<annotator>/test/<image>/`. One writer per file, so a save and a `close` never overwrite each
other.

### What AnnOverview relies on in AnnoLoop's output

AnnOverview has no code link to AnnoLoop; it reads the folders through `annocell.folder`. Its brief is
[../AnnOverview/AnnOverview.md](../AnnOverview/AnnOverview.md). Keep these true when changing the export (as of
2026-11-03):

- **Fields of `annotation.json` it reads:** `format_version`, `annotator`, `round`, `mode`, `source` (shown as the
  slide), `visible_channels`, `colours`, `closed_on`. A `format_version` other than the installed one is shown as a
  problem ("export it again"), never guessed at.
- **Shared image = the same image folder name in the same round** under several annotators: one primary
  (`"role": "primary"` in `annotation.json`, or no `role`) and the others (`"role": "other"`, D19). Only the primary
  version goes into training. AnnOverview still shows a shared image as complete only when every version is, and
  never into training: it must learn `role` (see "Next").
- **`image.tif` uncompressed**, as `annocell.folder.write_image` writes it. The picture is drawn from a memory map of
  a few rows per block (about 1/30 of a whole slide); a compressed file is read whole, which is slow over SMB.
- **Colours: the marker red (F4/80), the structure channels green and blue** (NaK-ATPase, DAPI), as in the
  `export_round` example. The pictures dim green to 70 % and blue to 50 % by colour, not by channel name, so other
  colours are drawn at full strength.
- **Scores:** `<root>/scores/<model>.json` with `round` (0 = the starting model) and `images: [{image, annotator,
  AP@...}]` (renamed from `regions` / `region` on 2026-11-04); every `AP@*` key is averaged over the test images.
- **Test images** are complete once the expert pressed Done (`is_complete`), like training images.

## Scope of AnnoLoop

**In scope**
1. **Ingest** a slide (OME-TIFF or CZI) into a SpatialData `.zarr`, channels renamed from the config.
2. **Starting segmentation** of each image: Cellpose-SAM on DAPI + NaK-ATPase + F4/80 (round 1), the latest
   fine-tuned model afterwards.
3. **Prepare a round**: give images to annotators, export one AnnoCell folder per image, record the settings used.
4. **Close a round**, and **collect** it: read the corrections and the review state back.
5. **Statistics from the raw files**, in a second step: cells drawn, edited, removed, and cleared by the "all OK"
   button, per image, round and annotator.
6. **Train**: fine-tune Cellpose on the complete images, recording the config and channels with the model.
7. **Score** a model (on the test set, later), and **propose** the next round with the new model.

**Out of scope**
- The annotation interface (AnnoCell) and the overview page (AnnOverview).
- Analysis of the final segmentation (feature extraction, classification of cells): that is
  [spatiato](https://github.com/vibspatial/spatialdata), a separate project; AnnoLoop only writes the segmentation
  into a SpatialData `.zarr`.
- Any monitoring of annotators (D8).

## How it runs

- **Scripts, not notebooks.** One command-line entry point per step, taking the global settings file and the round
  file (D9); no interaction, so it can run as a cluster (SLURM) job, with its log. Safe to re-run: a step that is done
  is skipped or says why it refuses.
- **No notebooks in AnnoLoop** (D16): notebooks stay in macroseg.
- **Where.** The cluster does the heavy steps: starting segmentation (about 8 minutes for a whole slide on a Mac
  GPU), training, whole-slide segmentation. The Mac is for development and small tests. Which cluster, and how the
  annotators' folders reach the cluster and the annotators' computers, is open (O6).
- **Nothing problem-specific in code**: marker names, channels, models and paths live in YAML and in `.env`
  (`ANNOLOOP_INPUT`, `ANNOLOOP_RUNS`, `ANNOLOOP_ANNOTATIONS`, `ANNOLOOP_MODELS`). Every trained model records the
  config and the channels it was trained with, and every exported image records the settings and the code version.
- The repository holds code and configs only; images, annotations and model weights live on storage.

## One round

| Step | Tool | Where | State |
|---|---|---|---|
| 1. Ingest and starting segmentation | AnnoLoop (`annoloop prepare`) | cluster | built 2026-11-04, part of step 2: one store per image, only the proposing segmentation |
| 2. Assign images, export folders | AnnoLoop (`annoloop prepare`) | cluster | built 2026-11-04: ingests, segments and exports each image of the round file |
| 3. Annotate | AnnoCell | annotators' computers | exists |
| 4. Watch progress | AnnOverview | project lead's Mac | exists |
| 5. Close the round, collect | AnnoLoop (`annoloop close`, `load_annotations`) | cluster | close is a command; collect is a function |
| 6. Statistics | AnnoLoop (`annoloop stats`) | anywhere | per round, or per round and annotator; O2 still to build |
| 7. Train, score | AnnoLoop (`annoloop train`, `annoloop score`) | cluster | built 2026-11-04 |
| 8. Next round | AnnoLoop | cluster | the new model segments the next images, back to step 2 |

## Decided

| # | Decision | From |
|---|---|---|
| D1 | Annotators get **whole images**, with the **raw channels**, unthresholded: they set contrast and display in napari themselves. Our F4/80 threshold and any statistics are never shown to them, so they judge macrophages from the raw signal | user, 2026-10-29 |
| D2 | A round gives **3–4 images per annotator**; **5 annotators** in round 1 (placeholder names, set in the settings file), so about 15–20 images. The annotators estimate about one hour per image; nobody prescribes or checks the time | user, 2026-10-29 |
| D3 | One **primary** annotator per image per round; an image may also have other annotators (D19) | tool rule + user |
| D4 | **Many images are coming**, with the same markers (maybe more; only DAPI, NaK-ATPase and F4/80 are used) | user, 2026-10-29 |
| D5 | The starting masks come from **Cellpose-SAM (`cpsam_v2`) on DAPI + NaK-ATPase + F4/80 together**, run on every image | user, 2026-10-29 |
| D6 | An image is complete when **every cell is OK and the annotator pressed Done** (AnnoCell refuses Done with cells not OK; any change takes it back); AnnoCell's "mark all remaining in view as OK" button (Shift+O) is allowed. An annotator can check the macrophages carefully and clear the rest quickly | user, 2026-10-29 |
| D7 | Record, per cell, **how it became OK**: by the button or individually (clicked, edited or drawn); the last action wins. Hand-drawn and edited counts already exist (`origin` in `cells.csv`). Built 2026-11-03: `ok_by` (`hand` / `bulk`) in `cells.csv`, counts `ok_hand` / `ok_bulk` in the review | user, 2026-10-29 |
| D8 | **No monitoring of annotators:** no time per cell or session, no view position, no history of corrections. The raw record is what they leave: `start_masks.tif`, `masks.tif`, `cells.csv`, `annotation.json`. Statistics are **computed afterwards, in a second step**, from those files, so a definition can change without redoing any annotation | user, 2026-10-29: "I don't want to check up on my colleagues" |
| D9 | **Settings in YAML, one global file and one per round**, like the segmentation configs. Global: what must not change between rounds (store paths, channels, starting model). Per round: round number, annotators, which images each gets. Each exported image records the values used. They hold our choices, not the annotators' display settings | user, 2026-10-29 |
| D10 | AnnoLoop **starts fresh**: nothing from macroseg's annotations or models is reused (not the corrected crop, not the round-1 model) | user, 2026-10-29 |
| D11 | Development is in **AnnoLoop**. macroseg only shows results to colleagues (its notebooks need not be kept in step). Steps are notebooks for now and scripts on the GPU cluster later: logic goes in modules a script can call | user, 2026-10-29 |
| D12 | The **test set is dealt with later**; possibly a smaller image annotated manually from scratch | user, 2026-10-30 |
| D14 | **Each round trains from the base model** (`start.model`, `cpsam_v2`) on every complete image of rounds 1..N: `annoloop train` makes `model_r<NN>`. Training **sequentially** (the previous model continued on round N's images only, `annoloop train --from model_r<NN-1>`, `model_r<NN>_seq`) is kept for later, to compare the two once all annotations exist | user, 2026-11-04 |
| D17 | **Each image is annotated in one round only**: `export_round` and `annoloop prepare` refuse an image already given in another round (by anyone). Two corrections of the same pixels cannot both go into training, and blending them is not built. Re-correcting an image with a better model, to measure progress on it, could be allowed later as a deliberate choice | user, 2026-11-04 |
| D16 | **No notebooks in AnnoLoop**: they stay in macroseg. **"Test image"**, not "test region", also in AnnoCell's API and AnnOverview (`af.images`, `af.ImageFolder`, `test_images_folder`, score files `images` / `image`) | user, 2026-11-04 |
| D15 | **The baseline work is not AnnoLoop's**: choosing the starting segmentation (which channels, measures without ground truth) is done in macroseg, and its choice goes into `start` in the global settings; AnnoLoop keeps only the one proposing segmentation per round. **Terminology: image**, since annotation and training are on whole images (test images stay regions) | user, 2026-11-04 |
| D18 | **Macrophages are marked by hand on test images only**, never on training images: in a test image the expert presses M on a cell to mark it a macrophage (M again unmarks it); every unmarked cell counts as not a macrophage (two values: no "unsure", as decided before), so marking adds no completeness rule; the expert decides from the raw signal (D1: no F4/80 threshold or pre-marking shown). It gives a ground truth for scoring models on macrophages alone (recall and IoU of the macrophages) and for checking the F4/80 threshold of the slide statistics. AnnoCell for training images stays as it is: no M key there (the mark was removed on 2026-11-03), and the mark is never used for training. Needs: AnnoCell (the M key and a per-cell mark in test mode only, a folder format change), AnnoLoop scoring (the user writes the macrophage metrics) | user, 2026-11-06 |
| D19 | **Shared images (was O1): one primary annotator and 0–n other annotators per image**, in the same round (`annotator` and `other_annotators` in the round file). Everyone gets the same image and proposal, one folder each; `annotation.json` records `role`: `primary` or `other`. **Only the primary's version goes into training**; the others' versions are never trained on and never blended: they are for measuring how annotators differ (annotator bias), an analysis kept separate and not built yet. Other annotators only in the primary's round (D17 unchanged); not for test images for now | user, 2026-11-09 |
| D13 | **Settings keys (was O5)**, in `annoloop.settings` and the examples `configs/annoloop.yaml`, `configs/rounds/round_01.yaml`. Global: `paths` (`input`, `runs`, `annotations`, `models`), `element`, `level`, `channels` (raw name -> short name), `start` (`labels`, `model`, `channels`, `diameter`), `display` (`visible_channels`, `colours`), `training`. Round: `round`, `model` (a fine-tuned model in `paths.models`, or null for `start`), `images: {<name>: {path, crop, annotator, other_annotators: [..]}}`; `path` relative to `paths.input`; `other_annotators` optional (D19). The image name is its folder for every annotator and its store `<runs>/<name>.zarr`. Each exported `annotation.json` gets `settings: {global, round, model}` | Claude, 2026-11-04: built, to be confirmed by the user |

## Open

| # | Question | Proposal | Against it |
|---|---|---|---|
| O1 | **Annotator bias** from the shared images (D19): how many images, by how many annotators, and the measure | 1–2 images per round given to all 5; agreement from matching cells by overlap (AP@0.5, Dice of matched cells), as in [docs/concurrent_annotations.md](docs/concurrent_annotations.md), per pair of annotators and against the proposal | each shared image costs 5 annotation sessions |
| O2 | Progress measure across rounds | from the raw files (D8): cells drawn or edited by hand, per round and annotator; possibly divided by the F4/80 blobs of the same image, which come from pixels and do not move with the model | blobs are a rough proxy for macrophages (about five cells each, ±30 %); bulk-OK'd cells must not dilute it (D7) |
| O3 | Which images go in which round | not decided; only one slide (Exp201) is here now | the statistics in "Slide statistics" could describe macrophage load per image, never shown to annotators (D1) |
| O4 | How many images are available now, and how large | unknown | decides how many rounds the first batch lasts |
| O6 | Which cluster, and how annotators' folders and the cluster's storage connect | open; earlier plans mention the IRC / VIB GPU servers and the VIB Data Core | decides paths, transfers and the job scripts |

## Facts this rests on

- Slide Exp201 SPC331: 6 channels, 7705 × 6272 px at 0.17 µm (about 1310 × 1066 µm). Its 3-channel segmentation
  (`membrane_dapi_f480`) has 21,639 cells; it was written into macroseg's whole-slide store
  (`data/runs/exp201_mouse_neuroblastoma.zarr`) on 2026-10-29: 484 s Cellpose plus 5 s measuring on the Mac GPU.
- Cellpose-SAM always works on 256 px patches (`train_seg` refuses any other `bsize`, `eval` cuts any image into
  256 px blocks with overlap). No image size is required by Cellpose; `annoloop.train.training_set` cuts every
  complete image into 512 px training tiles itself.
- `export_round` already exports a whole store as one image (all channels raw, proposed masks, read-only comparison
  masks). `training_set` reads whole images and takes only complete ones.
- AnnoCell's folder format (version 2): `cells.csv` has `state` (`unreviewed` / `ok`), `origin` (`proposed` /
  `edited` / `drawn`), and `ok_by` (`hand` / `bulk`); no macrophage mark (removed 2026-11-03). The counts
  of the corrections (kept, edited, deleted, drawn, OK by hand / in bulk, `start_cells`) are AnnoCell's
  `review.json`, the one definition: `load_annotations` reads it as `review.*` and `summarize_rounds` sums it;
  `close_round` adds `start_score` (the proposal scored against the correction) to the `annotation.json` of every
  complete image; AnnoLoop writes no `status` or `role` (completeness is always computed from `review.json`).
- Complete means `annocell.folder.is_complete(read_review(folder))`: `review.json` has `completed_at`, at least one
  cell, and no cell unreviewed. Only AnnoCell's Done button sets `completed_at` (`save(..., done=True)`); any later
  change clears it. Every cell OK without Done is not complete, so it is left out of training. AnnoCell does not let
  an annotator go to the next image without Done, and asks before closing napari or opening another folder.
- Closing a round while an annotator has it open is safe: AnnoCell reads `annotation.json` before each save, and if
  `closed_on` is set it writes nothing, turns the image read-only and tells the annotator.
- Blending several annotators of one image is planned, not built: [docs/concurrent_annotations.md](docs/concurrent_annotations.md).

## Slide statistics (description only; notebook `00_tiles` in macroseg)

- 10.9 % of the F4/80 pixels are exactly 0 (DAPI: none), spread evenly over the slide; read as background.
- The F4/80 histogram has a background lump (median 93, robust sigma 59) and a long tail to about 8000, with no second
  peak, so Otsu fails (it sits at the zero spike). A threshold has to be relative to the background: median + 5 robust
  sigmas is about 390, with 4.6 % of the pixels positive. The 5 is a convention, not derived from the data; it could be
  checked later on the cells annotators draw (nobody marks macrophages by hand).
- Per-cell macrophage counts depend on a segmentation that has not been verified; they agree with the pixel blobs
  only roughly (Spearman 0.88 per 1024 px tile), and the two are not independent (the segmentation uses F4/80).

## Set aside

An earlier plan cut the slide into 1024 px tiles (macrophage-rich tiles only, shuffled assignment, a shared
calibration tile, a held-out test tile). It was dropped for annotation when annotators said they can work on whole
images. The code is kept in macroseg in case (`macroseg.tiles`, `notebooks/00_tiles.ipynb`), and the tile
arithmetic adapts to whole images easily.

## Known issues

- Separated from macroseg on 2026-11-04 (the user: the two "have to work independently"): the package is `annoloop`,
  the path variables `ANNOLOOP_*` (each machine's `.env` must be renamed), the code version in every record
  `code_commit` (was `macroseg_commit`; nothing real had been written yet). macroseg's `docs/design.md` and its
  spatiato notebook (05) were removed, and the notebooks' outputs (run in the macroseg checkout) cleared. AnnoLoop
  neither imports nor installs macroseg, and macroseg does not depend on AnnoLoop.
- Cleaned up on 2026-11-04 (D15): the baseline work (choosing the starting segmentation: multi-segmentation
  experiment configs, `config.py`, `evaluate`, `histograms`, `view`, `marker_windows`, the comparison helpers of
  `accuracy`, notebooks 01, 01b, 03, 04) is removed; it stays in macroseg. `segmentation.Segmentation` is the one
  description of a segmentation (start model in the settings, or `from_model`); `export_round` / `export_test` take
  explicit arguments; training reads the images from the AnnoCell folders, not the stores; scoring counts only
  complete test images; "ROI" is now "image" (or "test image"). AnnoCell's API followed: `af.images()` /
  `af.ImageFolder` (was `rois()` / `Roi`), `test_images_folder`, and the score files' `images` / `image` keys.
- There are no notebooks, by decision (D16).

## Built: settings and the `annoloop` commands (2026-11-04)

- `annoloop.settings`: `load_settings`, `load_round` (strict; unknown keys, undefined channels, bad annotator or
  image names fail before anything runs).
- `annoloop.loop`, command `annoloop` (`[project.scripts]`): `annoloop prepare <global> <round> [--device]
  [--dry-run]` checks the whole round first (closed round, image already given to someone not listed for it in the
  round, missing raw file or model), then per image: ingests into `<runs>/<name>.zarr` once (written as
  `.partial.zarr` and renamed, so a killed job leaves nothing half-made), runs only the proposing segmentation once
  (redone if its intensity table is missing), and calls `export_round` for the annotator and the other annotators
  (an existing folder is kept with its corrections). A store made from another file, level or crop is refused. `annoloop close <global>
  <round>` closes the round.
- `annoloop train <global> <round> [--from M] [--name N] [--device]` (D14): `training_set` restricted to the rounds
  used, the global `training` settings; `model.json` also records `rounds`, `from_model`, the settings files and the
  `trainer` (the training loop; `train(..., trainer=)` takes another one with Cellpose's `train_seg` contract). A
  model that exists is skipped; a folder without `model.json` (an unfinished training) is removed and redone. A round
  still open is trained on with a warning.
- `annoloop score <global> (--model M | --start)`: `score_model` / `score_test_images` (the start model as round 0).
- `annoloop stats <global> [--by-annotator] [--csv F]`: `summarize_rounds(..., by=)`.
- The environment needs Python 3.13 (`uv sync --python 3.13 --extra imaging --extra dev`): `pylibczirw` has no
  3.14 wheel.

## Code needed

- Statistics per image, round and annotator from the review (drawn, edited, deleted, OK by hand / in bulk):
  `summarize_rounds` sums them per round or per round and annotator; the O2 progress measure is still to write.
- SLURM job scripts, once O6 is decided (the user asked to wait, 2026-11-04).
- Our own training loop: written (2026-11-05, `annoloop.trainer.train_seg`, branch `trainer`) and made the default
  of `train()` at the user's request. On 512 px tiles, the same contract as Cellpose's `train_seg`; on 4 real tiles
  over 12 epochs its loss curve follows Cellpose's (MPS is not deterministic: run-to-run noise is larger than the
  gap). Still needed, planned for 2026-11-06, before merging `trainer` into `main`: (1) unit tests
  (`tests/test_trainer.py`: normalisation incl. a flat channel, `_process_input` errors, the schedule, the loss
  perfect / random, the loop with a tiny net: loss falls, weights change), the user's to write. Dropped (4):
  `net.diam_labels`, which Cellpose 4 writes during training but never reads. Done the same day: (2) stop on a NaN or infinite loss (`FloatingPointError`, nothing saved); (3) `try/finally` so the
  net always gets its dtype back; (5) autocast only on CUDA, the schedule always `n_epochs` long, a torch seed.
  Then a longer comparison on the cluster.
- Then, a second version of the loop on whole images instead of tiles (2026-11-05): many random 256 px crops per
  image per epoch, with normalisation and flows computed on the whole image, as the model sees images when it
  segments, and with the arrows of cells cut by a crop pointing to their real centre. Needs: `training_set` /
  `train()` passing whole images (a new contract), a mask per image of the regions near test images that no crop
  may touch, and lazy loading if memory grows. Judged by the scores on the test images against the tile version.
- Shared images: built 2026-11-09 (D19): `other_annotators` in the round file, `role` in `annotation.json`,
  `training_set` trains on the primary versions only. Still to do: AnnOverview reads `role` (a shared image goes
  into training once its primary version is complete); the annotator-bias analysis (O1).
