# macroseg

Fine-tuning Cellpose to segment macrophages in MACSIMA data, built on
[SpatialData](https://spatialdata.scverse.org/) and [Harpy](https://harpy.readthedocs.io/).
Segmentations are written to SpatialData `.zarr` stores and analysed in
[spatiato](https://github.com/vibspatial/spatiato).

See [docs/design.md](docs/design.md) for goals, workflow and decisions.

## Where things run

**Everything runs on Mewtwo; the Mac is only a screen for napari.**

| | Mewtwo (`/srv/data/massimoz/macroseg`) | Mac |
|---|---|---|
| Edit code, git commit / push | VS Code Remote (`mewtwo-code /srv/data/massimoz/macroseg`) | – |
| Notebooks, tests, pipeline | yes | – |
| Data and results | `/srv/data/massimoz/macroseg_runs/<name>.zarr` (+ `<name>.log`) | copies in `data/runs/` |
| napari, Cellpose GUI | – (no display) | `macroseg-view <name>`, notebook 02 |

Paths differ per machine, so they live in `.env` at the repo root (not in git; copy `.env.example`):
`MACROSEG_INPUT` (raw images), `MACROSEG_RUNS` (`.zarr` stores), `MACROSEG_ANNOTATIONS` (annotation rounds),
and on the Mac `MACROSEG_REMOTE_RUNS` / `MACROSEG_REMOTE_ANNOTATIONS` (the same folders on Mewtwo, for rsync).
Configs use them as `${MACROSEG_RUNS}`, so one config works on both machines (`macroseg.paths`).

```bash
# on Mewtwo, in the repo
uv sync --extra imaging --extra dev                                         # environment (once)
.venv/bin/python -m macroseg.baseline configs/<experiment>.yaml [--overwrite] # import, segment, measure
.venv/bin/python -m pytest

# on the Mac (shell function in ~/.zshrc; not available in the Claude app's ! commands)
macroseg-view                         # list results on Mewtwo
macroseg-view <name> [--channels DAPI F480]   # mirror <name>.zarr into data/runs and open napari
```

Cellpose weights live in `/srv/data/massimoz/.cache/cellpose/models` (`CELLPOSE_LOCAL_MODELS_PATH`,
set in `~/.profile` and `~/.bashrc` on Mewtwo), not in the shared home.

## Pipeline (`macroseg.baseline`)

1. **Ingest** (`ingest.py`): OME-TIFF (lazy, any pyramid level, optional crop) or CZI → SpatialData,
   channels renamed to short names, pixel size kept in `sdata.attrs["pixel_size_um"]`.
2. **Write** the image to the `.zarr` given in the config (`--overwrite` replaces an existing store).
3. **Segmentations**, one per entry under `segmentation` in the config, each written as a labels
   element with the entry's name:
   - a Cellpose-SAM run (`cpsam_v2`) on up to 3 channels, chunk-wise through Harpy, one chunk at a
     time on the GPU;
   - or the expansion of an earlier segmentation by `distance_um` (objects stop where they meet).
4. **Measure** every segmentation: table `<name>_intensities` with, per object, the mean of every
   channel (`.X`), `sum_<channel>`, `max_<channel>`, area in pixels (`shapeSize`), `cell_ID` and
   centre (`.obsm["spatial"]`).

Exp201 whole slide (6 channels, 7705 × 6272): about 4 minutes on the RTX 4090.

## Evaluation without ground truth (`macroseg.evaluate`, notebook 01b)

Uses only the channels used to segment (DAPI and the generic membrane marker):

| Measure | Definition | Blind spot |
|---|---|---|
| Nuclei per cell | nuclei of the reference nuclear segmentation inside each cell (≥ 50 % of the nucleus area): 0, 1 or 2+; plus nuclei in no cell | not outlines; trivially perfect for grown nuclei |
| Border ratio | per cell: mean membrane signal on the border pixels (2 px band) / mean on the inside pixels | favours segmentations that used the membrane channel |
| Shape | area, solidity (area / convex hull area) | some cell types are genuinely irregular |

Optional: marker separation of a population marker (`marker=`), off by default.
`plot_intensity_per_cell` (mean / sum / max per cell) is descriptive, for the later classification phase.

## Annotation rounds (`macroseg.annotation`, notebook 02) and ground-truth measures (`macroseg.accuracy`, notebook 03)

One folder per image and round, `$MACROSEG_ANNOTATIONS/round_<NN>/<image>/`: the image for the Cellpose GUI
(3 channels), `<image>_seg.npy` (masks, corrected in the GUI), `start_masks.tif` (masks as exported) and
`annotation.json` (round, role `train` / `test`, provenance, and after `finish` the status, the changes and
the score of the starting masks). Measures against the corrected masks: AP (Cellpose's definition),
precision and recall at IoU thresholds, and recall on a subset of cells.

## Notebooks (kernel `.venv`)

| Notebook | Runs on | Does |
|---|---|---|
| `01_baseline.ipynb` | Mewtwo | the pipeline step by step, with per-pixel histograms of every channel |
| `01b_evaluation.ipynb` | Mewtwo | initial check: which segmentation to start from (measures above, cells to inspect in napari) |
| `02_cellpose_gui.ipynb` | Mac | copies a store, exports the chosen segmentation for the Cellpose GUI, opens it, records the round |
| `03_accuracy.ipynb` | Mac or Mewtwo | annotation rounds, and every segmentation scored against the corrected masks |
| `05_spatiato_annotation.ipynb` | Mewtwo | later phase: crops a whole-slide store for annotation in spatiato |

## Layout

```
configs/          one YAML per experiment: input image, channels, segmentations, output .zarr
docs/             design and (later) annotation guidelines
notebooks/        step-by-step runs (pipeline, evaluation)
src/macroseg/     config, paths, ingest, histograms, baseline, evaluate, view, annotation, accuracy
tests/            tests on synthetic data
```

## Configs

Marker names and channel order are never hard-coded. The parts used by the pipeline:

```yaml
input:
  path: ${MACROSEG_INPUT}/<file>.tif   # ${...}: from .env
  level: 0                  # pyramid level (0 = full resolution)
  crop: null                # [y0, y1, x0, x1] for a quick test; null = whole image
  channels:                 # optional: file name -> short name; omitted = first word of each file name
    "Na_K_ATPase (APC C2)": NaK_ATPase
zarr: ${MACROSEG_RUNS}/<name>.zarr
image:
  element: macsima_image
segmentation:               # run in this order; the name is the labels element
  nuclei:          {channels: [DAPI]}
  nuclei_expanded: {expand: nuclei, distance_um: 3}
  membrane:        {channels: [NaK_ATPase]}
  membrane_dapi:   {channels: [NaK_ATPase, DAPI]}     # optional per entry: diameter, model
```

The `channels` (Cellpose input for fine-tuning) and `annotation` sections are for later phases.
Load a config with `macroseg.config.load_config(path)`; unknown keys and invalid values fail loudly.

## Next

- Inspect suspicious cells (2+ nuclei, nuclei in no cell) in napari.
- spatiato for cell selection: work on a separate local copy (`data/spatiato/<name>.zarr`) and back it
  up to `/srv/data/massimoz/macroseg_annotations/`, because spatiato saves into the store's tables and
  `macroseg-view` mirrors `data/runs` with `--delete`.
- A small hand-annotated ground truth (also needed for fine-tuning).
