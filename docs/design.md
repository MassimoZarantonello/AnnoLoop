# macroseg: design

## Goal

Fine-tune **Cellpose** with manual annotations so it segments **macrophages** correctly in
**MACSIMA** data (mouse neuroblastoma and liver cancer; markers include DAPI and F4/80, full
panel not decided yet). The improved masks are written as a new `labels` element in a
SpatialData `.zarr` store and analysed in **spatiato** (feature extraction, classification).

macroseg and spatiato are separate projects. They exchange data only through SpatialData
`.zarr` stores.

## Stack

- **SpatialData**: on-disk format for images, labels, shapes and tables.
- **Harpy**: `hp.io.macsima` to read MACSIMA data; `hp.im.segment` with
  `hp.im.cellpose_callable(pretrained_model=...)` to run baseline and fine-tuned models
  chunk-wise on whole slides.
- **Cellpose 4** (Cellpose-SAM): baseline model and fine-tuning (`cellpose.train.train_seg`).
- **spatiato**: annotation UI (phase 1) and downstream analysis (phase 2).

## Principles

- **Nothing problem-specific in code.** Marker names, channel order, tissue, element names
  and training parameters live in YAML configs under `configs/`. A new cell type should
  need a new config and new annotations, not new code.
- **Every trained model records the config it was trained with**, in particular the
  Cellpose input channels and their order.
- **The repo holds code and configs only.** Data, ROIs, annotations and model weights live
  on cluster / Mewtwo storage.
- **Heavy work stays off the paid GUI cluster.** Annotation sessions only open prepared data.

## Phases

### Phase 1: build training data and fine-tune Cellpose

1. **Ingest**: MACSIMA → SpatialData `.zarr` (`hp.io.macsima`).
2. **ROI**: crop regions of interest into small `.zarr` stores that contain only the channels
   needed for annotation.
3. **Baseline**: segment ROIs with the stock Cellpose model.
4. **Candidates**: pre-select candidate macrophages (e.g. high mean F4/80 per cell).
5. **Annotate in spatiato** (biologists, VIB Data Core GUI cluster):
   - confirm or reject candidates;
   - mark missed macrophages;
   - redraw badly segmented macrophages (missed, split, merged, wrong outline) as polygons
     with the Annotation widget (`shapes` element);
   - review the **whole** ROI, then mark it as completed. In a completed ROI, untouched
     cells count as correct.
6. **Merge**: rasterize the polygons and burn them into the baseline mask. Remove only the
   pixels a polygon covers from overlapping baseline objects, and keep the remainders as
   separate cells if they are large enough (so a neighbour fused to a macrophage is not lost).
7. **Dataset**: export (image, mask) pairs; split train / validation / test **by sample or
   patient**, with both tissues in the test set.
8. **Train** (GPU cluster): fine-tune Cellpose starting from the generalist model.
9. **Evaluate**: average precision on held-out test ROIs, plus recall / precision on
   macrophages only; compare with the baseline model.
10. **Iterate** (human-in-the-loop): the new model pre-segments fresh ROIs, biologists only
    correct its errors, retrain. Stop when test metrics stop improving.

Macrophage marks (confirmed / redrawn) are required on test and validation ROIs, where the
macrophage-specific metric needs them, and optional on training ROIs: Cellpose training only
needs correct masks.

Quality control: occasionally have a test ROI annotated **without** pre-selection, to check
that candidates are not being accepted uncritically.

### Phase 2: analysis in spatiato

1. **Segment** whole slides with the fine-tuned model via `hp.im.segment`.
2. In spatiato: **Feature Extraction** on all cells, then **Object Classification**:
   biologists label a few cells, the random forest classifies the rest. Phase 1 macrophage
   marks can seed the classifier labels.

## Modules

| Module | Does | Runs on |
|---|---|---|
| `config` | load and validate YAML configs | anywhere |
| `ingest` | MACSIMA → SpatialData `.zarr` | Mewtwo |
| `roi` | crop ROIs, keep annotation channels only | Mewtwo |
| `baseline` | stock Cellpose on ROIs via Harpy | GPU cluster |
| `candidates` | pre-select candidate macrophages | anywhere |
| `merge` | polygons + baseline → full instance mask | anywhere |
| `dataset` | export training pairs, sample-level split | anywhere |
| `train` | Cellpose fine-tuning | GPU cluster |
| `evaluate` | AP overall and macrophage-only | GPU cluster / anywhere |
| `segment` | fine-tuned model on whole slides via Harpy | GPU cluster |

All modules are developed and tested on Mewtwo (VS Code Remote) on small crops or synthetic ROIs
before running on whole slides; the Mac only displays results in napari.

## Compute

| Step | Where |
|---|---|
| Development, tests | Mewtwo (VS Code Remote) |
| Viewing results (napari) | Mac |
| Ingest, ROI cropping | Mewtwo |
| Baseline, training, whole-slide segmentation | GPU cluster |
| Annotation (prepared environment, no compute in session) | VIB Data Core (GUI, paid) |
| Feature extraction, classification | spatiato |

## Open questions

- Does VIB Data Core share a filesystem with the GPU cluster? If not, how do data move?
- Apptainer or a shared Python environment on VIB Data Core? Can spatiato be installed there?
- How are GUI sessions billed (time, resources, storage)?
- Number of biologists annotating (unknown; the per-ROI, per-annotator layout scales either way).
- Final marker panel, and where the MACSIMA data will be stored.
