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
| Data and results | `/srv/data/massimoz/macroseg_runs/<name>.zarr` | copies in `data/runs/` |
| napari | – (no display) | `macroseg-view <name>` |

Configs hold Mewtwo paths only.

```bash
# on Mewtwo, in the repo
uv sync --extra imaging --extra dev                                   # environment (once)
.venv/bin/python -m macroseg.baseline configs/<experiment>.yaml       # import, segment, measure
.venv/bin/python -m pytest

# on the Mac (shell function in ~/.zshrc)
macroseg-view                         # list results on Mewtwo
macroseg-view <name> [--channels DAPI F480]   # copy <name>.zarr to data/runs and open napari
```

For a quick test, set `input.crop: [y0, y1, x0, x1]` in the config. Without `input.channels`,
channels are named by the first word of their file name (`"Na_K_ATPase (APC C2)"` -> `Na_K_ATPase`).

## Layout

```
configs/          one YAML per experiment: input image, channels, segmentation, output .zarr
docs/             design and (later) annotation guidelines
notebooks/        step-by-step runs (histograms, segmentation)
src/macroseg/     package code
tests/            tests on synthetic data
```

## Configs

Marker names and channel order are never hard-coded. Load an experiment config with:

```python
from macroseg.config import load_config

cfg = load_config("configs/example_macrophages.yaml")
cfg.channels.cellpose  # Cellpose input channels, in order
```
