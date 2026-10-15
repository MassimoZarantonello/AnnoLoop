# macroseg

Fine-tuning Cellpose to segment macrophages in MACSIMA data, built on
[SpatialData](https://spatialdata.scverse.org/) and [Harpy](https://harpy.readthedocs.io/).
Segmentations are written to SpatialData `.zarr` stores and analysed in
[spatiato](https://github.com/vibspatial/spatiato).

See [docs/design.md](docs/design.md) for goals, workflow and decisions.

## Layout

```
configs/          one YAML per experiment: channels, markers, element names, training params
docs/             design and (later) annotation guidelines
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
