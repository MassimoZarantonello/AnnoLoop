# Task: our own training loop

> **Done (2026-11-05).** The loop is `annoloop.trainer.train_seg` and it is the default of `annoloop train`.
> On 4 real Exp201 tiles (start masks as labels), 12 epochs on the Mac GPU, its loss curve follows Cellpose's loop
> and ends at the same value. Cellpose's loop is still there: `train(..., trainer=cellpose.train.train_seg)`.
> Each model records the loop that trained it (`trainer` in `model.json`). The brief below is kept as written.

`annoloop train` fine-tuned Cellpose-SAM through Cellpose's own loop, `cellpose.train.train_seg`. The task is to
write that loop ourselves, in plain PyTorch, so we understand it and can change it. Examples: give hand-drawn cells
more weight, or draw tiles with macrophages more often.

The rest of the pipeline stays the same. `annoloop.train.train` already takes a `trainer=` argument, and it records
which loop trained each model (`trainer` in `model.json`).

## What to write

`src/annoloop/trainer.py`, with one function that has the same contract as Cellpose's:

```python
def train_seg(net, train_data, train_labels, channel_axis, n_epochs, learning_rate, weight_decay,
              batch_size, save_path, model_name, **ignored):
    ...
    return weights_path, train_losses, test_losses   # Path, array of n_epochs, array (zeros: no test set)
```

- `net`: `CellposeModel(...).net`, a `torch.nn.Module` already on its device (`net.device`).
- `train_data`: a list of tiles, each a `(3, 512, 512)` uint16 array: NaK-ATPase, DAPI and F4/80.
  `train_labels`: the masks, each a `(512, 512)` uint32 array where 0 is background and 1, 2, ... are the cells.
- Save the weights with `net.save_model(save_path / "models" / model_name)` and return that path.
  `train` then moves them next to `model.json`.

To try it, call `train(data, models_root, "try1", settings, trainer=trainer.train_seg)` from a script on a few
tiles. A `--trainer` flag for `annoloop train` can come later.

## The five pieces (each is a few lines)

Read Cellpose's version side by side with this: `.venv/lib/python3.13/site-packages/cellpose/train.py`, from
`def train_seg` (about line 309) to the end. Cellpose is version 4.2.1.1 here.

1. **Prepare the data, once.**
   - Normalise each tile channel by channel: `cellpose.transforms.normalize_img(img, axis=0)`. This scales each
     channel from its 1st to its 99th percentile, so 0 to 1 roughly.
   - Turn the masks into what the network predicts: `cellpose.dynamics.labels_to_flows(list_of_masks,
     device=net.device)`. It gives one `(4, H, W)` float array per tile: `[labels, cell probability (0/1), flow Y,
     flow X]`. The flows point from every pixel of a cell to its centre, and this is the core idea of Cellpose.
   - Drop the tiles with fewer than 5 cells, as Cellpose does (`min_train_masks=5`).
2. **Make batches with augmentation.** Each epoch, shuffle the tiles. For each batch of `batch_size` tiles, call
   `cellpose.transforms.random_rotate_and_resize(imgs, lbls=flows, bsize=256, scale_range=0.5,
   device=net.device)`. It returns the torch tensors `X` of shape `(B, 3, 256, 256)` and `lbl` of shape
   `(B, 4, 256, 256)`: a random 256 px crop, rotated, flipped and rescaled by 0.75 to 1.25. Cellpose-SAM only
   works on 256 px patches, which is why every tile is only seen through one crop per epoch.
3. **Forward pass and loss.** `y = net(X)[0]` has shape `(B, 3, 256, 256)`: flow Y, flow X, then the cell
   probability as a logit. The loss is:
   ```python
   flow_loss = nn.MSELoss()(y[:, :2], 5 * lbl[:, 2:]) / 2          # flows are scaled by 5
   prob_loss = nn.BCEWithLogitsLoss()(y[:, 2], (lbl[:, 1] > 0.5).float())
   loss = flow_loss + prob_loss
   ```
   Then `optimizer.zero_grad(); loss.backward(); optimizer.step()`, and add `loss.item() * len(X)` to the epoch's
   loss.
4. **Optimizer and learning-rate schedule.**
   - Use `torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)`, and set the rate by
     hand each epoch (`param_group["lr"] = LR[epoch]`).
   - For 100 epochs the schedule is: a linear warm-up from 0 over 10 epochs, flat until epoch 50, then halved every
     5 epochs.
   - Cellpose builds this array in a few lines (`LR = np.linspace(...)`); write it as a small function you can
     test.
5. **Precision.** The SAM network is stored in bfloat16. Cellpose sets `net.dtype = torch.float32` before
   training and puts it back afterwards, and runs the forward pass inside
   `torch.autocast(device_type=net.device.type, dtype=net.dtype)`. Do the same, or the gradients are poor.

Call `net.train()` before the loop. Use `np.random.seed(epoch)` at the start of each epoch if you want to compare
runs exactly with Cellpose's loop.

## How to know it works

- **A unit test without the real network** (`tests/test_trainer.py`): use a tiny module, for example a
  `nn.Conv2d(3, 3, 1)` wrapped so that `forward` returns `(y,)`, with `device`, `dtype` and `save_model`
  attributes. Train it on a few synthetic tiles on the CPU and check three things: the loss goes down, the weights
  file exists, and `train_losses` has `n_epochs` values. `tests/test_train.py` and `tests/test_loop.py` show how the
  tiles are made.
- **Against Cellpose:** on the same small set of tiles (say 20) and with the same seeds, train for 10 to 20 epochs
  with both loops. The loss curves should be close. Then score both models on a test image (`annoloop score`).
- **Where to run it:** the real network has about 300 M parameters. On the Mac (MPS), only run a few tiles and a
  few epochs to check that it runs. Real training goes on the cluster.

## Things worth knowing first

- PyTorch basics: modules, `train()` and `eval()`, tensors on a device, and `zero_grad` / `backward` / `step`. The
  official "Learn the Basics" tutorial on pytorch.org covers all of it in about an hour.
- What Cellpose predicts and why: flows plus cell probability, and how masks are rebuilt from them by following the
  flows. The Cellpose paper (Stringer et al., 2021, Nature Methods) explains it in its Figure 1.
- How our tiles are made: `annoloop.train.training_set` and `_tiles`. These are 512 px tiles of the complete
  images, renumbered per tile, with tiles near a test image dropped.

## Once it works: ideas it opens up

- Weight the loss by how a cell became OK: hand-drawn and edited cells are surer than bulk-OK'd ones. `cells.csv`
  has `origin` and `ok_by`.
- Draw tiles with many F4/80+ cells more often (Cellpose's `train_probs` does the same).
- Log the loss per epoch to the job log, to watch a cluster run.
