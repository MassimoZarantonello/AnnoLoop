"""Our own training loop for Cellpose-SAM, with the contract of Cellpose's ``train.train_seg`` (see
docs/training_loop.md). Use it with ``annoloop.train.train(..., trainer=trainer.train_seg)``."""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import dynamics
from cellpose.transforms import random_rotate_and_resize
from torch import nn

train_logger = logging.getLogger(__name__)


def _normalize_tile(tile: np.ndarray) -> np.ndarray:
    """Scale each channel of a (channels, y, x) tile so its 1st percentile becomes 0 and its 99th becomes 1.

    Values outside are kept (not clipped), as Cellpose does when it segments. A flat channel (1st = 99th
    percentile) becomes 0 everywhere, its few bright pixels included."""
    tile = tile.astype(np.float32)
    low, high = np.percentile(tile, [1, 99], axis=(1, 2), keepdims=True)
    spread = high - low
    spread[spread == 0] = 1  # a flat channel: avoid 0 / 0
    out = (tile - low) / spread
    out[(high == low)[:, 0, 0]] = 0
    return out.astype(np.float32)


def _process_input(train_data, train_masks, device, min_cells=5):
    """Normalised tiles and their flows ((4, H, W): labels, inside a cell, dy, dx), without the tiles with fewer
    than ``min_cells`` cells. Masks are not changed."""
    if len(train_data) != len(train_masks):
        raise ValueError(f"{len(train_data)} tiles but {len(train_masks)} masks.")
    norm_tiles = []
    flows = []
    for i, (tile, mask) in enumerate(zip(train_data, train_masks, strict=True)):
        if tile.ndim != 3 or tile.shape[1:] != mask.shape:
            raise ValueError(f"Tile {i}: image {tile.shape} does not match mask {mask.shape}.")
        if np.count_nonzero(np.unique(mask)) < min_cells:
            continue
        norm_tiles.append(_normalize_tile(tile))
        # labels_to_flows renumbers the masks it gets in place: give it a copy.
        flows.append(dynamics.labels_to_flows([mask.copy()], device=device)[0])
    train_logger.info(
        "%d of %d tiles kept (at least %d cells), %d dropped",
        len(norm_tiles),
        len(train_data),
        min_cells,
        len(train_data) - len(norm_tiles),
    )
    if not norm_tiles:
        raise ValueError(f"No tile with at least {min_cells} cells.")
    return norm_tiles, flows


def _lr_schedule(learning_rate, n_epochs) -> np.ndarray:
    """Learning rate of each epoch, as Cellpose's: a linear warm-up from 0 over 10 epochs, flat, then halved 10
    times (every 5 epochs over the last 50 from 100 epochs, every 10 over the last 100 above 300 epochs).
    Below 10 epochs the array is 10 long (only the first ``n_epochs`` are used)."""
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))
    return LR


_mse = nn.MSELoss(reduction="mean")
_bce = nn.BCEWithLogitsLoss(reduction="mean")


def _compute_loss(y, lbl):
    """Loss of a batch.

    - ``y`` (prediction): (B, 3, 256, 256), the network's output for B crops of 256 x 256 px.
      Channel 0 = dy, 1 = dx (arrows, scaled x5), 2 = logit of "inside a cell".
    - ``lbl`` (targets): (B, 3, 256, 256), the flows of the same crops after augmentation, without the labels.
      Channel 0 = inside a cell, 1 = dy, 2 = dx (arrows, length <= 1). "Inside" is interpolated by the
      augmentation, so not exactly 0/1: hence ``> 0.5``. Counting from the end, the same indices work on the
      4 channels of ``labels_to_flows`` (labels first).
    """
    flow_loss = _mse(y[:, -3:-1], 5.0 * lbl[:, -2:]) / 2.0
    prob_loss = _bce(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    return flow_loss + prob_loss


def train_seg(
    net,
    train_data,
    train_labels,
    channel_axis=0,
    n_epochs=100,
    learning_rate=1e-5,
    weight_decay=0.1,
    batch_size=1,
    save_path=None,
    model_name=None,
    **ignored,
):
    """Fine-tune ``net`` on the tiles; the same arguments and return value as Cellpose's ``train.train_seg``.

    - ``net``: ``CellposeModel(...).net``, on its device (``net.device``). If it is in bfloat16, it is trained in
      float32 and converted back at the end; the saved weights are float32.
    - ``train_data``: list of (3, H, W) uint16 tiles (channels first: ``channel_axis`` must be 0).
    - ``train_labels``: list of (H, W) masks, 0 = background, 1, 2, ... = cells.
    - ``n_epochs``, ``learning_rate``, ``weight_decay``, ``batch_size``: the ``training`` settings.
    - ``save_path``, ``model_name``: the weights go to ``<save_path>/models/<model_name>``.

    Returns (weights path, training loss of each epoch, test losses: zeros, there is no test set).
    """
    if channel_axis != 0:
        raise ValueError(f"Tiles must have their channels first (channel_axis=0), got {channel_axis}.")
    device = net.device
    bsize = 256  # Cellpose-SAM works on 256 px crops only
    scale_range = 0.5  # random rescaling of the crops by 0.75 to 1.25
    original_net_dtype = net.dtype
    if net.dtype == torch.bfloat16:
        train_logger.info("Network in bfloat16: trained in float32, converted back at the end")
        net.dtype = torch.float32

    train_images, train_flows = _process_input(train_data, train_labels, device)
    nimg = len(train_images)
    LR = _lr_schedule(learning_rate, n_epochs)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    (save_path / "models").mkdir(parents=True, exist_ok=True)
    filename = save_path / "models" / model_name
    train_logger.info("Saving the model to %s", filename)

    net.train()
    train_losses = np.zeros(n_epochs)
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        rperm = np.random.permutation(nimg)  # every tile once per epoch, in a random order
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]

        for k in range(0, nimg, batch_size):
            inds = rperm[k : k + batch_size]  # the last batch may be smaller
            imgs = [train_images[i] for i in inds]
            # without channel 0 (the labels): the augmentation takes (inside a cell, dy, dx)
            lbls = [train_flows[i][1:] for i in inds]
            # augmentations: the same random crop, rotation, flip and rescaling for images and flows
            X, lbl = random_rotate_and_resize(imgs, lbls=lbls, bsize=bsize, scale_range=scale_range, device=device)[:2]

            # Only the forward pass and the loss under autocast; the optimisation steps outside it.
            with torch.autocast(device_type=device.type, dtype=net.dtype):
                y = net(X)[0]
                loss = _compute_loss(y, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses[iepoch] += loss.item() * len(X)  # a sum over tiles: the last batch may be smaller

        train_losses[iepoch] /= nimg  # the epoch's mean loss per tile
        if iepoch == 5 or iepoch % 10 == 0 or iepoch == n_epochs - 1:
            train_logger.info(
                "Epoch %d/%d: loss %.4f, learning rate %.2e, %.0f s",
                iepoch + 1,
                n_epochs,
                train_losses[iepoch],
                LR[iepoch],
                time.time() - t0,
            )

    # Save first, while the network is still float32 (the saved weights are float32), then convert it back.
    net.save_model(filename)
    if original_net_dtype != torch.float32:
        train_logger.info("Network converted back to %s", original_net_dtype)
        net.dtype = original_net_dtype
    return filename, train_losses, np.zeros(n_epochs)  # no test set: zero test losses
