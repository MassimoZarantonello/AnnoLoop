"""Experiments on a "shared image": the same crop segmented by two annotators, alice and bob.

No shared image has been annotated yet, so two segmentations already in the Exp201 store play alice and bob.
Each ``# %%`` line starts a cell: "Run Cell" in VS Code runs it alone. Or run the whole file:

    .venv/bin/python scripts/show_shared_image.py
"""

# %% Imports and settings

import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
from skimage.segmentation import expand_labels
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

STORE = "/Users/massimoz/Documents/macroseg/data/runs/exp201_mouse_neuroblastoma.zarr"

# The crop: top-left corner (Y, X) and side length, in pixels.
Y, X = 4500, 3600
SIZE = 512


def show(picture, title, **imshow_options):
    """Open a new figure with one picture."""
    plt.figure(figsize=(6, 6))
    plt.imshow(picture, **imshow_options)
    plt.title(title)
    plt.axis("off")


# %% 1. Read the crop

sdata = sd.read_zarr(STORE, selection=("images", "labels"))  # pyright: ignore[reportArgumentType]
window = {"y": slice(Y, Y + SIZE), "x": slice(X, X + SIZE)}

# The image: 3 channels, shape (3, 512, 512).
channels = ["F480", "NaK_ATPase", "DAPI"]
image = sdata.images["macsima_image"].sel(c=channels).isel(window).values

# The labels: shape (512, 512). 0 = background, 1, 2, 3, ... = one number per cell.
alice_lb = sdata.labels["membrane_dapi_f480"].isel(window).values
bob_lb = sdata.labels["membrane_dapi"].isel(window).values

print("image:", image.shape)
print("alice:", len(np.unique(alice_lb)) - 1, "cells")
print("bob:", len(np.unique(bob_lb)) - 1, "cells")


# %% 2. Each channel raw, in grey

for name, plane in zip(channels, image):
    show(plane, name + " (raw)", cmap="gray")


# %% 3. Each channel stretched to 0..1


def stretch(plane, low_percent=1, high_percent=99.5):
    """The plane scaled to 0..1 between two percentiles."""
    plane = plane.astype(float)
    low, high = np.percentile(plane, (low_percent, high_percent))
    return np.clip((plane - low) / (high - low), 0, 1)


stretched = [stretch(plane) for plane in image]
for name, plane in zip(channels, stretched):
    show(plane, name + " (stretched)", cmap="gray")


# %% 5. Cells only, one colour per cell

show(label2rgb(alice_lb, bg_label=0), "alice: cells only")
show(label2rgb(bob_lb, bg_label=0), "bob: cells only")

# %%

bg_a = np.where(alice_lb==0,1,0)
bg_b = np.where(bob_lb==0,1,0)
# # show((bg_a == bg_b)==0, "", cmap="Blues")
# show((bg_a & bg_b), "INTERSECTION", cmap="gray")

# show((bg_a | bg_b), "UNION", cmap="gray")

zeros = np.zeros_like(bg_a, dtype=float)
two = np.stack([bg_a, zeros, bg_b], axis=-1)

show(two,"Red: Alice, Blue: Bob")

# %%
alice_array = np.array(alice_lb)
bob_array = np.array(bob_lb)
a_cells_ids = np.unique(alice_array) 
b_cells_ids = np.unique(bob_array) 
# show(alice_array == a_cells_ids[0],"", cmap="gray")
# show(bob_array == b_cells_ids[0],"",cmap="gray")
p_max_iou = 0
max_iou = 0
i = 0
cells = {}
for a in a_cells_ids[1:]:
    for b in b_cells_ids[1:]:
        i+=1
        mask = np.astype((alice_array == a),int) + (bob_array == b)
        intersection = (mask == 2).sum()
        union = (mask > 0).sum()
        max_iou = max((intersection / union), p_max_iou)
        if (max_iou > p_max_iou) & (max_iou > 0.8):
            cells[a] = b
        p_max_iou = max_iou
    p_max_iou = 0
    max_iou = 0
print(i)
print(len(cells))
#%% 
red = np.isin(alice_array, list(cells.keys()))      # all of Alice's matched cells
blue = np.isin(bob_array, list(cells.values()))     # all of Bob's matched cells
green = np.zeros_like(red)

overlay = np.stack([red, green, blue], axis=-1).astype(float)
show(overlay, "Red: Alice, Blue: Bob, Magenta: both", cmap="hsv")

# %%
def match_cells(alice, bob, min_iou=0.5):
    a, b = alice.ravel(), bob.ravel()

    # pixels shared by each pair of cells that touch at all
    both = (a > 0) & (b > 0)
    pairs, inter = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    pa, pb = pairs

    # area of every cell
    area_a = np.bincount(a)
    area_b = np.bincount(b)

    iou = inter / (area_a[pa] + area_b[pb] - inter)
    keep = iou >= min_iou
    return dict(zip(pa[keep].tolist(), pb[keep].tolist())), iou[keep]

cells, ious = match_cells(alice_array, bob_array)

a_ids = np.unique(alice_array)[1:]
b_ids = np.unique(bob_array)[1:]
only_alice = np.setdiff1d(a_ids, list(cells.keys()))
only_bob = np.setdiff1d(b_ids, list(cells.values()))

# 1. agreement map: matched cells grey, Alice-only red, Bob-only blue, outlines on top
rgb = np.zeros(alice_array.shape + (3,))
rgb[np.isin(alice_array, list(cells.keys()))] = 0.35          # matched (Alice's outline of it)
rgb[np.isin(alice_array, only_alice)] = [1, 0.2, 0.2]         # only Alice
rgb[np.isin(bob_array, only_bob)] = [0.2, 0.4, 1]             # only Bob
rgb[find_boundaries(alice_array, mode="inner")] = [1, 0.5, 0.5]
rgb[find_boundaries(bob_array, mode="inner")] = [0.5, 0.7, 1]

# 2. matched cells coloured by their IoU
iou_map = np.full(alice_array.shape, np.nan)
lut = np.full(alice_array.max() + 1, np.nan)
lut[list(cells.keys())] = ious
iou_map = lut[alice_array]

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(rgb)
ax[0].set_title(f"grey: matched ({len(cells)})   red: only Alice ({len(only_alice)})   "
                f"blue: only Bob ({len(only_bob)})")
im = ax[1].imshow(iou_map, cmap="viridis", vmin=0.5, vmax=1)
ax[1].set_title("matched cells by IoU")
fig.colorbar(im, ax=ax[1], fraction=0.046)
for a_ in ax[:2]:
    a_.axis("off")

# 3. how good the matches are
ax[2].hist(ious, bins=np.linspace(0.5, 1, 26), color="grey")
ax[2].axvline(np.median(ious), color="k", ls="--", label=f"median {np.median(ious):.2f}")
ax[2].set_xlabel("IoU of matched cells")
ax[2].set_ylabel("cells")
ax[2].legend()

plt.tight_layout()
plt.show()

print(f"Alice {len(a_ids)} cells, Bob {len(b_ids)} cells, matched {len(cells)} "
      f"({len(cells) / len(a_ids):.0%} of Alice, {len(cells) / len(b_ids):.0%} of Bob)")

# %%
print(cells)
# %%
plt.quiver([0],[0], )