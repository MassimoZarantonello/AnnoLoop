# Concurrent annotations: blending several annotators into one mask

Status: plan agreed, not built. The points under *To decide* (thresholds, blanked area) are still open. The annotation tool is
[AnnoCell](../../AnnoCell) (its `docs/annotation_tool.md`).

## Setting

- Several annotators annotate **every ROI**, **independently**, all starting from **the same proposal**: the same
  segmentation by the model (the same `start_masks.tif`). Every cell starts unreviewed (yellow); an annotator's image
  takes part only when it is **complete** (no unreviewed cell left): every cell of it is a decision, never an
  abstention.
- There is no adjudicator: the annotations are **blended automatically** into one mask per ROI, because Cellpose
  trains on exactly one instance mask per image and has no "ignore" label.
- The same blending gives the agreement between annotators (the noise ceiling: model differences smaller than it are
  not real).

## Where it lives

Only in the **training code** (`macroseg.train`, when the training set is built). The annotation tool and the round
folders do not change: each annotator's folder holds only that annotator's decisions, nobody sees a consensus, and no
consensus mask is written next to the annotations. The blending is computed when training starts, from the
annotators' complete folders, and recorded with the model (in `model.json`: annotators per ROI, kept / dropped /
disputed groups, blanked fraction), so the training set can be rebuilt exactly.

The same blending function is also needed to **score models on a test ROI annotated by several annotators** (the
consensus is the ground truth there, with disputed areas left out); scoring calls it from the training code rather
than having its own copy.

## IDs: names, not identities

In each annotator's `masks.tif` every pixel holds a number, 0 for background and one number per cell. The numbers are
only name tags, given separately in each file: an untouched or edited proposal cell keeps its proposal number, but a
drawn cell gets the next free number, so every annotator's first drawn cell has the same number wherever it is.
**Number 17 in A's file and number 17 in B's file need not be the same cell.** Identity is decided by **overlap**;
the consensus then gets its own new numbers.

## The plan

### 1. Match: group the versions of each cell

- Compare every cell with the overlapping cells of the **other** annotators (never with cells of the same annotator:
  those are different cells by definition), by IoU = pixels both cover / pixels at least one covers.
- Pairs with **IoU ≥ 0.5** are the same cell. Above 0.5 a cell can match at most one cell of another annotator (it
  cannot share more than half with two cells that do not overlap each other).
- Join the pairs into groups **from the strongest overlap down**, with **at most one cell per annotator per group**: a
  pair that would put a second cell of the same annotator into a group is skipped (that annotator would vote twice,
  e.g. after splitting a cell). A cell left alone is a group of its own.
- All cells are matched by overlap, proposal cells included (untouched ones match with IoU 1); the proposal numbers
  can serve as a consistency check.

### 2. Decide, per group

| Group | Decision |
|---|---|
| fewer than half of the annotators have a cell in it | **dropped**: the majority said "no cell here"; its pixels are background, not blanked |
| at least half are in it, and at least half of the annotators agree with the medoid | **kept**, with the medoid's outline |
| at least half are in it, but fewer than half agree with the medoid | **disputed**: a cell exists, nobody agrees where it ends; its area is blanked in training |

- **Medoid**: the most typical version, the one with the highest mean IoU with the other versions in its group; the
  consensus cell is that annotator's cell exactly as drawn (always a smooth, human outline). Seen as vectors of 0 / 1
  over the group's pixels, it is the member of the group closest to all others in Jaccard distance (1 − IoU).
- **Agree with the medoid**: IoU ≥ 0.7 with it; the medoid counts itself.
- **IDs**: every kept cell gets a new sequential number.

### 3. Training

- Kept cells are the mask.
- Disputed areas are **blanked**: background in the mask *and* in the image, filled per channel with the median of
  the image's background pixels, grown by ~2 px so no membrane fragment is left; except pixels a kept cell covers. A
  training tile where blanking covers more than ~5 % is dropped instead.
- On the test ROI, disputed areas are left out of the score on both sides, not blanked.

## Two examples

The cells drawn as stretches along one row of pixels. Three annotators A, B, C; C drew **two** cells (C10, C9) where
the others drew one.

### Case 1: nobody agrees — disputed

```
pixels:  0    20   40   60   80   100  120  140
A        [=========================]                 0–100
B             [=========================]            25–125
C10      [=============]                             0–55
C9                     [================]            60–125
```

| Pair | Shared | Union | IoU |
|---|---|---|---|
| A–B | 25–100 = 75 | 0–125 = 125 | 0.60 |
| B–C9 | 60–125 = 65 | 25–125 = 100 | 0.65 |
| A–C10 | 0–55 = 55 | 0–100 = 100 | 0.55 |
| A–C9 | 60–100 = 40 | 0–125 = 125 | 0.32 |
| B–C10 | 25–55 = 30 | 0–125 = 125 | 0.24 |

- Groups, strongest first: B–C9 gives {B, C9}; A–B gives {A, B, C9}; A–C10 would put a second C cell in the group,
  skipped: {C10} alone.
- {A, B, C9}: 3 of 3 in it. Medoid: A (0.60 + 0.32) / 2 = 0.46, **B (0.60 + 0.65) / 2 = 0.63**, C9 (0.32 + 0.65) / 2
  = 0.49. Agree with B (IoU ≥ 0.7): only B itself, 1 of 3 → **disputed**: all three put the cell in different places.
- {C10}: 1 of 3 → dropped.

### Case 2: A and B agree, C split the cell by mistake — kept

```
pixels:  0    20   40   60   80   100  120  140
A        [=========================]                 0–100
B         [=========================]                5–105
C10      [=============]                             0–55
C9                     [================]            60–125
```

| Pair | IoU |
|---|---|
| A–B | 95 / 105 = 0.90 |
| A–C10 | 55 / 100 = 0.55 |
| B–C10 | 50 / 105 = 0.48 (no match) |
| B–C9 | 45 / 120 = 0.38 (no match) |
| A–C9 | 40 / 125 = 0.32 (no match) |

- Groups: A–B gives {A, B}; A–C10 (C not in the group yet) gives {A, B, C10}; C9 matches nothing: {C9}.
- {A, B, C10}: medoid **A (0.90 + 0.55) / 2 = 0.73**, B (0.90 + 0.48) / 2 = 0.69, C10 (0.55 + 0.48) / 2 = 0.52.
  Agree with A: A and B (0.90), 2 of 3 → **kept, with A's outline**. C's split is outvoted.
- {C9}: 1 of 3 → dropped. Its pixels beyond the kept cell (100–125) are background: A and B both said so, 2 of 3.

## Number of annotators

| Annotators | "At least half" | Effect |
|---|---|---|
| 2 | 1 of 2 | any cell either annotator has is kept (a cell deleted by one and kept by the other survives); a group of 2 is disputed if their IoU is below 0.7; the medoid of 2 is a tie (both have the same mean IoU): a fixed rule picks, e.g. the edited version over the untouched proposal |
| 3 | 2 of 3 | true majority |
| 4 | 2 of 4 | ties lean towards keeping |
| 5 | 3 of 5 | true majority |

An odd number, 3 or more, avoids ties.

## Comparing annotators

Per pair of annotators, only where at least one of them changed something (elsewhere they agree by construction, from
the shared proposal): AP@0.5 (do they agree on which cells exist, splits and merges included) and the mean Dice of
the matched cells (how close the outlines are). Per annotator: agreement with the consensus of the *others* (leave
one out), to spot an annotator who reads the guidelines differently. Per ROI: the number of kept, dropped and
disputed groups.

The consensus inherits the proposal's bias (everyone was anchored to it): the test ROI drawn from scratch, by several
annotators and blended the same way, stays essential.

## Decided

- **At least 2 annotators per ROI**, the minimum for which the method works; 3 or more (odd) when possible, for a real
  majority.
- **No uncertain state**: the annotator toggles OK with `O` (green / back to yellow), deletes, draws. A cell they are
  unsure about must still be decided (OK or deleted): the image counts only when every cell is OK. Blanking is only
  for disputed areas between annotators, never per cell.

## To decide

- **Thresholds**, proposed as a start, to adjust after the first real round:
  - **0.5**: two annotators' cells are the same cell (below it, one cell could match two);
  - **0.7**: an annotator agrees with the chosen outline (medoid); higher = stricter, more disputed cells;
  - **5 %**: a training tile where blanking would cover more is left out instead.
- **The blanked area of a disputed group**: everything any annotator covered (the union: safe, blanks a bit more), or
  only what most of them covered (blanks less, but an edge some drew stays visible as background).

## Alternatives considered

- **Pixel majority** (a pixel is in the cell if half or more of the group's annotators include it): the pixel-wise
  median of the versions, the mask closest to their average (the centroid, which per pixel is the fraction of
  annotators who included it). Uses every annotator's pixels, but builds an outline nobody drew, which can be ragged
  where they disagree at the edge; it would need a ragged check (perimeter, pieces, holes, solidity, compared with the
  versions). The unrounded centroid stays useful as an agreement map.
- **STAPLE** (Warfield et al., 2004): weights each annotator by an estimated reliability instead of one vote each.
