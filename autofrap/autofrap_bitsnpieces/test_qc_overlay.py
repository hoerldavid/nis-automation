"""
one-off visual test for qc.save_qc_overlay, on the copied
0013_ch1.tif image + 0013_ch1_cp_masks.tif cellpose labels
(run: python autofrap/autofrap_bitsnpieces/test_qc_overlay.py)

Feeds the overlay exactly what autofrap() would produce:
default half-cell stimulation mask, next_stimulatable_cell selection,
and the polygons mask_to_polygon sends to NIS.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # autofrap/

import matplotlib.pyplot as plt
import numpy as np
import tifffile

import detection
import qc
from autofrap import next_stimulatable_cell

ROOT = os.path.dirname(os.path.dirname(HERE))
IMAGE = os.path.join(ROOT, '0013_ch1.tif')
LABELS = os.path.join(ROOT, '0013_ch1_cp_masks.tif')

image = tifffile.imread(IMAGE)
labels = tifffile.imread(LABELS).astype(int)
n_obj = len(set(labels.ravel().tolist())) - 1

stim = detection.default_stimulation_mask(labels)
cell = next_stimulatable_cell(labels, set(), stim)
cell_poly = detection.mask_to_polygon(detection.cell_mask(labels, cell))
stim_poly = detection.mask_to_polygon(
    detection.cell_mask(labels, cell, stim))

print(f'{n_obj} objects, selected cell {cell} '
      f'({len(cell_poly)} + {len(stim_poly)} polygon vertices)')

# full artifact, as autofrap() would save it
qc.save_qc_overlay(
    image, labels, os.path.join(ROOT, '0013_qc_selected.png'),
    stimulation_mask=stim, cell_id=cell,
    cell_poly=cell_poly, stim_poly=stim_poly,
    caption=f'test  cell {cell} of {n_obj}')

# no selection: just image + labels + stim mask
qc.save_qc_overlay(
    image, labels, os.path.join(ROOT, '0013_qc_noselect.png'),
    stimulation_mask=stim,
    caption=f'test  no selection ({n_obj} objects)')

# highlight a different cell (largest area) to check generality
big = max((l for l in np.unique(labels) if l > 0),
          key=lambda l: (labels == l).sum())
qc.save_qc_overlay(
    image, labels, os.path.join(ROOT, '0013_qc_bigcell.png'),
    stimulation_mask=stim, cell_id=big,
    caption=f'test  largest cell {big}')

print('wrote 0013_qc_selected.png, 0013_qc_noselect.png, '
      '0013_qc_bigcell.png')

# --- coordinate check on a synthetic case (known expected pixels) ---
# 300x300; the fixed-font-size legend (top-left, ~rows 10-150, cols
# 10-200) keeps the checked regions clear, so the geometry sits
# bottom-right: gray square 20:290, one label 150:280 (centroid
# (215,215), so its ID text stays out of the stim box), stim 220:280,
# cell_poly = label boundary, stim_poly = stim boundary
simg = np.zeros((300, 300), float)
simg[20:290, 20:290] = 1.0
slbl = np.zeros((300, 300), int)
slbl[150:280, 150:280] = 1
sstim = np.zeros((300, 300), bool)
sstim[220:280, 220:280] = True
syn_path = os.path.join(ROOT, '0013_qc_synthetic.png')
qc.save_qc_overlay(
    simg, slbl, syn_path, stimulation_mask=sstim, cell_id=1,
    cell_poly=[(150, 150), (280, 150), (280, 280), (150, 280)],
    stim_poly=[(220, 220), (280, 220), (280, 280), (220, 280)])

out = plt.imread(syn_path)  # (h, w, 4) float 0..1
out = (out[..., :3] * 255).round().astype(int)
assert out.shape == (300, 300, 3), out.shape
failures = []

# orange stim fill: 0.3 alpha orange (255,165,0) over the bright square
# -> (255, 228, 179); sampled well inside the box, away from its border
# lines and from the legend (top-left)
box = out[240:270, 240:270].astype(int)
tinted = ((box[..., 0] > 240) & (200 < box[..., 1])
          & (box[..., 1] < 245) & (150 < box[..., 2]) & (box[..., 2] < 200))
if tinted.sum() < 600:
    failures.append(f'stim fill: only {tinted.sum()} tinted px in box')

# (selected-cell highlight is its cyan ID text + the ROI polygons;
# no contour any more - visual check by eye, see 0013_qc_*.png)

# cyan cell_poly along y = 150 (top edge), sampled right of the legend
cyan = [out[150, x] for x in range(210, 241)]
if not any(p[0] < 120 and p[1] > 180 and p[2] > 180 for p in cyan):
    failures.append('no cyan cell_poly pixels along y=150')

# magenta stim_poly along y = 280 (bottom edge, x = 220..280)
magenta = [out[280, x] for x in range(226, 251)]
if not any(p[0] > 150 and p[1] < 100 and p[2] > 150 for p in magenta):
    failures.append('no magenta stim_poly pixels along y=280')

# outside the gray square stays black (no stray layers)
if out[5, 5].max() > 30:
    failures.append(f'corner (5,5) not dark: {out[5, 5]}')

# RGB(A) input (y, x, 3) is shown as-is, no grayscale/clipping
rgb = np.zeros((300, 300, 3), np.uint8)
rgb[200:280, 200:280] = (255, 0, 0)
rgb_path = os.path.join(ROOT, '0013_qc_rgb.png')
qc.save_qc_overlay(rgb, slbl, rgb_path)
outr = (plt.imread(rgb_path)[..., :3] * 255).round().astype(int)
# (240,240) is inside the red square, clear of label contour/text/legend
if abs(int(outr[240, 240, 0]) - 255) > 10 or outr[240, 240, 1] > 10 \
        or outr[240, 240, 2] > 10:
    failures.append(f'RGB input not shown as-is: {tuple(outr[240, 240])}')

if failures:
    for f in failures:
        print('FAIL', f)
    sys.exit(1)
print('synthetic coordinate check: all layers land where expected')
