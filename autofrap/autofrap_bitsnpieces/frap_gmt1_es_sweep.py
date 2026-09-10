"""
one-off: cellpose (remote server, diameter=70) on the t=0 frame of every
time series in FRAP_GMT1_ESC/ (GFP-tagged nuclear protein; files are
single-channel (T, Y, X)).

Runs detection on all 21 files, prints per-file object counts, and saves
a contact sheet (clipped grayscale + label contours, grouped by
60min / 90min / wo) for a visual reliability check.

Run from the repo root:
    CELLPOSE_SERVER_URL=http://localhost:8000 \
        python autofrap/autofrap_bitsnpieces/frap_gmt1_es_sweep.py
"""
import glob
import os
import re
import sys
import time

# Ensure the repo root is on sys.path
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nd2 import ND2File
from skimage.measure import find_contours

from autofrap.detection import remote_detect_objects

DATA_DIR = 'FRAP_GMT1_ESC'
DIAMETER = 70
OUT = 'FRAP_GMT1_ESC_t0_cellpose_contact_sheet.png'


def clip(img):
    lo, hi = np.percentile(img, [1, 99.5])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    return float(lo), float(hi)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.nd2')))
    assert files, f'no nd2 files in {DATA_DIR}'
    print(f'{len(files)} files, diameter={DIAMETER}', flush=True)

    items = []  # (group, idx, elapsed, n_obj, img, labels)
    for f in files:
        m = re.search(r'\b(60min|90min|wo)\b', f)
        group = m.group(1) if m else '?'
        with ND2File(f) as n:
            img = np.array(n.read_frame(0))  # copy out of the file handle
        t0 = time.time()
        labels = remote_detect_objects(img,
                                       server_url=os.environ.get(
                                           'CELLPOSE_SERVER_URL',
                                           'http://10.163.69.12:8000'),
                                       diameter=DIAMETER)
        dt = time.time() - t0
        n_obj = int(labels.max())
        print(f'{os.path.basename(f)[:52]:52s} {n_obj:3d} objects  ({dt:.1f} s)',
              flush=True)
        items.append((group, os.path.basename(f), dt, n_obj, img, labels))

    # summary
    counts = sorted(it[3] for it in items)
    print(f'\nobjects per file: min={counts[0]} '
          f'median={counts[len(counts)//2]} max={counts[-1]}')
    zeros = [it[1] for it in items if it[3] == 0]
    print('zero-object files:', zeros if zeros else 'none')
    per_group = {}
    for it in items:
        per_group[it[0]] = per_group.get(it[0], 0) + it[3]
    print('total objects per group:', dict(sorted(per_group.items())))

    # contact sheet: 5 columns, rows per group order (60min, 90min, wo)
    order = sorted(items, key=lambda it: (
        {'60min': 0, '90min': 1, 'wo': 2}.get(it[0], 3), it[1]))
    ncol = 5
    nrow = (len(order) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 3.2, nrow * 3.2), dpi=100)
    axes = np.atleast_1d(axes).ravel()
    for ax, it in zip(axes, order):
        group, name, dt, n_obj, img, labels = it
        lo, hi = clip(img)
        ax.imshow(img, cmap='gray', vmin=lo, vmax=hi)
        for prop_label in np.unique(labels):
            if prop_label == 0:
                continue
            for contour in find_contours(labels == prop_label, 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color='white',
                        lw=0.6, alpha=0.9)
        ax.set_title(f'{group} {name[-5:]}: {n_obj}', fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(order):]:
        ax.axis('off')
    fig.suptitle(f'FRAP_GMT1_ESC t=0, cellpose diameter={DIAMETER}', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=100)
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
