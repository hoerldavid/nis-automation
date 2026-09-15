"""
one-off: segment small bright GFP clusters *within* the cellpose nuclei of
the FRAP_GMT1_ESC time series (t=0 frame of every file).

Motivation: some cells show a uniform nuclear GFP distribution, others
carry discrete bright clusters. For cluster-targeted FRAP we need a
stimulation mask that (a) finds the clusters and (b) leaves uniform nuclei
empty (so the pipeline skips them).

Deliberately simple: per-nucleus Otsu thresholding on the nucleus pixels +
connected-component filtering (min area, max area fraction of the nucleus,
contrast of the cluster mean vs the nucleus median). No training, no
local/thresholding beyond Otsu.

Per file:
  - t=0 frame
  - cellpose (remote server, diameter=70) -> nucleus labels
  - border-touching nuclei discarded (pipeline convention)
  - per nucleus: Otsu on its pixels -> candidate clusters -> keep components
    with area >= MIN_CLUSTER_AREA and component mean >= CONTRAST * nucleus
    median; if the *total* kept area exceeds MAX_CLUSTER_FRAC * nucleus area
    the whole nucleus is set to 0 (coarse-diffuse / diffuse-heavy), like
    uniform nuclei

The production version of this logic (without the stats bookkeeping) lives
in autofrap.mask_utils: clusters_in_object() + cluster_stim_mask() (a
drop-in stim_mask_fun for build_detector; one cluster per cell via
largest/most_central_region_per_label; uniform nuclei stay empty and are
skipped by next_stimulatable_cell). Not yet wired into the pipeline. This
script self-checks per file that its per-nucleus classification agrees with
mask_utils.cluster_stim_mask.

Outputs (written under test_data/, which is git-ignored):
  - FRAP_GMT1_ESC_clusters_stats.csv  (one row per nucleus)
  - FRAP_GMT1_ESC_clusters_contact_sheet.png (all files, nucleus + cluster
    contours; title marks how many nuclei carry clusters)
  - FRAP_GMT1_ESC_clusters_zooms.png  (zoom crops: cluster-bearing nuclei
    and a few uniform ones, cluster mask overlaid)

Run from the repo root:
    CELLPOSE_SERVER_URL=http://localhost:8000 \
        python autofrap/autofrap_bitsnpieces/frap_gmt1_es_clusters.py
"""
import csv
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
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label
from skimage.segmentation import clear_border, relabel_sequential

from autofrap.core.detection import remote_detect_objects
from autofrap.core.image.mask import cluster_stim_mask

DATA_DIR = 'test_data/FRAP_GMT1_ESC'
DIAMETER = 70             # cellpose diameter, established for this family
MIN_CLUSTER_AREA = 15     # px, below this = noise speck
MAX_CLUSTER_FRAC = 0.2    # *total* cluster area must be < 20 % of the nucleus
CONTRAST = 1.5            # cluster mean must exceed CONTRAST * nucleus median

OUT_STATS = 'test_data/FRAP_GMT1_ESC_clusters_stats.csv'
OUT_SHEET = 'test_data/FRAP_GMT1_ESC_clusters_contact_sheet.png'
OUT_ZOOMS = 'test_data/FRAP_GMT1_ESC_clusters_zooms.png'


def clip(img):
    lo, hi = np.percentile(img, [1, 99.5])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    return float(lo), float(hi)


def cluster_stats_in_nucleus(img, nuc_mask, min_area, max_frac, contrast):
    """
    Stats-carrying twin of mask_utils.clusters_in_object (same filters,
    same order: per-cluster min area + contrast, then the total-area cap)
    used for the tuning CSV: also returns the per-nucleus numbers behind
    the clustered / coarse / uniform decision.

    Returns (cluster_mask, stats): cluster_mask is a 2D bool (same shape as
    img), True on kept cluster pixels; stats is a dict with the per-nucleus
    numbers. A coarse nucleus has an empty cluster_mask (like uniform
    nuclei), stats['coarse'] is True and its dropped regions are still
    listed in stats['cluster_areas'] for the CSV.
    """
    vals = img[nuc_mask]
    nuc_area = int(nuc_mask.sum())
    p50 = float(np.percentile(vals, 50))
    p99 = float(np.percentile(vals, 99))

    stats = {'nucleus_area': nuc_area, 'otsu': None, 'nuc_p50': p50,
             'nuc_p99': p99, 'n_candidates': 0, 'coarse': False,
             'total_frac': 0.0,
             'cluster_areas': [], 'cluster_fracs': [], 'contrasts': []}

    if vals.max() == vals.min():  # perfectly flat -> nothing to threshold
        return np.zeros(img.shape, bool), stats

    thr = float(threshold_otsu(vals))
    stats['otsu'] = thr
    cand = (img > thr) & nuc_mask
    lab = label(cand)
    n_lab = int(lab.max())
    stats['n_candidates'] = int(n_lab)

    kept = []  # (area, mean, component mask)
    for i in range(1, n_lab + 1):
        comp = lab == i
        area = int(comp.sum())
        mean = float(img[comp].mean())
        if area >= min_area and mean >= contrast * p50:
            kept.append((area, mean, comp))

    total_area = sum(a for a, _, _ in kept)
    stats['total_frac'] = round(total_area / nuc_area, 3)
    stats['coarse'] = total_area > max_frac * nuc_area
    for area, mean, comp in kept:
        stats['cluster_areas'].append(area)
        stats['cluster_fracs'].append(round(area / nuc_area, 3))
        stats['contrasts'].append(round(mean / p50, 2))
    if stats['coarse']:
        return np.zeros(img.shape, bool), stats

    keep = np.zeros(img.shape, bool)
    for _, _, comp in kept:
        keep |= comp
    return keep, stats


def process_file(f, server_url):
    m = re.search(r'\b(60min|90min|wo)\b', f)
    group = m.group(1) if m else '?'
    with ND2File(f) as n:
        img = np.array(n.read_frame(0))  # copy out of the file handle

    t0 = time.time()
    labels = remote_detect_objects(img, server_url=server_url,
                                   diameter=DIAMETER)
    if labels.max() == 0:
        return None
    labels = relabel_sequential(clear_border(labels))[0]
    dt = time.time() - t0

    # self-check against the production function (mask_utils)
    prod = cluster_stim_mask(labels, img,
                             min_cluster_area=MIN_CLUSTER_AREA,
                             max_cluster_frac=MAX_CLUSTER_FRAC,
                             contrast=CONTRAST, pick='largest')

    rows = []
    cluster_overlay = np.zeros(img.shape, bool)
    for i in range(1, labels.max() + 1):
        nuc_mask = labels == i
        keep, stats = cluster_stats_in_nucleus(img, nuc_mask,
                                               MIN_CLUSTER_AREA,
                                               MAX_CLUSTER_FRAC, CONTRAST)
        assert bool(keep.any()) == bool((prod & nuc_mask).any())
        cluster_overlay |= keep
        rows.append({
            'file': os.path.basename(f), 'group': group,
            'nucleus_id': i, 'clustered': int(bool(keep.any())),
            **stats,
        })
    return {
        'file': os.path.basename(f), 'group': group, 'img': img,
        'labels': labels, 'clusters': cluster_overlay, 'rows': rows,
        'dt': dt,
    }


def main():
    server_url = os.environ.get('CELLPOSE_SERVER_URL', 'http://localhost:8000')
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.nd2')))
    assert files, f'no nd2 files in {DATA_DIR}'
    print(f'{len(files)} files, diameter={DIAMETER} '
          f'min_cluster_area={MIN_CLUSTER_AREA}px max_cluster_frac={MAX_CLUSTER_FRAC} '
          f'(total per nucleus) contrast={CONTRAST}', flush=True)

    results = []
    for f in files:
        res = process_file(f, server_url)
        if res is None:
            print(f'{os.path.basename(f)[:52]:52s} no nuclei', flush=True)
            continue
        n_clustered = sum(r['clustered'] for r in res['rows'])
        n_coarse = sum(r['coarse'] for r in res['rows'])
        fracs = [x for r in res['rows'] if r['clustered']
                 for x in r['cluster_fracs']]
        fstr = (f'cluster frac {min(fracs):.2f}-{max(fracs):.2f}'
                if fracs else 'no clusters')
        print(f'{res["file"][:52]:52s} {len(res["rows"]):2d} nuclei, '
              f'{n_clustered:2d} clustered, {n_coarse:2d} coarse '  # noqa: E203 (column alignment with the line above)
              f'({fstr})  ({res["dt"]:.1f} s)',
              flush=True)
        results.append(res)

    # ---- stats CSV
    with open(OUT_STATS, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'group', 'nucleus_id', 'clustered', 'coarse',
                    'total_frac', 'nucleus_area', 'otsu', 'nuc_p50', 'nuc_p99',
                    'n_candidates', 'n_clusters', 'cluster_areas',
                    'cluster_fracs', 'contrasts'])
        for res in results:
            for r in res['rows']:
                w.writerow([r['file'], r['group'], r['nucleus_id'],
                            r['clustered'], int(r['coarse']), r['total_frac'],
                            r['nucleus_area'],
                            None if r['otsu'] is None else round(r['otsu'], 1),
                            round(r['nuc_p50'], 1), round(r['nuc_p99'], 1),
                            r['n_candidates'], len(r['cluster_areas']),
                            ';'.join(str(a) for a in r['cluster_areas']),
                            ';'.join(str(x) for x in r['cluster_fracs']),
                            ';'.join(str(c) for c in r['contrasts'])])
    print(f'saved {OUT_STATS}')

    # ---- contact sheet (whole images, nucleus + cluster contours)
    order = sorted(results, key=lambda r: (
        {'60min': 0, '90min': 1, 'wo': 2}.get(r['group'], 3), r['file']))
    ncol = 5
    nrow = (len(order) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 3.2, nrow * 3.2), dpi=100)
    axes = np.atleast_1d(axes).ravel()
    for ax, res in zip(axes, order):
        img, labels, clusters = res['img'], res['labels'], res['clusters']
        lo, hi = clip(img)
        ax.imshow(img, cmap='gray', vmin=lo, vmax=hi)
        for i in range(1, labels.max() + 1):
            for c in find_contours(labels == i, 0.5):
                ax.plot(c[:, 1], c[:, 0], color='white', lw=0.6, alpha=0.9)
        for c in find_contours(clusters, 0.5):
            ax.plot(c[:, 1], c[:, 0], color='yellow', lw=1.0)
        n_clustered = sum(r['clustered'] for r in res['rows'])
        n_coarse = sum(r['coarse'] for r in res['rows'])
        ax.set_title(f'{res["group"]} {res["file"][-5:]}: '
                     f'{len(res["rows"])}/{n_clustered}/{n_coarse}',
                     fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(order):]:
        ax.axis('off')
    fig.suptitle(f'FRAP_GMT1_ESC t=0: nucleus contours (white), '
                 f'cluster contours (yellow) — title: nuclei/clustered/coarse',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_SHEET, dpi=100)
    plt.close(fig)
    print(f'saved {OUT_SHEET}')

    # ---- zoom panel: best-clustered nucleus per file + a couple of uniform
    picks = []  # (res, nucleus_id, clustered)
    for res in order:
        by_id = {r['nucleus_id']: r for r in res['rows']}
        clustered = [i for i, r in by_id.items() if r['clustered']]
        uniform = [i for i, r in by_id.items()
                   if not r['clustered'] and not r['coarse']]
        if clustered:
            best = max(clustered, key=lambda i: len(by_id[i]['cluster_areas']))
            picks.append((res, best, True))
    if picks:  # 2 examples of uniform nuclei for contrast
        for res in order[:3]:
            uniform = [r['nucleus_id'] for r in res['rows']
                       if not r['clustered'] and not r['coarse']]
            if uniform:
                picks.append((res, uniform[0], False))
                if len([p for p in picks if not p[2]]) >= 2:
                    break

    npicks = len(picks)
    ncol = 4
    nrow = (npicks + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 4.0, nrow * 4.0), dpi=110)
    axes = np.atleast_1d(axes).ravel()
    for ax, (res, nid, clustered) in zip(axes, picks):
        img, labels, clusters = res['img'], res['labels'], res['clusters']
        m = labels == nid
        ys, xs = np.where(m)
        pad = 12
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
        lo, hi = clip(img)
        ax.imshow(img[y0:y1, x0:x1], cmap='gray', vmin=lo, vmax=hi)
        cm = (clusters & m)[y0:y1, x0:x1]
        if cm.any():
            ax.imshow(np.ma.masked_where(~cm, cm), cmap='autumn',
                      alpha=0.55, vmin=0, vmax=1)
        for c in find_contours(m[y0:y1, x0:x1], 0.5):
            ax.plot(c[:, 1], c[:, 0], color='white', lw=0.8)
        for c in find_contours(cm, 0.5):
            ax.plot(c[:, 1], c[:, 0], color='yellow', lw=1.0)
        row = next(r for r in res['rows'] if r['nucleus_id'] == nid)
        n_cl = len(row['cluster_areas'])
        ax.set_title(f'{res["group"]} {res["file"][-5:]} n{nid}: '
                     f'{"UNIFORM" if not clustered else f"{n_cl} cluster(s)"}',
                     fontsize=9)
        ax.set_axis_off()
    for ax in axes[npicks:]:
        ax.axis('off')
    fig.suptitle('zooms: nucleus (white), clusters (yellow overlay), '
                 'uniform nuclei marked', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_ZOOMS, dpi=110)
    plt.close(fig)
    print(f'saved {OUT_ZOOMS}')


if __name__ == '__main__':
    main()
