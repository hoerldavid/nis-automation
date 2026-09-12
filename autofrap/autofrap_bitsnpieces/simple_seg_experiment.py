"""
One-off experiment: simple threshold + watershed nucleus segmentation

A cellpose-free detector for faint nuclear staining (DAPI surveys and
GFP-tagged nuclear proteins), using only scipy/skimage operations.

Pipeline (parameters in the block below; all length scales derive from
CELL_SIGMA):

    1. highpass: image - gaussian(image, BG_SIGMA)          [mode='reflect']
    2. matched filter: gaussian(highpass, SMOOTH_SIGMA) - nucleus-scale
       smoothing; blurs punctate chromatin/clusters into solid blobs and
       keeps dim out-of-focus nuclei (a blurred blob matches a smooth
       template well)
    3. threshold: OTSU_FRAC x Otsu of the non-negative matched-filter
       image (full Otsu would sit at the level of the bright puncta).
       Optional per-dataset clip_pct: Otsu is then computed on a histogram
       clipped at the clip percentile - a small bright population (e.g.
       GFP clusters) no longer pushes the Otsu split up. The clipping
       only reshapes the histogram: as long as the resulting threshold is
       below the clip level (it is), the mask `sm > thr` is identical
       clipped or not.
    4. clean-up: binary closing x CLOSE_ITERS, drop blobs < MIN_OBJECT_AREA
    5. distance transform + local maxima as markers (peaks must be >=
       MARKER_REL of the per-blob max; >= MARKER_MIN_DIST apart) ->
       watershed to split touching nuclei
    6. merge over-splits: adjacent segments whose shared boundary sits
       deep inside the blob (median distance on the boundary >=
       MERGE_RIDGE_FRAC of the pair's max inscribed radius) are merged,
       iteratively
    7. drop merged-looking objects: erode each object ERODE_ITERS px
       (kills thin spiky tails), then drop it if the eroded object's
       area/convex-hull ratio < MIN_ERODED_EXTENT or erosion pinches it
       into two pieces

Inputs MUST be float: scipy ndimage.gaussian_filter returns the input
dtype, so with uint16 the high-pass underflows (wraps) and negative
values clip to 0. The high-pass is also what makes the OTSU_FRAC scaling
meaningful - without it, Otsu returns an absolute threshold that includes
the background offset, and OTSU_FRAC x that lands below the background
(whole image selected). If the high-pass is ever dropped, re-anchor the
threshold to the background level, e.g. median + frac x (otsu - median).

Outputs per file (test_data/simple_seg[_gmt1]/):
    <name>_{cleaned_mask,distance,watershed}.png
    <name>_contact_sheet.png  - all intermediate stages
    <name>_overlay.png        - final contours + IDs (dropped objects in
                                magenta) on the image
and a summary line per file. Run from the repo root:
  python -m autofrap.autofrap_bitsnpieces.simple_seg_experiment           (20260901 grid, DAPI)
  python -m autofrap.autofrap_bitsnpieces.simple_seg_experiment gmt1     (FRAP_GMT1_ESC t=0, GFP-DNMT1)

Results (final object counts after merging + extent filter):
  - 20260901 DAPI grid: 19/17/19/12 for fov01-04 vs 21/17/18/13 from
    cellpose at diameter=70. The extent filter correctly drops non-convex
    merged clusters (e.g. fov01's 3-nucleus blob); *convex* merged
    clusters still pass (an area cap vs the per-file median would catch
    them); the faintest out-of-focus objects remain missed.
  - FRAP_GMT1_ESC t=0 (GFP-DNMT1, no DNA stain; treated 60min/90min files
    show drug-induced clustering): the pipeline transfers (masks track
    C-shaped/kidney nuclei well), gmt1 runs with clip_pct=95 because the
    bright clusters push the Otsu split up (90min_003: threshold 60.8
    without clipping vs 14.0 with). But the extent filter - tuned on
    round DAPI nuclei - misfires on this family's C-shaped/ring-like
    single nuclei and drops many (e.g. 90min_001: 12 watershed -> 5
    final). It should become a per-family setting (or a valley-depth /
    local-contrast criterion) before this could be a real detector for
    that sample.

Alternative ideas explored (not adopted): pure matched-filter peak finding
(Gaussian template + NMS) finds the dim blobs in its response but a global
relative peak threshold drops them; Li/isodata thresholds are much lower
than Otsu (recover more faint objects, risk more noise); iterative
template subtraction (detect peak, subtract Gaussian, repeat) and a
local/region-adaptive threshold are the obvious next steps if the
faintest objects matter. A small local CNN (stardist-class, tens of
s/frame on CPU) is the fallback if classical tuning keeps fighting the
sample family.
"""
import glob
import os
import sys

# Ensure the repo root is on sys.path
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage import filters, morphology
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential, watershed

import autofrap.nd2_helpers as nd2h

# --- parameters ---------------------------------------------------------
# One length scale: the Gaussian sigma that blurs a typical nucleus into a
# single smooth blob (~half the nucleus diameter; 16 px at this pixel
# scale, i.e. ~2 um). All other length scales are derived from it:
CELL_SIGMA = 16.0
BG_SIGMA = 2 * CELL_SIGMA            # background estimate: must be larger
                                     # than a cell (slow illumination
                                     # variation)
SMOOTH_SIGMA = CELL_SIGMA            # matched filter / nucleus-scale
                                     # smoothing
MARKER_MIN_DIST = int(CELL_SIGMA)    # min distance between watershed
                                     # markers (the MARKER_REL height
                                     # filter does the real work)
MIN_OBJECT_AREA = int(round(np.pi * (CELL_SIGMA / 3) ** 2))
                                     # ~1/3 of the cell radius: smaller
                                     # blobs are noise
MIN_SEG_AREA = MIN_OBJECT_AREA
ERODE_ITERS = max(1, int(round(CELL_SIGMA / 8)))  # ~2 px; removes tails
                                     # thinner than ~4 px

# dimensionless knobs (ratios, scale-free - should transfer between
# pixel sizes without changes):
OTSU_FRAC = 0.3        # threshold at this fraction of the Otsu value
CLOSE_ITERS = 2        # binary closing iterations (3x3)
MARKER_REL = 0.6       # keep only distance peaks at >= this fraction of the
                       # per-blob maximum (drops secondary peaks inside a
                       # single nucleus, keeps both peaks of two touching ones)
MERGE_RIDGE_FRAC = 0.45  # merge adjacent segments if the shared-boundary
                       # distance ridge is >= this fraction of the pair's
                       # max inscribed radius (boundary deep inside the blob
                       # = over-split, not two touching nuclei)
MIN_ERODED_EXTENT = 0.90  # drop objects whose eroded area/convex-hull
                       # ratio is below this (merged blobs; tuned on round
                       # DAPI nuclei - misfires on C-shaped GFP nuclei)


def _merge_over_splits(lab, dist):
    """
    iteratively merge adjacent labels whose shared boundary sits deep inside
    the blob (over-splits of one nucleus, not two touching nuclei)
    """
    merged = 0
    for _ in range(50):
        props = regionprops(lab)
        best = None
        for i, a in enumerate(props):
            ma = lab == a.label
            da = dist[ma].max()
            for b in props[i + 1:]:
                mb = lab == b.label
                db = dist[mb].max()
                rmax = max(da, db)
                if rmax <= 0:
                    continue
                adj = binary_dilation(ma, np.ones((3, 3))) & mb
                if not adj.any():
                    continue
                frac = np.median(dist[adj]) / rmax
                if frac >= MERGE_RIDGE_FRAC and (best is None or frac > best[0]):
                    best = (frac, a.label, b.label)
        if best is None:
            break
        lab[lab == best[2]] = best[1]
        merged += 1
    return lab, merged


def _extent_filter(lab):
    """
    drop objects that look like merged nuclei

    Erode each object (kills thin spiky tails that inflate the convex
    hull), then the area/convex-hull ratio of the eroded object must be
    >= MIN_ERODED_EXTENT. Objects erosion pinsches into two pieces are
    dropped as well.
    """
    keep, dropped = [], []
    for rp in regionprops(lab):
        er = ndimage.binary_erosion(lab == rp.label, iterations=ERODE_ITERS)
        comps = label(er)
        if comps.max() == 0 or comps.max() > 1:
            dropped.append(rp.label)
            continue
        extent = er.sum() / regionprops(comps)[0].area_convex
        (keep if extent >= MIN_ERODED_EXTENT else dropped).append(rp.label)
    lab = np.isin(lab, list(keep)) * lab
    return lab, dropped


def segment(image, clip_pct=None):
    """
    simple threshold + watershed segmentation

    clip_pct: if given, Otsu is computed on the highpass clipped to its
    <clip_pct> percentile (keeps a small bright population from pushing
    the Otsu split up); the resulting threshold is applied to the
    unclipped image.

    Returns (labels, intermediates); intermediates carries 'highpass',
    'smooth', 'otsu_thr', 'mask' (cleaned binary), 'distance', 'markers',
    'n_markers', 'n_merged', 'dropped' (label IDs removed by the extent
    filter) and 'dropped_map' (the pre-filter label map, in which the
    dropped IDs are still valid - use it to draw the dropped contours).
    """
    image = image.astype(float)  # float required: gaussian_filter keeps dtype
    bg = ndimage.gaussian_filter(image, sigma=BG_SIGMA, mode='reflect')
    hp = image - bg
    sm = ndimage.gaussian_filter(hp, sigma=SMOOTH_SIGMA, mode='reflect')
    if clip_pct is None:
        thr_hist = np.clip(sm, 0, None)
    else:
        thr_hist = np.clip(sm, 0, np.percentile(sm, clip_pct))
    thr = filters.threshold_otsu(thr_hist) * OTSU_FRAC
    mask = sm > thr
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)),
                                  iterations=CLOSE_ITERS)
    mask = morphology.remove_small_objects(mask, max_size=MIN_OBJECT_AREA)

    dist = ndimage.distance_transform_edt(mask)
    markers = peak_local_max(dist, min_distance=MARKER_MIN_DIST,
                             labels=mask, threshold_rel=MARKER_REL,
                             exclude_border=True)
    # label the marker pixels (1..M) for watershed
    mlab = np.zeros(image.shape, dtype=int)
    mlab[markers[:, 0], markers[:, 1]] = np.arange(1, len(markers) + 1)
    lab = watershed(-dist, mask=mask, markers=mlab, connectivity=1)

    # drop tiny segments
    keep = set(r.label for r in regionprops(lab) if r.area > MIN_SEG_AREA)
    lab = np.isin(lab, list(keep)) * lab
    lab, _, _ = relabel_sequential(lab)
    n_watershed = lab.max()

    # merge over-splits, then drop merged-looking objects
    lab, n_merged = _merge_over_splits(lab, dist)
    dropped_map = lab  # keep for drawing the dropped contours
    lab, dropped = _extent_filter(lab)
    lab, _, _ = relabel_sequential(lab)
    return lab, dict(highpass=hp, smooth=sm, otsu_thr=thr, mask=mask,
                     distance=dist, markers=markers,
                     n_markers=len(markers), n_merged=n_merged,
                     n_watershed=n_watershed, dropped=dropped,
                     dropped_map=dropped_map)


def load_image(nd2_file):
    """
    channel 0 of a survey file; for files with a T dimension (FRAP time
    series, e.g. FRAP_GMT1_ESC) the t=0 frame
    """
    try:
        return nd2h.read_channel(nd2_file, 0).copy()
    except ValueError:
        import nd2
        with nd2.ND2File(nd2_file) as f:
            assert 'T' in f.sizes and 'C' not in f.sizes, nd2_file
            return f.asarray()[0].copy()  # (T, Y, X) -> frame 0


def _pctile_clip(img, lo=1.0, hi=99.5):
    lo_v, hi_v = np.percentile(img, [lo, hi])
    return np.clip(img, lo_v, hi_v)


def _contour(ax, lab, ids, color='lime', lw=1.2):
    from skimage.measure import find_contours
    for i in ids:
        for c in find_contours(lab == i, 0.5):
            ax.plot(c[:, 1], c[:, 0], color, lw=lw)


def plot_contact_sheet(image, inter, labels, path):
    h, w = image.shape
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
    thr = inter['otsu_thr']

    axes[0, 0].imshow(_pctile_clip(image), cmap='gray')
    axes[0, 0].set_title('image (1-99.5%)')
    axes[0, 1].imshow(_pctile_clip(inter['highpass'], 0, 99.5), cmap='gray')
    axes[0, 1].set_title(f'highpass (gauss {BG_SIGMA} px)')
    axes[0, 2].imshow(_pctile_clip(inter['smooth'], 0, 99.5), cmap='gray')
    axes[0, 2].set_title(
        f'smooth (sigma {SMOOTH_SIGMA}), thr={OTSU_FRAC}\u00d7Otsu={thr:.1f}')
    axes[0, 3].imshow(inter['mask'], cmap='gray')
    axes[0, 3].set_title('cleaned binary mask')

    axes[1, 0].imshow(inter['distance'], cmap='hot')
    axes[1, 0].set_title('distance transform')
    axes[1, 1].imshow(inter['distance'], cmap='hot')
    mk = inter['markers']
    axes[1, 1].plot(mk[:, 1], mk[:, 0], 'c+', ms=6)
    axes[1, 1].set_title(f'markers ({inter["n_markers"]})')
    axes[1, 2].imshow(labels, cmap='tab20')
    axes[1, 2].set_title(
        f'{labels.max()} objects ({inter["n_watershed"]} watershed, '
        f'{inter["n_merged"]} merged, {len(inter["dropped"])} dropped)')
    axes[1, 3].imshow(_pctile_clip(image), cmap='gray')
    _contour(axes[1, 3], labels, np.unique(labels)[1:])
    _contour(axes[1, 3], inter['dropped_map'], inter['dropped'],
             color='magenta')
    axes[1, 3].set_title('overlay on image (magenta = dropped)')

    for ax in axes.ravel():
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overlay(image, labels, inter, path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(_pctile_clip(image), cmap='gray')
    _contour(ax, labels, np.unique(labels)[1:], color='lime')
    _contour(ax, inter['dropped_map'], inter['dropped'], color='magenta')
    props = regionprops(labels)
    for rp in props:
        cy, cx = rp.centroid
        ax.text(cx, cy, str(rp.label), color='cyan', fontsize=8,
                ha='center', va='center')
    ax.set_title(f'simple threshold + watershed: {labels.max()} objects '
                 f'(+{len(inter["dropped"])} dropped, magenta)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


DATASETS = {
    # dataset: (glob pattern, output dir, name cleanup, clip_pct)
    'grid': (
        'test_acquisitions/autofrap_grid/20260901_160216/*/*survey*.nd2',
        'test_data/simple_seg',
        lambda n: n.replace('_survey.nd2', ''),
        None),  # no clipping: validated settings for the DAPI surveys
    'gmt1': (
        'test_acquisitions/FRAP_GMT1_ESC/*.nd2',
        'test_data/simple_seg_gmt1',
        lambda n: n.replace('.nd2', '').replace(' ', '_'),
        95),    # clip bright clusters out of the Otsu histogram
}


def main(dataset='grid'):
    pattern, out_rel, clean_name, clip_pct = DATASETS[dataset]
    files = sorted(glob.glob(os.path.join(_here, pattern)))
    files = [f for f in files
             if not os.path.basename(f).startswith('._')]  # macOS junk
    if not files:
        raise SystemExit('no survey files found')
    out_dir = os.path.join(_here, out_rel)
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        image = load_image(f).astype(float)
        name = clean_name(os.path.basename(f))
        labels, inter = segment(image, clip_pct=clip_pct)
        areas = np.array([r.area for r in regionprops(labels)])
        print(f'{name}: {inter["n_watershed"]} watershed -> '
              f'{inter["n_watershed"] - inter["n_merged"]} after merging -> '
              f'{labels.max()} final '
              f'(dropped merged-looking: {inter["dropped"]}); '
              f'areas min/med/max = '
              f'{int(areas.min())}/{int(np.median(areas))}/{int(areas.max())} px')
        for tag, arr in [('cleaned_mask', inter['mask']),
                         ('distance', inter['distance']),
                         ('watershed', labels)]:
            im = arr.astype(np.float32)
            if tag == 'distance':
                im = _pctile_clip(im, 0, 99.5)
            plt.imsave(os.path.join(out_dir, f'{name}_{tag}.png'), im,
                       cmap='gray' if tag != 'watershed' else 'tab20')
        plot_contact_sheet(image, inter, labels,
                           os.path.join(out_dir, f'{name}_contact_sheet.png'))
        plot_overlay(image, labels, inter,
                     os.path.join(out_dir, f'{name}_overlay.png'))
    print(f'plots in {out_dir}/')


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'grid'
    if dataset not in DATASETS:
        raise SystemExit(f'unknown dataset {dataset!r} '
                         f'(choose from {list(DATASETS)})')
    main(dataset)
