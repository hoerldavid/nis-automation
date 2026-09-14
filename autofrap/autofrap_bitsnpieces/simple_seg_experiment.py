"""
One-off experiment: simple threshold + watershed nucleus segmentation

Now uses the refactored core detector in autofrap.core.detection.simple_seg.
Plotting and dataset looping stay here.
"""
import glob
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops

import autofrap.nd2_helpers as nd2h
from autofrap.core.simple_seg import SimpleSegParams, detect_objects

# --- params for the two datasets ---
GRID_PARAMS = SimpleSegParams(cell_sigma=16.0, otsu_frac=0.3, min_eroded_extent=0.90)
GMT1_PARAMS = SimpleSegParams(cell_sigma=16.0, otsu_frac=0.3, min_eroded_extent=0.70)

def load_image(nd2_file):
    try:
        return nd2h.read_channel(nd2_file, 0).copy()
    except ValueError:
        import nd2
        with nd2.ND2File(nd2_file) as f:
            assert 'T' in f.sizes and 'C' not in f.sizes, nd2_file
            return f.asarray()[0].copy()

def _pctile_clip(img, lo=1.0, hi=99.5):
    lo_v, hi_v = np.percentile(img, [lo, hi])
    return np.clip(img, lo_v, hi_v)

def _contour(ax, lab, ids, color='lime', lw=1.2):
    from skimage.measure import find_contours
    for i in ids:
        for c in find_contours(lab == i, 0.5):
            ax.plot(c[:, 1], c[:, 0], color, lw=lw)

def plot_contact_sheet(image, inter, labels, path, params):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
    thr = inter['otsu_thr']
    d = params.derived()
    axes[0, 0].imshow(_pctile_clip(image), cmap='gray')
    axes[0, 0].set_title('image (1-99.5%)')
    axes[0, 1].imshow(_pctile_clip(inter['highpass'], 0, 99.5), cmap='gray')
    axes[0, 1].set_title(f'highpass (gauss {d["bg_sigma"]} px)')
    axes[0, 2].imshow(_pctile_clip(inter['smooth'], 0, 99.5), cmap='gray')
    axes[0, 2].set_title(f'smooth (sigma {d["smooth_sigma"]}), thr={params.otsu_frac}×Otsu={thr:.1f}')
    axes[0, 3].imshow(inter['mask'], cmap='gray')
    axes[0, 3].set_title('cleaned binary mask')
    axes[1, 0].imshow(inter['distance'], cmap='hot')
    axes[1, 0].set_title('distance transform')
    axes[1, 1].imshow(inter['distance'], cmap='hot')
    mk = inter['markers']
    if mk.size:
        axes[1, 1].plot(mk[:, 1], mk[:, 0], 'c+', ms=6)
    axes[1, 1].set_title(f'markers ({inter["n_markers"]})')
    axes[1, 2].imshow(labels, cmap='tab20')
    axes[1, 2].set_title(f'{labels.max()} objects ({inter["n_watershed"]} watershed, {inter["n_merged"]} merged, {len(inter["dropped"])} dropped)')
    axes[1, 3].imshow(_pctile_clip(image), cmap='gray')
    _contour(axes[1, 3], labels, np.unique(labels)[1:])
    _contour(axes[1, 3], inter['dropped_map'], inter['dropped'], color='magenta')
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
        ax.text(cx, cy, str(rp.label), color='cyan', fontsize=8, ha='center', va='center')
    ax.set_title(f'simple threshold + watershed: {labels.max()} objects (+{len(inter["dropped"])} dropped, magenta)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

DATASETS = {
    'grid': (
        'test_acquisitions/autofrap_grid/20260901_160216/*/*survey*.nd2',
        'test_data/simple_seg',
        lambda n: n.replace('_survey.nd2', ''),
        GRID_PARAMS,
        None),
    'gmt1': (
        'test_acquisitions/FRAP_GMT1_ESC/*.nd2',
        'test_data/simple_seg_gmt1',
        lambda n: n.replace('.nd2', '').replace(' ', '_'),
        GMT1_PARAMS,
        95),
}

def main(dataset='grid'):
    pattern, out_rel, clean_name, params, clip_pct = DATASETS[dataset]
    files = sorted(glob.glob(os.path.join(_here, pattern)))
    files = [f for f in files if not os.path.basename(f).startswith('._')]
    if not files:
        raise SystemExit('no survey files found')
    out_dir = os.path.join(_here, out_rel)
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        image = load_image(f).astype(float)
        name = clean_name(os.path.basename(f))
        labels, inter = detect_objects(image, params, clip_pct=clip_pct, return_intermediates=True)
        areas = np.array([r.area for r in regionprops(labels)])
        print(f'{name}: {inter["n_watershed"]} watershed -> {inter["n_watershed"] - inter["n_merged"]} after merging -> {labels.max()} final (dropped {inter["dropped"]}); areas min/med/max = {int(areas.min()) if areas.size else 0}/{int(np.median(areas)) if areas.size else 0}/{int(areas.max()) if areas.size else 0} px')
        for tag, arr in [('cleaned_mask', inter['mask']), ('distance', inter['distance']), ('watershed', labels)]:
            im = arr.astype(np.float32)
            if tag == 'distance':
                im = _pctile_clip(im, 0, 99.5)
            plt.imsave(os.path.join(out_dir, f'{name}_{tag}.png'), im, cmap='gray' if tag != 'watershed' else 'tab20')
        plot_contact_sheet(image, inter, labels, os.path.join(out_dir, f'{name}_contact_sheet.png'), params)
        plot_overlay(image, labels, inter, os.path.join(out_dir, f'{name}_overlay.png'))
    print(f'plots in {out_dir}/')

if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'grid'
    if dataset not in DATASETS:
        raise SystemExit(f'unknown dataset {dataset!r} (choose from {list(DATASETS)})')
    main(dataset)
