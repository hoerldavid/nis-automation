"""
Quick sanity + timing tests for bbox-local mask utilities.
Run with: PYTHONPATH=/workspace python autofrap/autofrap_bitsnpieces/test_mask_bbox_perf.py
"""
import time
import numpy as np
from skimage.draw import ellipse, disk
from autofrap.core.image.mask import (
    half_object_stim_mask,
    random_circle_stim_mask,
    largest_region_per_label,
    most_central_region_per_label,
    mask_to_polygon,
)

def make_labels(shape=(1024, 1024), n=20, radius_range=(30, 80), seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=int)
    for i in range(1, n+1):
        r = int(rng.integers(*radius_range))
        cy = int(rng.integers(r, shape[0]-r))
        cx = int(rng.integers(r, shape[1]-r))
        rr, cc = disk((cy, cx), r, shape=shape)
        labels[rr, cc] = i
    return labels

def test_half_object():
    labels = make_labels(shape=(1024,1024), n=50, seed=1)
    t0 = time.perf_counter()
    stim = half_object_stim_mask(labels)
    t1 = time.perf_counter()
    # sanity: stim is subset of labels>0
    assert np.all(stim <= (labels>0))
    # each label gets ~half area
    from skimage.measure import regionprops
    for rp in regionprops(labels):
        obj = labels == rp.label
        stim_obj = stim & obj
        # area should be roughly half, within 10%
        if obj.sum() > 0:
            frac = stim_obj.sum() / obj.sum()
            assert 0.35 < frac < 0.65, f"label {rp.label} frac {frac}"
    print(f"half_object_stim_mask: {t1-t0:.4f}s, ok")

def test_random_circle():
    labels = make_labels(shape=(1024,1024), n=50, seed=2)
    t0 = time.perf_counter()
    stim = random_circle_stim_mask(labels, area_fraction=0.25, seed=42)
    t1 = time.perf_counter()
    # sanity: stim inside labels
    assert np.all(stim <= (labels>0))
    # each label has at most one connected component
    from skimage.measure import label
    for lbl in np.unique(labels):
        if lbl==0: continue
        comp = label(stim & (labels==lbl), connectivity=1)
        assert comp.max() <= 1, f"label {lbl} has {comp.max()} components"
    print(f"random_circle_stim_mask: {t1-t0:.4f}s, ok")

def test_largest_central():
    labels = make_labels(shape=(512,512), n=30, seed=3)
    # create mask with two blobs per label
    mask = np.zeros_like(labels, dtype=bool)
    from skimage.measure import regionprops
    for rp in regionprops(labels):
        minr,minc,maxr,maxc = rp.bbox
        # put two small disks inside bbox
        rr, cc = disk((minr+5, minc+5), 3, shape=labels.shape)
        mask[rr, cc] = True
        rr, cc = disk((maxr-5, maxc-5), 3, shape=labels.shape)
        mask[rr, cc] = True
    # also ensure mask overlaps label
    mask &= (labels>0)
    t0 = time.perf_counter()
    reduced = largest_region_per_label(labels, mask)
    t1 = time.perf_counter()
    # reduced should be subset
    assert np.all(reduced <= mask)
    # most central
    t0 = time.perf_counter()
    reduced2 = most_central_region_per_label(labels, mask)
    t1 = time.perf_counter()
    assert np.all(reduced2 <= mask)
    print(f"largest_region_per_label + most_central_region_per_label: ok")

def test_mask_to_polygon():
    # create a simple ellipse mask
    mask = np.zeros((200,200), dtype=bool)
    rr, cc = ellipse(100,100,60,40)
    mask[rr, cc] = True
    t0 = time.perf_counter()
    poly = mask_to_polygon(mask, tolerance=2.0)
    t1 = time.perf_counter()
    assert len(poly) > 0
    print(f"mask_to_polygon: {t1-t0:.4f}s, {len(poly)} vertices, ok")

if __name__ == "__main__":
    test_half_object()
    test_random_circle()
    test_largest_central()
    test_mask_to_polygon()
    print("All bbox-local tests passed.")
