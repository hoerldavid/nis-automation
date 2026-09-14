"""
Timing comparison: binary dilation vs EDT for surround intensity filter.
Synthetic data with ~100 objects.
"""
import time
import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops
from skimage.morphology import dilation, disk as skdisk
from scipy.ndimage import distance_transform_edt

def make_synthetic(n_objects=100, shape=(1024,1024), seed=0):
    rng = np.random.default_rng(seed)
    image = rng.normal(loc=100, scale=10, size=shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    for i in range(1, n_objects+1):
        r = rng.integers(15, 30)
        cy = rng.integers(r, shape[0]-r)
        cx = rng.integers(r, shape[1]-r)
        rr, cc = disk((cy,cx), r, shape=shape)
        labels[rr, cc] = i
        image[rr, cc] += rng.uniform(20, 80)
    return image, labels

def filter_surround_dilation(labels, image, distance_px, channel=0, metric='mean'):
    if image.ndim == 3:
        img = image[channel]
    else:
        img = image
    selem = skdisk(distance_px)
    good = []
    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        r0 = max(0, minr - distance_px)
        c0 = max(0, minc - distance_px)
        r1 = min(img.shape[0], maxr + distance_px)
        c1 = min(img.shape[1], maxc + distance_px)
        img_crop = img[r0:r1, c0:c1]
        mask_crop = (labels[r0:r1, c0:c1] == rp.label)
        dilated = dilation(mask_crop, footprint=selem)
        ring = np.logical_xor(dilated, mask_crop)
        if not np.any(ring):
            continue
        vals = img_crop[ring]
        val = float(vals.mean()) if metric=='mean' else float(np.median(vals))
        good.append(val)
    return good

def filter_surround_edt(labels, image, distance_px, channel=0, metric='mean'):
    if image.ndim == 3:
        img = image[channel]
    else:
        img = image
    good = []
    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        r0 = max(0, minr - distance_px)
        c0 = max(0, minc - distance_px)
        r1 = min(img.shape[0], maxr + distance_px)
        c1 = min(img.shape[1], maxc + distance_px)
        img_crop = img[r0:r1, c0:c1]
        mask_crop = (labels[r0:r1, c0:c1] == rp.label)
        dist = distance_transform_edt(~mask_crop)
        ring = (dist > 0) & (dist <= distance_px)
        if not np.any(ring):
            continue
        vals = img_crop[ring]
        val = float(vals.mean()) if metric=='mean' else float(np.median(vals))
        good.append(val)
    return good

image, labels = make_synthetic(n_objects=120, shape=(1024,1024), seed=1)
radii = [5, 10, 15, 20]

print(f"Objects: {len(np.unique(labels))-1}, image shape: {image.shape}")
print("\nRadius | Dilation [s] | EDT [s] | Speedup EDT/Dilation")
print("-"*55)
for r in radii:
    t0 = time.perf_counter()
    filter_surround_dilation(labels, image, distance_px=r)
    t_dil = time.perf_counter() - t0

    t0 = time.perf_counter()
    filter_surround_edt(labels, image, distance_px=r)
    t_edt = time.perf_counter() - t0

    speedup = t_dil / t_edt if t_edt>0 else float('nan')
    print(f"{r:6d} | {t_dil:11.4f} | {t_edt:7.4f} | {speedup:14.2f}")

print("\nDone.")
