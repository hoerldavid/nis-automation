import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops
from skimage.morphology import dilation, disk as skdisk
from scipy.ndimage import distance_transform_edt

def make_synthetic(n_objects=30, shape=(512,512), seed=2):
    rng = np.random.default_rng(seed)
    image = rng.normal(loc=100, scale=10, size=shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    for i in range(1, n_objects+1):
        r = rng.integers(10, 25)
        cy = rng.integers(r, shape[0]-r)
        cx = rng.integers(r, shape[1]-r)
        rr, cc = disk((cy,cx), r, shape=shape)
        labels[rr, cc] = i
        image[rr, cc] += rng.uniform(20, 80)
    return image, labels

def surround_mean_dilation(labels, image, distance_px):
    img = image
    selem = skdisk(distance_px)
    vals = []
    for rp in regionprops(labels):
        if rp.label == 0: continue
        minr, minc, maxr, maxc = rp.bbox
        r0 = max(0, minr - distance_px)
        c0 = max(0, minc - distance_px)
        r1 = min(img.shape[0], maxr + distance_px)
        c1 = min(img.shape[1], maxc + distance_px)
        img_crop = img[r0:r1, c0:c1]
        mask_crop = (labels[r0:r1, c0:c1] == rp.label)
        dilated = dilation(mask_crop, footprint=selem)
        ring = np.logical_xor(dilated, mask_crop)
        vals.append(float(img_crop[ring].mean()) if np.any(ring) else np.nan)
    return np.array(vals)

def surround_mean_edt(labels, image, distance_px):
    img = image
    vals = []
    for rp in regionprops(labels):
        if rp.label == 0: continue
        minr, minc, maxr, maxc = rp.bbox
        r0 = max(0, minr - distance_px)
        c0 = max(0, minc - distance_px)
        r1 = min(img.shape[0], maxr + distance_px)
        c1 = min(img.shape[1], maxc + distance_px)
        img_crop = img[r0:r1, c0:c1]
        mask_crop = (labels[r0:r1, c0:c1] == rp.label)
        dist = distance_transform_edt(~mask_crop)
        ring = (dist > 0) & (dist <= distance_px)
        vals.append(float(img_crop[ring].mean()) if np.any(ring) else np.nan)
    return np.array(vals)

image, labels = make_synthetic()
for r in [5, 10, 15, 20]:
    v_dil = surround_mean_dilation(labels, image, r)
    v_edt = surround_mean_edt(labels, image, r)
    ok = np.allclose(v_dil, v_edt, equal_nan=True)
    maxdiff = np.nanmax(np.abs(v_dil - v_edt))
    print(f"r={r:2d}  match={ok}  max abs diff={maxdiff:.6e}")
