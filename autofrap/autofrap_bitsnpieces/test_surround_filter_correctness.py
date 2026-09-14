"""
Check that surround filter actually distinguishes objects by surrounding intensity.
"""
import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops
from skimage.morphology import dilation, disk as skdisk
from scipy.ndimage import distance_transform_edt

def filter_surround(labels, image, distance_px, threshold, metric='mean', method='edt'):
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
        if method == 'dilation':
            selem = skdisk(distance_px)
            dilated = dilation(mask_crop, footprint=selem)
            ring = np.logical_xor(dilated, mask_crop)
        else:
            dist = distance_transform_edt(~mask_crop)
            ring = (dist > 0) & (dist <= distance_px)
        if not np.any(ring):
            continue
        vals = img_crop[ring]
        val = float(vals.mean()) if metric == 'mean' else float(np.median(vals))
        if val > threshold:
            good.append(rp.label)
    return good

# synthetic image with two background levels
shape = (256, 256)
image = np.full(shape, 30, dtype=np.float32)
image[:, shape[1]//2:] = 150  # right half bright background

labels = np.zeros(shape, dtype=np.int32)
# place objects in left low background
for i, (cy,cx) in enumerate([(80,60), (180,70), (120,100)], start=1):
    rr, cc = disk((cy,cx), 15, shape=shape)
    labels[rr, cc] = i

# place objects in right high background
for i, (cy,cx) in enumerate([(80,180), (180,200), (120,220)], start=4):
    rr, cc = disk((cy,cx), 15, shape=shape)
    labels[rr, cc] = i

print("Background left ~30, right ~150")
print("Objects 1-3 in left, 4-6 in right")

threshold = 80
good_dil = filter_surround(labels, image, distance_px=10, threshold=threshold, method='dilation')
good_edt = filter_surround(labels, image, distance_px=10, threshold=threshold, method='edt')

print(f"Dilation method kept labels: {good_dil}")
print(f"EDT method kept labels: {good_edt}")

assert set(good_dil) == {4,5,6}, "Dilation filter did not select right side objects"
assert set(good_edt) == {4,5,6}, "EDT filter did not select right side objects"
print("Both methods correctly filtered by surrounding intensity.")
