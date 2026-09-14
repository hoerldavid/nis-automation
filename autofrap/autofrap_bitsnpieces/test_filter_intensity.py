"""
Synthetic test for filter_intensity_inside

Creates a 256x256 image with three circular objects of different mean intensity
and checks that the filter keeps only the objects above threshold.
"""
import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops

# --- the function under test ---
def filter_intensity_inside(labels, image, channel=0, metric='mean', threshold=0.0):
    if image.ndim == 3:
        img = image[channel]
    else:
        img = image

    if metric not in ('mean', 'median'):
        raise ValueError("metric must be 'mean' or 'median'")

    good = []
    for rp in regionprops(labels, intensity_image=img):
        if rp.label == 0:
            continue
        if metric == 'mean':
            val = rp.mean_intensity
        else:
            mask = labels == rp.label
            val = float(np.median(img[mask]))
        if val > threshold:
            good.append(int(rp.label))
    return good

# --- synthetic data ---
rng = np.random.default_rng(42)
shape = (256, 256)
image = rng.normal(loc=50, scale=5, size=shape).astype(np.float32)

labels = np.zeros(shape, dtype=np.int32)

# object 1: bright
cy, cx, r = 80, 80, 20
rr, cc = disk((cy, cx), r, shape=shape)
image[rr, cc] += 200  # bright
labels[rr, cc] = 1

# object 2: dim
cy, cx, r = 80, 180, 20
rr, cc = disk((cy, cx), r, shape=shape)
image[rr, cc] += 20  # dim
labels[rr, cc] = 2

# object 3: medium
cy, cx, r = 180, 128, 25
rr, cc = disk((cy, cx), r, shape=shape)
image[rr, cc] += 100  # medium
labels[rr, cc] = 3

# compute reference intensities
props = regionprops(labels, intensity_image=image)
ref = {p.label: (p.mean_intensity, np.median(image[labels==p.label])) for p in props if p.label>0}
print("Reference intensities:")
for lab, (mean_i, med_i) in ref.items():
    print(f"  label {lab}: mean={mean_i:.1f}, median={med_i:.1f}")

# --- test mean filter ---
threshold = 120.0
good_mean = filter_intensity_inside(labels, image, metric='mean', threshold=threshold)
print(f"\nMean filter threshold > {threshold}: kept labels {good_mean}")
assert set(good_mean) == {1, 3}, "Mean filter failed"

# --- test median filter ---
threshold_med = 130.0
good_med = filter_intensity_inside(labels, image, metric='median', threshold=threshold_med)
print(f"Median filter threshold > {threshold_med}: kept labels {good_med}")
# object 1 is bright, object 3 is medium ~150, object 2 is dim
assert set(good_med) == {1, 3}, "Median filter failed"

# --- test multi-channel ---
image3 = np.stack([image, image*0.5], axis=0)
good_ch1 = filter_intensity_inside(labels, image3, channel=1, metric='mean', threshold=80.0)
print(f"\nMulti-channel channel=1 mean >80: kept labels {good_ch1}")
# channel 1 is half intensity: obj1 ~125, obj3 ~75, so only 1 passes
assert set(good_ch1) == {1}, "Multi-channel filter failed"

print("\nAll tests passed.")
