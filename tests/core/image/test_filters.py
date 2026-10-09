"""
Unit tests for the intensity filters in autofrap.core.image.mask:
filter_intensity_inside and filter_intensity_surround.

The surround tests keep the provenance of the 20260914 research that
motivated the shipped implementation: the EDT-based ring is checked
against a naive dilation-based reference implementation.
"""
import unittest

import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops
from skimage.morphology import dilation, disk as skdisk

from autofrap.core.image.mask import (filter_intensity_inside,
                                       filter_intensity_surround)


def make_intensity_scene():
    """256x256, background ~50, three disk objects: bright / dim / medium."""
    rng = np.random.default_rng(42)
    shape = (256, 256)
    image = rng.normal(loc=50, scale=5, size=shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    for lab, (cy, cx, r, add) in enumerate(
            [(80, 80, 20, 200.0),    # object 1: bright
             (80, 180, 20, 20.0),    # object 2: dim
             (180, 128, 25, 100.0)],  # object 3: medium
            start=1):
        rr, cc = disk((cy, cx), r, shape=shape)
        image[rr, cc] += add
        labels[rr, cc] = lab
    return image, labels


class TestFilterIntensityInside(unittest.TestCase):
    """filter_intensity_inside: keep labels whose mean/median intensity
    inside the object is above the threshold."""

    def setUp(self):
        self.image, self.labels = make_intensity_scene()
        # reference intensities: means ~250 / ~70 / ~150

    def test_mean_filter(self):
        good = filter_intensity_inside(self.labels, self.image,
                                       metric='mean', threshold=120.0)
        self.assertEqual(set(good), {1, 3})

    def test_median_filter(self):
        # object 1 is bright, object 3 is medium ~150, object 2 is dim
        good = filter_intensity_inside(self.labels, self.image,
                                       metric='median', threshold=130.0)
        self.assertEqual(set(good), {1, 3})

    def test_multichannel(self):
        image3 = np.stack([self.image, self.image * 0.5], axis=0)
        # channel 1 is half intensity: obj1 ~125, obj3 ~75, so only 1 passes
        good = filter_intensity_inside(self.labels, image3, channel=1,
                                       metric='mean', threshold=80.0)
        self.assertEqual(set(good), {1})

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            filter_intensity_inside(self.labels, self.image, metric='max')


def make_synthetic(n_objects=30, shape=(512, 512), seed=2):
    rng = np.random.default_rng(seed)
    image = rng.normal(loc=100, scale=10, size=shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    for i in range(1, n_objects + 1):
        r = rng.integers(10, 25)
        cy = rng.integers(r, shape[0] - r)
        cx = rng.integers(r, shape[1] - r)
        rr, cc = disk((cy, cx), r, shape=shape)
        labels[rr, cc] = i
        image[rr, cc] += rng.uniform(20, 80)
    return image, labels


def surround_mean_dilation(labels, image, distance_px):
    """Naive reference implementation (dilation-based ring), kept as the
    provenance for the exact EDT method the shipped filter uses."""
    selem = skdisk(distance_px)
    vals = []
    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        r0 = max(0, minr - distance_px)
        c0 = max(0, minc - distance_px)
        r1 = min(image.shape[0], maxr + distance_px)
        c1 = min(image.shape[1], maxc + distance_px)
        img_crop = image[r0:r1, c0:c1]
        mask_crop = (labels[r0:r1, c0:c1] == rp.label)
        dilated = dilation(mask_crop, footprint=selem)
        ring = np.logical_xor(dilated, mask_crop)
        vals.append(float(img_crop[ring].mean()) if np.any(ring) else np.nan)
    return np.array(vals)


class TestFilterIntensitySurroundMatchesDilationReference(unittest.TestCase):
    """The shipped EDT-based ring must agree with the naive dilation-based
    reference (the 20260914 research question) for every threshold."""

    def test_edt_ring_equals_dilation_ring(self):
        image, labels = make_synthetic()
        all_labels = [rp.label for rp in regionprops(labels)]
        for r in (5, 10, 15, 20):
            ref = surround_mean_dilation(labels, image, r)
            # thresholds: every reference value plus the midpoints between
            # them, so each object's decision is exercised on both sides
            values = sorted(set(ref[~np.isnan(ref)]))
            thresholds = (values
                          + [(a + b) / 2 for a, b in zip(values, values[1:])]
                          + [values[0] - 1, values[-1] + 1])
            for t in thresholds:
                with self.subTest(distance_px=r, threshold=round(t, 6)):
                    want = {lab for lab, v in zip(all_labels, ref)
                            if not np.isnan(v) and v > t}
                    got = set(filter_intensity_surround(
                        labels, image, distance_px=r, threshold=t))
                    self.assertEqual(got, want)


class TestFilterIntensitySurroundScenario(unittest.TestCase):
    """Two background levels: objects in the bright half of the image
    must be kept, objects in the dim half dropped."""

    def test_bright_surround_kept_dim_dropped(self):
        shape = (256, 256)
        image = np.full(shape, 30, dtype=np.float32)
        image[:, shape[1] // 2:] = 150  # right half bright background
        labels = np.zeros(shape, dtype=np.int32)
        # objects 1-3 in the left (low background)
        for i, (cy, cx) in enumerate([(80, 60), (180, 70), (120, 100)], start=1):
            rr, cc = disk((cy, cx), 15, shape=shape)
            labels[rr, cc] = i
        # objects 4-6 in the right (high background)
        for i, (cy, cx) in enumerate([(80, 180), (180, 200), (120, 220)], start=4):
            rr, cc = disk((cy, cx), 15, shape=shape)
            labels[rr, cc] = i

        for metric in ('mean', 'median'):
            with self.subTest(metric=metric):
                good = filter_intensity_surround(labels, image, distance_px=10,
                                                 threshold=80, metric=metric)
                self.assertEqual(set(good), {4, 5, 6})

    def test_unknown_metric_raises(self):
        image, labels = make_intensity_scene()
        with self.assertRaises(ValueError):
            filter_intensity_surround(labels, image, metric='max')


if __name__ == '__main__':
    unittest.main()
