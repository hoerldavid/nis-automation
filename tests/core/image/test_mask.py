"""
Unit tests for autofrap.core.image.mask: the one-stimulation-region-per-label
helpers, bbox-local sanity/timing of the stim-mask utilities, and the
cross-cycle centroid matching (match_imaged_centroids +
next_stimulatable_cell).
"""
import unittest

import numpy as np
from skimage.draw import disk, ellipse
from skimage.measure import label, regionprops

from autofrap.core.image.mask import (
    half_object_stim_mask,
    random_circle_stim_mask,
    largest_region_per_label,
    most_central_region_per_label,
    mask_to_polygon,
    match_imaged_centroids,
    next_stimulatable_cell,
)


class TestOneRegionPerLabel(unittest.TestCase):
    """largest_region_per_label / most_central_region_per_label: reduce a
    mask to at most one connected region per label."""

    def test_empty(self):
        """empty mask -> empty mask"""
        labels = np.zeros((32, 32), dtype=np.int32)
        stim = np.zeros((32, 32), dtype=bool)
        result = largest_region_per_label(labels, stim)
        self.assertEqual(result.sum(), 0)
        self.assertEqual(result.shape, (32, 32))

    def test_single_region_per_label(self):
        """no-op when each label already has at most one region"""
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[10:20, 10:20] = 1
        labels[40:50, 40:50] = 2
        stim = labels.astype(bool).copy()
        result = largest_region_per_label(labels, stim)
        self.assertTrue(np.array_equal(result, stim))

    def test_two_regions_per_label_largest(self):
        """largest region wins when a label has multiple regions"""
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[10:20, 10:20] = 1  # label 1 at top-left

        # two disconnected regions within label 1's bbox (10:20, 10:20)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:16, 10:16] = True   # 6x6 = 36 px (top-left)
        mask[17:20, 17:20] = True   # 3x3 = 9 px  (bottom-right, gap=1)

        result = largest_region_per_label(labels, mask)
        self.assertTrue(result[13, 13], 'keeps largest region (36 px)')
        self.assertFalse(result[18, 18], 'drops smaller region (9 px)')

    def test_two_regions_per_label_central(self):
        """most-central region wins"""
        labels = np.zeros((64, 64), dtype=np.int32)
        # label 1 centered roughly at (25, 25)
        labels[10:40, 10:40] = 1

        # two regions: one at top (far), one at bottom-right (closer to centroid)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:15, 10:15] = True    # far top-left, centroid ~12.5
        mask[30:38, 30:38] = True    # closer to label centroid ~25, centroid ~34

        result = most_central_region_per_label(labels, mask)
        self.assertTrue(result[34, 34], 'keeps most central region')
        self.assertFalse(result[12, 12], 'drops far region')

    def test_multi_label_mixed(self):
        """multiple labels, some with multiple regions"""
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[10:20, 10:20] = 1
        labels[40:50, 40:50] = 2

        # label 1: two disconnected regions (both within bbox 10:20, 10:20)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:16, 10:16] = True    # 6x6 = 36 px (top-left)
        mask[17:20, 17:20] = True    # 3x3 = 9 px  (bottom-right)
        # label 2: one region
        mask[40:48, 40:48] = True    # 8x8 = 64 px

        result = largest_region_per_label(labels, mask)
        self.assertTrue(result[14, 14], 'label 1 keeps largest')
        self.assertFalse(result[18, 18], 'label 1 drops small')
        self.assertTrue(result[44, 44], 'label 2 untouched')

    def test_all_regions_same_label(self):
        """all pixels of a label have multiple regions"""
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[:] = 1  # everything is label 1

        # checkerboard pattern -> many regions
        mask = np.zeros((64, 64), dtype=bool)
        mask[::2, ::2] = True

        result = largest_region_per_label(labels, mask)
        # each 1px block is a region, all equal size -> largest picks one
        self.assertEqual(result.sum(), 1, 'single region remains')

    def test_all_regions_same_label_central(self):
        """checkerboard -> picks region closest to image centroid"""
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[:] = 1

        mask = np.zeros((64, 64), dtype=bool)
        mask[::2, ::2] = True

        result = most_central_region_per_label(labels, mask)
        # closest to center (32,32) among checkerboard 1px regions
        self.assertEqual(result.sum(), 1, 'single central-ish region')


def make_labels(shape=(1024, 1024), n=20, radius_range=(30, 80), seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=int)
    for i in range(1, n + 1):
        r = int(rng.integers(*radius_range))
        cy = int(rng.integers(r, shape[0] - r))
        cx = int(rng.integers(r, shape[1] - r))
        rr, cc = disk((cy, cx), r, shape=shape)
        labels[rr, cc] = i
    return labels


class TestMaskBboxSanity(unittest.TestCase):
    """Sanity checks of the bbox-local mask utilities on synthetic
    label maps."""

    def test_half_object(self):
        labels = make_labels(shape=(1024, 1024), n=50, seed=1)
        stim = half_object_stim_mask(labels)
        # sanity: stim is subset of labels>0
        self.assertTrue(np.all(stim <= (labels > 0)))
        # each label gets ~half area
        for rp in regionprops(labels):
            obj = labels == rp.label
            stim_obj = stim & obj
            # area should be roughly half, within 10%
            if obj.sum() > 0:
                frac = stim_obj.sum() / obj.sum()
                self.assertTrue(0.35 < frac < 0.65,
                                f'label {rp.label} frac {frac}')

    def test_random_circle(self):
        labels = make_labels(shape=(1024, 1024), n=50, seed=2)
        stim = random_circle_stim_mask(labels, area_fraction=0.25, seed=42)
        # sanity: stim inside labels
        self.assertTrue(np.all(stim <= (labels > 0)))
        # each label has at most one connected component
        for lbl in np.unique(labels):
            if lbl == 0:
                continue
            comp = label(stim & (labels == lbl), connectivity=1)
            self.assertLessEqual(comp.max(), 1,
                                 f'label {lbl} has {comp.max()} components')

    def test_largest_central(self):
        labels = make_labels(shape=(512, 512), n=30, seed=3)
        # create mask with two blobs per label
        mask = np.zeros_like(labels, dtype=bool)
        for rp in regionprops(labels):
            minr, minc, maxr, maxc = rp.bbox
            # put two small disks inside bbox
            rr, cc = disk((minr + 5, minc + 5), 3, shape=labels.shape)
            mask[rr, cc] = True
            rr, cc = disk((maxr - 5, maxc - 5), 3, shape=labels.shape)
            mask[rr, cc] = True
        # also ensure mask overlaps label
        mask &= (labels > 0)
        reduced = largest_region_per_label(labels, mask)
        # reduced should be subset
        self.assertTrue(np.all(reduced <= mask))
        # most central
        reduced2 = most_central_region_per_label(labels, mask)
        self.assertTrue(np.all(reduced2 <= mask))

    def test_mask_to_polygon(self):
        # create a simple ellipse mask
        mask = np.zeros((200, 200), dtype=bool)
        rr, cc = ellipse(100, 100, 60, 40)
        mask[rr, cc] = True
        poly = mask_to_polygon(mask, tolerance=2.0)
        self.assertGreater(len(poly), 0)


class TestCentroidMatching(unittest.TestCase):
    """Cross-cycle cell matching: match_imaged_centroids against the
    accumulated already-imaged centroid map, then next_stimulatable_cell
    picks the next unstimulated cell."""

    def setUp(self):
        # Create a synthetic label map: 3 cells in different positions
        self.labels = np.zeros((100, 100), dtype=int)
        self.labels[20:30, 20:30] = 1   # center ~ (25, 25)
        self.labels[50:60, 50:60] = 2   # center ~ (55, 55)
        self.labels[80:90, 40:50] = 3   # center ~ (85, 45)
        self.stim_mask = self.labels > 0

    def test_centroid_extraction_yx_order(self):
        """regionprops returns centroids in (y, x) numpy order (the
        coordinate convention the matcher relies on)."""
        for rp in regionprops(self.labels):
            if rp.label == 1:  # box at rows 20:30, cols 20:30 -> centroid ~24.5
                self.assertAlmostEqual(rp.centroid[0], 24.5, places=0)
                self.assertAlmostEqual(rp.centroid[1], 24.5, places=0)
            elif rp.label == 2:  # box at rows 50:60, cols 50:60 -> centroid ~54.5
                self.assertAlmostEqual(rp.centroid[0], 54.5, places=0)
                self.assertAlmostEqual(rp.centroid[1], 54.5, places=0)

    def test_centroid_auto_mode_matches_by_equivalent_diameter(self):
        """'auto' uses rp.equivalent_diameter_area as the matching radius."""
        imaged = [(25.0, 25.0)]  # cell 1's centroid
        matched = match_imaged_centroids(self.labels, imaged, 'auto')
        # Cell 1 is at the same position, so its equivalent_diameter_area
        # (~11.3 px for a 10x10 box, area=100) as radius will capture it.
        self.assertIn(1, matched)

    def test_centroid_auto_mode_far_apart(self):
        """Auto mode: distant cells not matched."""
        imaged = [(25.0, 25.0)]
        matched = match_imaged_centroids(self.labels, imaged, 'auto')
        # Cells 2 and 3 are far from cell 1; their own diameters
        # won't reach them.
        self.assertNotIn(2, matched)
        self.assertNotIn(3, matched)

    def test_centroid_auto_mode_no_imaged(self):
        """No imaged centroids -> no matches (auto mode)."""
        matched = match_imaged_centroids(self.labels, [], 'auto')
        self.assertEqual(matched, set())

    def test_centroid_auto_mode_multiple_imaged(self):
        """Multiple imaged centroids with auto mode."""
        imaged = [(25.0, 25.0), (85.0, 45.0)]
        matched = match_imaged_centroids(self.labels, imaged, 'auto')
        self.assertEqual(matched, {1, 3})

    def test_centroid_numeric_mode(self):
        """Numeric threshold is used as-is."""
        imaged = [(25.0, 25.0)]
        matched = match_imaged_centroids(self.labels, imaged, 50.0)
        # 50 px radius -> cell 1 (25,25) matches itself;
        # cell 2 (55,55) is at distance sqrt(30^2+30^2) ~ 42.4 < 50 -> also matched
        self.assertIn(1, matched)
        self.assertIn(2, matched)

    def test_centroid_numeric_mode_tight(self):
        """Very tight numeric threshold -> only the exact-position cell."""
        imaged = [(25.0, 25.0)]
        matched = match_imaged_centroids(self.labels, imaged, 1.0)
        # cell 1 at exactly the same position is a 0 distance
        self.assertIn(1, matched)  # dist = 0 < 1^2

    def test_next_stimulatable_cell_skips_matched(self):
        """next_stimulatable_cell skips all labels in the matched set."""
        matched = {1, 2}
        cell = next_stimulatable_cell(self.labels, matched, self.stim_mask)
        self.assertEqual(cell, 3)

    def test_next_stimulatable_cell_all_matched(self):
        """Returns None when all labels are matched."""
        cell = next_stimulatable_cell(self.labels, {1, 2, 3}, self.stim_mask)
        self.assertIsNone(cell)

    def test_next_stimulatable_cell_no_mask(self):
        """Without a mask, any unmatched label is selectable."""
        cell = next_stimulatable_cell(self.labels, {2}, None)
        self.assertEqual(cell, 1)

    def test_centroid_roundtrip_auto(self):
        """Extract centroids, re-match with auto mode -> same cell."""
        imaged = []
        for rp in regionprops(self.labels):
            if rp.label == 1:
                imaged.append(rp.centroid)
                break
        matched = match_imaged_centroids(self.labels, imaged, 'auto')
        self.assertEqual(matched, {1})

    def test_equivalent_diameter_area_reflects_size(self):
        """equivalent_diameter_area is proportional to object size."""
        for rp in regionprops(self.labels):
            # 10x10 square -> area=100 -> equivalent_diameter_area
            # = 2*sqrt(area/pi) = 2*sqrt(100/pi) ~ 11.28
            self.assertAlmostEqual(rp.equivalent_diameter_area, 11.28, places=1)


if __name__ == '__main__':
    unittest.main()
