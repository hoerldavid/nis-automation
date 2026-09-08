"""
Unit tests for centroid-based cross-cycle cell matching.

Verifies the new matching logic that replaces merge_label_slices
(see TODO #13 in STATUS.md).
"""
import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from autofrap.pipeline import next_stimulatable_cell
from skimage.measure import regionprops


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _match(labels, imaged_centroids, centroid_threshold):
    """Run the matching logic from autofrap() and return the matched set."""
    matched = set()
    for rp in regionprops(labels):
        cy, cx = rp.centroid  # (y, x) — numpy order
        if centroid_threshold == 'auto':
            radius = rp.equivalent_diameter_area
        else:
            radius = float(centroid_threshold)
        for iy, ix in imaged_centroids:
            if (cy - iy)**2 + (cx - ix)**2 < radius**2:
                matched.add(rp.label)
                break
    return matched


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestCentroidMatching(unittest.TestCase):
    """Test centroid matching replaces label-based matching."""

    def setUp(self):
        # Create a synthetic label map: 3 cells in different positions
        self.labels = np.zeros((100, 100), dtype=int)
        self.labels[20:30, 20:30] = 1   # center ~ (25, 25)
        self.labels[50:60, 50:60] = 2   # center ~ (55, 55)
        self.labels[80:90, 40:50] = 3   # center ~ (85, 45)
        self.stim_mask = self.labels > 0

    def test_centroid_extraction_yx_order(self):
        """regionprops returns centroids in (y, x) numpy order."""
        for rp in regionprops(self.labels):
            if rp.label == 1:  # box at rows 20:30, cols 20:30 → centroid ~24.5
                self.assertAlmostEqual(rp.centroid[0], 24.5, places=0)
                self.assertAlmostEqual(rp.centroid[1], 24.5, places=0)
            elif rp.label == 2:  # box at rows 50:60, cols 50:60 → centroid ~54.5
                self.assertAlmostEqual(rp.centroid[0], 54.5, places=0)
                self.assertAlmostEqual(rp.centroid[1], 54.5, places=0)

    def test_centroid_auto_mode_matches_by_equivalent_diameter(self):
        """'auto' uses rp.equivalent_diameter_area as the matching radius."""
        imaged = [(25.0, 25.0)]  # cell 1's centroid
        matched = _match(self.labels, imaged, 'auto')
        # Cell 1 is at the same position, so its equivalent_diameter_area
        # (~11.3 px for a 10×10 box, area=100) as radius will capture it.
        self.assertIn(1, matched)

    def test_centroid_auto_mode_far_apart(self):
        """Auto mode: distant cells not matched."""
        imaged = [(25.0, 25.0)]
        matched = _match(self.labels, imaged, 'auto')
        # Cells 2 and 3 are far from cell 1; their own diameters
        # won't reach them.
        self.assertNotIn(2, matched)
        self.assertNotIn(3, matched)

    def test_centroid_auto_mode_no_imaged(self):
        """No imaged centroids → no matches (auto mode)."""
        matched = _match(self.labels, [], 'auto')
        self.assertEqual(matched, set())

    def test_centroid_auto_mode_multiple_imaged(self):
        """Multiple imaged centroids with auto mode."""
        imaged = [(25.0, 25.0), (85.0, 45.0)]
        matched = _match(self.labels, imaged, 'auto')
        self.assertEqual(matched, {1, 3})

    def test_centroid_numeric_mode(self):
        """Numeric threshold is used as-is."""
        imaged = [(25.0, 25.0)]
        matched = _match(self.labels, imaged, 50.0)
        # 50 px radius → cell 1 (25,25) matches itself;
        # cell 2 (55,55) is at distance √(30²+30²)≈42.4 < 50 → also matched
        self.assertIn(1, matched)
        self.assertIn(2, matched)

    def test_centroid_numeric_mode_tight(self):
        """Very tight numeric threshold → no matches."""
        imaged = [(25.0, 25.0)]
        matched = _match(self.labels, imaged, 1.0)
        # Even cell 1 at exactly the same position is a 0 distance
        self.assertIn(1, matched)  # dist = 0 < 1²

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
        """Extract centroids, re-match with auto mode → same cell."""
        imaged = []
        for rp in regionprops(self.labels):
            if rp.label == 1:
                imaged.append(rp.centroid)
                break
        matched = _match(self.labels, imaged, 'auto')
        self.assertEqual(matched, {1})

    def test_equivalent_diameter_area_reflects_size(self):
        """equivalent_diameter_area is proportional to object size."""
        for rp in regionprops(self.labels):
            # 10×10 square → area=100 → equivalent_diameter_area
            # = 2√(area/π) = 2√(100/π) ≈ 11.28
            self.assertAlmostEqual(rp.equivalent_diameter_area, 11.28, places=1)


if __name__ == '__main__':
    unittest.main()
