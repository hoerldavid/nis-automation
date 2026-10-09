"""
Tests for autofrap.core.image.qc.save_qc_overlay.

The synthetic class needs no data files. The real-data class feeds the
overlay exactly what autofrap() would produce, on the copied
test_data/0013_ch1.tif image + 0013_ch1_cp_masks.tif labels; it is
skipped when those files are not present (they live only on some
machines). Overlays are written to a temp dir - the visual artifacts
that used to land in test_data/ are not needed for the assertions.
"""
import os
import tempfile
import unittest

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from autofrap.core.image import mask as mask_utils
from autofrap.core.image.mask import next_stimulatable_cell
from autofrap.core.image.qc import save_qc_overlay

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         os.pardir, os.pardir, os.pardir))
TEST_DATA = os.path.join(REPO_ROOT, 'test_data')
IMAGE = os.path.join(TEST_DATA, '0013_ch1.tif')
LABELS = os.path.join(TEST_DATA, '0013_ch1_cp_masks.tif')


class TestQcOverlaySynthetic(unittest.TestCase):
    """Pixel-level checks on a synthetic 300x300 case (known expected
    pixels). 300x300; the fixed-font-size legend (top-left, ~rows 10-150,
    cols 10-200) keeps the checked regions clear, so the geometry sits
    bottom-right: gray square 20:290, one label 150:280 (centroid
    (215,215), so its ID text stays out of the stim box), stim 220:280,
    cell_poly = label boundary, stim_poly = stim boundary."""

    def test_layers_land_where_expected(self):
        simg = np.zeros((300, 300), float)
        simg[20:290, 20:290] = 1.0
        slbl = np.zeros((300, 300), int)
        slbl[150:280, 150:280] = 1
        sstim = np.zeros((300, 300), bool)
        sstim[220:280, 220:280] = True

        with tempfile.TemporaryDirectory() as tmp:
            syn_path = os.path.join(tmp, 'qc_synthetic.png')
            save_qc_overlay(
                simg, slbl, syn_path, stimulation_mask=sstim, cell_id=1,
                cell_poly=[(150, 150), (280, 150), (280, 280), (150, 280)],
                stim_poly=[(220, 220), (280, 220), (280, 280), (220, 280)])

            out = plt.imread(syn_path)  # (h, w, 4) float 0..1
            self.assertEqual(out.shape[:2], (300, 300))
            out = (out[..., :3] * 255).round().astype(int)

            # orange stim fill: 0.3 alpha orange (255,165,0) over the bright
            # square -> (255, 228, 179); sampled well inside the box, away
            # from its border lines and from the legend (top-left)
            box = out[240:270, 240:270]
            tinted = ((box[..., 0] > 240) & (200 < box[..., 1])
                      & (box[..., 1] < 245) & (150 < box[..., 2])
                      & (box[..., 2] < 200))
            self.assertGreater(tinted.sum(), 600,
                               f'stim fill: only {tinted.sum()} tinted px in box')

            # (selected-cell highlight is its cyan ID text + the ROI polygons;
            # no contour any more - visual check by eye on the saved PNG)

            # cyan cell_poly along y = 150 (top edge), right of the legend
            cyan = [out[150, x] for x in range(210, 241)]
            self.assertTrue(any(p[0] < 120 and p[1] > 180 and p[2] > 180
                                for p in cyan),
                            'no cyan cell_poly pixels along y=150')

            # magenta stim_poly along y = 280 (bottom edge, x = 220..280)
            magenta = [out[280, x] for x in range(226, 251)]
            self.assertTrue(any(p[0] > 150 and p[1] < 100 and p[2] > 150
                                for p in magenta),
                            'no magenta stim_poly pixels along y=280')

            # outside the gray square stays black (no stray layers)
            self.assertLessEqual(out[5, 5].max(), 30,
                                 f'corner (5,5) not dark: {out[5, 5]}')

            # RGB(A) input (y, x, 3) is shown as-is, no grayscale/clipping
            rgb = np.zeros((300, 300, 3), np.uint8)
            rgb[200:280, 200:280] = (255, 0, 0)
            rgb_path = os.path.join(tmp, 'qc_rgb.png')
            save_qc_overlay(rgb, slbl, rgb_path)
            outr = (plt.imread(rgb_path)[..., :3] * 255).round().astype(int)
            # (240,240) is inside the red square, clear of contour/text/legend
            self.assertAlmostEqual(int(outr[240, 240, 0]), 255, delta=10)
            self.assertLessEqual(outr[240, 240, 1], 10)
            self.assertLessEqual(outr[240, 240, 2], 10)


@unittest.skipUnless(os.path.isfile(IMAGE) and os.path.isfile(LABELS),
                     'needs test_data/0013_ch1.tif + 0013_ch1_cp_masks.tif')
class TestQcOverlayRealData(unittest.TestCase):
    """The full artifact, as autofrap() would save it, on the real test
    data: random-circle stimulation mask, next_stimulatable_cell
    selection, and the polygons mask_to_polygon sends to NIS."""

    def test_full_artifact_written(self):
        image = tifffile.imread(IMAGE)
        labels = tifffile.imread(LABELS).astype(int)
        n_obj = len(set(labels.ravel().tolist())) - 1
        self.assertGreater(n_obj, 0)

        stim = mask_utils.random_circle_stim_mask(labels, seed=0)
        cell = next_stimulatable_cell(labels, set(), stim)
        self.assertIsNotNone(cell)
        cell_poly = mask_utils.mask_to_polygon(
            mask_utils.cell_mask(labels, cell))
        stim_poly = mask_utils.mask_to_polygon(
            mask_utils.cell_mask(labels, cell, stim))
        self.assertTrue(cell_poly)
        self.assertTrue(stim_poly)

        with tempfile.TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, n) for n in
                     ('qc_selected.png', 'qc_noselect.png', 'qc_bigcell.png')]

            # full artifact, as autofrap() would save it
            save_qc_overlay(
                image, labels, paths[0], stimulation_mask=stim, cell_id=cell,
                cell_poly=cell_poly, stim_poly=stim_poly,
                caption=f'test  cell {cell} of {n_obj}')

            # no selection: just image + labels + stim mask
            save_qc_overlay(
                image, labels, paths[1], stimulation_mask=stim,
                caption=f'test  no selection ({n_obj} objects)')

            # highlight a different cell (largest area) to check generality
            big = max((l for l in np.unique(labels) if l > 0),
                      key=lambda l: (labels == l).sum())
            save_qc_overlay(
                image, labels, paths[2], stimulation_mask=stim, cell_id=big,
                caption=f'test  largest cell {big}')

            for p in paths:
                self.assertTrue(os.path.isfile(p), f'{p} not written')
                self.assertGreater(os.path.getsize(p), 1000,
                                   f'{p} suspiciously small')


if __name__ == '__main__':
    unittest.main()
