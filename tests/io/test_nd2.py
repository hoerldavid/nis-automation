"""
Tests for autofrap.io.nd2.read_channel: axis-order-independent channel
selection, multi-channel selection, dimension validation (T/P rejected,
Z only with a projection), and the axis logic on synthetic arrays
(no-C files and odd axis orders that no nd2 file here can provide).

The file-based test classes are skipped when the data files are not
present (test_acquisitions/ and 01.nd2 live only on some machines).
"""
import os
import unittest

import nd2
import numpy as np

from autofrap.io import nd2 as nd2_helpers

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         os.pardir, os.pardir))
SURVEY = os.path.join(REPO_ROOT, 'test_acquisitions', 'autofrap_out',
                      '20260824_125948_c01_survey.nd2')   # (C, Y, X), 3 ch
FRAP = os.path.join(REPO_ROOT, 'test_acquisitions', 'autofrap_out',
                    '20260824_125948_c01_frap.nd2')       # (T, Y, X), no C
ZSTACK = os.path.join(REPO_ROOT, '01.nd2')                # (Z, C, Y, X)


@unittest.skipUnless(os.path.isfile(SURVEY), f'needs {SURVEY}')
class TestReadChannelSurveyFile(unittest.TestCase):
    """survey file (C, Y, X): same results as the old f.asarray()[channel]"""

    def test_channel_selection(self):
        with nd2.ND2File(SURVEY) as f:
            full = f.asarray()
        for ch in range(full.shape[0]):
            with self.subTest(channel=ch):
                img = nd2_helpers.read_channel(SURVEY, ch)
                self.assertEqual(img.shape, full.shape[1:])
                self.assertEqual(img.dtype, full.dtype)
                self.assertTrue(np.array_equal(img, full[ch]))

    def test_multi_channel_selection(self):
        with nd2.ND2File(SURVEY) as f:
            full = f.asarray()
        multi = nd2_helpers.read_channel(SURVEY, (0, 1))
        self.assertEqual(multi.shape, (2, *full.shape[1:]))
        self.assertTrue(np.array_equal(multi, full[[0, 1]]))

    def test_multi_channel_order_kept(self):
        with nd2.ND2File(SURVEY) as f:
            full = f.asarray()
        reordered = nd2_helpers.read_channel(SURVEY, (1, 0))
        self.assertTrue(np.array_equal(reordered, full[[1, 0]]))

    def test_channel_out_of_range(self):
        with self.assertRaises((ValueError, IndexError)):
            nd2_helpers.read_channel(SURVEY, 5)

    def test_old_behavior_equivalence(self):
        """equivalence with the old implementation across all copied
        survey files in the directory"""
        survey_dir = os.path.dirname(SURVEY)
        n_eq = 0
        for fn in sorted(os.listdir(survey_dir)):
            if not fn.endswith('_survey.nd2'):
                continue
            path = os.path.join(survey_dir, fn)
            with nd2.ND2File(path) as f:
                full = f.asarray()
            for ch in range(full.shape[0]):
                with self.subTest(file=fn, channel=ch):
                    self.assertTrue(np.array_equal(
                        nd2_helpers.read_channel(path, ch), full[ch]))
                n_eq += 1
        self.assertGreater(n_eq, 0, 'no survey files found to compare')


@unittest.skipUnless(os.path.isfile(FRAP), f'needs {FRAP}')
class TestReadChannelFrapFile(unittest.TestCase):
    """FRAP timeseries (T, Y, X): unsupported dimension -> error"""

    def test_timeseries_rejected(self):
        with self.assertRaises(ValueError) as cm:
            nd2_helpers.read_channel(FRAP, 0)
        self.assertIn('T', str(cm.exception))


@unittest.skipUnless(os.path.isfile(ZSTACK), f'needs {ZSTACK}')
class TestReadChannelZStack(unittest.TestCase):
    """z-stack (Z, C, Y, X): error by default, max projection on request"""

    def test_rejected_by_default(self):
        with self.assertRaises(ValueError) as cm:
            nd2_helpers.read_channel(ZSTACK, 0)
        self.assertIn('Z', str(cm.exception))
        self.assertIn('z_projection', str(cm.exception))

    def test_unknown_projection_rejected(self):
        with self.assertRaises(ValueError) as cm:
            nd2_helpers.read_channel(ZSTACK, 0, z_projection='mean')
        self.assertIn('z_projection', str(cm.exception))

    def test_max_projection(self):
        with nd2.ND2File(ZSTACK) as f:
            full = f.asarray()
            proj = full.max(axis=0)
        img = nd2_helpers.read_channel(ZSTACK, 0, z_projection='max')
        self.assertEqual(img.shape, proj.shape[1:])
        self.assertTrue(np.array_equal(img, proj[0]))

    def test_max_projection_multi_channel(self):
        with nd2.ND2File(ZSTACK) as f:
            full = f.asarray()
            proj = full.max(axis=0)
        multi = nd2_helpers.read_channel(ZSTACK, (1, 0), z_projection='max')
        self.assertEqual(multi.shape, (2, *proj.shape[1:]))
        self.assertTrue(np.array_equal(multi, proj[[1, 0]]))


class TestExtractAxisLogic(unittest.TestCase):
    """axis logic on synthetic arrays (no nd2 writer in the nd2 package:
    no-C files and axis orders that no file here can provide)"""

    def setUp(self):
        rng = np.random.default_rng(0)
        y, x = 6, 7
        self.base_cyx = rng.integers(0, 400, (3, y, x), dtype=np.uint16)  # (C, Y, X)
        self.a = self.base_cyx[0]
        self.b = np.transpose(self.base_cyx, (1, 0, 2))                  # (Y, C, X)
        self.z = rng.integers(0, 400, (4, y, x), dtype=np.uint16)         # (Z, Y, X)
        self.base_czyx = rng.integers(0, 400, (3, 4, y, x), dtype=np.uint16)  # (C, Z, Y, X)
        self.zc = np.transpose(self.base_czyx, (1, 0, 2, 3))             # (Z, C, Y, X)

    def test_yx_unchanged(self):
        """(Y, X), ch 0 -> unchanged"""
        self.assertTrue(np.array_equal(
            nd2_helpers._extract(self.a, ['Y', 'X'], [0], None), self.a))

    def test_zyx_max_projection(self):
        """(Z, Y, X), ch 0, max -> (y, x)"""
        self.assertTrue(np.array_equal(
            nd2_helpers._extract(self.z, ['Z', 'Y', 'X'], [0], 'max'),
            self.z.max(axis=0)))

    def test_cyx_single_channel(self):
        """(C, Y, X), ch 0 -> (y, x)"""
        self.assertTrue(np.array_equal(
            nd2_helpers._extract(self.base_cyx, ['C', 'Y', 'X'], [0], None),
            self.base_cyx[0]))

    def test_cyx_multi_channel(self):
        """(C, Y, X), ch (0, 1) -> (2, y, x)"""
        self.assertTrue(np.array_equal(
            nd2_helpers._extract(self.base_cyx, ['C', 'Y', 'X'], [0, 1], None),
            np.stack([self.base_cyx[0], self.base_cyx[1]])))

    def test_zcyx_max_projection(self):
        """(Z, C, Y, X), ch 1, max -> (y, x)"""
        self.assertTrue(np.array_equal(
            nd2_helpers._extract(self.zc, ['Z', 'C', 'Y', 'X'], [1], 'max'),
            self.zc[:, 1, :, :].max(axis=0)))

    def test_czyx_max_projection_multi(self):
        """(C, Z, Y, X), ch (0, 1), max -> (2, y, x)"""
        self.assertTrue(np.array_equal(
            nd2_helpers._extract(self.base_czyx, ['C', 'Z', 'Y', 'X'],
                                 [0, 1], 'max'),
            np.stack([self.base_czyx[0].max(axis=0),
                      self.base_czyx[1].max(axis=0)])))

    def test_ycx_odd_order(self):
        """(Y, C, X), ch 0 -> (y, x) (odd axis order)"""
        self.assertTrue(np.array_equal(
            nd2_helpers._extract(self.b, ['Y', 'C', 'X'], [0], None),
            self.base_cyx[0]))


if __name__ == '__main__':
    unittest.main()
