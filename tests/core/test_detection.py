"""
Unit tests for autofrap.core.detection: build_detector's return-tuple
contract (visualization_fun), the filter_function parameter, and
load_detector_file (user-supplied detector file import).

Synthetic data only - no nd2 files, no detector server needed.
"""
import os
import tempfile
import unittest
import warnings

import numpy as np

import autofrap
from autofrap.core.detection import build_detector, load_detector_file
from autofrap.core.image import mask as mask_utils
from autofrap.core.image.segmentation import dummy_detect_objects

# absolute path to the example detector (resolved via the package, so it
# works no matter where the suite is invoked from)
EXAMPLE_DETECTOR = os.path.join(os.path.dirname(autofrap.__file__),
                                'detectors', 'example_detector.py')


def load2d(f):
    return np.zeros((64, 48), dtype=np.uint16)


def loadrgb(f):
    return np.zeros((3, 64, 48), dtype=np.uint16)  # (c, y, x)


def maskfun(labels, image):
    return mask_utils.half_object_stim_mask(labels)


class TestBuildDetectorVisualization(unittest.TestCase):
    """build_detector()'s visualization_fun contract: the return-tuple
    positions (2 = mask, 3 = viz, fixed) for all combinations of
    stim_mask_fun / visualization_fun (visualization_fun=False disables
    the default visualization), and that a failing visualization degrades
    to no viz with a warning instead of raising (the labels/mask contract
    checks must still raise)."""

    def test_default_visualization_produced(self):
        """visualization_fun omitted -> default visualization is produced."""
        res = build_detector(load2d, dummy_detect_objects)('x')
        self.assertEqual(len(res), 3)
        self.assertIsNone(res[1])
        self.assertEqual(res[2].shape, (64, 48))

    def test_no_mask_no_viz_1tuple(self):
        """no mask, no viz -> (labels,)  [the 1-tuple autofrap() requires]"""
        res = build_detector(load2d, dummy_detect_objects,
                             visualization_fun=False)('x')
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].shape, (64, 48))
        self.assertTrue(np.issubdtype(res[0].dtype, np.integer))

    def test_mask_no_viz_2tuple(self):
        """mask, no viz -> (labels, mask)"""
        res = build_detector(load2d, dummy_detect_objects,
                             stim_mask_fun=maskfun,
                             visualization_fun=False)('x')
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[1].shape, (64, 48))
        self.assertEqual(res[1].dtype, bool)
        self.assertTrue(res[1].any())

    def test_viz_no_mask_keeps_mask_position(self):
        """viz, no mask -> (labels, None, viz)  [position 2 stays the mask]"""
        res = build_detector(load2d, dummy_detect_objects,
                             visualization_fun=lambda image: image)('x')
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertIsNone(res[1])
        self.assertEqual(res[2].shape, (64, 48))

    def test_mask_and_viz_3tuple(self):
        """mask + viz -> (labels, mask, viz)"""
        res = build_detector(load2d, dummy_detect_objects,
                             stim_mask_fun=maskfun,
                             visualization_fun=lambda image: image)('x')
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertIsNotNone(res[1])
        self.assertIsNotNone(res[2])

    def test_multichannel_load_2d_viz(self):
        """multi-channel (c, y, x) load, 2D grayscale viz from a channel"""
        res = build_detector(loadrgb, dummy_detect_objects,
                             visualization_fun=lambda image: image[0])('x')
        self.assertEqual(len(res), 3)
        self.assertEqual(res[2].shape, (64, 48))

    def test_multichannel_load_rgb_viz(self):
        """multi-channel load, scientific (c, y, x) -> display (y, x, 3)"""
        res = build_detector(loadrgb, dummy_detect_objects,
                             visualization_fun=lambda image:
                             np.transpose(image, (1, 2, 0))
                             .astype(np.uint8))('x')
        self.assertEqual(len(res), 3)
        self.assertEqual(res[2].shape, (64, 48, 3))

    def test_bad_viz_shape_dropped_with_warning(self):
        """bad viz shape ((y, 1)) -> warn + drop, run survives"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = build_detector(load2d, dummy_detect_objects,
                                 stim_mask_fun=maskfun,
                                 visualization_fun=lambda image:
                                 image[..., :1])('x')
        self.assertEqual(len(res), 2)
        self.assertEqual(len(w), 1)
        self.assertIn('visualization', str(w[0].message))

    def test_raising_viz_dropped_with_warning(self):
        """raising viz -> warn + drop, run survives"""
        def boom(image):
            raise RuntimeError('nope')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = build_detector(load2d, dummy_detect_objects,
                                 stim_mask_fun=maskfun,
                                 visualization_fun=boom)('x')
        self.assertEqual(len(res), 2)
        self.assertEqual(len(w), 1)
        self.assertIn('visualization_fun failed', str(w[0].message))

    def test_labels_dtype_check_still_raises(self):
        """the real contract checks still raise (viz must not swallow them)"""
        with self.assertRaises(ValueError):
            build_detector(load2d, lambda image: image.astype(bool),
                           visualization_fun=lambda image: image)('x')

    def test_mask_shape_check_still_raises(self):
        """a mis-shaped stim mask still raises ValueError"""
        with self.assertRaises(ValueError):
            build_detector(load2d, dummy_detect_objects,
                           stim_mask_fun=lambda labels, image:
                           np.zeros((10, 10), dtype=bool))('x')


class TestFilterFunction(unittest.TestCase):
    """Tests for filter_function in build_detector.

    filter_function receives (labels, image), returns a set/list of "good"
    label IDs; labels not in the set are zeroed before clear_border/relabelling.
    """

    def _make_detector(self, **kwargs):
        """Helper: build a detector from dummy_detect_objects."""
        load_fun = lambda f: np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        return build_detector(load_fun, dummy_detect_objects,
                              visualization_fun=False, **kwargs)

    def test_filter_function_keeps_selected_labels(self):
        """filter_function keeps labels in the returned set."""
        det = self._make_detector(filter_function=lambda labels, image: {1})
        labels, = det(None)  # survey_file is not used by dummy
        # After filter keeps only label 1, clear_border keeps it (not
        # touching border), relabel gives 1. Label 2 is removed.
        unique = np.unique(labels)
        self.assertEqual(len(unique), 2)  # 0 (background) + 1 (kept)
        self.assertIn(1, unique)

    def test_filter_function_removes_unselected_labels(self):
        """Labels not in filter_function result are zeroed."""
        # filter keeps only label 1; label 2 should be removed
        det = self._make_detector(filter_function=lambda labels, image: {1})
        labels, = det(None)
        self.assertNotIn(2, np.unique(labels))

    def test_filter_function_empty_result(self):
        """Empty filter result -> only background labels."""
        det = self._make_detector(filter_function=lambda labels, image: [])
        labels, = det(None)
        unique = np.unique(labels)
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0], 0)

    def test_filter_function_gets_raw_detector_labels(self):
        """filter_function sees the raw detector label IDs, not post-clear_border."""
        # dummy_detect_objects puts label 1 in the upper-left third (not border)
        # and label 2 in the lower-right quadrant (not border).
        # Filter checks label IDs directly.
        received_labels = None

        def capture_filter(labels, image):
            nonlocal received_labels
            received_labels = labels.copy()
            return {1}

        det = self._make_detector(filter_function=capture_filter)
        det(None)

        # received_labels should have exactly labels 1 and 2 (dummy's two objects)
        unique = np.unique(received_labels)
        self.assertEqual(len(unique), 3)  # 0, 1, 2
        self.assertEqual(set(unique), {0, 1, 2})

    def test_filter_function_gets_image(self):
        """filter_function receives the loaded image."""
        received_image = None

        def capture_filter(labels, image):
            nonlocal received_image
            received_image = image
            return {1}

        def load_with_shape(f):
            return np.zeros((50, 60), dtype=np.uint16)

        det = build_detector(load_with_shape, dummy_detect_objects,
                             visualization_fun=False,
                             filter_function=capture_filter)
        det(None)  # survey_file is ignored by load_with_shape

        self.assertIsNotNone(received_image)
        self.assertEqual(received_image.shape, (50, 60))

    def test_filter_function_with_multi_channel_image(self):
        """filter_function can index multi-channel images."""
        def keep_high_expression(labels, image):
            # image is (c, y, x); use channel 1 as "expression" channel
            from skimage.measure import regionprops
            good = []
            for rp in regionprops(labels):
                if image[1][labels == rp.label].mean() > 100:
                    good.append(rp.label)
            return good

        def load_multi_channel(f):
            return np.stack([
                np.zeros((50, 60), dtype=np.uint16),  # ch 0
                np.full((50, 60), 200, dtype=np.uint16),  # ch 1 = high expression
            ], axis=0)

        det = build_detector(load_multi_channel, dummy_detect_objects,
                             visualization_fun=False,
                             filter_function=keep_high_expression)
        labels, = det(None)
        unique = np.unique(labels)
        # Both dummy objects should survive (ch1 = 200 > 100 for both)
        self.assertIn(1, unique)

    def test_filter_function_set_vs_list(self):
        """filter_function works with both set and list inputs."""
        det_set = self._make_detector(filter_function=lambda labels, image: {1})
        det_list = self._make_detector(filter_function=lambda labels, image: [1])

        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        labels_set, = det_set(img)
        labels_list, = det_list(img)

        self.assertTrue(np.array_equal(labels_set, labels_list))

    def test_no_filter_function(self):
        """Without filter_function, behavior is unchanged."""
        det = self._make_detector()
        labels, = det(None)
        # Both dummy objects survive (neither touches border)
        unique = np.unique(labels)
        self.assertEqual(len(unique), 3)  # 0, 1, 2


class TestLoadDetectorFile(unittest.TestCase):
    """Tests for load_detector_file."""

    def test_load_example_detector(self):
        """Can load the example detector file."""
        detection_fun = load_detector_file(EXAMPLE_DETECTOR)
        self.assertTrue(callable(detection_fun))

    def test_load_nonexistent_file(self):
        """Missing file raises an error."""
        with self.assertRaises(FileNotFoundError):
            load_detector_file('nonexistent_detector.py')

    def test_load_file_without_detection_fun(self):
        """File without detection_fun raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w',
                                         delete=False) as f:
            f.write('x = 42\n')
            tmp = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                load_detector_file(tmp)
            self.assertIn('detection_fun', str(cm.exception))
        finally:
            os.unlink(tmp)

    def test_load_file_with_non_callable_detection_fun(self):
        """File with non-callable detection_fun raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w',
                                         delete=False) as f:
            f.write('detection_fun = "not a function"\n')
            tmp = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                load_detector_file(tmp)
            self.assertIn('callable', str(cm.exception))
        finally:
            os.unlink(tmp)

    def test_load_file_with_callable_detection_fun(self):
        """File with callable detection_fun is returned."""
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w',
                                         delete=False) as f:
            f.write('detection_fun = lambda f: (f, f, f)\n')
            tmp = f.name
        try:
            result = load_detector_file(tmp)
            self.assertTrue(callable(result))
        finally:
            os.unlink(tmp)


if __name__ == '__main__':
    unittest.main()
