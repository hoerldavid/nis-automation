"""
Unit tests for build_detector's filter_function parameter.

filter_function receives (labels, image), returns a set/list of "good"
label IDs; labels not in the set are zeroed before clear_border/relabelling.
"""
import os
import sys
import unittest
import numpy as np
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from autofrap.detection import (
    build_detector,
    dummy_detect_objects,
)


class TestFilterFunction(unittest.TestCase):
    """Tests for filter_function in build_detector."""

    def _make_detector(self, **kwargs):
        """Helper: build a detector from dummy_detect_objects."""
        load_fun = lambda f: np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        return build_detector(load_fun, dummy_detect_objects, **kwargs)

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
        image = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        labels, = det(None)
        self.assertNotIn(2, np.unique(labels))

    def test_filter_function_empty_result(self):
        """Empty filter result → only background labels."""
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
            for rp in regionprops(labels, intensity_image=image[1]):
                if rp.mean_intensity > 100:
                    good.append(rp.label)
            return good

        def load_multi_channel(f):
            return np.stack([
                np.zeros((50, 60), dtype=np.uint16),  # ch 0
                np.full((50, 60), 200, dtype=np.uint16),  # ch 1 = high expression
            ], axis=0)

        det = build_detector(load_multi_channel, dummy_detect_objects,
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
        image = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        labels, = det(None)
        # Both dummy objects survive (neither touches border)
        unique = np.unique(labels)
        self.assertEqual(len(unique), 3)  # 0, 1, 2


if __name__ == '__main__':
    unittest.main()
