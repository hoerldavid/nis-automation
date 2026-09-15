"""
Unit tests for load_detector_file (user-supplied detector file import).
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from autofrap.core.detection import load_detector_file


# Absolute path to the example detector (avoids path doubling issues)
EXAMPLE_DETECTOR = os.path.join(
    ROOT, 'autofrap', 'detectors', 'example_detector.py')


class TestLoadDetectorFile(unittest.TestCase):
    """Tests for load_detector_file."""

    def test_load_example_detector(self):
        """Can load the example detector file."""
        detection_fun = load_detector_file(EXAMPLE_DETECTOR)
        self.assertTrue(callable(detection_fun))

    def test_load_dummy_detector(self):
        """Can load the dummy detector file."""
        from autofrap import detectors
        path = os.path.join(
            ROOT, 'autofrap', 'detectors', 'dummy_detector.py')
        detection_fun = load_detector_file(path)
        self.assertTrue(callable(detection_fun))
        labels, stim = detection_fun('/dev/null')
        self.assertEqual(labels.shape, (512, 512))

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
