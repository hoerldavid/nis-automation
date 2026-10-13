"""
Tests for the offline detector runner CLI (--list-detector-args):
the parameter listing for composed and hand-rolled detectors, the
module-docstring display, and the exit codes. No image files, no
detection calls - only detector loading and output formatting.
"""
import contextlib
import io
import os
import tempfile
import unittest

import autofrap
from autofrap.detectors import cli

DETECTORS_DIR = os.path.join(os.path.dirname(autofrap.__file__), "detectors")
SIMPLE_SEG = os.path.join(DETECTORS_DIR, "simple_seg_detector.py")
EXAMPLE = os.path.join(DETECTORS_DIR, "example_detector.py")


class TestListDetectorArgs(unittest.TestCase):

    def _run(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(argv)
        return out.getvalue()

    def test_composed_detector_listing(self):
        """build_detector detectors: keys grouped by step, defaults
        from the sub-function signatures, module docstring appended."""
        out = self._run(["--list-detector-args", "--detector", SIMPLE_SEG])
        self.assertIn("simple_seg_detector: tunable parameters", out)
        self.assertIn("detector_fun:", out)
        self.assertIn("cell_sigma", out)
        self.assertIn("default: 16.0", out)
        # tier 2: the module docstring
        self.assertIn("Otsu + watershed detector", out)

    def test_plain_detector_listing(self):
        """hand-rolled detection_fun: no step groups; only the survey
        file parameter -> (none)."""
        out = self._run(["--list-detector-args", "--detector", EXAMPLE])
        self.assertIn("(none)", out)
        self.assertIn("Example custom detector file", out)

    def test_listing_needs_no_image(self):
        """--list-detector-args runs without the positional image."""
        out = self._run(["--list-detector-args", "--detector", SIMPLE_SEG])
        self.assertIn("cell_sigma", out)

    def test_missing_image_without_flag_exits_2(self):
        """the positional image is required unless --list-detector-args."""
        with self.assertRaises(SystemExit) as cm:
            cli.main(["--detector", SIMPLE_SEG])
        self.assertEqual(cm.exception.code, 2)

    def test_broken_detector_exits_1(self):
        """a detector file without a callable detection_fun exits 1."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.py")
            with open(path, "w") as fh:
                fh.write('"doc"\n\ndetection_fun = 42\n')
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--list-detector-args", "--detector", path])
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
