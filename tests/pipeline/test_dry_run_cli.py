"""
Integration test for the pipeline CLI layer and the dry-run tool.

test_offline_pipeline.py drives autofrap_loop_outer() directly; this
module covers the layer above it - argument parsing, detector-file
loading (--detector), the default position build and the CLI exit codes
- end-to-end under FakeNIS with a stub detector. No microscope, no
detector server, no data files: the FakeNIS survey sources are opaque
bytes and the stub detector never reads them. (Per the suite's scope:
the pipeline logic is tested here, not the detectors - read_channel and
load_detector_file have their own unit tests.)
"""
import logging
import os
import tempfile
import unittest

from autofrap.microscope.fake_nis import FakeNIS
from autofrap.pipeline import autofrap as af
from autofrap.pipeline import dry_run
from autofrap.pipeline.autofrap import SPIRAL_DEFAULT_POSITIONS

STUB_DETECTOR = '''\
"""Stub detector for the CLI integration test: fixed canvas, two objects,
left-half stim mask; ignores the survey file (FakeNIS sources are opaque
bytes)."""
import numpy as np

from autofrap.core.image.segmentation import dummy_detect_objects
from autofrap.core.image.mask import half_object_stim_mask


def detection_fun(survey_file):
    image = np.zeros((512, 512), dtype=np.uint16)
    labels = dummy_detect_objects(image)
    return labels, half_object_stim_mask(labels)
'''


class TestBuildArgv(unittest.TestCase):
    """the dry-run tool's argv building (pure - no FakeNIS, no pipeline)"""

    def test_preset_expansion(self):
        argv = dry_run.build_argv('dummy', ['a.nd2'], 'out', 'nm', [])
        i = argv.index('--detector')
        self.assertTrue(argv[i + 1].endswith('example_detector.py'))
        self.assertEqual(argv[argv.index('--out') + 1], 'out')
        self.assertEqual(argv[argv.index('--name') + 1], 'nm')

    def test_cap_defaults_to_source_count(self):
        sources = [f's{i}.nd2' for i in range(6)]
        argv = dry_run.build_argv('dummy', sources, 'out', 'nm', [])
        self.assertEqual(argv[argv.index('--max-positions') + 1], '6')

    def test_cap_at_most_cli_default(self):
        sources = [f's{i}.nd2' for i in range(100)]
        argv = dry_run.build_argv('dummy', sources, 'out', 'nm', [])
        self.assertEqual(argv[argv.index('--max-positions') + 1],
                         str(SPIRAL_DEFAULT_POSITIONS))

    def test_explicit_max_positions_respected(self):
        argv = dry_run.build_argv('dummy', ['a.nd2'], 'out', 'nm',
                                  ['--max-positions', '5'])
        self.assertEqual(argv.count('--max-positions'), 1)
        self.assertEqual(argv[argv.index('--max-positions') + 1], '5')

    def test_explicit_max_positions_equals_form(self):
        argv = dry_run.build_argv('dummy', ['a.nd2'], 'out', 'nm',
                                  ['--max-positions=7'])
        self.assertNotIn('--max-positions', argv)
        self.assertIn('--max-positions=7', argv)

    def test_grid_flags_opt_out(self):
        for flag in ('--grid', '--nx', '--ny'):
            with self.subTest(flag=flag):
                argv = dry_run.build_argv('dummy', ['a.nd2'], 'out', 'nm',
                                          [flag, '2'])
                self.assertNotIn('--max-positions', argv)

    def test_spacing_alone_still_capped(self):
        argv = dry_run.build_argv('dummy', ['a.nd2', 'b.nd2'], 'out', 'nm',
                                  ['--spacing', '0.8'])
        self.assertEqual(argv[argv.index('--max-positions') + 1], '2')


class _ListHandler(logging.Handler):
    """collect (levelname, message) records for assertions"""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


class TestCliEndToEnd(unittest.TestCase):
    """the real pipeline CLI (main) end-to-end under FakeNIS, stub detector"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        # stub detector file, loaded through the CLI's --detector path
        self.detector = os.path.join(self.tmp, 'stub_detector.py')
        with open(self.detector, 'w') as f:
            f.write(STUB_DETECTOR)
        # opaque fake survey sources
        self.sources = []
        for i in range(3):
            p = os.path.join(self.tmp, f'src{i}.nd2')
            with open(p, 'wb') as f:
                f.write(f'fake-nd2-{i}'.encode())
            self.sources.append(p)
        self.out = os.path.join(self.tmp, 'out')

    def _run_cli(self, extra_argv):
        """run the real CLI under FakeNIS.

        Returns (result, log_records): result is None on exit 0;
        SystemExit(1/130) propagates on abort / clean stop.
        """
        argv = ['--detector', self.detector, '--out', self.out,
                '--name', 'cli', '--no-timestamp'] + extra_argv
        # keep the CLI's console logging out of the suite output, and
        # restore the root logger afterwards (main() reconfigures it via
        # basicConfig(force=True))
        logger = logging.getLogger('autofrap')
        handler = _ListHandler()
        logger.addHandler(handler)
        old_propagate = logger.propagate
        logger.propagate = False
        root = logging.getLogger()
        old_handlers = root.handlers[:]
        old_level = root.level
        try:
            with FakeNIS(self.sources):
                result = af.main(argv)
        finally:
            logger.removeHandler(handler)
            logger.propagate = old_propagate
            root.handlers = old_handlers
            root.setLevel(old_level)
        return result, handler.records

    def test_happy_path(self):
        result, records = self._run_cli(['--max-positions', '3'])
        self.assertIsNone(result)  # exit 0
        self.assertTrue(any('Run done' in m for _, m in records))
        run_dir = os.path.join(self.out, 'cli')
        for i in (1, 2, 3):
            for kind in ('survey', 'frap'):
                self.assertTrue(os.path.isfile(os.path.join(
                    run_dir, f'fov{i:02d}_cycle01_{kind}.nd2')),
                    f'fov{i:02d}_cycle01_{kind}.nd2 missing')
            self.assertTrue(os.path.isfile(os.path.join(
                run_dir, f'fov{i:02d}_cycle01_survey_qc.png')),
                f'fov{i:02d} QC overlay missing')

    def test_invalid_name_exits_1(self):
        with self.assertRaises(SystemExit) as cm:
            self._run_cli(['--max-positions', '1', '--name', 'bad name!'])
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
